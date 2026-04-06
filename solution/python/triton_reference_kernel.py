"""
Copy of ``solution/triton/kernel.py`` for packaging with ``main.py``.

Used by ``DSA_BASELINE_FP32_REFERENCE=1`` so Modal runs can call the same Triton
gather + FP32 path as the triton submission without a second language pack.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_H = 64
_D = 128
_PS = 64
_K_TOPK = 2048


@triton.jit
def _gather_dequant_kernel(
    cache_ptr,
    bt_ptr,
    k_out_ptr,
    B,
    S,
    D: tl.constexpr,
    P,
    PS: tl.constexpr,
    HDS: tl.constexpr,
    actual_pages,
    stride_bt0,
    stride_bt1,
):
    token_idx = tl.program_id(0)
    d_offs = tl.arange(0, D)

    b = token_idx // S
    s = token_idx % S

    page_col = s // PS
    bt_addr = bt_ptr + b * stride_bt0 + page_col * stride_bt1
    page_id = tl.load(bt_addr)
    page_id = tl.minimum(page_id, P - 1)

    t = s % PS
    page_byte = page_id.to(tl.int64) * (PS * HDS)
    fp8_byte = page_byte + t.to(tl.int64) * D + d_offs.to(tl.int64)

    u8 = tl.load(cache_ptr + fp8_byte)
    fp8v = u8.to(tl.float8e4nv, bitcast=True)
    fv = fp8v.to(tl.float32)

    scale_off = page_byte + PS * D + t.to(tl.int64) * 4
    scale = tl.load((cache_ptr + scale_off).to(tl.pointer_type(tl.float32)))

    out_off = token_idx.to(tl.int64) * D + d_offs.to(tl.int64)
    tl.store(k_out_ptr + out_off, fv * scale)


@torch.inference_mode()
def kernel(
    q_index_fp8: torch.Tensor,
    k_index_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    topk_indices: torch.Tensor,
) -> None:
    assert q_index_fp8.is_cuda
    device = q_index_fp8.device

    B, H, D = q_index_fp8.shape
    P, PS, _, HDS = k_index_cache_fp8.shape
    K_topk = topk_indices.size(1)

    assert H == _H and D == _D and PS == _PS and K_topk == _K_TOPK
    assert HDS == D + 4

    topk_indices.fill_(-1)

    if B == 0:
        return

    seq_cpu = seq_lens.detach().to("cpu")
    if seq_cpu.dtype in (torch.int32, torch.int64):
        sl_list = seq_cpu.view(-1).tolist()
    else:
        sl_list = [int(x) for x in seq_cpu.view(-1)]
    max_seq_len = max(sl_list) if sl_list else 0
    if max_seq_len == 0:
        return

    max_pages_needed = (max_seq_len + PS - 1) // PS
    actual_pages = min(max_pages_needed, block_table.size(1))
    S = actual_pages * PS

    bt = block_table[:, :actual_pages].to(device=device, dtype=torch.int32).clamp(0, P - 1).contiguous()
    cache_u8 = (
        k_index_cache_fp8
        if k_index_cache_fp8.dtype == torch.uint8
        else k_index_cache_fp8.view(torch.uint8)
    ).contiguous()

    K_batched = torch.empty((B, S, D), dtype=torch.float32, device=device)

    grid = (B * S,)
    _gather_dequant_kernel[grid](
        cache_u8,
        bt,
        K_batched,
        B,
        S,
        D,
        P,
        PS,
        HDS,
        actual_pages,
        bt.stride(0),
        bt.stride(1),
    )

    q_f32 = q_index_fp8.to(torch.float32)
    logits = torch.bmm(q_f32, K_batched.transpose(1, 2))
    weighted = torch.relu(logits) * weights.unsqueeze(2).contiguous()

    for b in range(B):
        sl = sl_list[b]
        if sl == 0:
            continue
        actual_topk = min(K_topk, sl)
        scores = weighted[b, :, :sl].sum(dim=0)
        _, topk_local = torch.topk(scores, actual_topk, dim=-1, largest=True, sorted=True)
        topk_local_i32 = topk_local.to(torch.int32)

        bt_b = bt[b]
        page_idx = topk_local_i32 // PS
        offset = topk_local_i32 % PS
        global_page = bt_b[page_idx.long()]
        topk_global = global_page * PS + offset

        topk_indices[b, :actual_topk] = topk_global
