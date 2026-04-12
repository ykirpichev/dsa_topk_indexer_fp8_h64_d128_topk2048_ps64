"""
DSA Top-K indexer — pure PyTorch, paged flat layout + masks (deep_gemm-style).

Perf vs naive full-length matmul:
- Truncate sequence axis to min(context_slots, max(seq_lens)) — same tokens as reference, fewer columns.
- Single batched matmul (no head expand), TF32, optional bf16 matmul + torch.compile.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import torch

_USE_BF16_MATMUL = os.environ.get("FIB_MATMUL_BF16", "1").lower() not in ("0", "false", "no")
_COMPILE_ENABLED = os.environ.get("FIB_TORCH_COMPILE", "1").lower() not in ("0", "false", "no")

_compiled_hot: Optional[Callable] = None


def _dequant_fp8_kv_cache(k_index_cache_fp8: torch.Tensor) -> torch.Tensor:
    """Dequantize FP8 KV cache from deep_gemm layout to float32. See dataset definition."""
    k_uint8 = k_index_cache_fp8.view(torch.uint8)
    num_pages, page_size, _num_heads, head_dim_sf = k_uint8.shape
    head_dim = head_dim_sf - 4

    kv_flat = k_uint8.view(num_pages, page_size * head_dim_sf)
    fp8_bytes = kv_flat[:, : page_size * head_dim].contiguous()
    fp8_tensor = fp8_bytes.view(num_pages, page_size, head_dim).view(torch.float8_e4m3fn)
    fp8_float = fp8_tensor.to(torch.float32)

    scale_bytes = kv_flat[:, page_size * head_dim :].contiguous()
    scale = scale_bytes.view(num_pages, page_size, 4).view(torch.float32)

    return fp8_float * scale


def _hot_path(
    q: torch.Tensor,
    k_eff: torch.Tensor,
    weights: torch.Tensor,
    valid: torch.Tensor,
    global_token: torch.Tensor,
    k_take: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _USE_BF16_MATMUL:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            scores = torch.matmul(q, k_eff.transpose(-1, -2))
        scores = scores.float()
    else:
        scores = torch.matmul(q, k_eff.transpose(-1, -2))

    final_scores = (torch.relu(scores) * weights.unsqueeze(-1)).sum(dim=1)
    final_scores = final_scores.masked_fill(~valid, float("-inf"))

    _values, topk_flat = torch.topk(final_scores, k=k_take, dim=-1)
    topk_global = global_token.gather(1, topk_flat)
    good = torch.isfinite(_values) & valid.gather(1, topk_flat)
    return topk_global, good


def _get_hot_path() -> callable:
    global _compiled_hot
    if _compiled_hot is not None:
        return _compiled_hot
    if torch.cuda.is_available() and _COMPILE_ENABLED:
        try:
            _compiled_hot = torch.compile(_hot_path, dynamic=True, mode="reduce-overhead")
        except Exception:
            _compiled_hot = _hot_path
    else:
        _compiled_hot = _hot_path
    return _compiled_hot


@torch.no_grad()
def kernel(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table):
    batch_size, num_index_heads, index_head_dim = q_index_fp8.shape
    _num_pages, page_size, _, _ = k_index_cache_fp8.shape
    topk = 2048

    assert num_index_heads == 64
    assert index_head_dim == 128
    assert page_size == 64

    device = q_index_fp8.device
    bt = block_table.to(torch.long)

    torch.set_float32_matmul_precision("high")

    q = q_index_fp8.to(torch.float32).contiguous()
    k_all = _dequant_fp8_kv_cache(k_index_cache_fp8)

    _, max_num_pages = bt.shape
    t_flat = max_num_pages * page_size

    max_len = int(seq_lens.max().item())
    if max_len == 0:
        return (torch.full((batch_size, topk), -1, dtype=torch.int32, device=device),)

    t_eff = min(t_flat, max_len)
    j = torch.arange(t_eff, device=device, dtype=torch.long)
    slot = j // page_size
    off = j % page_size

    phys_page = bt[:, slot]
    k_flat = k_all[phys_page, off].contiguous()
    global_token = (phys_page * page_size + off).contiguous()

    valid = j.unsqueeze(0) < seq_lens.to(device).unsqueeze(1)

    k_take = min(topk, t_eff)
    hot = _get_hot_path()
    topk_global, good = hot(q, k_flat, weights, valid, global_token, k_take)

    topk_indices = torch.full((batch_size, topk), -1, dtype=torch.int32, device=device)
    topk_indices[:, :k_take] = torch.where(
        good,
        topk_global,
        torch.tensor(-1, device=device, dtype=torch.long),
    ).to(torch.int32)

    empty = seq_lens == 0
    if empty.any():
        topk_indices[empty] = -1

    return (topk_indices,)
