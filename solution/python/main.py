"""
HF discussion #2 style baseline: **DeepGEMM** ``fp8_paged_mqa_logits`` + **torch.topk**
page-table mapping (not FlashInfer radix top-k — the harness compares indices to the
trace ``definition.reference``, which follows ``torch.topk`` tie order).

Workarounds for undefined tail logits when ``clean_logits=False``: mask columns
``>= seq_len`` with ``-inf``; flatten ``[B,1]`` seq_lens; cap pages / ``max_context_len``
like the Triton indexer; ``clamp`` block-table page ids.

``DSA_BASELINE_FP32_REFERENCE=1``: DeepGEMM off; same Triton gather + FP32 ``bmm`` as
``triton_reference_kernel.py`` (for profiling; may still differ from trace golden).

See: https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest/discussions/2/files
"""

from __future__ import annotations

import os

import torch
import deep_gemm

try:
    from . import triton_reference_kernel as _triton_ref
except ImportError:
    import triton_reference_kernel as _triton_ref


def _flatten_seq_lens(seq_lens: torch.Tensor) -> torch.Tensor:
    """``[B]`` or ``[B, 1]`` → 1-D ``int32`` on device (FlashInfer / schedule)."""
    s = seq_lens
    if s.dim() == 2 and s.size(1) == 1:
        s = s.squeeze(1)
    return s


def _local_tokens_to_global(
    out: torch.Tensor,
    local_idx: torch.Tensor,
    bt_row: torch.Tensor,
    n: int,
    ps: int,
) -> None:
    li = local_idx[:n].to(torch.int64)
    g = bt_row.to(torch.int64)[li // ps] * ps + (li % ps)
    out[:n].copy_(g.to(out.dtype))


def _topk_from_scores_torch(
    scores_2d: torch.Tensor,
    seq_lens_1d: torch.Tensor,
    block_table_i32: torch.Tensor,
    page_size: int,
    topk: int,
) -> torch.Tensor:
    """Same index semantics as ``solution/triton/kernel.py`` (``torch.topk``, sorted)."""
    b, _ = scores_2d.shape
    device = scores_2d.device
    out = torch.full((b, topk), -1, device=device, dtype=torch.int32)
    sl_list = [int(x) for x in seq_lens_1d.cpu()]
    for bi in range(b):
        sl_i = sl_list[bi]
        if sl_i == 0:
            continue
        tk = min(topk, sl_i)
        _, idx = scores_2d[bi, :sl_i].topk(tk, dim=-1, largest=True, sorted=True)
        loc = idx.to(torch.int32).contiguous()
        _local_tokens_to_global(out[bi], loc, block_table_i32[bi], tk, page_size)
    return out


def _reference_topk_fp32(
    q_index_fp8: torch.Tensor,
    k_index_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Triton gather + FP32 ``torch.bmm`` (``triton_reference_kernel``)."""
    out = torch.empty(
        (q_index_fp8.size(0), topk),
        device=q_index_fp8.device,
        dtype=torch.int32,
    )
    _triton_ref.kernel(
        q_index_fp8,
        k_index_cache_fp8,
        weights,
        seq_lens,
        block_table,
        out,
    )
    return out


def _mask_padded_logits(logits: torch.Tensor, context_lens: torch.Tensor) -> torch.Tensor:
    """Invalidate columns past each row's sequence length (undefined if ``clean_logits=False``)."""
    R, lmax = logits.shape
    lens = context_lens.to(device=logits.device, dtype=torch.int32).reshape(R, 1)
    col = torch.arange(lmax, device=logits.device, dtype=torch.int32).view(1, lmax)
    return logits.masked_fill(col >= lens, float("-inf"))


@torch.no_grad()
def run(
    q_index_fp8: torch.Tensor,
    k_index_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
) -> tuple[torch.Tensor]:
    """DeepGEMM FP8 logits + ``torch.topk`` + block-table global indices."""
    batch_size, num_index_heads, index_head_dim = q_index_fp8.shape
    num_pages, page_size, _, _ = k_index_cache_fp8.shape
    topk = 2048

    assert num_index_heads == 64
    assert index_head_dim == 128
    assert page_size == 64

    device = q_index_fp8.device

    if os.environ.get("DSA_BASELINE_FP32_REFERENCE", "").lower() in ("1", "true", "yes"):
        return (
            _reference_topk_fp32(
                q_index_fp8,
                k_index_cache_fp8,
                weights,
                seq_lens,
                block_table,
                topk,
            ),
        )

    q_index_fp8_4d = q_index_fp8.unsqueeze(1)
    k_index_cache_uint8 = k_index_cache_fp8.view(torch.uint8)

    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    seq_lens_i32 = _flatten_seq_lens(seq_lens).to(device=device, dtype=torch.int32).contiguous()
    # Match Triton reference: only use pages needed for max seq len, capped by block_table width.
    sl_cpu = seq_lens_i32.detach().cpu()
    max_sl = int(sl_cpu.max().item()) if sl_cpu.numel() else 0
    if max_sl == 0:
        return (torch.full((batch_size, topk), -1, device=device, dtype=torch.int32),)

    max_pages_needed = (max_sl + page_size - 1) // page_size
    ap = min(max_pages_needed, int(block_table.shape[1]))
    max_context_len = ap * page_size
    block_table_i32 = (
        block_table[:, :ap].to(device=device, dtype=torch.int32).clamp(0, num_pages - 1).contiguous()
    )

    schedule_meta = deep_gemm.get_paged_mqa_logits_metadata(seq_lens_i32, page_size, num_sms)

    # clean_logits=True crashes on some smoke workloads on B200; keep False and mask.
    logits = deep_gemm.fp8_paged_mqa_logits(
        q_index_fp8_4d,
        k_index_cache_uint8,
        weights,
        seq_lens_i32,
        block_table_i32,
        schedule_meta,
        max_context_len,
        clean_logits=False,
    )
    logits = _mask_padded_logits(logits, seq_lens_i32)

    # FlashInfer radix top-k can disagree with the trace reference (``torch.topk``) on ties;
    # the harness compares indices exactly vs ``definition.reference`` output.
    topk_indices = _topk_from_scores_torch(
        logits, seq_lens_i32, block_table_i32, page_size, topk
    )

    return (topk_indices,)


__all__ = ["run"]
