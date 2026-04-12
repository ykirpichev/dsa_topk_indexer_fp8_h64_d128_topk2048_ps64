"""
DSA Top-K indexer — pure PyTorch, paged flat layout + masks (deep_gemm-style).

Optimizations:
- Page list without full-length phys_page: uniq from bt[:, :num_slots] only.
- Optional chunked matmul + running top-k merge (FIB_CHUNK_TOKENS) to avoid peak [B,H,T] memory.

Tunables: FIB_MATMUL_BF16, FIB_TORCH_COMPILE, FIB_COMPILE_MIN_TOKENS, FIB_TORCH_COMPILE_MODE,
FIB_CHUNK_TOKENS (0 = one-shot matmul; default 4096 when t_eff larger).
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import torch

_USE_BF16_MATMUL = os.environ.get("FIB_MATMUL_BF16", "1").lower() not in ("0", "false", "no")
_COMPILE_ENABLED = os.environ.get("FIB_TORCH_COMPILE", "1").lower() not in ("0", "false", "no")
_COMPILE_MIN_TOKENS = int(os.environ.get("FIB_COMPILE_MIN_TOKENS", "256"))
# Chunk token axis to cap peak activations; 0 disables chunking (always one-shot).
_CHUNK_TOKENS = int(os.environ.get("FIB_CHUNK_TOKENS", "4096"))

_compiled_hot: Optional[Callable] = None


def _dequant_fp8_kv_cache(k_index_cache_fp8: torch.Tensor, out_bf16: bool) -> torch.Tensor:
    """Dequantize FP8 KV cache from deep_gemm layout to float32 or bf16."""
    k_uint8 = k_index_cache_fp8.view(torch.uint8)
    num_pages, page_size, _num_heads, head_dim_sf = k_uint8.shape
    head_dim = head_dim_sf - 4

    kv_flat = k_uint8.view(num_pages, page_size * head_dim_sf)
    fp8_bytes = kv_flat[:, : page_size * head_dim].contiguous()
    fp8_tensor = fp8_bytes.view(num_pages, page_size, head_dim).view(torch.float8_e4m3fn)
    fp8_float = fp8_tensor.to(torch.float32)

    scale_bytes = kv_flat[:, page_size * head_dim :].contiguous()
    scale = scale_bytes.view(num_pages, page_size, 4).view(torch.float32)

    x = fp8_float * scale
    if out_bf16:
        return x.to(torch.bfloat16)
    return x


def _hot_path_nocompile(
    q_mm: torch.Tensor,
    k_eff: torch.Tensor,
    weights: torch.Tensor,
    valid: torch.Tensor,
    global_token: torch.Tensor,
    k_take: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.matmul(q_mm, k_eff.transpose(-1, -2)).float()
    final_scores = (torch.relu(scores) * weights.unsqueeze(-1)).sum(dim=1)
    final_scores = final_scores.masked_fill(~valid, float("-inf"))
    _values, topk_flat = torch.topk(final_scores, k=k_take, dim=-1)
    topk_global = global_token.gather(1, topk_flat)
    good = torch.isfinite(_values) & valid.gather(1, topk_flat)
    return topk_global, good


def _get_compiled_hot() -> Callable:
    global _compiled_hot
    if _compiled_hot is not None:
        return _compiled_hot
    if not (torch.cuda.is_available() and _COMPILE_ENABLED):
        _compiled_hot = _hot_path_nocompile
        return _compiled_hot
    try:
        mode = os.environ.get("FIB_TORCH_COMPILE_MODE", "reduce-overhead")
        _compiled_hot = torch.compile(_hot_path_nocompile, dynamic=True, mode=mode)
    except Exception:
        _compiled_hot = _hot_path_nocompile
    return _compiled_hot


def _build_page_tables(
    bt: torch.Tensor,
    num_pages_total: int,
    k_index_cache_fp8: torch.Tensor,
    use_bf16: bool,
    num_slots: int,
):
    """uniq/dequant/mapper from block_table slice only (no full-length arange)."""
    uniq = torch.unique(bt[:, :num_slots].flatten())
    k_uniq = _dequant_fp8_kv_cache(k_index_cache_fp8[uniq], out_bf16=use_bf16)
    device = bt.device
    mapper = torch.full((num_pages_total,), -1, dtype=torch.long, device=device)
    mapper[uniq] = torch.arange(uniq.numel(), device=device, dtype=torch.long)
    return k_uniq, mapper


def _kernel_chunked(
    q_mm: torch.Tensor,
    weights: torch.Tensor,
    bt: torch.Tensor,
    k_uniq: torch.Tensor,
    mapper: torch.Tensor,
    seq_lens: torch.Tensor,
    batch_size: int,
    page_size: int,
    t_eff: int,
    k_take: int,
    chunk: int,
    device: torch.device,
):
    """Stream token columns in chunks; running top-k merge (exact same scores as one-shot)."""
    best_scores = torch.full((batch_size, k_take), float("-inf"), device=device)
    best_j = torch.full((batch_size, k_take), -1, dtype=torch.long, device=device)
    seq_2d = seq_lens.to(device).unsqueeze(1)

    for t0 in range(0, t_eff, chunk):
        t1 = min(t0 + chunk, t_eff)
        j_chunk = torch.arange(t0, t1, device=device, dtype=torch.long)
        slot_chunk = j_chunk // page_size
        off_chunk = j_chunk % page_size
        phys_chunk = bt[:, slot_chunk]
        row = mapper[phys_chunk]
        off_exp = off_chunk.unsqueeze(0).expand(batch_size, -1)
        k_chunk = k_uniq[row, off_exp].contiguous()

        valid_chunk = j_chunk.unsqueeze(0) < seq_2d
        scores = torch.matmul(q_mm, k_chunk.transpose(-1, -2)).float()
        final_chunk = (torch.relu(scores) * weights.unsqueeze(-1)).sum(dim=1)
        final_chunk = final_chunk.masked_fill(~valid_chunk, float("-inf"))

        chunk_j = j_chunk.unsqueeze(0).expand(batch_size, -1)
        merged_val = torch.cat([best_scores, final_chunk], dim=-1)
        merged_j = torch.cat([best_j, chunk_j], dim=-1)
        best_scores, idx = torch.topk(merged_val, k=k_take, dim=-1)
        best_j = merged_j.gather(1, idx)

    slot_idx = (best_j // page_size).clamp(0, bt.shape[1] - 1)
    phys = torch.gather(bt, 1, slot_idx)
    off_b = best_j % page_size
    topk_global = phys * page_size + off_b
    valid_j = (best_j >= 0) & (best_j < seq_2d.expand_as(best_j))
    good = torch.isfinite(best_scores) & valid_j
    return topk_global, good


@torch.no_grad()
def kernel(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table):
    batch_size, num_index_heads, index_head_dim = q_index_fp8.shape
    num_pages_total, page_size, _, _ = k_index_cache_fp8.shape
    topk = 2048

    assert num_index_heads == 64
    assert index_head_dim == 128
    assert page_size == 64

    device = q_index_fp8.device
    bt = block_table.to(torch.long)

    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    use_bf16 = _USE_BF16_MATMUL and torch.cuda.is_available()
    q = q_index_fp8.to(torch.float32).contiguous()
    q_mm = q.to(torch.bfloat16) if use_bf16 else q

    _, max_num_pages = bt.shape
    t_flat = max_num_pages * page_size

    max_len = int(seq_lens.max().item())
    if max_len == 0:
        return (torch.full((batch_size, topk), -1, dtype=torch.int32, device=device),)

    t_eff = min(t_flat, max_len)
    num_slots = (t_eff + page_size - 1) // page_size
    k_take = min(topk, t_eff)

    k_uniq, mapper = _build_page_tables(bt, num_pages_total, k_index_cache_fp8, use_bf16, num_slots)

    chunk = _CHUNK_TOKENS
    use_chunks = chunk > 0 and t_eff > chunk

    if use_chunks:
        topk_global, good = _kernel_chunked(
            q_mm,
            weights,
            bt,
            k_uniq,
            mapper,
            seq_lens,
            batch_size,
            page_size,
            t_eff,
            k_take,
            chunk,
            device,
        )
    else:
        j = torch.arange(t_eff, device=device, dtype=torch.long)
        slot = j // page_size
        off = j % page_size
        phys_page = bt[:, slot]
        row = mapper[phys_page]
        off_exp = off.unsqueeze(0).expand(batch_size, -1)
        k_flat = k_uniq[row, off_exp].contiguous()
        global_token = (phys_page * page_size + off).contiguous()
        valid = j.unsqueeze(0) < seq_lens.to(device).unsqueeze(1)

        if use_bf16 and t_eff >= _COMPILE_MIN_TOKENS and _COMPILE_ENABLED:
            hot = _get_compiled_hot()
        else:
            hot = _hot_path_nocompile

        topk_global, good = hot(q_mm, k_flat, weights, valid, global_token, k_take)

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
