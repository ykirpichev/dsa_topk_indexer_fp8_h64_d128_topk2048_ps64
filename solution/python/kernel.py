"""
B200 Python: Triton FP8 gather + CuTe FP8 UMMA + **Triton** fused ``logits *= k_scale``.

``DSA_FP8_TMMA_MM=1``: UMMA then ``triton_scale_logits.scale_logits_by_k`` (no fragile CuTe broadcast).
"""

from __future__ import annotations

import os

import torch

try:
    from .blackwell_fp8_batched_mm import fp8_batched_mm_hs as _fp8_mm
    from .blackwell_fp8_batched_mm import _N_TILE as _N_TILE_FP8
    from .triton_gather_fp8 import gather_k_fp8_scaled as _gather_k_fp8
    from .triton_scale_logits import scale_logits_by_k as _scale_logits
except ImportError:
    from blackwell_fp8_batched_mm import fp8_batched_mm_hs as _fp8_mm
    from blackwell_fp8_batched_mm import _N_TILE as _N_TILE_FP8
    from triton_gather_fp8 import gather_k_fp8_scaled as _gather_k_fp8
    from triton_scale_logits import scale_logits_by_k as _scale_logits

_USE_FP8_TMMA = os.environ.get("DSA_FP8_TMMA_MM", "").lower() in ("1", "true", "yes")


def _pages_fp32(cache_u8: torch.Tensor, p: int, ps: int, d: int, hds: int) -> torch.Tensor:
    kv = cache_u8.view(torch.uint8).view(p, ps * hds)
    fp8 = kv[:, : ps * d].contiguous().view(p, ps, d).view(torch.float8_e4m3fn).to(torch.float32)
    sc = kv[:, ps * d :].contiguous().view(p, ps, 4).view(torch.float32)
    return fp8 * sc


def _gather_k_f32(
    cache_u8: torch.Tensor,
    bt_i32: torch.Tensor,
    b: int,
    s_pad: int,
    d: int,
    p: int,
    ps: int,
    hds: int,
) -> torch.Tensor:
    dev = cache_u8.device
    k_all = _pages_fp32(cache_u8, p, ps, d, hds)
    bb = torch.arange(b, device=dev, dtype=torch.int64).view(b, 1).expand(b, s_pad)
    ss = torch.arange(s_pad, device=dev, dtype=torch.int64).view(1, s_pad).expand(b, s_pad)
    pid = bt_i32[bb, ss // ps].to(torch.int64).clamp(0, p - 1)
    return k_all[pid, ss % ps]


def _local_tokens_to_global(out: torch.Tensor, local_idx: torch.Tensor, bt_row: torch.Tensor, n: int, ps: int) -> None:
    li = local_idx[:n].to(torch.int64)
    g = bt_row.to(torch.int64)[li // ps] * ps + (li % ps)
    out[:n].copy_(g.to(out.dtype))


@torch.no_grad()
def kernel(
    q_index_fp8: torch.Tensor,
    k_index_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    topk_indices: torch.Tensor,
) -> None:
    b = int(q_index_fp8.size(0))
    n_heads = int(q_index_fp8.size(1))
    d = int(q_index_fp8.size(2))
    p = int(k_index_cache_fp8.size(0))
    ps = int(k_index_cache_fp8.size(1))
    hds = int(k_index_cache_fp8.size(3))
    k_topk = int(topk_indices.size(1))

    topk_indices.fill_(-1)
    cache_u8 = (
        k_index_cache_fp8
        if k_index_cache_fp8.dtype == torch.uint8
        else k_index_cache_fp8.view(torch.uint8)
    ).contiguous()

    sl = [int(x) for x in seq_lens.cpu()]
    if not sl or max(sl) == 0:
        return

    ap = min((max(sl) + ps - 1) // ps, int(block_table.size(1)))
    s_pad = ap * ps
    bt = block_table[:, :ap].to(torch.int32).clamp(0, p - 1).contiguous()
    dev = q_index_fp8.device

    if _USE_FP8_TMMA and s_pad % _N_TILE_FP8 == 0:
        k_u8, k_scale = _gather_k_fp8(cache_u8, bt, b, s_pad, p, ps, hds)
        k_fp8 = k_u8.view(torch.float8_e4m3fn)
        q_fp8 = q_index_fp8.view(torch.float8_e4m3fn) if q_index_fp8.dtype != torch.float8_e4m3fn else q_index_fp8
        logits = torch.empty((b, n_heads, s_pad), device=dev, dtype=torch.float32)
        _fp8_mm(q_fp8.contiguous(), k_fp8.contiguous(), logits)
        _scale_logits(logits, k_scale)
    else:
        k_b = torch.empty((b, s_pad, d), device=dev, dtype=torch.float32)
        k_b.copy_(_gather_k_f32(cache_u8, bt, b, s_pad, d, p, ps, hds))
        q = q_index_fp8.to(torch.float32)
        logits = torch.bmm(q, k_b.transpose(1, 2).contiguous())

    col = torch.arange(s_pad, device=dev, dtype=torch.int64).view(1, s_pad)
    sl_t = seq_lens.to(dev).to(torch.int64).view(b, 1)
    valid = col < sl_t
    w = logits.relu() * weights.unsqueeze(2)
    scores = (w * valid.unsqueeze(1).to(w.dtype)).sum(dim=1).masked_fill(~valid, float("-inf"))

    for bi in range(b):
        sl_i = sl[bi]
        if sl_i == 0:
            continue
        tk = min(k_topk, sl_i)
        _, idx = scores[bi, :].topk(tk, dim=-1, largest=True, sorted=True)
        loc = idx.to(torch.int32).contiguous()
        _local_tokens_to_global(topk_indices[bi], loc, bt[bi], tk, ps)


__all__ = ["kernel"]
