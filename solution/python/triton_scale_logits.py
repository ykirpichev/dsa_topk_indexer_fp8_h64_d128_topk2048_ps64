"""Triton: in-place ``logits[b,h,s] *= k_scale[b,s]`` (broadcast on head dim)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _scale_logits_kernel(
    logits_ptr,
    scale_ptr,
    B,
    H,
    S,
    stride_l_b,
    stride_l_h,
    stride_l_s,
    stride_sc_b,
    stride_sc_s,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK, dtype=tl.int64)
    n = B.to(tl.int64) * H.to(tl.int64) * S.to(tl.int64)
    mask = offs < n
    s_idx = offs % S.to(tl.int64)
    t1 = offs // S.to(tl.int64)
    h_idx = t1 % H.to(tl.int64)
    b_idx = t1 // H.to(tl.int64)
    sc_off = b_idx * stride_sc_b.to(tl.int64) + s_idx * stride_sc_s.to(tl.int64)
    sc = tl.load(scale_ptr + sc_off, mask=mask, other=0.0)
    l_off = b_idx * stride_l_b.to(tl.int64) + h_idx * stride_l_h.to(tl.int64) + s_idx * stride_l_s.to(tl.int64)
    v = tl.load(logits_ptr + l_off, mask=mask, other=0.0)
    tl.store(logits_ptr + l_off, v * sc, mask=mask)


def scale_logits_by_k(logits: torch.Tensor, k_scale: torch.Tensor) -> None:
    b, h, s = logits.shape
    assert k_scale.shape == (b, s) and logits.is_cuda
    logits_c = logits if logits.is_contiguous() else logits.contiguous()
    if logits_c.data_ptr() != logits.data_ptr():
        raise RuntimeError("scale_logits_by_k expects contiguous logits")
    n = b * h * s
    BLOCK = 256
    grid = (triton.cdiv(n, BLOCK),)
    _scale_logits_kernel[grid](
        logits_c,
        k_scale,
        b,
        h,
        s,
        logits_c.stride(0),
        logits_c.stride(1),
        logits_c.stride(2),
        k_scale.stride(0),
        k_scale.stride(1),
        BLOCK=BLOCK,
        num_warps=4,
    )


__all__ = ["scale_logits_by_k"]
