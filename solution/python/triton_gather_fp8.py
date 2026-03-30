"""Triton: paged KV gather → FP8 [B,S,D] (raw cache bytes) + per-token K scales [B,S]. B200 D=128."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_D: tl.constexpr = 128


@triton.jit
def _gather_k_fp8_kernel(
    cache_ptr,
    bt_ptr,
    k_fp8_out_ptr,
    k_scale_out_ptr,
    B,
    S,
    P,
    PS: tl.constexpr,
    HDS: tl.constexpr,
    stride_bt0,
    stride_bt1,
    stride_k0,
    stride_k1,
    stride_k2,
    stride_sc0,
    stride_sc1,
):
    token_idx = tl.program_id(0)
    d_offs = tl.arange(0, _D)

    b = token_idx // S
    s = token_idx % S
    page_col = s // PS
    bt_addr = bt_ptr + b * stride_bt0 + page_col * stride_bt1
    page_id = tl.load(bt_addr)
    page_id = tl.minimum(page_id, P - 1)
    t = s % PS

    page_byte = page_id.to(tl.int64) * (PS * HDS)
    fp8_byte = page_byte + t.to(tl.int64) * _D + d_offs.to(tl.int64)
    u8 = tl.load(cache_ptr + fp8_byte)
    out_off = token_idx.to(tl.int64) * _D + d_offs.to(tl.int64)
    tl.store(k_fp8_out_ptr + out_off, u8)

    scale_off = page_byte + PS * _D + t.to(tl.int64) * 4
    sc = tl.load((cache_ptr + scale_off).to(tl.pointer_type(tl.float32)))
    sc_addr = k_scale_out_ptr + b * stride_sc0 + s * stride_sc1
    tl.store(sc_addr, sc)


def gather_k_fp8_scaled(
    cache_u8: torch.Tensor,
    bt_i32: torch.Tensor,
    b: int,
    s_pad: int,
    p: int,
    ps: int,
    hds: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns ``k_fp8`` [B,S,D] uint8 view of e4m3, ``k_scale`` [B,S] float32."""
    dev = cache_u8.device
    k_u8 = torch.empty((b, s_pad, _D), device=dev, dtype=torch.uint8)
    k_sc = torch.empty((b, s_pad), device=dev, dtype=torch.float32)
    grid = (b * s_pad,)
    _gather_k_fp8_kernel[grid](
        cache_u8,
        bt_i32,
        k_u8,
        k_sc,
        b,
        s_pad,
        p,
        ps,
        hds,
        bt_i32.stride(0),
        bt_i32.stride(1),
        k_u8.stride(0),
        k_u8.stride(1),
        k_u8.stride(2),
        k_sc.stride(0),
        k_sc.stride(1),
    )
    return k_u8, k_sc


__all__ = ["gather_k_fp8_scaled", "_D"]
