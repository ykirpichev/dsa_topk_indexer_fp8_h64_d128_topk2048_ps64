"""B200 FP8 batched GEMM for DSA TopK.

- **Default (`MmaFP8Op`)**: 64×128 CTA tile, plain FP8 UMMA (matches H=64).
- **MXF8 (`MmaMXF8Op`)** — opt-in ``DSA_FP8_MXF8_MM=1``: tcgen05 **block-scaled** FP8; CTA **M must be 128**
  (we pad Q to 128×D), **K block = 32** with **Float8E8M0FNU** scales. KV cache stores one **f32 scale per
  token**; we **broadcast** that scale across each token’s K blocks (same value every 32 K elements in SFB),
  which matches “one scale row per token” semantics.

MXF8 uses internal ``tiled_mma_partition`` for **SFA/SFB** (not on ``ThrMma`` yet in public API).
"""

from __future__ import annotations

import functools
import os
from typing import Callable

import cutlass
import cutlass.cute as cute
import cutlass.cute.algorithm as cute_alg
import torch
from cutlass import Float32, Float8E4M3FN, Float8E8M0FNU, Int32
from cutlass._mlir.dialects import cute as _cute_ir
from cutlass.cute.core import _pack_coord
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.utils.blackwell_helpers import (
    make_blockscaled_trivial_tiled_mma,
    make_trivial_tiled_mma,
)

_H = 64
_K = 128
_N_TILE = 128
_MXF8_M = 128  # BlockScaled MXF8 requires M=128 for CtaGroup.ONE
_SF_VEC = 32
_COMPILE_OPTS = "--generate-line-info --enable-tvm-ffi"


def _use_mxf8() -> bool:
    # Experimental: SFB layout must match MXF8 atom tiling; may not match naive [B,S,K] broadcast.
    return os.environ.get("DSA_FP8_MXF8_MM", "").lower() in ("1", "true", "yes")


class Fp8BatchedMmHsKernel:
    """Plain FP8 UMMA, M=64."""

    @cute.jit
    def __call__(
        self,
        m_q: cute.Tensor,
        m_k: cute.Tensor,
        m_c: cute.Tensor,
        b: Int32,
        s: Int32,
        stream,
    ):
        n_tiles = cute.ceil_div(s, Int32(_N_TILE))
        self._k(m_q, m_k, m_c, b, s, n_tiles).launch(
            grid=[b * n_tiles, 1, 1], block=[128, 1, 1], stream=stream
        )

    @cute.kernel
    def _k(
        self,
        m_q: cute.Tensor,
        m_k: cute.Tensor,
        m_c: cute.Tensor,
        b: Int32,
        s: Int32,
        n_tiles: Int32,
    ):
        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        bi = bidx // n_tiles
        n_tile = bidx - bi * n_tiles

        tiled_mma = make_trivial_tiled_mma(
            Float8E4M3FN,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            Float32,
            tcgen05.CtaGroup.ONE,
            (_H, _N_TILE),
            tcgen05.OperandSource.SMEM,
        )
        thr = tiled_mma.get_slice(tidx)
        g_a = cute.local_tile(m_q, (_H, _K), (bi, 0))
        g_b = cute.local_tile(m_k, (_N_TILE, _K), (bi, n_tile))
        g_c = cute.local_tile(m_c, (_H, _N_TILE), (bi, n_tile))
        t_a = thr.partition_A(g_a)
        t_b = thr.partition_B(g_b)
        t_c = thr.partition_C(g_c)
        acc = cute.zeros_like(t_c, Float32)
        t_ag = cute.local_tile(t_a, (None, None, _K), (None, None, 0))
        t_bg = cute.local_tile(t_b, (None, None, _K), (None, None, 0))
        r_a = cute.make_rmem_tensor_like(t_ag, Float8E4M3FN)
        r_b = cute.make_rmem_tensor_like(t_bg, Float8E4M3FN)
        cute.autovec_copy(t_ag, r_a)
        cute.autovec_copy(t_bg, r_b)
        cute_alg.gemm(tiled_mma, acc, r_a, r_b, acc)
        cute.autovec_copy(acc, t_c)


class Fp8Mxf8BatchedMmKernel:
    """MXF8 block-scaled UMMA: M=128 CTA tile, SFA/SFB in TMEM via atom fields."""

    @cute.jit
    def __call__(
        self,
        m_q_pad: cute.Tensor,
        m_k: cute.Tensor,
        m_sfa: cute.Tensor,
        m_sfb: cute.Tensor,
        m_c_pad: cute.Tensor,
        b: Int32,
        s: Int32,
        stream,
    ):
        n_tiles = cute.ceil_div(s, Int32(_N_TILE))
        self._k(m_q_pad, m_k, m_sfa, m_sfb, m_c_pad, b, s, n_tiles).launch(
            grid=[b * n_tiles, 1, 1], block=[128, 1, 1], stream=stream
        )

    @cute.kernel
    def _k(
        self,
        m_q_pad: cute.Tensor,
        m_k: cute.Tensor,
        m_sfa: cute.Tensor,
        m_sfb: cute.Tensor,
        m_c_pad: cute.Tensor,
        b: Int32,
        s: Int32,
        n_tiles: Int32,
    ):
        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        bi = bidx // n_tiles
        n_tile = bidx - bi * n_tiles

        tiled_mma = make_blockscaled_trivial_tiled_mma(
            Float8E4M3FN,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            Float8E8M0FNU,
            _SF_VEC,
            tcgen05.CtaGroup.ONE,
            (_MXF8_M, _N_TILE),
            tcgen05.OperandSource.SMEM,
        )
        thr_idx = _pack_coord(tidx)
        atom = tiled_mma._trait.value

        g_a = cute.local_tile(m_q_pad, (_MXF8_M, _K), (bi, 0))
        g_b = cute.local_tile(m_k, (_N_TILE, _K), (bi, n_tile))
        g_c = cute.local_tile(m_c_pad, (_MXF8_M, _N_TILE), (bi, n_tile))
        g_sfa = cute.local_tile(m_sfa, (_MXF8_M, _K), (bi, 0))
        g_sfb = cute.local_tile(m_sfb, (_N_TILE, _K), (bi, n_tile))

        t_a = _cute_ir.tiled_mma_partition(
            _cute_ir.MmaOperand.A, atom, g_a.value, thr_idx
        )
        t_b = _cute_ir.tiled_mma_partition(
            _cute_ir.MmaOperand.B, atom, g_b.value, thr_idx
        )
        t_c = _cute_ir.tiled_mma_partition(
            _cute_ir.MmaOperand.C, atom, g_c.value, thr_idx
        )
        t_sfa = _cute_ir.tiled_mma_partition(
            _cute_ir.MmaOperand.SFA, atom, g_sfa.value, thr_idx
        )
        t_sfb = _cute_ir.tiled_mma_partition(
            _cute_ir.MmaOperand.SFB, atom, g_sfb.value, thr_idx
        )

        t_a = cute.make_tensor(t_a)
        t_b = cute.make_tensor(t_b)
        t_c = cute.make_tensor(t_c)
        t_sfa = cute.make_tensor(t_sfa)
        t_sfb = cute.make_tensor(t_sfb)

        acc = cute.zeros_like(t_c, Float32)
        t_ag = cute.local_tile(t_a, (None, None, _K), (None, None, 0))
        t_bg = cute.local_tile(t_b, (None, None, _K), (None, None, 0))
        t_sfag = cute.local_tile(t_sfa, (None, None, _K), (None, None, 0))
        t_sfbg = cute.local_tile(t_sfb, (None, None, _K), (None, None, 0))

        r_a = cute.make_rmem_tensor_like(t_ag, Float8E4M3FN)
        r_b = cute.make_rmem_tensor_like(t_bg, Float8E4M3FN)
        r_sfa = cute.make_rmem_tensor_like(t_sfag, Float8E8M0FNU)
        r_sfb = cute.make_rmem_tensor_like(t_sfbg, Float8E8M0FNU)

        cute.autovec_copy(t_ag, r_a)
        cute.autovec_copy(t_bg, r_b)
        cute.autovec_copy(t_sfag, r_sfa)
        cute.autovec_copy(t_sfbg, r_sfb)

        tiled_mma.set(tcgen05.Field.SFA, r_sfa.iterator)
        tiled_mma.set(tcgen05.Field.SFB, r_sfb.iterator)
        cute_alg.gemm(tiled_mma, acc, r_a, r_b, acc)
        cute.autovec_copy(acc, t_c)


@functools.cache
def _compiled_dense() -> Callable:
    os.environ.setdefault("CUTE_DSL_ARCH", "sm_100a")
    sb, ss = cute.sym_int(), cute.sym_int()
    k = Fp8BatchedMmHsKernel()
    m_q = cute.runtime.make_fake_compact_tensor(
        Float8E4M3FN, (sb, _H, _K), stride_order=(2, 1, 0), assumed_align=16
    )
    m_k = cute.runtime.make_fake_compact_tensor(
        Float8E4M3FN, (sb, ss, _K), stride_order=(2, 1, 0), assumed_align=16
    )
    m_c = cute.runtime.make_fake_compact_tensor(
        Float32, (sb, _H, ss), stride_order=(2, 1, 0), assumed_align=16
    )
    st = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(k, m_q, m_k, m_c, sb, ss, st, options=_COMPILE_OPTS)


@functools.cache
def _compiled_mxf8() -> Callable:
    os.environ.setdefault("CUTE_DSL_ARCH", "sm_100a")
    sb, ss = cute.sym_int(), cute.sym_int()
    k = Fp8Mxf8BatchedMmKernel()
    m_q = cute.runtime.make_fake_compact_tensor(
        Float8E4M3FN, (sb, _MXF8_M, _K), stride_order=(2, 1, 0), assumed_align=16
    )
    m_k = cute.runtime.make_fake_compact_tensor(
        Float8E4M3FN, (sb, ss, _K), stride_order=(2, 1, 0), assumed_align=16
    )
    m_sfa = cute.runtime.make_fake_compact_tensor(
        Float8E8M0FNU, (sb, _MXF8_M, _K), stride_order=(2, 1, 0), assumed_align=16
    )
    m_sfb = cute.runtime.make_fake_compact_tensor(
        Float8E8M0FNU, (sb, ss, _K), stride_order=(2, 1, 0), assumed_align=16
    )
    m_c = cute.runtime.make_fake_compact_tensor(
        Float32, (sb, _MXF8_M, ss), stride_order=(2, 1, 0), assumed_align=16
    )
    st = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        k, m_q, m_k, m_sfa, m_sfb, m_c, sb, ss, st, options=_COMPILE_OPTS
    )


def _f32_to_e8m0(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float32).to(torch.float8_e8m0fnu)


def _prep_mxf8_inputs(
    q_fp8: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale_f32: torch.Tensor,
    q_scale_f32: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad Q/C to M=128; build SFA/SFB e8m0 (K broadcast per token for SFB)."""
    b, h, d = q_fp8.shape
    _, s, d2 = k_fp8.shape
    assert h == _H and d == _K and d2 == _K and k_scale_f32.shape == (b, s)

    q_pad = torch.zeros((b, _MXF8_M, _K), device=q_fp8.device, dtype=q_fp8.dtype)
    q_pad[:, :_H, :].copy_(q_fp8)

    if q_scale_f32 is None:
        sfa = torch.ones((b, _MXF8_M, _K), device=q_fp8.device, dtype=torch.float32)
    else:
        assert q_scale_f32.shape == (b, h)
        sfa = torch.ones((b, _MXF8_M, _K), device=q_fp8.device, dtype=torch.float32)
        sfa[:, :_H, :].copy_(q_scale_f32.unsqueeze(2).expand(-1, -1, _K))

    sfb = k_scale_f32.unsqueeze(2).expand(-1, -1, _K)
    sfa_e = _f32_to_e8m0(sfa)
    sfb_e = _f32_to_e8m0(sfb)

    logits_pad = torch.empty((b, _MXF8_M, s), device=q_fp8.device, dtype=torch.float32)
    return q_pad, k_fp8, sfa_e, sfb_e, logits_pad


def fp8_batched_mm_hs(
    q_fp8: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale_f32: torch.Tensor,
    c_out: torch.Tensor,
    *,
    q_scale_f32: torch.Tensor | None = None,
) -> None:
    """``c_out[b,:H,:]`` = matmul with optional MXF8 block scaling from ``k_scale_f32`` [B,S]."""
    b, h, d = q_fp8.shape
    _, s, d2 = k_fp8.shape
    assert (
        h == _H
        and d == _K
        and d2 == _K
        and c_out.shape == (b, h, s)
        and s % _N_TILE == 0
    )

    if _use_mxf8():
        q_pad, k8, sfa_e, sfb_e, logits_pad = _prep_mxf8_inputs(
            q_fp8.contiguous(),
            k_fp8.contiguous(),
            k_scale_f32.contiguous(),
            q_scale_f32,
        )
        fn = _compiled_mxf8()
        mq = from_dlpack(q_pad, assumed_align=16, enable_tvm_ffi=True)
        mq.element_type = Float8E4M3FN
        mk = from_dlpack(k8, assumed_align=16, enable_tvm_ffi=True)
        mk.element_type = Float8E4M3FN
        msfa = from_dlpack(sfa_e, assumed_align=16, enable_tvm_ffi=True)
        msfa.element_type = Float8E8M0FNU
        msfb = from_dlpack(sfb_e, assumed_align=16, enable_tvm_ffi=True)
        msfb.element_type = Float8E8M0FNU
        mc = from_dlpack(logits_pad, assumed_align=16, enable_tvm_ffi=True)
        fn(mq, mk, msfa, msfb, mc, Int32(b), Int32(s))
        torch.cuda.synchronize()
        c_out.copy_(logits_pad[:, :_H, :])
        return

    q8 = q_fp8.contiguous()
    k8 = k_fp8.contiguous()
    fn = _compiled_dense()
    mq = from_dlpack(q8, assumed_align=16, enable_tvm_ffi=True)
    mq.element_type = Float8E4M3FN
    mk = from_dlpack(k8, assumed_align=16, enable_tvm_ffi=True)
    mk.element_type = Float8E4M3FN
    mc = from_dlpack(c_out, assumed_align=16, enable_tvm_ffi=True)
    fn(mq, mk, mc, Int32(b), Int32(s))
    torch.cuda.synchronize()
    c_out.mul_(k_scale_f32.unsqueeze(1))


__all__ = [
    "fp8_batched_mm_hs",
    "_H",
    "_K",
    "_N_TILE",
    "_use_mxf8",
]
