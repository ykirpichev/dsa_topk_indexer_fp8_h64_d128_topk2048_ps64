# Batched FP8 GEMM helper structured after:
#   https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/blackwell/tutorial_gemm
# Patterns used: `cutlass.cute.compile(..., options="--generate-line-info")`,
# `from_dlpack(..., enable_tvm_ffi=True)` for TVM-FFI runners (flashinfer-bench),
# `tcgen05` + `make_trivial_tiled_mma` like the Blackwell dense GEMM tutorials.

from __future__ import annotations

import functools
import os
from typing import Callable

import cutlass
import cutlass.cute as cute
import cutlass.cute.algorithm as cute_alg
import torch
from cutlass import Float32, Float8E4M3FN, Int32
from cutlass.cute.nvgpu.tcgen05 import CtaGroup, OperandMajorMode, OperandSource
from cutlass.cute.runtime import from_dlpack
from cutlass.utils.blackwell_helpers import make_trivial_tiled_mma

_MM = 64
_K = 128
_N_TILE = 128

_COMPILE_OPTS = "--generate-line-info --enable-tvm-ffi"


class Fp8BatchedMmHsKernel:
    """One CTA per (batch, N-tile); tcgen05 FP8 UMMA, FP32 accumulate (tutorial-style tiling)."""

    @cute.jit
    def __call__(
        self,
        m_q: cute.Tensor,
        m_k: cute.Tensor,
        m_c: cute.Tensor,
        B: Int32,
        S: Int32,
        stream,
    ):
        num_n_tiles = cute.ceil_div(S, Int32(_N_TILE))
        self._device_kernel(m_q, m_k, m_c, B, S, num_n_tiles).launch(
            grid=[B * num_n_tiles, 1, 1],
            block=[128, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def _device_kernel(
        self,
        m_q: cute.Tensor,
        m_k: cute.Tensor,
        m_c: cute.Tensor,
        B: Int32,
        S: Int32,
        num_n_tiles: Int32,
    ):
        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        bi = bidx // num_n_tiles
        n_tile = bidx - bi * num_n_tiles

        tiled_mma = make_trivial_tiled_mma(
            Float8E4M3FN,
            OperandMajorMode.K,
            OperandMajorMode.K,
            Float32,
            CtaGroup.ONE,
            (_MM, _N_TILE),
            OperandSource.SMEM,
        )
        thr_mma = tiled_mma.get_slice(tidx)

        gA = cute.local_tile(m_q, (_MM, _K), (bi, 0))
        gB = cute.local_tile(m_k, (_N_TILE, _K), (bi, n_tile))
        gC = cute.local_tile(m_c, (_MM, _N_TILE), (bi, n_tile))

        tCgA = thr_mma.partition_A(gA)
        tCgB = thr_mma.partition_B(gB)
        tCgC = thr_mma.partition_C(gC)

        tCrC = cute.zeros_like(tCgC, Float32)

        tAgA = cute.local_tile(tCgA, (None, None, _K), (None, None, 0))
        tBgB = cute.local_tile(tCgB, (None, None, _K), (None, None, 0))
        tArA = cute.make_rmem_tensor_like(tAgA, Float8E4M3FN)
        tBrB = cute.make_rmem_tensor_like(tBgB, Float8E4M3FN)
        cute.autovec_copy(tAgA, tArA)
        cute.autovec_copy(tBgB, tBrB)
        cute_alg.gemm(tiled_mma, tCrC, tArA, tBrB, tCrC)

        cute.autovec_copy(tCrC, tCgC)


@functools.cache
def compiled_fp8_batched_mm_hs() -> Callable:
    os.environ.setdefault("CUTE_DSL_ARCH", "sm_100a")

    sym_b = cute.sym_int()
    sym_s = cute.sym_int()

    kobj = Fp8BatchedMmHsKernel()
    m_q = cute.runtime.make_fake_compact_tensor(
        Float8E4M3FN,
        (sym_b, _MM, _K),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    m_k = cute.runtime.make_fake_compact_tensor(
        Float8E4M3FN,
        (sym_b, sym_s, _K),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    m_c = cute.runtime.make_fake_compact_tensor(
        Float32,
        (sym_b, _MM, sym_s),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        kobj,
        m_q,
        m_k,
        m_c,
        sym_b,
        sym_s,
        stream_fake,
        options=_COMPILE_OPTS,
    )


def quantize_rows_f32_to_fp8(x_f32: torch.Tensor) -> torch.Tensor:
    e_max = float(torch.finfo(torch.float8_e4m3fn).max)
    amax = x_f32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / e_max
    return (x_f32 / scale).clamp(-e_max, e_max).to(torch.float8_e4m3fn)


def fp8_batched_mm_hs(q_f32: torch.Tensor, k_f32: torch.Tensor, c_out: torch.Tensor) -> None:
    """c_out[b,h,s] = sum_d q[b,h,d] * k[b,s,d] via Blackwell tcgen05 FP8 UMMA."""
    b, h, d = q_f32.shape
    _, s, d2 = k_f32.shape
    assert h == _MM and d == _K and d2 == _K
    assert c_out.shape == (b, h, s)
    assert s % _N_TILE == 0

    q_fp8 = quantize_rows_f32_to_fp8(q_f32).contiguous()
    k_fp8 = quantize_rows_f32_to_fp8(k_f32).contiguous()

    fn = compiled_fp8_batched_mm_hs()
    # Tutorial-style DLPack bridge; TVM-FFI matches flashinfer-bench isolated workers.
    m_q = from_dlpack(q_fp8, assumed_align=16, enable_tvm_ffi=True)
    m_q.element_type = Float8E4M3FN
    m_k = from_dlpack(k_fp8, assumed_align=16, enable_tvm_ffi=True)
    m_k.element_type = Float8E4M3FN
    m_c = from_dlpack(c_out, assumed_align=16, enable_tvm_ffi=True)

    fn(m_q, m_k, m_c, Int32(b), Int32(s))


__all__ = [
    "Fp8BatchedMmHsKernel",
    "compiled_fp8_batched_mm_hs",
    "fp8_batched_mm_hs",
    "quantize_rows_f32_to_fp8",
    "_MM",
    "_K",
    "_N_TILE",
]
