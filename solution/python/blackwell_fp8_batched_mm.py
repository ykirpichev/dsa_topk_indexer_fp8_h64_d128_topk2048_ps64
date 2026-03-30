"""B200 batched FP8 GEMM: tcgen05 FP8 UMMA, FP32 acc (GMEM→RMEM operands).

K-scale broadcast is applied by ``triton_scale_logits.scale_logits_by_k`` (fused Triton epilogue)."""

from __future__ import annotations

import functools
import os
from typing import Callable

import cutlass
import cutlass.cute as cute
import cutlass.cute.algorithm as cute_alg
import torch
from cutlass import Float32, Float8E4M3FN, Int32
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.utils.blackwell_helpers import make_trivial_tiled_mma

_MM, _K, _N_TILE = 64, 128, 128
_COMPILE_OPTS = "--generate-line-info --enable-tvm-ffi"


class Fp8BatchedMmHsKernel:
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
            (_MM, _N_TILE),
            tcgen05.OperandSource.SMEM,
        )
        thr = tiled_mma.get_slice(tidx)
        g_a = cute.local_tile(m_q, (_MM, _K), (bi, 0))
        g_b = cute.local_tile(m_k, (_N_TILE, _K), (bi, n_tile))
        g_c = cute.local_tile(m_c, (_MM, _N_TILE), (bi, n_tile))
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


@functools.cache
def _compiled() -> Callable:
    os.environ.setdefault("CUTE_DSL_ARCH", "sm_100a")
    sb, ss = cute.sym_int(), cute.sym_int()
    k = Fp8BatchedMmHsKernel()
    m_q = cute.runtime.make_fake_compact_tensor(
        Float8E4M3FN, (sb, _MM, _K), stride_order=(2, 1, 0), assumed_align=16
    )
    m_k = cute.runtime.make_fake_compact_tensor(
        Float8E4M3FN, (sb, ss, _K), stride_order=(2, 1, 0), assumed_align=16
    )
    m_c = cute.runtime.make_fake_compact_tensor(
        Float32, (sb, _MM, ss), stride_order=(2, 1, 0), assumed_align=16
    )
    st = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(k, m_q, m_k, m_c, sb, ss, st, options=_COMPILE_OPTS)


def fp8_batched_mm_hs(q_fp8: torch.Tensor, k_fp8: torch.Tensor, c_out: torch.Tensor) -> None:
    b, h, d = q_fp8.shape
    _, s, d2 = k_fp8.shape
    assert h == _MM and d == _K and d2 == _K and c_out.shape == (b, h, s) and s % _N_TILE == 0

    q8 = q_fp8.contiguous()
    k8 = k_fp8.contiguous()
    fn = _compiled()
    mq = from_dlpack(q8, assumed_align=16, enable_tvm_ffi=True)
    mq.element_type = Float8E4M3FN
    mk = from_dlpack(k8, assumed_align=16, enable_tvm_ffi=True)
    mk.element_type = Float8E4M3FN
    mc = from_dlpack(c_out, assumed_align=16, enable_tvm_ffi=True)
    fn(mq, mk, mc, Int32(b), Int32(s))


__all__ = ["fp8_batched_mm_hs", "_MM", "_K", "_N_TILE"]
