# Batched FP8 GEMM helper structured after:
#   https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/blackwell/tutorial_gemm
# and CuTe **experimental** TMA APIs (see
#   examples/python/CuTeDSL/experimental/blackwell/dense_gemm_cute_pipeline.py):
#   `cute.experimental.jit` / `cute.experimental.kernel`, `cute_ext.tma_load`, `get_cta_v_map_ab`.
#
# Patterns: `cutlass.cute.compile(..., --enable-tvm-ffi)`, `from_dlpack(..., enable_tvm_ffi=True)`,
# `tcgen05` UMMA + TMEM accumulator (tutorial fp16_gemm_0-style mainloop).

from __future__ import annotations

import functools
import os
from typing import Callable

import cutlass
import cutlass.cute as cute
import cutlass.cute.algorithm as cute_alg
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import torch
from cutlass import Float32, Float8E4M3FN, Int32
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.utils.blackwell_helpers import make_trivial_tiled_mma, make_smem_layout_a, make_smem_layout_b

_MM = 64
_K = 128
_N_TILE = 128

_COMPILE_OPTS = "--generate-line-info --enable-tvm-ffi"

# `cutlass.cute.experimental` is only shipped for CUDA toolkit 13.1+ (stub otherwise).
try:
    import cutlass.cute.experimental as cute_ext  # noqa: F401

    _CUTE_EXPERIMENTAL_AVAILABLE = True
except NotImplementedError:
    cute_ext = None  # type: ignore[assignment]
    _CUTE_EXPERIMENTAL_AVAILABLE = False

ab_stages = 2
acc_stage = 1
threads_per_cta = 128


@cute.struct
class _SharedStorage:
    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ab_stages * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, acc_stage * 2]
    tmem_holding_buf: cutlass.Int32


class Fp8BatchedMmHsKernel:
    """GMEM→RMEM path: one CTA per (batch, N-tile); tcgen05 FP8 UMMA (no TMA setup)."""

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
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            Float32,
            tcgen05.CtaGroup.ONE,
            (_MM, _N_TILE),
            tcgen05.OperandSource.SMEM,
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


if _CUTE_EXPERIMENTAL_AVAILABLE:

    class Fp8BatchedMmHsExperimentalKernel:
        """TMA via `cute_ext.tma_load` + SMEM UMMA + TMEM acc; GMEM tensors M×K×L / N×K×L / M×N×L."""

        mma_tiler_mnk = (_MM, _N_TILE, _K)

        @cute_ext.jit
        def __call__(
            self,
            m_a_mkl: cute.Tensor,
            m_b_nkl: cute.Tensor,
            m_c_mnl: cute.Tensor,
            stream,
        ):
            tiled_mma = make_trivial_tiled_mma(
                Float8E4M3FN,
                tcgen05.OperandMajorMode.K,
                tcgen05.OperandMajorMode.K,
                Float32,
                tcgen05.CtaGroup.ONE,
                (_MM, _N_TILE),
                tcgen05.OperandSource.SMEM,
            )
            a_smem_layout = make_smem_layout_a(
                tiled_mma, self.mma_tiler_mnk, m_a_mkl.element_type, ab_stages
            )
            b_smem_layout = make_smem_layout_b(
                tiled_mma, self.mma_tiler_mnk, m_b_nkl.element_type, ab_stages
            )
            a_smem_layout_one_stage = cute.select(a_smem_layout, mode=[0, 1, 2])
            b_smem_layout_one_stage = cute.select(b_smem_layout, mode=[0, 1, 2])

            op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
            tma_atom_a, m_a_for_tma = cpasync.make_tiled_tma_atom_A(
                op,
                m_a_mkl,
                a_smem_layout_one_stage,
                self.mma_tiler_mnk,
                tiled_mma,
            )
            tma_atom_b, m_b_for_tma = cpasync.make_tiled_tma_atom_B(
                op,
                m_b_nkl,
                b_smem_layout_one_stage,
                self.mma_tiler_mnk,
                tiled_mma,
            )

            m_shape = m_c_mnl.shape[0]
            n_shape = m_c_mnl.shape[1]
            grid_shape = cute.ceil_div((m_shape, n_shape, 1), self.mma_tiler_mnk[:2])

            self._device_kernel(
                tiled_mma,
                tma_atom_a,
                m_a_for_tma,
                tma_atom_b,
                m_b_for_tma,
                m_c_mnl,
                a_smem_layout,
                b_smem_layout,
            ).launch(
                grid=grid_shape,
                block=(threads_per_cta, 1, 1),
                stream=stream,
            )

        @cute_ext.kernel
        def _device_kernel(
            self,
            tiled_mma: cute.TiledMma,
            tma_atom_a: cute.CopyAtom,
            m_a_mkl_tma: cute.Tensor,
            tma_atom_b: cute.CopyAtom,
            m_b_nkl_tma: cute.Tensor,
            m_c_mnl: cute.Tensor,
            a_smem_layout: cute.ComposedLayout,
            b_smem_layout: cute.ComposedLayout,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            warp_idx = cute.arch.warp_idx()
            warp_idx = cute.arch.make_warp_uniform(warp_idx)
            bidx, bidy, _ = cute.arch.block_idx()
            mma_coord_mnk = (bidx, bidy, None)

            smem = utils.SmemAllocator()
            storage = smem.allocate(_SharedStorage)
            s_a = smem.allocate_tensor(
                element_type=Float8E4M3FN,
                layout=a_smem_layout.outer,
                byte_alignment=128,
                swizzle=a_smem_layout.inner,
            )
            s_b = smem.allocate_tensor(
                element_type=Float8E4M3FN,
                layout=b_smem_layout.outer,
                byte_alignment=128,
                swizzle=b_smem_layout.inner,
            )

            tmem_alloc_barrier = pipeline.NamedBarrier(
                barrier_id=1,
                num_threads=threads_per_cta,
            )
            tmem = utils.TmemAllocator(
                storage.tmem_holding_buf,
                barrier_for_retrieve=tmem_alloc_barrier,
            )
            tmem.allocate(512)

            if warp_idx == 0:
                cpasync.prefetch_descriptor(tma_atom_a)
                cpasync.prefetch_descriptor(tma_atom_b)

            num_tma_copy_bytes = cute.size_in_bytes(
                Float8E4M3FN, cute.select(a_smem_layout, mode=[0, 1, 2])
            ) + cute.size_in_bytes(Float8E4M3FN, cute.select(b_smem_layout, mode=[0, 1, 2]))
            ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
                num_stages=ab_stages,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                tx_count=num_tma_copy_bytes,
                barrier_storage=storage.ab_mbar_ptr.data_ptr(),
            ).make_participants()
            acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
                num_stages=acc_stage,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    threads_per_cta,
                ),
                barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            ).make_participants()

            g_a = cute.local_tile(
                m_a_mkl_tma, self.mma_tiler_mnk, mma_coord_mnk, proj=(1, None, 1)
            )
            g_b = cute.local_tile(
                m_b_nkl_tma, self.mma_tiler_mnk, mma_coord_mnk, proj=(None, 1, 1)
            )
            g_c = cute.local_tile(
                m_c_mnl, self.mma_tiler_mnk, mma_coord_mnk, proj=(1, 1, None)
            )
            thr_mma = tiled_mma.get_slice(0)
            t_cg_a = thr_mma.partition_A(g_a)
            t_cg_b = thr_mma.partition_B(g_b)
            t_cg_c = thr_mma.partition_C(g_c)
            t_cr_a = tiled_mma.make_fragment_A(s_a)
            t_cr_b = tiled_mma.make_fragment_B(s_b)
            acc_shape = tiled_mma.partition_shape_C(self.mma_tiler_mnk[:2])
            t_ct_acc = tiled_mma.make_fragment_C(acc_shape)

            t_as_a, t_ag_a = cpasync.tma_partition(
                tma_atom_a,
                0,
                cute.make_layout(1),
                cute.group_modes(s_a, 0, 3),
                cute.group_modes(t_cg_a, 0, 3),
            )
            t_bs_b, t_bg_b = cpasync.tma_partition(
                tma_atom_b,
                0,
                cute.make_layout(1),
                cute.group_modes(s_b, 0, 3),
                cute.group_modes(t_cg_b, 0, 3),
            )

            tma_operation_type = cute_ext.OperationTypeEnum.SM90_TMA_LOAD
            a_cta_v_map = cute_ext.get_cta_v_map_ab(
                m_a_mkl_tma, self.mma_tiler_mnk, tiled_mma, "A"
            )
            b_cta_v_map = cute_ext.get_cta_v_map_ab(
                m_b_nkl_tma, self.mma_tiler_mnk, tiled_mma, "B"
            )

            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            t_ct_acc = cute.make_tensor(tmem_ptr, t_ct_acc.layout)

            subtile_cnt = 4
            epi_tiler = (
                (
                    cute.size(t_ct_acc, mode=[0, 0]),
                    cute.size(t_ct_acc, mode=[0, 1]) // subtile_cnt,
                ),
            )
            t_ct_acc_epi = cute.zipped_divide(t_ct_acc, epi_tiler)
            g_c_epi = cute.zipped_divide(t_cg_c, epi_tiler)

            tmem_atom = cute.make_copy_atom(
                tcgen05.Ld32x32bOp(tcgen05.Repetition.x64),
                Float32,
            )
            tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_atom, t_ct_acc_epi[None, 0])
            tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)
            t_ct_c = tmem_thr_copy.partition_S(t_ct_acc_epi)
            t_cg_c_epi = tmem_thr_copy.partition_D(g_c_epi)
            t_cr_acc = cute.make_rmem_tensor(t_cg_c_epi[None, None, 0].shape, Float32)

            num_k_tiles = cute.size(g_a, mode=[2])
            if warp_idx == 0:
                acc_empty = acc_producer.acquire_and_advance()
                for _k_tile_idx in cutlass.range(
                    num_k_tiles, prefetch_stages=ab_stages - 2
                ):
                    ab_empty = ab_producer.acquire_and_advance()
                    g_a_k = t_ag_a[(None, ab_empty.count)]
                    g_b_k = t_bg_b[(None, ab_empty.count)]
                    cute_ext.tma_load(
                        g_a_k,
                        t_as_a[(None, ab_empty.index)],
                        ab_empty.barrier.value,
                        cta_v_map=a_cta_v_map,
                        update_expect_tx=False,
                        tma_operation_type=tma_operation_type,
                    )
                    cute_ext.tma_load(
                        g_b_k,
                        t_bs_b[(None, ab_empty.index)],
                        ab_empty.barrier.value,
                        cta_v_map=b_cta_v_map,
                        update_expect_tx=False,
                        tma_operation_type=tma_operation_type,
                    )
                    ab_full = ab_consumer.wait_and_advance()
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    num_k_blocks = cute.size(t_cr_a, mode=[2])
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_coord = (None, None, k_block_idx, ab_full.index)
                        cute.gemm(
                            tiled_mma,
                            t_ct_acc,
                            t_cr_a[k_block_coord],
                            t_cr_b[k_block_coord],
                            t_ct_acc,
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    ab_full.release()
                acc_empty.commit()

            tmem.relinquish_alloc_permit()
            acc_full = acc_consumer.wait_and_advance()
            for i in cutlass.range(cute.size(t_ct_c, mode=[2])):
                cute.copy(tmem_tiled_copy, t_ct_c[None, None, i], t_cr_acc)
                cute.autovec_copy(t_cr_acc, t_cg_c_epi[None, None, i])
            acc_full.release()

            pipeline.sync(barrier_id=1)
            tmem.free(tmem_ptr)

else:

    class Fp8BatchedMmHsExperimentalKernel:  # type: ignore[no-redef]
        """Placeholder when `cutlass.cute.experimental` is unavailable (CUDA toolkit < 13.1)."""

        pass


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
        (sym_s, sym_b, _K),
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


def _compiled_fp8_batched_mm_hs_experimental_impl() -> Callable:
    os.environ.setdefault("CUTE_DSL_ARCH", "sm_100a")

    sym_m = _MM
    sym_k = _K
    sym_s = cute.sym_int()
    sym_b = cute.sym_int()

    kobj = Fp8BatchedMmHsExperimentalKernel()
    m_a = cute.runtime.make_fake_compact_tensor(
        Float8E4M3FN,
        (sym_m, sym_k, sym_b),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    m_b = cute.runtime.make_fake_compact_tensor(
        Float8E4M3FN,
        (sym_s, sym_k, sym_b),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    m_c = cute.runtime.make_fake_compact_tensor(
        Float32,
        (sym_m, sym_s, sym_b),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        kobj,
        m_a,
        m_b,
        m_c,
        stream_fake,
        options=_COMPILE_OPTS,
    )


if _CUTE_EXPERIMENTAL_AVAILABLE:
    compiled_fp8_batched_mm_hs_experimental = functools.cache(
        _compiled_fp8_batched_mm_hs_experimental_impl
    )
else:

    def compiled_fp8_batched_mm_hs_experimental() -> Callable:  # type: ignore[misc]
        raise RuntimeError(
            "cutlass.cute.experimental requires CUDA toolkit 13.1+ (host driver/toolkit)."
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

    use_ext = (
        _CUTE_EXPERIMENTAL_AVAILABLE
        and os.environ.get("DSA_FP8_TMMA_MM_EXT", "").lower() in ("1", "true", "yes")
    )

    if use_ext:
        # M×K×L / N×K×L / M×N×L layouts for `get_cta_v_map_ab` (matches CUTLASS GEMM examples).
        q_mkl = q_fp8.permute(1, 2, 0).contiguous()
        k_nkl = k_fp8.permute(1, 2, 0).contiguous()
        # Keep a view of `c_out` so the epilogue writes the caller's tensor (no extra copy).
        c_mnl = c_out.permute(1, 2, 0)

        fn = compiled_fp8_batched_mm_hs_experimental()
        m_a = from_dlpack(q_mkl, assumed_align=16, enable_tvm_ffi=True)
        m_a.element_type = Float8E4M3FN
        m_b = from_dlpack(k_nkl, assumed_align=16, enable_tvm_ffi=True)
        m_b.element_type = Float8E4M3FN
        m_c = from_dlpack(c_mnl, assumed_align=16, enable_tvm_ffi=True)
        fn(m_a, m_b, m_c)
        return

    fn = compiled_fp8_batched_mm_hs()
    m_q = from_dlpack(q_fp8, assumed_align=16, enable_tvm_ffi=True)
    m_q.element_type = Float8E4M3FN
    m_k = from_dlpack(k_fp8, assumed_align=16, enable_tvm_ffi=True)
    m_k.element_type = Float8E4M3FN
    m_c = from_dlpack(c_out, assumed_align=16, enable_tvm_ffi=True)

    fn(m_q, m_k, m_c, Int32(b), Int32(s))


__all__ = [
    "Fp8BatchedMmHsKernel",
    "Fp8BatchedMmHsExperimentalKernel",
    "compiled_fp8_batched_mm_hs",
    "compiled_fp8_batched_mm_hs_experimental",
    "fp8_batched_mm_hs",
    "quantize_rows_f32_to_fp8",
    "_CUTE_EXPERIMENTAL_AVAILABLE",
    "_MM",
    "_K",
    "_N_TILE",
]
