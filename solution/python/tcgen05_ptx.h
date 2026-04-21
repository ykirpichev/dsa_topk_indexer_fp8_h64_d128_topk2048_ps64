// SPDX-License-Identifier: BSD-3-Clause
//
// Inline-PTX wrappers for the subset of SM100 (Blackwell) instructions we
// need to drive a FP8 MQA-logits kernel:
//
//   * Tensor Memory (TMEM) allocation  (tcgen05.alloc / dealloc /
//                                       relinquish_alloc_permit)
//   * FP8 UMMA                         (tcgen05.mma.cta_group::1.kind::f8f6f4)
//   * TMEM readback                    (tcgen05.ld.sync.aligned.16x256b.xN)
//   * TMEM commit + wait               (tcgen05.commit, tcgen05.wait::ld)
//   * Fences                           (tcgen05.fence::{before,after}_thread_sync)
//   * cp.async                         (cp.async.ca / commit / wait)
//   * mbarrier                         (init / inval / arrive / try_wait_parity)
//
// The `mbarrier` and `cp.async` wrappers are SM80+; the `tcgen05.*` wrappers
// require sm_100a.  All bodies are guarded with `#if __CUDA_ARCH__ >= 1000`
// so the kernel.cu translation unit can still compile for sm_90 (Modal
// exposes sm_100 which shares these as "generic Blackwell" but not the
// architecture-specific `a` extensions).  Unguarded callers on older arches
// will fall through to trap stubs that abort cleanly so any mistake surfaces
// at runtime rather than silently miscompiling.

#pragma once

#include <cstdint>

#include "umma_desc.h"

namespace dsa_ptx {

// ---------------------------------------------------------------------------
// Common: SMEM generic-to-shared address conversion
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint32_t smem_ptr_to_uint(const void* smem_ptr) {
    // Convert a generic CUDA pointer residing in __shared__ memory to the
    // 32-bit shared-memory address expected by PTX instructions that take
    // `[reg]` operands.  Cheaper than `__cvta_generic_to_shared` because it
    // bypasses the generic aperture test; only valid for addresses known to
    // be in SMEM.
    uint32_t addr;
    asm volatile("{ .reg .u64 u64_addr;\n"
                 "  cvta.to.shared.u64 u64_addr, %1;\n"
                 "  cvt.u32.u64 %0, u64_addr; }\n"
                 : "=r"(addr)
                 : "l"(smem_ptr));
    return addr;
}

// ---------------------------------------------------------------------------
// cp.async (SM_80+): 16-byte async copy from global to shared
// ---------------------------------------------------------------------------

__device__ __forceinline__ void
cp_async_16B(uint32_t smem_addr, const void* gmem_ptr) {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                 :
                 : "r"(smem_addr), "l"(gmem_ptr));
#else
    (void)smem_addr; (void)gmem_ptr;
    __trap();
#endif
}

__device__ __forceinline__ void
cp_async_4B(uint32_t smem_addr, const void* gmem_ptr) {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                 :
                 : "r"(smem_addr), "l"(gmem_ptr));
#else
    (void)smem_addr; (void)gmem_ptr;
    __trap();
#endif
}

__device__ __forceinline__ void cp_async_commit_group() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n");
#endif
}

// `cp.async.wait_group N;`   -- wait until at most N groups are still
// pending.  Template argument because PTX takes an immediate.
template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

// ---------------------------------------------------------------------------
// mbarrier (SM_80+)
// ---------------------------------------------------------------------------

__device__ __forceinline__ void mbarrier_init(uint32_t mbar_smem_addr, uint32_t count) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
                 :
                 : "r"(mbar_smem_addr), "r"(count));
#else
    (void)mbar_smem_addr; (void)count;
#endif
}

// Mirrors `cutlass::arch::fence_barrier_init()` -- required after
// `mbarrier.init` on SM90+ so other threads see the initialized barrier
// before any arrive / wait occurs.  Without it, `tcgen05.commit` may
// arrive on an uninitialized mbarrier and the kernel traps (Xid 43).
__device__ __forceinline__ void fence_barrier_init() {
#if __CUDA_ARCH__ >= 900
    asm volatile("fence.mbarrier_init.release.cluster;\n");
#elif __CUDA_ARCH__ >= 800
    __threadfence_block();
#endif
}

__device__ __forceinline__ void mbarrier_inval(uint32_t mbar_smem_addr) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n"
                 :
                 : "r"(mbar_smem_addr));
#else
    (void)mbar_smem_addr;
#endif
}

__device__ __forceinline__ uint64_t
mbarrier_arrive(uint32_t mbar_smem_addr) {
#if __CUDA_ARCH__ >= 800
    uint64_t state;
    asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];\n"
                 : "=l"(state)
                 : "r"(mbar_smem_addr));
    return state;
#else
    (void)mbar_smem_addr;
    return 0;
#endif
}

// Parity-based wait: `phase` toggles between 0 and 1 every time the barrier
// completes; see PTX ISA 9.7.12.15.6.
__device__ __forceinline__ bool
mbarrier_try_wait_parity(uint32_t mbar_smem_addr, uint32_t phase) {
#if __CUDA_ARCH__ >= 800
    uint32_t done;
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                 "selp.b32 %0, 1, 0, p;\n"
                 "}\n"
                 : "=r"(done)
                 : "r"(mbar_smem_addr), "r"(phase));
    return done != 0u;
#else
    (void)mbar_smem_addr; (void)phase;
    return true;
#endif
}

// Busy-wait helper.  Use when the cost of an extra spin is negligible.
__device__ __forceinline__ void
mbarrier_wait_parity(uint32_t mbar_smem_addr, uint32_t phase) {
    while (!mbarrier_try_wait_parity(mbar_smem_addr, phase)) { /* spin */ }
}

// `cp.async.mbarrier.arrive` -- a cp.async issuer can tie its completion to
// an mbarrier increment, which is how producer warps signal consumer warps
// without an explicit `cp.async.wait_group`.
__device__ __forceinline__ void
cp_async_mbarrier_arrive_noinc(uint32_t mbar_smem_addr) {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n"
                 :
                 : "r"(mbar_smem_addr));
#else
    (void)mbar_smem_addr;
#endif
}

// ---------------------------------------------------------------------------
// TMEM allocation (SM_100a)
// ---------------------------------------------------------------------------
//
// `tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [dst], ncols;`
// Stores the resulting 32-bit TMEM pointer into shared memory at `dst`.
// Must be issued by a single fully-active warp of the CTA.

__device__ __forceinline__ void
tcgen05_alloc_1sm(uint32_t dst_smem_addr, uint32_t num_columns) {
#if __CUDA_ARCH__ >= 1000
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
        :
        : "r"(dst_smem_addr), "r"(num_columns));
#else
    (void)dst_smem_addr; (void)num_columns;
    __trap();
#endif
}

__device__ __forceinline__ void
tcgen05_dealloc_1sm(uint32_t tmem_addr, uint32_t num_columns) {
#if __CUDA_ARCH__ >= 1000
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n"
                 :
                 : "r"(tmem_addr), "r"(num_columns));
#else
    (void)tmem_addr; (void)num_columns;
    __trap();
#endif
}

__device__ __forceinline__ void
tcgen05_relinquish_alloc_permit_1sm() {
#if __CUDA_ARCH__ >= 1000
    asm volatile(
        "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n");
#else
    __trap();
#endif
}

// ---------------------------------------------------------------------------
// tcgen05 fences (SM_100a)
// ---------------------------------------------------------------------------

__device__ __forceinline__ void tcgen05_fence_before_thread_sync() {
#if __CUDA_ARCH__ >= 1000
    asm volatile("tcgen05.fence::before_thread_sync;\n");
#endif
}

__device__ __forceinline__ void tcgen05_fence_after_thread_sync() {
#if __CUDA_ARCH__ >= 1000
    asm volatile("tcgen05.fence::after_thread_sync;\n");
#endif
}

// ---------------------------------------------------------------------------
// tcgen05.commit: signal an mbarrier when outstanding MMA/LD work retires.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void
tcgen05_commit_1sm(uint32_t mbar_smem_addr) {
#if __CUDA_ARCH__ >= 1000
    // Matches `cutlass::arch::umma_arrive` (cutlass/arch/barrier.h:766):
    // the shared-memory qualifier is `::cluster` not `::cta` so the
    // mbarrier can live in another CTA in the cluster; for cluster_size=1
    // (our case) both resolve identically.
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 "
        "[%0];\n"
        :
        : "r"(mbar_smem_addr));
#else
    (void)mbar_smem_addr;
    __trap();
#endif
}

// ---------------------------------------------------------------------------
// tcgen05.mma.cta_group::1.kind::f8f6f4  (dense FP8 UMMA)
// ---------------------------------------------------------------------------
//
// `[tmem_c] += (scale_c ? desc_a * desc_b : desc_a * desc_b)`
//
// desc_a / desc_b : 64-bit SmemDescriptor values
// instr_desc_hi   : high 32 bits of the 64-bit operand passed as the 4th
//                   operand -- i.e. `((uint64_t)instr_desc) << 32`, and we
//                   pass the `>> 32` part as an immediate register.
// accumulate      : if non-zero, += into [tmem_c] ; if zero, overwrite.

__device__ __forceinline__ void
tcgen05_mma_f8f6f4_ss(uint32_t tmem_c,
                      uint64_t desc_a,
                      uint64_t desc_b,
                      uint32_t instr_desc_hi,
                      uint32_t accumulate) {
#if __CUDA_ARCH__ >= 1000
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f8f6f4 "
        "  [%0], %1, %2, %3, p;\n"
        "}\n"
        :
        : "r"(tmem_c),
          "l"(desc_a),
          "l"(desc_b),
          "r"(instr_desc_hi),
          "r"(accumulate));
#else
    (void)tmem_c; (void)desc_a; (void)desc_b;
    (void)instr_desc_hi; (void)accumulate;
    __trap();
#endif
}

// Convenience: take the full 32-bit InstrDescriptor and materialise it into
// the high 32 bits slot expected by the PTX instruction.
__device__ __forceinline__ void
tcgen05_mma_f8f6f4_ss(uint32_t                   tmem_c,
                      dsa_umma::SmemDescriptor   desc_a,
                      dsa_umma::SmemDescriptor   desc_b,
                      dsa_umma::InstrDescriptor  instr_desc,
                      uint32_t                   accumulate) {
    // The PTX takes only the upper 32 bits of a 64-bit value as the
    // instruction descriptor slot; CUTLASS wraps this as
    // `static_cast<uint64_t>(static_cast<uint32_t>(desc)) << 32` and passes
    // the upper 32 bits as the third immediate.  We compute those upper 32
    // bits directly to save a shift.
    tcgen05_mma_f8f6f4_ss(
        tmem_c,
        static_cast<uint64_t>(desc_a),
        static_cast<uint64_t>(desc_b),
        static_cast<uint32_t>(instr_desc.desc_),
        accumulate);
}

// ---------------------------------------------------------------------------
// tcgen05.wait: block the issuing thread until prior tcgen05 loads retire.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void tcgen05_wait_ld() {
#if __CUDA_ARCH__ >= 1000
    asm volatile("tcgen05.wait::ld.sync.aligned;\n");
#else
    __trap();
#endif
}

// ---------------------------------------------------------------------------
// tcgen05.ld.sync.aligned.16x256b.x{1,2,4,8,16,32}.b32
// ---------------------------------------------------------------------------
//
// Reads a column-sliced chunk of TMEM back into 32-bit registers owned by
// the issuing warp.  The `16x256b` shape means each row is 256 bits wide and
// we read 16 rows; `xN` means N adjacent column groups.  Each call returns
// `4 * N` uint32 registers per lane (16 rows x 256b = 4 uint32 per lane for
// a 32-lane warp); for a single-warp `tcgen05.ld.16x256b.x4.b32`, each lane
// receives 16 uint32 values.
//
// The register allocation mapping (which lane owns which (row, col) pair)
// is documented in the PTX ISA 10.2 section on `tcgen05.ld`:
//   https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
//
// We only need the x4 flavour for our 64x64 FP32 accumulator (covers
// rows [0..16) × cols [0..256b*4)/4 = 16 per lane); we will call it four
// times per 64-row tile to cover all rows.

__device__ __forceinline__ void
tcgen05_ld_16x256b_x4_b32(uint32_t tmem_addr,
                          uint32_t (&out)[16]) {
#if __CUDA_ARCH__ >= 1000
    asm volatile(
        "tcgen05.ld.sync.aligned.16x256b.x4.b32 "
        "{ %0, %1, %2, %3, %4, %5, %6, %7, "
        "  %8, %9, %10, %11, %12, %13, %14, %15 }, [%16];\n"
        : "=r"(out[0]),  "=r"(out[1]),  "=r"(out[2]),  "=r"(out[3]),
          "=r"(out[4]),  "=r"(out[5]),  "=r"(out[6]),  "=r"(out[7]),
          "=r"(out[8]),  "=r"(out[9]),  "=r"(out[10]), "=r"(out[11]),
          "=r"(out[12]), "=r"(out[13]), "=r"(out[14]), "=r"(out[15])
        : "r"(tmem_addr));
#else
    (void)tmem_addr;
    for (int i = 0; i < 16; ++i) out[i] = 0u;
    __trap();
#endif
}

// `tcgen05.ld.sync.aligned.32x32b.x64.b32`
//   Reads 32 DP rows × 32 bits × 64 column-tiles = 8192 bytes per warp of
//   TMEM into 64 32-bit registers per lane.  This is the shape DeepGEMM's
//   FP8 MQA-logits kernel uses in its math warpgroup epilogue
//   (CUTLASS `SM100_TMEM_LOAD_32dp32b64x`).  When 4 warps of a warpgroup
//   issue this with the same tmem_addr, the hardware distributes DP rows
//   0..31 to warp 0, 32..63 to warp 1, ..., 96..127 to warp 3 — exactly
//   the 128-token-per-warpgroup accumulator layout produced by an
//   `UMMA_M=128, UMMA_N=64` `tcgen05.mma.kind::f8f6f4`.
__device__ __forceinline__ void
tcgen05_ld_32x32b_x64_b32(uint32_t tmem_addr, uint32_t (&out)[64]) {
#if __CUDA_ARCH__ >= 1000
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x64.b32 "
        "{ %0, %1, %2, %3, %4, %5, %6, %7, "
        "  %8, %9, %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31, "
        " %32, %33, %34, %35, %36, %37, %38, %39, "
        " %40, %41, %42, %43, %44, %45, %46, %47, "
        " %48, %49, %50, %51, %52, %53, %54, %55, "
        " %56, %57, %58, %59, %60, %61, %62, %63 }, "
        "[%64];\n"
        : "=r"(out[ 0]), "=r"(out[ 1]), "=r"(out[ 2]), "=r"(out[ 3]),
          "=r"(out[ 4]), "=r"(out[ 5]), "=r"(out[ 6]), "=r"(out[ 7]),
          "=r"(out[ 8]), "=r"(out[ 9]), "=r"(out[10]), "=r"(out[11]),
          "=r"(out[12]), "=r"(out[13]), "=r"(out[14]), "=r"(out[15]),
          "=r"(out[16]), "=r"(out[17]), "=r"(out[18]), "=r"(out[19]),
          "=r"(out[20]), "=r"(out[21]), "=r"(out[22]), "=r"(out[23]),
          "=r"(out[24]), "=r"(out[25]), "=r"(out[26]), "=r"(out[27]),
          "=r"(out[28]), "=r"(out[29]), "=r"(out[30]), "=r"(out[31]),
          "=r"(out[32]), "=r"(out[33]), "=r"(out[34]), "=r"(out[35]),
          "=r"(out[36]), "=r"(out[37]), "=r"(out[38]), "=r"(out[39]),
          "=r"(out[40]), "=r"(out[41]), "=r"(out[42]), "=r"(out[43]),
          "=r"(out[44]), "=r"(out[45]), "=r"(out[46]), "=r"(out[47]),
          "=r"(out[48]), "=r"(out[49]), "=r"(out[50]), "=r"(out[51]),
          "=r"(out[52]), "=r"(out[53]), "=r"(out[54]), "=r"(out[55]),
          "=r"(out[56]), "=r"(out[57]), "=r"(out[58]), "=r"(out[59]),
          "=r"(out[60]), "=r"(out[61]), "=r"(out[62]), "=r"(out[63])
        : "r"(tmem_addr));
#else
    (void)tmem_addr;
    for (int i = 0; i < 64; ++i) out[i] = 0u;
    __trap();
#endif
}

} // namespace dsa_ptx
