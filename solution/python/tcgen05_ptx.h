// SPDX-License-Identifier: BSD-3-Clause
//
// Inline-PTX wrappers for the SM100 (Blackwell) instructions this kernel
// needs: TMEM alloc, FP8 UMMA, TMEM readback, commit/wait, tcgen05 fences,
// cp.async and mbarrier.
//
// sm_100a only: solution.py builds a single `arch=compute_100a,code=sm_100a`
// target, so there are no arch guards or fallback stubs here. Compiling this
// header for an older arch fails at the PTX level, which is the intent.

#pragma once

#include <cstdint>

#include "umma_desc.h"

namespace dsa_ptx {

// Generic pointer to the 32-bit shared address PTX `[reg]` operands want.
// Cheaper than `__cvta_generic_to_shared` (no aperture test), and valid only
// for pointers already known to be in SMEM.
__device__ __forceinline__ uint32_t smem_ptr_to_uint(const void* smem_ptr) {
    uint32_t addr;
    asm volatile("{ .reg .u64 u64_addr;\n"
                 "  cvta.to.shared.u64 u64_addr, %1;\n"
                 "  cvt.u32.u64 %0, u64_addr; }\n"
                 : "=r"(addr)
                 : "l"(smem_ptr));
    return addr;
}

// cp.async: 16-byte async copy, global to shared.
__device__ __forceinline__ void
cp_async_16B(uint32_t smem_addr, const void* gmem_ptr) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                 :
                 : "r"(smem_addr), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;\n");
}

// Waits until at most N groups are pending; templated because PTX wants an
// immediate.
template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void mbarrier_init(uint32_t mbar_smem_addr, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
                 :
                 : "r"(mbar_smem_addr), "r"(count));
}

// Required after `mbarrier.init` on SM90+: without it `tcgen05.commit` can
// arrive on a barrier other threads have not seen initialised yet, and the
// kernel traps with Xid 43.
__device__ __forceinline__ void fence_barrier_init() {
    asm volatile("fence.mbarrier_init.release.cluster;\n");
}

__device__ __forceinline__ uint64_t
mbarrier_arrive(uint32_t mbar_smem_addr) {
    uint64_t state;
    asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];\n"
                 : "=l"(state)
                 : "r"(mbar_smem_addr));
    return state;
}

// `phase` flips between 0 and 1 each time the barrier completes.
__device__ __forceinline__ bool
mbarrier_try_wait_parity(uint32_t mbar_smem_addr, uint32_t phase) {
    uint32_t done;
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                 "selp.b32 %0, 1, 0, p;\n"
                 "}\n"
                 : "=r"(done)
                 : "r"(mbar_smem_addr), "r"(phase));
    return done != 0u;
}

__device__ __forceinline__ void
mbarrier_wait_parity(uint32_t mbar_smem_addr, uint32_t phase) {
    while (!mbarrier_try_wait_parity(mbar_smem_addr, phase)) { /* spin */ }
}

// Warpgroup register reconfiguration: reassigns up to N registers per thread.
// All 128 threads of the warpgroup must issue it together, and N must be in
// [24, 256] with step 8.

template <uint32_t N>
__device__ __forceinline__ void warpgroup_reg_alloc() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(N));
}

template <uint32_t N>
__device__ __forceinline__ void warpgroup_reg_dealloc() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(N));
}

// Syncs one warpgroup without stalling the other: `num_threads` must arrive at
// barrier `id` (0..15) before any of them proceeds.

__device__ __forceinline__ void named_barrier_sync(int id, int num_threads) {
    asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(num_threads));
}

// TMEM allocation. Writes the resulting 32-bit TMEM pointer to SMEM at `dst`,
// and must be issued by a single fully-active warp of the CTA.

__device__ __forceinline__ void
tcgen05_alloc_1sm(uint32_t dst_smem_addr, uint32_t num_columns) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
        :
        : "r"(dst_smem_addr), "r"(num_columns));
}

__device__ __forceinline__ void
tcgen05_dealloc_1sm(uint32_t tmem_addr, uint32_t num_columns) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n"
                 :
                 : "r"(tmem_addr), "r"(num_columns));
}

__device__ __forceinline__ void
tcgen05_relinquish_alloc_permit_1sm() {
    asm volatile(
        "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n");
}

__device__ __forceinline__ void tcgen05_fence_before_thread_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;\n");
}

__device__ __forceinline__ void tcgen05_fence_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;\n");
}

// Signals an mbarrier once outstanding MMA/LD work retires. The `::cluster`
// qualifier allows the mbarrier to live in another CTA of the cluster; at
// cluster_size=1 it resolves the same as `::cta`.
__device__ __forceinline__ void
tcgen05_commit_1sm(uint32_t mbar_smem_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 "
        "[%0];\n"
        :
        : "r"(mbar_smem_addr));
}

// Dense FP8 UMMA. desc_a / desc_b are 64-bit SmemDescriptors, instr_desc_hi is
// the high half of the descriptor operand, and `accumulate` chooses between
// += into [tmem_c] and overwrite.

__device__ __forceinline__ void
tcgen05_mma_f8f6f4_ss(uint32_t tmem_c,
                      uint64_t desc_a,
                      uint64_t desc_b,
                      uint32_t instr_desc_hi,
                      uint32_t accumulate) {
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
}

// Same, taking a full InstrDescriptor.
__device__ __forceinline__ void
tcgen05_mma_f8f6f4_ss(uint32_t                   tmem_c,
                      dsa_umma::SmemDescriptor   desc_a,
                      dsa_umma::SmemDescriptor   desc_b,
                      dsa_umma::InstrDescriptor  instr_desc,
                      uint32_t                   accumulate) {
    tcgen05_mma_f8f6f4_ss(
        tmem_c,
        static_cast<uint64_t>(desc_a),
        static_cast<uint64_t>(desc_b),
        static_cast<uint32_t>(instr_desc.desc_),
        accumulate);
}

// Blocks the issuing thread until prior tcgen05 loads retire.
__device__ __forceinline__ void tcgen05_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;\n");
}

// TMEM readback: 32 DP rows x 32 bits x 64 column-tiles = 8192 bytes per warp,
// into 64 registers per lane. Issued by all 4 warps of a warpgroup with the
// same tmem_addr, the hardware hands DP rows 0..31 to warp 0, 32..63 to warp 1
// and so on -- exactly the accumulator layout an UMMA_M=128, UMMA_N=64
// tcgen05.mma produces.
__device__ __forceinline__ void
tcgen05_ld_32x32b_x64_b32(uint32_t tmem_addr, uint32_t (&out)[64]) {
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
}

} // namespace dsa_ptx
