// SMEM-layout and UMMA-descriptor helpers shared by both Stage-1 kernels.

#pragma once

#include <cstdint>

#include "tcgen05_ptx.h"
#include "umma_desc.h"

namespace {

// UMMA's SWIZZLE_128B layout: within each run of 8 16-byte vectors, vector i
// lives at i ^ ((i >> 3) & 7). The fill and the descriptor must agree.
__device__ __forceinline__ int swizzle_16b(int idx) {
    return idx ^ ((idx >> 3) & 7);
}

// K-major SWIZZLE_128B operand descriptor for an SMEM tile whose consecutive
// rows are `sbo_bytes` apart. `smem_base` must be 1024-byte aligned.
__device__ __forceinline__ dsa_umma::SmemDescriptor
make_kmajor_desc(const void* smem_base, uint32_t sbo_bytes) {
    dsa_umma::SmemDescriptor d{};
    d.desc_                = 0;
    d.version_             = 1;
    d.lbo_mode_            = 0;
    d.base_offset_         = 0;
    d.layout_type_         = static_cast<uint8_t>(dsa_umma::LayoutType::SWIZZLE_128B);
    d.leading_byte_offset_ = 0;
    d.stride_byte_offset_  = static_cast<uint16_t>(sbo_bytes >> 4);
    d.start_address_       = static_cast<uint16_t>(
        dsa_ptx::smem_ptr_to_uint(smem_base) >> 4);
    return d;
}

// Same tile shifted along K; descriptor addresses are in 16-byte units.
__device__ __forceinline__ dsa_umma::SmemDescriptor
advance_k(dsa_umma::SmemDescriptor d, int k_offset_bytes) {
    d.start_address_ = static_cast<uint16_t>(
        d.start_address_ + static_cast<uint16_t>(k_offset_bytes >> 4));
    return d;
}

// Warp 0 allocates TMEM, lane 0 runs `init_mbars` (one or more mbarrier.init),
// then the warp fences so a later tcgen05.commit cannot arrive early.
// `init_mbars` is only invoked from lane 0.
template <class InitMbars>
__device__ __forceinline__ void alloc_tmem_init_mbars(
    uint32_t* tmem_ptr_smem,
    uint32_t num_cols,
    int warp_id,
    int lane,
    InitMbars init_mbars)
{
    if (warp_id == 0) {
        dsa_ptx::tcgen05_alloc_1sm(dsa_ptx::smem_ptr_to_uint(tmem_ptr_smem), num_cols);
        dsa_ptx::tcgen05_relinquish_alloc_permit_1sm();
        if (lane == 0) init_mbars();
        dsa_ptx::fence_barrier_init();
    }
}

// Warp 0, lane 0: K-loop of tcgen05.mma (K step 32 bytes) then commit to `mbar`.
template <int KIters>
__device__ __forceinline__ void issue_umma_k_commit(
    uint32_t tmem_addr,
    dsa_umma::SmemDescriptor a_base,
    dsa_umma::SmemDescriptor b_base,
    dsa_umma::InstrDescriptor instr,
    uint32_t mbar_addr,
    int warp_id,
    int lane)
{
    constexpr int kKStep = 32;
    if (warp_id == 0) {
        dsa_ptx::tcgen05_fence_after_thread_sync();
        if (lane == 0) {
#pragma unroll
            for (int k = 0; k < KIters; ++k) {
                const int k_off = k * kKStep;
                dsa_ptx::tcgen05_mma_f8f6f4_ss(
                    tmem_addr,
                    advance_k(a_base, k_off),
                    advance_k(b_base, k_off),
                    instr, (k == 0) ? 0u : 1u);
            }
            dsa_ptx::tcgen05_commit_1sm(mbar_addr);
        }
    }
}

}  // namespace
