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

}  // namespace
