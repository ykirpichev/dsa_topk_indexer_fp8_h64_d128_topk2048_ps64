// Stage 1+2 bypass. When the whole paged context fits inside top-K there is
// nothing to rank: the answer is the block-table transform of [0, seq_len)
// padded with -1, independent of Q, K and weights. 69 of the 128 contest
// workloads land here, so launch overhead dominates this kernel's cost.

#pragma once

#include <cuda_runtime.h>

#include "dsa_config.cuh"

namespace {

// 128 threads with 4 vectorised int4 stores each (512 int4 = kTopK) beat 256
// threads with scalar stores: half the warp-scheduler init at launch and half
// the HBM transactions. smem_bt[] is zero-padded instead of bounds-checked --
// those lanes emit -1 regardless, their token index being >= seq_len.
__global__ __launch_bounds__(128)
void topk_fast_path_kernel(
    const int* __restrict__ seq_lens,
    const int* __restrict__ block_table,
    int max_num_pages,
    int* __restrict__ out_indices)
{
    constexpr int kInts4PerRow   = kTopK / 4;           // 512 int4 per row
    constexpr int kInt4PerThread = kInts4PerRow / 128;  // 4 int4 per thread

    const int b   = blockIdx.x;
    const int tid = threadIdx.x;

    __shared__ int smem_bt[kFastPathMaxPages];
    if (tid < kFastPathMaxPages) {
        smem_bt[tid] = (tid < max_num_pages)
            ? block_table[b * max_num_pages + tid]
            : 0;
    }

    const int seq_len = seq_lens[b];
    int4*     out4    = reinterpret_cast<int4*>(
        out_indices + static_cast<size_t>(b) * kTopK);

    __syncthreads();

#pragma unroll
    for (int s = 0; s < kInt4PerThread; ++s) {
        const int i4   = tid + s * 128;                     // 0..511
        const int base = i4 << 2;                            // 0..2044, stride 4

        const int page0 = (base + 0) >> 6;                   // / 64
        const int page1 = (base + 1) >> 6;
        const int page2 = (base + 2) >> 6;
        const int page3 = (base + 3) >> 6;
        const int slot0 = (base + 0) & 63;
        const int slot1 = (base + 1) & 63;
        const int slot2 = (base + 2) & 63;
        const int slot3 = (base + 3) & 63;

        int4 v;
        v.x = (base + 0 < seq_len) ? (smem_bt[page0] << 6) + slot0 : -1;
        v.y = (base + 1 < seq_len) ? (smem_bt[page1] << 6) + slot1 : -1;
        v.z = (base + 2 < seq_len) ? (smem_bt[page2] << 6) + slot2 : -1;
        v.w = (base + 3 < seq_len) ? (smem_bt[page3] << 6) + slot3 : -1;
        out4[i4] = v;
    }
}

}  // namespace
