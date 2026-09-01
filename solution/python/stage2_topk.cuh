// Stage 2: top-K (k = 2048) selection over the Stage-1 logits, then the
// block-table transform from token indices to paged KV slots. Two-round radix
// select over fp16 keys mapped to comparison-ordered uint16, with the ordered
// keys cached in shared memory when the row fits.

#pragma once

#include <cuda_fp16.h>

#include "dsa_config.cuh"

namespace {

// fp16 bits reordered so that unsigned integer comparison matches float
// ordering: flip the sign bit for positives, invert everything for negatives.
__device__ __forceinline__ uint16_t HalfToOrderedU16(__half h) {
    const uint16_t bits = __half_as_ushort(h);
    return (bits & 0x8000u) ? static_cast<uint16_t>(~bits)
                            : static_cast<uint16_t>(bits ^ 0x8000u);
}

// Ordered keys are cached in 32 KB of SMEM during round 0 and re-read by
// rounds 1+, the gt-count pass and both emit passes, saving four HBM reads and
// four conversions per element. Every contest workload fits, but the
// non-cached path is what keeps longer rows correct.
constexpr int kStage2MaxCachedLen = 16384;

__global__ __launch_bounds__(kStage2Threads)
void topk_page_table_transform_kernel(
    const __half* __restrict__ logits,
    const int*    __restrict__ seq_lens,
    const int*    __restrict__ block_table,
    int max_len, int max_num_pages, int top_k,
    int* __restrict__ out_indices)
{
    const int b   = blockIdx.x;
    const int tid = threadIdx.x;

    const int seq_len    = seq_lens[b];
    const __half* row_lg = logits + static_cast<size_t>(b) * max_len;
    int*          row_out= out_indices + static_cast<size_t>(b) * top_k;
    const int*    row_bt = block_table + static_cast<size_t>(b) * max_num_pages;

    __shared__ uint32_t smem_hist[kRadix];
    __shared__ uint32_t smem_suffix[kRadix];
    __shared__ uint16_t smem_prefix;
    __shared__ uint32_t smem_remaining;
    __shared__ uint32_t smem_found_bucket;
    __shared__ uint32_t smem_found_remaining;
    __shared__ int      smem_gt_count;
    __shared__ int      smem_emit_counter;
    __shared__ uint16_t smem_ordered[kStage2MaxCachedLen];

    if (seq_len <= top_k) {
        for (int i = tid; i < top_k; i += blockDim.x) {
            if (i < seq_len)
                row_out[i] = row_bt[i / kPageSize] * kPageSize + (i % kPageSize);
            else
                row_out[i] = -1;
        }
        return;
    }

    if (tid == 0) { smem_prefix = 0u; smem_remaining = (uint32_t)top_k; }
    __syncthreads();

    const bool use_ordered_cache = (seq_len <= kStage2MaxCachedLen);
    const int  cached_len        = use_ordered_cache ? seq_len : 0;

#pragma unroll
    for (int round = 0; round < kRadixRounds; ++round) {
        const int shift = kOrderedBits - 8 - round * 8;
        const uint16_t prefix_mask =
            (round == 0) ? uint16_t{0}
                         : static_cast<uint16_t>(0xFFFFu << (kOrderedBits - round * 8));

        for (int i = tid; i < kRadix; i += blockDim.x) smem_hist[i] = 0u;
        __syncthreads();

        const uint16_t prefix = smem_prefix;
        if (round == 0 && use_ordered_cache) {
            // prefix_mask == 0 on round 0, so every key is accepted.
            for (int i = tid; i < seq_len; i += blockDim.x) {
                const uint16_t ordered = HalfToOrderedU16(row_lg[i]);
                smem_ordered[i] = ordered;
                atomicAdd(&smem_hist[(ordered >> shift) & 0xFFu], 1u);
            }
        } else if (use_ordered_cache) {
#pragma unroll 4
            for (int i = tid; i < cached_len; i += blockDim.x) {
                const uint16_t ordered = smem_ordered[i];
                if ((uint16_t)(ordered & prefix_mask) == prefix)
                    atomicAdd(&smem_hist[(ordered >> shift) & 0xFFu], 1u);
            }
        } else {
#pragma unroll 4
            for (int i = tid; i < seq_len; i += blockDim.x) {
                const uint16_t ordered = HalfToOrderedU16(row_lg[i]);
                if ((uint16_t)(ordered & prefix_mask) == prefix)
                    atomicAdd(&smem_hist[(ordered >> shift) & 0xFFu], 1u);
            }
        }
        __syncthreads();

        if (tid < kRadix) smem_suffix[tid] = smem_hist[tid];
        __syncthreads();
        for (int stride = 1; stride < kRadix; stride <<= 1) {
            uint32_t v = 0u;
            if (tid < kRadix) {
                v = smem_suffix[tid];
                if (tid + stride < kRadix) v += smem_suffix[tid + stride];
            }
            __syncthreads();
            if (tid < kRadix) smem_suffix[tid] = v;
            __syncthreads();
        }

        if (tid == 0) { smem_found_bucket = 0u; smem_found_remaining = smem_remaining; }
        __syncthreads();

        if (tid < kRadix) {
            const uint32_t cge = smem_suffix[tid];
            const uint32_t cgt = (tid + 1 < kRadix) ? smem_suffix[tid + 1] : 0u;
            if (cge >= smem_remaining && cgt < smem_remaining) {
                smem_found_bucket    = (uint32_t)tid;
                smem_found_remaining = smem_remaining - cgt;
            }
        }
        __syncthreads();

        if (tid == 0) {
            smem_prefix    = (uint16_t)(smem_prefix | (smem_found_bucket << shift));
            smem_remaining = smem_found_remaining;
        }
        __syncthreads();
    }

    const uint16_t pivot = smem_prefix;

    if (tid == 0) { smem_gt_count = 0; smem_emit_counter = 0; }
    __syncthreads();

    auto ordered_at = [&](int i) -> uint16_t {
        return use_ordered_cache ? smem_ordered[i]
                                 : HalfToOrderedU16(row_lg[i]);
    };

    {
        int local = 0;
        for (int i = tid; i < seq_len; i += blockDim.x)
            if (ordered_at(i) > pivot) local++;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            local += __shfl_xor_sync(0xffffffffu, local, off);
        if ((tid & 31) == 0) atomicAdd(&smem_gt_count, local);
    }
    __syncthreads();

    const int gt_total = smem_gt_count;

    for (int i = tid; i < seq_len; i += blockDim.x) {
        if (ordered_at(i) > pivot) {
            const int pos = atomicAdd(&smem_emit_counter, 1);
            if (pos < gt_total) row_out[pos] = i;
        }
    }
    __syncthreads();

    for (int i = tid; i < seq_len; i += blockDim.x) {
        if (ordered_at(i) == pivot) {
            const int pos = atomicAdd(&smem_emit_counter, 1);
            if (pos < top_k) row_out[pos] = i;
        }
    }
    __syncthreads();

    for (int i = tid; i < top_k; i += blockDim.x) {
        const int tok = row_out[i];
        if (tok >= 0 && tok < seq_len)
            row_out[i] = row_bt[tok / kPageSize] * kPageSize + (tok % kPageSize);
        else
            row_out[i] = -1;
    }
}

}  // namespace
