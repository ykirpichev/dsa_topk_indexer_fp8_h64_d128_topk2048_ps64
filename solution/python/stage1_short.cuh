// Stage 1 for mid-size contexts (33 <= max_num_pages < 64): one KV page per
// UMMA tile, one tile per CTA. Here the persistent kernel's per-CTA setup
// (Q load, TMEM alloc, mbarrier init) costs more than it saves, and the flat
// grid saturates the SMs sooner.

#pragma once

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "dsa_config.cuh"
#include "stage1_common.cuh"
#include "tcgen05_ptx.h"
#include "umma_desc.h"

namespace {

__global__ __launch_bounds__(128)
void paged_mqa_logits_umma_kernel_short(
    const __nv_fp8_e4m3* __restrict__ q_fp8,
    const uint8_t*       __restrict__ kv_cache,
    const float*         __restrict__ weights,
    const int*           __restrict__ seq_lens,
    const int*           __restrict__ block_table,
    int                                max_num_pages,
    int                                max_kv_tiles,
    __half*              __restrict__  logits_out)
{
    constexpr int kUMMA_M    = 128;
    constexpr int kUMMA_N    = kNumHeads;
    constexpr int kUMMA_K_B  = 32;
    constexpr int kKIters    = kHeadDim / kUMMA_K_B;
    constexpr int kThreads   = 128;
    constexpr int kTmemCols  = kUMMA_N;
    constexpr uint32_t kSboBytes = 1024u;

    const int tile_idx = blockIdx.x;
    const int b        = blockIdx.y;
    const int tid      = threadIdx.x;
    const int warp_id  = tid / 32;
    const int lane     = tid % 32;

    const int seq_len = seq_lens[b];

    // Rows that fit in top-K are emitted by Stage 2's per-row fast path
    // straight from (seq_lens, block_table); it never reads these logits.
    if (seq_len <= kTopK) return;

    const int kv_base = tile_idx * kBlockKv;
    __half*   logits_b = logits_out + static_cast<size_t>(b) *
                                        static_cast<size_t>(max_kv_tiles) * kBlockKv;

    // Unreachable given grid.x == max_kv_tiles; kept for the same reason as the
    // redundant bounds in stage1_persistent_ws.cuh.
    if (kv_base >= max_kv_tiles * kBlockKv) return;

    const __half kHalfNegInf = __ushort_as_half(static_cast<unsigned short>(0xFC00));
    if (kv_base >= seq_len) {
        for (int i = tid; i < kBlockKv; i += kThreads)
            logits_b[kv_base + i] = kHalfNegInf;
        return;
    }

    // Q [64, 128] and one K page, both in the 128B swizzle UMMA expects.
    // The MMA is M=128 but this kernel only has 64 real K rows, so the
    // upper half of smem_k is zero and cannot contribute to the accumulator.
    __shared__ __align__(1024) __nv_fp8_e4m3 smem_q[kNumHeads * kHeadDim];
    __shared__ __align__(1024) __nv_fp8_e4m3 smem_k[kUMMA_M * kHeadDim];
    __shared__ float   smem_kscale[kBlockKv];
    __shared__ float   smem_w[kNumHeads];
    __shared__ uint32_t smem_tmem_ptr;
    __shared__ __align__(8) uint64_t smem_mbar;

    {
        const uint4* src = reinterpret_cast<const uint4*>(
            q_fp8 + static_cast<size_t>(b) * kNumHeads * kHeadDim);
        uint4* dst = reinterpret_cast<uint4*>(smem_q);
        constexpr int kVec = (kNumHeads * kHeadDim) / 16;
#pragma unroll 4
        for (int i = tid; i < kVec; i += kThreads) dst[swizzle_16b(i)] = src[i];
    }

    const int page_idx = block_table[b * max_num_pages + tile_idx];
    const uint8_t* page_ptr = kv_cache + static_cast<size_t>(page_idx) * kPageBytes;

    {
        const uint4* src = reinterpret_cast<const uint4*>(page_ptr);
        uint4* dst = reinterpret_cast<uint4*>(smem_k);
        constexpr int kVecReal  = (kBlockKv * kHeadDim) / 16;
        constexpr int kVecTotal = (kUMMA_M * kHeadDim) / 16;
        const uint4 zero4 = make_uint4(0u, 0u, 0u, 0u);
#pragma unroll 4
        for (int i = tid; i < kVecReal; i += kThreads)
            dst[swizzle_16b(i)] = src[i];
#pragma unroll 4
        for (int i = tid; i < (kVecTotal - kVecReal); i += kThreads)
            dst[swizzle_16b(i + kVecReal)] = zero4;
    }

    if (tid < kBlockKv)
        smem_kscale[tid] = reinterpret_cast<const float*>(page_ptr + kScaleOffsetBytes)[tid];
    if (tid < kNumHeads)
        smem_w[tid] = weights[b * kNumHeads + tid];

    __syncthreads();

    // One UMMA (K·Q^T into TMEM) for this page, then every thread reads its
    // 64 accumulators back. Warp 0 deallocates TMEM before the logit stores.
    alloc_tmem_init_mbars(&smem_tmem_ptr, kTmemCols, warp_id, lane, [&] {
        dsa_ptx::mbarrier_init(dsa_ptx::smem_ptr_to_uint(&smem_mbar), 1);
    });
    __syncthreads();

    const uint32_t tmem_addr = smem_tmem_ptr;

    {
        const auto instr = dsa_umma::make_instr_desc_f8f6f4(
            dsa_umma::F8F6F4Format::E4M3, dsa_umma::F8F6F4Format::E4M3,
            dsa_umma::CFormat::F32,
            dsa_umma::Major::K, dsa_umma::Major::K,
            kUMMA_M, kUMMA_N);
        issue_umma_k_commit<kKIters>(
            tmem_addr,
            make_kmajor_desc(smem_k, kSboBytes),
            make_kmajor_desc(smem_q, kSboBytes),
            instr,
            dsa_ptx::smem_ptr_to_uint(&smem_mbar),
            warp_id, lane);
    }

    dsa_ptx::mbarrier_wait_parity(dsa_ptx::smem_ptr_to_uint(&smem_mbar), 0u);
    dsa_ptx::tcgen05_fence_after_thread_sync();

    uint32_t regs[kUMMA_N];
    dsa_ptx::tcgen05_ld_32x32b_x64_b32(tmem_addr, regs);
    dsa_ptx::tcgen05_wait_ld();
    (void)warp_id; (void)lane;

    __syncthreads();
    if (warp_id == 0) {
        dsa_ptx::tcgen05_fence_before_thread_sync();
        dsa_ptx::tcgen05_dealloc_1sm(tmem_addr, kTmemCols);
    }

    // logit = scale * sum_h ReLU(acc[h]) * weight[h]. Threads past seq_len
    // write FP16 -inf so Stage 2's radix select cannot pick the padding.
    if (tid < kBlockKv) {
        const float* acc = reinterpret_cast<const float*>(regs);
        float acc_relu = 0.0f;
#pragma unroll
        for (int h = 0; h < kNumHeads; ++h) {
            const float r = acc[h] > 0.0f ? acc[h] : 0.0f;
            acc_relu = fmaf(r, smem_w[h], acc_relu);
        }
        const int   kv_abs = kv_base + tid;
        const float scaled = acc_relu * smem_kscale[tid];
        logits_b[kv_abs]   = (kv_abs < seq_len) ? __float2half_rn(scaled) : kHalfNegInf;
    }
}

}  // namespace
