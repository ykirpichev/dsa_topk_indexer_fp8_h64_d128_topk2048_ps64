// Stage 1 for large contexts (max_num_pages >= kPersistentPageThreshold):
// warp-specialised and persistent, so Q, weights, TMEM and the mbarriers are
// set up once per CTA and reused across all of its tile-pairs.

#pragma once

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "dsa_config.cuh"
#include "stage1_common.cuh"
#include "tcgen05_ptx.h"
#include "umma_desc.h"

namespace {

// 256 threads = 2 warpgroups. Warps 0-3 (math, 232 regs via setmaxnreg.inc)
// own TMEM, issue UMMA, read the accumulator back and emit logits. Warps 4-7
// (producer, down to 40 regs) stream K/kscale into smem_k through the
// kKVStages-deep cp.async pipeline. Handoff is two per-stage count=1
// mbarriers, K_ready and K_done, waited on by parity, so the producer fills
// stage s+1 while math drains stage s.

__global__ __launch_bounds__(256)
void paged_mqa_logits_umma_kernel_persistent_ws(
    const __nv_fp8_e4m3* __restrict__ q_fp8,
    const uint8_t*       __restrict__ kv_cache,
    const float*         __restrict__ weights,
    const int*           __restrict__ seq_lens,
    const int*           __restrict__ block_table,
    int                               max_num_pages,
    int                               max_kv_tile_pairs,
    int                               tiles_per_cta,
    __half*              __restrict__ logits_out)
{
    constexpr int kUMMA_M    = kPagesPerUMMA * kBlockKv;  // 128
    constexpr int kUMMA_N    = kNumHeads;                 // 64
    constexpr int kUMMA_K_B  = 32;
    constexpr int kKIters    = kHeadDim / kUMMA_K_B;
    constexpr int kMathThreads     = 128;
    constexpr int kProducerThreads = 128;
    constexpr int kTmemCols  = kUMMA_N;
    constexpr uint32_t kSboBytes = 1024u;
    constexpr int kMathBarrierId     = 1;
    constexpr int kProducerBarrierId = 2;

    const int cta_x     = blockIdx.x;
    const int b         = blockIdx.y;
    const int tid       = threadIdx.x;
    const int warp_id   = tid / 32;
    const int lane      = tid % 32;
    const bool is_math  = (warp_id < 4);
    const int  math_tid = tid;
    const int  prod_tid = tid - kMathThreads;

    const int seq_len = seq_lens[b];
    if (seq_len <= kTopK) return;

    const int tile_pair_begin = cta_x * tiles_per_cta;
    int tile_pair_end         = tile_pair_begin + tiles_per_cta;
    if (tile_pair_end > max_kv_tile_pairs) tile_pair_end = max_kv_tile_pairs;

    __half* logits_b = logits_out + static_cast<size_t>(b) *
                                    static_cast<size_t>(max_kv_tile_pairs * kUMMA_M);

    const __half kHalfNegInf = __ushort_as_half(static_cast<unsigned short>(0xFC00));

    // Both this and the range check in describe_tile below are unreachable --
    // the host sets num_splits = ceil(max_kv_tile_pairs / tiles_per_cta) -- but
    // they change how the compiler schedules the pipeline. Removing the
    // describe_tile one alone costs 11.6% on B200; this one measured within
    // noise but is kept for the same reason.
    if (tile_pair_begin >= max_kv_tile_pairs) return;


    __shared__ __align__(1024) __nv_fp8_e4m3 smem_q[kNumHeads * kHeadDim];
    __shared__ __align__(1024) __nv_fp8_e4m3 smem_k[kKVStages][kUMMA_M * kHeadDim];
    __shared__ float    smem_kscale[kKVStages][kUMMA_M];
    __shared__ float    smem_w[kNumHeads];
    __shared__ uint32_t smem_tmem_ptr;
    __shared__ __align__(8) uint64_t smem_k_ready[kKVStages];
    __shared__ __align__(8) uint64_t smem_k_done[kKVStages];
    __shared__ __align__(8) uint64_t smem_umma_done;

    struct TileDesc {
        bool           any_valid;
        bool           page1_valid;
        const uint8_t* page_ptr_0;
        const uint8_t* page_ptr_1;
    };

    auto describe_tile = [&](int tp) -> TileDesc {
        TileDesc d{};
        if (tp < tile_pair_begin || tp >= tile_pair_end) return d;
        const int kv_base = tp * kUMMA_M;
        if (kv_base >= seq_len) return d;
        d.any_valid = true;
        const int page0_btidx = tp * kPagesPerUMMA;
        const int page1_btidx = tp * kPagesPerUMMA + 1;
        d.page1_valid = (page1_btidx < max_num_pages) &&
                        (kv_base + kBlockKv < seq_len);
        const int page0_idx = block_table[b * max_num_pages + page0_btidx];
        const int page1_idx = d.page1_valid
            ? block_table[b * max_num_pages + page1_btidx] : 0;
        d.page_ptr_0 = kv_cache + static_cast<size_t>(page0_idx) * kPageBytes;
        d.page_ptr_1 = kv_cache + static_cast<size_t>(page1_idx) * kPageBytes;
        return d;
    };

    if (warp_id == 0) {
        const uint32_t tmem_ptr_smem = dsa_ptx::smem_ptr_to_uint(&smem_tmem_ptr);
        dsa_ptx::tcgen05_alloc_1sm(tmem_ptr_smem, kTmemCols);
        dsa_ptx::tcgen05_relinquish_alloc_permit_1sm();
        if (lane == 0) {
#pragma unroll
            for (int s = 0; s < kKVStages; ++s) {
                dsa_ptx::mbarrier_init(
                    dsa_ptx::smem_ptr_to_uint(&smem_k_ready[s]), 1);
                dsa_ptx::mbarrier_init(
                    dsa_ptx::smem_ptr_to_uint(&smem_k_done[s]), 1);
            }
            dsa_ptx::mbarrier_init(
                dsa_ptx::smem_ptr_to_uint(&smem_umma_done), 1);
        }
        dsa_ptx::fence_barrier_init();
    }

    if (is_math) {
        const uint4* src = reinterpret_cast<const uint4*>(
            q_fp8 + static_cast<size_t>(b) * kNumHeads * kHeadDim);
        uint4* dst = reinterpret_cast<uint4*>(smem_q);
        constexpr int kVec = (kNumHeads * kHeadDim) / 16;
#pragma unroll 4
        for (int i = math_tid; i < kVec; i += kMathThreads) {
            dst[swizzle_16b(i)] = src[i];
        }
        if (math_tid < kNumHeads) smem_w[math_tid] = weights[b * kNumHeads + math_tid];
    }

    __syncthreads();   // barriers init + Q/weights visible

    const int num_tiles = tile_pair_end - tile_pair_begin;

    if (!is_math) {
        // ======================= PRODUCER WARPGROUP =======================
        dsa_ptx::warpgroup_reg_dealloc<40>();

        auto prefetch_tile_producer = [&](int buf, int tp) {
            const TileDesc d = describe_tile(tp);
            if (d.any_valid) {
                constexpr int kVecHalf  = (kBlockKv * kHeadDim) / 16;
                constexpr int kVecTotal = (kUMMA_M  * kHeadDim) / 16;
                const uint32_t smem_k_base = dsa_ptx::smem_ptr_to_uint(&smem_k[buf][0]);
                const uint4* src0 = reinterpret_cast<const uint4*>(d.page_ptr_0);
#pragma unroll 4
                for (int i = prod_tid; i < kVecHalf; i += kProducerThreads) {
                    const int swz = swizzle_16b(i);
                    dsa_ptx::cp_async_16B(smem_k_base + swz * 16u, &src0[i]);
                }
                if (d.page1_valid) {
                    const uint4* src1 = reinterpret_cast<const uint4*>(d.page_ptr_1);
#pragma unroll 4
                    for (int i = prod_tid; i < kVecHalf; i += kProducerThreads) {
                        const int swz = swizzle_16b(i + kVecHalf);
                        dsa_ptx::cp_async_16B(smem_k_base + swz * 16u, &src1[i]);
                    }
                } else {
                    const uint4 zero4 = make_uint4(0u, 0u, 0u, 0u);
                    uint4* k_dst = reinterpret_cast<uint4*>(&smem_k[buf][0]);
#pragma unroll 4
                    for (int i = prod_tid; i < (kVecTotal - kVecHalf); i += kProducerThreads)
                        k_dst[swizzle_16b(i + kVecHalf)] = zero4;
                }
                if (prod_tid < kBlockKv) {
                    smem_kscale[buf][prod_tid] = reinterpret_cast<const float*>(
                        d.page_ptr_0 + kScaleOffsetBytes)[prod_tid];
                    smem_kscale[buf][prod_tid + kBlockKv] = d.page1_valid
                        ? reinterpret_cast<const float*>(
                              d.page_ptr_1 + kScaleOffsetBytes)[prod_tid]
                        : 0.0f;
                }
            }
            dsa_ptx::cp_async_commit_group();
        };

        uint32_t done_parity[kKVStages] = {};

        for (int i = 0; i < num_tiles; ++i) {
            const int buf = i % kKVStages;

            // No wait on the first visit to each buffer.
            if (i >= kKVStages) {
                dsa_ptx::mbarrier_wait_parity(
                    dsa_ptx::smem_ptr_to_uint(&smem_k_done[buf]),
                    done_parity[buf]);
                done_parity[buf] ^= 1u;
            }

            prefetch_tile_producer(buf, tile_pair_begin + i);

            // cp.async.wait_group only covers the calling thread, so the
            // warpgroup barrier is required before the single-thread arrive:
            // otherwise math can consume smem_k while other producer threads'
            // copies are still in flight.
            dsa_ptx::cp_async_wait_group<0>();
            dsa_ptx::named_barrier_sync(kProducerBarrierId, kProducerThreads);
            if (prod_tid == 0) {
                (void)dsa_ptx::mbarrier_arrive(
                    dsa_ptx::smem_ptr_to_uint(&smem_k_ready[buf]));
            }
        }

    } else {
        // ========================= MATH WARPGROUP =========================
        dsa_ptx::warpgroup_reg_alloc<232>();

        const uint32_t tmem_addr = smem_tmem_ptr;

        const auto instr = dsa_umma::make_instr_desc_f8f6f4(
            dsa_umma::F8F6F4Format::E4M3, dsa_umma::F8F6F4Format::E4M3,
            dsa_umma::CFormat::F32,
            dsa_umma::Major::K, dsa_umma::Major::K,
            kUMMA_M, kUMMA_N);

        dsa_umma::SmemDescriptor a_desc_base[kKVStages];
#pragma unroll
        for (int s = 0; s < kKVStages; ++s) {
            a_desc_base[s] = make_kmajor_desc(&smem_k[s][0], kSboBytes);
        }
        const auto b_desc_base = make_kmajor_desc(smem_q, kSboBytes);

        uint32_t umma_parity = 0u;
        uint32_t ready_parity[kKVStages] = {};

        for (int i = 0; i < num_tiles; ++i) {
            const int tp      = tile_pair_begin + i;
            const int buf     = i % kKVStages;
            const int kv_base = tp * kUMMA_M;

            dsa_ptx::mbarrier_wait_parity(
                dsa_ptx::smem_ptr_to_uint(&smem_k_ready[buf]),
                ready_parity[buf]);
            ready_parity[buf] ^= 1u;

            if (kv_base >= seq_len) {
                for (int j = math_tid; j < kUMMA_M; j += kMathThreads)
                    logits_b[kv_base + j] = kHalfNegInf;
                dsa_ptx::named_barrier_sync(kMathBarrierId, kMathThreads);
                if (math_tid == 0) {
                    (void)dsa_ptx::mbarrier_arrive(
                        dsa_ptx::smem_ptr_to_uint(&smem_k_done[buf]));
                }
                continue;
            }

            if (warp_id == 0) {
                dsa_ptx::tcgen05_fence_after_thread_sync();
                if (lane == 0) {
#pragma unroll
                    for (int k = 0; k < kKIters; ++k) {
                        const int k_off = k * kUMMA_K_B;
                        dsa_ptx::tcgen05_mma_f8f6f4_ss(
                            tmem_addr,
                            advance_k(a_desc_base[buf], k_off),
                            advance_k(b_desc_base, k_off),
                            instr, (k == 0) ? 0u : 1u);
                    }
                    dsa_ptx::tcgen05_commit_1sm(
                        dsa_ptx::smem_ptr_to_uint(&smem_umma_done));
                }
            }

            dsa_ptx::mbarrier_wait_parity(
                dsa_ptx::smem_ptr_to_uint(&smem_umma_done), umma_parity);
            umma_parity ^= 1u;
            dsa_ptx::tcgen05_fence_after_thread_sync();

            uint32_t regs[kUMMA_N];
            dsa_ptx::tcgen05_ld_32x32b_x64_b32(tmem_addr, regs);
            dsa_ptx::tcgen05_wait_ld();

            if (math_tid < kUMMA_M) {
                const float* acc = reinterpret_cast<const float*>(regs);
                float acc_relu = 0.0f;
#pragma unroll
                for (int h = 0; h < kNumHeads; ++h) {
                    const float r = acc[h] > 0.0f ? acc[h] : 0.0f;
                    acc_relu = fmaf(r, smem_w[h], acc_relu);
                }
                const int   kv_abs = kv_base + math_tid;
                const float scaled = acc_relu * smem_kscale[buf][math_tid];
                logits_b[kv_abs] = (kv_abs < seq_len) ? __float2half_rn(scaled) : kHalfNegInf;
            }

            // Release the buffer back to the producer.
            dsa_ptx::named_barrier_sync(kMathBarrierId, kMathThreads);
            if (math_tid == 0) {
                (void)dsa_ptx::mbarrier_arrive(
                    dsa_ptx::smem_ptr_to_uint(&smem_k_done[buf]));
            }
        }

        dsa_ptx::named_barrier_sync(kMathBarrierId, kMathThreads);
        if (warp_id == 0) {
            dsa_ptx::tcgen05_fence_before_thread_sync();
            dsa_ptx::tcgen05_dealloc_1sm(tmem_addr, kTmemCols);
        }
    }
}

}  // namespace
