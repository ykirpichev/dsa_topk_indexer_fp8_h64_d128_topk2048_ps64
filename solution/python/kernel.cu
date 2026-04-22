/*
 * CUDA DSA TopK Indexer — SM100a (Blackwell) UMMA fast path.
 * PyTorch extension build: compiled via torch.utils.cpp_extension.load().
 *
 * Pipeline (batch row `b`, context position `t`, 0 <= t < seq_lens[b]):
 *   S[h]       = sum_d Q[b,h,d] * K[page,slot,d]          (FP8 x FP8 -> FP32)
 *   logit[b,t] = scale[page,slot] * sum_h ReLU(S[h]) * weights[b,h]
 *   topk[b,:]  = top_k (k=2048) argmax of logit[b, 0:seq_lens[b]]
 *   topk[b,j]  = block_table[b, topk[b,j]/64] * 64 + (topk[b,j] % 64)
 */

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <array>
#include <mutex>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

#include "umma_desc.h"
#include "tcgen05_ptx.h"

namespace {

constexpr int kNumHeads        = 64;
constexpr int kHeadDim         = 128;
constexpr int kPageSize        = 64;
constexpr int kHeadDimWithSf   = 132;
constexpr int kPageBytes       = kPageSize * kHeadDimWithSf;  // 8448
constexpr int kScaleOffsetBytes= kPageSize * kHeadDim;        // 8192
constexpr int kTopK            = 2048;
constexpr int kBlockKv         = 64;
constexpr int kStage2Threads   = 1024;
constexpr int kRadix           = 256;
constexpr int kRadixRounds     = 2;
constexpr int kOrderedBits     = 16;

// Rank 1 (size-aware launch): two Stage-1 kernels are compiled, the host
// dispatcher picks between them based on per-row workload size.
//
//   * paged_mqa_logits_umma_kernel_short    — 1 page per UMMA, grid =
//       (max_num_pages, B). Lowest launch overhead, best for small workloads.
//   * paged_mqa_logits_umma_kernel_persistent — 2 pages per UMMA (kUMMA_M=128
//       both halves real), persistent CTAs that loop over tiles_per_cta
//       tile-pairs while Q/TMEM/mbar are loaded/alloc'd once. Multi-stage
//       cp.async K pipeline. Best for large workloads.
constexpr int kPagesPerUMMA = 2;

// Rank 5 (deeper K pipeline): persistent kernel uses a 3-stage cp.async
// pipeline for K / kscale. At iteration `i` UMMA consumes buf (i % kKVStages)
// while two newer prefetches (for tiles i+1 and i+2) are in flight into the
// other two buffers — fully hides HBM load latency for large num_tiles.
// 3 stages costs +16 KB smem vs 2 stages (~57 KB total, still well under the
// B200's 228 KB budget per CTA).
//
// We evaluated kKVStages=4 (mnp 40–63: 19.27 → 19.45 µs, +0.9%, other buckets
// flat) and found no benefit — the math path (TMEM readout + weighted sum +
// emit) is the current bottleneck, not producer latency. Staying at 3.
constexpr int kKVStages = 3;


// ============================================================================
// Stage 1 (short path): 1 page per UMMA tile, 1 tile per CTA.
// Kept verbatim from submission-v8; used for workloads where the persistent
// path's setup costs dominate.
// ============================================================================

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

    // A1: if this row fully fits in top-K, Stage 2's per-row fast path
    // will emit the output directly from (seq_lens, block_table) without
    // ever reading logits.  Skip the entire Stage-1 pipeline for this CTA.
    if (seq_len <= kTopK) return;

    const int kv_base = tile_idx * kBlockKv;
    __half*   logits_b = logits_out + static_cast<size_t>(b) *
                                        static_cast<size_t>(max_kv_tiles) * kBlockKv;

    if (kv_base >= max_kv_tiles * kBlockKv) return;

    const __half kHalfNegInf = __ushort_as_half(static_cast<unsigned short>(0xFC00));
    if (kv_base >= seq_len) {
        for (int i = tid; i < kBlockKv; i += kThreads)
            logits_b[kv_base + i] = kHalfNegInf;
        return;
    }

    __shared__ __align__(1024) __nv_fp8_e4m3 smem_q[kNumHeads * kHeadDim];
    __shared__ __align__(1024) __nv_fp8_e4m3 smem_k[kUMMA_M * kHeadDim];
    __shared__ float   smem_kscale[kBlockKv];
    __shared__ float   smem_w[kNumHeads];
    __shared__ uint32_t smem_tmem_ptr;
    __shared__ __align__(8) uint64_t smem_mbar;

    auto kSwizzle16B = [] (int idx) {
        return idx ^ ((idx >> 3) & 7);
    };

    {
        const uint4* src = reinterpret_cast<const uint4*>(
            q_fp8 + static_cast<size_t>(b) * kNumHeads * kHeadDim);
        uint4* dst = reinterpret_cast<uint4*>(smem_q);
        constexpr int kVec = (kNumHeads * kHeadDim) / 16;
#pragma unroll 4
        for (int i = tid; i < kVec; i += kThreads) dst[kSwizzle16B(i)] = src[i];
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
            dst[kSwizzle16B(i)] = src[i];
#pragma unroll 4
        for (int i = tid; i < (kVecTotal - kVecReal); i += kThreads)
            dst[kSwizzle16B(i + kVecReal)] = zero4;
    }

    if (tid < kBlockKv)
        smem_kscale[tid] = reinterpret_cast<const float*>(page_ptr + kScaleOffsetBytes)[tid];
    if (tid < kNumHeads)
        smem_w[tid] = weights[b * kNumHeads + tid];

    __syncthreads();

    if (warp_id == 0) {
        const uint32_t tmem_ptr_smem = dsa_ptx::smem_ptr_to_uint(&smem_tmem_ptr);
        dsa_ptx::tcgen05_alloc_1sm(tmem_ptr_smem, kTmemCols);
        dsa_ptx::tcgen05_relinquish_alloc_permit_1sm();
        if (lane == 0)
            dsa_ptx::mbarrier_init(dsa_ptx::smem_ptr_to_uint(&smem_mbar), 1);
        dsa_ptx::fence_barrier_init();
    }
    __syncthreads();

    const uint32_t tmem_addr = smem_tmem_ptr;

    if (warp_id == 0) {
        dsa_ptx::tcgen05_fence_after_thread_sync();
        if (lane == 0) {
            const auto instr = dsa_umma::make_instr_desc_f8f6f4(
                dsa_umma::F8F6F4Format::E4M3, dsa_umma::F8F6F4Format::E4M3,
                dsa_umma::CFormat::F32,
                dsa_umma::Major::K, dsa_umma::Major::K,
                kUMMA_M, kUMMA_N);

            const auto make_kmajor_desc = [&](const void* smem_base) {
                dsa_umma::SmemDescriptor d{};
                d.desc_                = 0;
                d.version_             = 1;
                d.lbo_mode_            = 0;
                d.base_offset_         = 0;
                d.layout_type_         = static_cast<uint8_t>(dsa_umma::LayoutType::SWIZZLE_128B);
                d.leading_byte_offset_ = 0;
                d.stride_byte_offset_  = static_cast<uint16_t>(kSboBytes >> 4);
                d.start_address_       = static_cast<uint16_t>(
                    dsa_ptx::smem_ptr_to_uint(smem_base) >> 4);
                return d;
            };

            auto a_desc_base = make_kmajor_desc(smem_k);
            auto b_desc_base = make_kmajor_desc(smem_q);

#pragma unroll
            for (int k = 0; k < kKIters; ++k) {
                dsa_umma::SmemDescriptor a = a_desc_base;
                dsa_umma::SmemDescriptor b = b_desc_base;
                const uint16_t k_step_shifted = static_cast<uint16_t>((k * kUMMA_K_B) >> 4);
                a.start_address_ = static_cast<uint16_t>(a.start_address_ + k_step_shifted);
                b.start_address_ = static_cast<uint16_t>(b.start_address_ + k_step_shifted);
                dsa_ptx::tcgen05_mma_f8f6f4_ss(tmem_addr, a, b, instr, (k == 0) ? 0u : 1u);
            }
            dsa_ptx::tcgen05_commit_1sm(dsa_ptx::smem_ptr_to_uint(&smem_mbar));
        }
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


// ============================================================================
// Stage 1 (persistent path): 2 pages per UMMA, persistent CTAs.
//
// Grid: (num_splits, B).  Each CTA owns tile-pairs
//   [cta_x * tiles_per_cta, min((cta_x+1) * tiles_per_cta, max_kv_tile_pairs)).
// Q, weights, TMEM, and mbarrier are allocated/loaded ONCE per CTA and reused
// across the inner tile-pair loop; with typical num_pages >> 1 this amortises
// the Q load cost dramatically vs the short path.
//
// smem_k is `kKVStages`-way buffered; at iteration i UMMA consumes buf
// (i % kKVStages) while cp.async prefetches for tiles i+1 .. i+(kKVStages-1)
// are in flight into the other buffers.  Each iteration issues exactly one
// prefetch (possibly empty, past the end) and calls
// cp.async.wait_group<kKVStages-1> to drain the oldest — this gives a clean
// sliding-window pipeline with no buffer aliasing.  kscale is buffered
// alongside K.
// ============================================================================

__global__ __launch_bounds__(128)
void paged_mqa_logits_umma_kernel_persistent(
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
    constexpr int kThreads   = 128;
    constexpr int kTmemCols  = kUMMA_N;
    constexpr uint32_t kSboBytes = 1024u;

    const int cta_x     = blockIdx.x;
    const int b         = blockIdx.y;
    const int tid       = threadIdx.x;
    const int warp_id   = tid / 32;
    const int lane      = tid % 32;

    const int seq_len = seq_lens[b];

    // A1: if this row fully fits in top-K, Stage 2's per-row fast path
    // will emit the output directly from (seq_lens, block_table) without
    // ever reading logits.  Skip the entire Stage-1 pipeline (incl. the
    // expensive TMEM alloc / mbar init / Q load) for this CTA.
    if (seq_len <= kTopK) return;

    const int tile_pair_begin = cta_x * tiles_per_cta;
    int tile_pair_end         = tile_pair_begin + tiles_per_cta;
    if (tile_pair_end > max_kv_tile_pairs) tile_pair_end = max_kv_tile_pairs;

    __half* logits_b = logits_out + static_cast<size_t>(b) *
                                    static_cast<size_t>(max_kv_tile_pairs * kUMMA_M);

    const __half kHalfNegInf = __ushort_as_half(static_cast<unsigned short>(0xFC00));

    if (tile_pair_begin >= max_kv_tile_pairs) return;

    __shared__ __align__(1024) __nv_fp8_e4m3 smem_q[kNumHeads * kHeadDim];
    __shared__ __align__(1024) __nv_fp8_e4m3 smem_k[kKVStages][kUMMA_M * kHeadDim];
    __shared__ float    smem_kscale[kKVStages][kUMMA_M];
    __shared__ float    smem_w[kNumHeads];
    __shared__ uint32_t smem_tmem_ptr;
    __shared__ __align__(8) uint64_t smem_mbar;

    auto kSwizzle16B = [] (int idx) {
        return idx ^ ((idx >> 3) & 7);
    };

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

    // Async K prefetch for tile `tp` into double-buffer slot `buf`.  Always
    // closes a commit group so the caller's wait_group accounting stays
    // simple, regardless of whether the tile is in range.
    auto prefetch_tile = [&](int buf, int tp) {
        const TileDesc d = describe_tile(tp);
        if (d.any_valid) {
            constexpr int kVecHalf  = (kBlockKv * kHeadDim) / 16;
            constexpr int kVecTotal = (kUMMA_M  * kHeadDim) / 16;
            const uint32_t smem_k_base = dsa_ptx::smem_ptr_to_uint(&smem_k[buf][0]);
            const uint4* src0 = reinterpret_cast<const uint4*>(d.page_ptr_0);
#pragma unroll 4
            for (int i = tid; i < kVecHalf; i += kThreads) {
                const int swz = kSwizzle16B(i);
                dsa_ptx::cp_async_16B(smem_k_base + swz * 16u, &src0[i]);
            }
            if (d.page1_valid) {
                const uint4* src1 = reinterpret_cast<const uint4*>(d.page_ptr_1);
#pragma unroll 4
                for (int i = tid; i < kVecHalf; i += kThreads) {
                    const int swz = kSwizzle16B(i + kVecHalf);
                    dsa_ptx::cp_async_16B(smem_k_base + swz * 16u, &src1[i]);
                }
            } else {
                const uint4 zero4 = make_uint4(0u, 0u, 0u, 0u);
                uint4* k_dst = reinterpret_cast<uint4*>(&smem_k[buf][0]);
#pragma unroll 4
                for (int i = tid; i < (kVecTotal - kVecHalf); i += kThreads)
                    k_dst[kSwizzle16B(i + kVecHalf)] = zero4;
            }

            if (tid < kBlockKv) {
                smem_kscale[buf][tid] = reinterpret_cast<const float*>(
                    d.page_ptr_0 + kScaleOffsetBytes)[tid];
                smem_kscale[buf][tid + kBlockKv] = d.page1_valid
                    ? reinterpret_cast<const float*>(
                          d.page_ptr_1 + kScaleOffsetBytes)[tid]
                    : 0.0f;
            }
        }
        dsa_ptx::cp_async_commit_group();
    };

    // -------- one-shot setup: Q, weights, TMEM alloc, mbar init --------
    {
        const uint4* src = reinterpret_cast<const uint4*>(
            q_fp8 + static_cast<size_t>(b) * kNumHeads * kHeadDim);
        uint4* dst = reinterpret_cast<uint4*>(smem_q);
        constexpr int kVec = (kNumHeads * kHeadDim) / 16;
#pragma unroll 4
        for (int i = tid; i < kVec; i += kThreads) dst[kSwizzle16B(i)] = src[i];
    }

    if (tid < kNumHeads) smem_w[tid] = weights[b * kNumHeads + tid];

    if (warp_id == 0) {
        const uint32_t tmem_ptr_smem = dsa_ptx::smem_ptr_to_uint(&smem_tmem_ptr);
        dsa_ptx::tcgen05_alloc_1sm(tmem_ptr_smem, kTmemCols);
        dsa_ptx::tcgen05_relinquish_alloc_permit_1sm();
        if (lane == 0)
            dsa_ptx::mbarrier_init(dsa_ptx::smem_ptr_to_uint(&smem_mbar), 1);
        dsa_ptx::fence_barrier_init();
    }

    // Pre-launch prefetches for the first (kKVStages - 1) tile-pairs.
    // Each call commits a group unconditionally (possibly empty past the
    // end), so the loop's wait_group<kKVStages-1> has a steady rhythm: it
    // always drains the single oldest group.
#pragma unroll
    for (int s = 0; s < kKVStages - 1; ++s) {
        prefetch_tile(s % kKVStages, tile_pair_begin + s);
    }

    __syncthreads();

    const uint32_t tmem_addr = smem_tmem_ptr;

    const auto instr = dsa_umma::make_instr_desc_f8f6f4(
        dsa_umma::F8F6F4Format::E4M3, dsa_umma::F8F6F4Format::E4M3,
        dsa_umma::CFormat::F32,
        dsa_umma::Major::K, dsa_umma::Major::K,
        kUMMA_M, kUMMA_N);

    const auto make_kmajor_desc = [&](const void* smem_base) {
        dsa_umma::SmemDescriptor d{};
        d.desc_                = 0;
        d.version_             = 1;
        d.lbo_mode_            = 0;
        d.base_offset_         = 0;
        d.layout_type_         = static_cast<uint8_t>(dsa_umma::LayoutType::SWIZZLE_128B);
        d.leading_byte_offset_ = 0;
        d.stride_byte_offset_  = static_cast<uint16_t>(kSboBytes >> 4);
        d.start_address_       = static_cast<uint16_t>(
            dsa_ptx::smem_ptr_to_uint(smem_base) >> 4);
        return d;
    };
    dsa_umma::SmemDescriptor a_desc_base[kKVStages];
#pragma unroll
    for (int s = 0; s < kKVStages; ++s) {
        a_desc_base[s] = make_kmajor_desc(&smem_k[s][0]);
    }
    const auto b_desc_base = make_kmajor_desc(smem_q);

    uint32_t parity = 0u;

    const int num_tiles = tile_pair_end - tile_pair_begin;

    for (int i = 0; i < num_tiles; ++i) {
        const int tp      = tile_pair_begin + i;
        const int buf     = i % kKVStages;
        const int kv_base = tp * kUMMA_M;

        // Issue the prefetch for tile (i + kKVStages - 1) BEFORE waiting,
        // so we always have (kKVStages - 1) newer groups in flight while
        // UMMA consumes tile i.  prefetch_tile commits an (empty) group
        // past the end, which keeps the in-flight count steady so
        // wait_group<kKVStages-1> drains exactly one group per iteration.
        const int prefetch_i = i + kKVStages - 1;
        prefetch_tile(prefetch_i % kKVStages, tile_pair_begin + prefetch_i);

        dsa_ptx::cp_async_wait_group<kKVStages - 1>();
        __syncthreads();

        if (kv_base >= seq_len) {
            for (int j = tid; j < kUMMA_M; j += kThreads)
                logits_b[kv_base + j] = kHalfNegInf;
            continue;
        }

        if (warp_id == 0) {
            dsa_ptx::tcgen05_fence_after_thread_sync();
            if (lane == 0) {
#pragma unroll
                for (int k = 0; k < kKIters; ++k) {
                    dsa_umma::SmemDescriptor a = a_desc_base[buf];
                    dsa_umma::SmemDescriptor b = b_desc_base;
                    const uint16_t k_step_shifted =
                        static_cast<uint16_t>((k * kUMMA_K_B) >> 4);
                    a.start_address_ = static_cast<uint16_t>(a.start_address_ + k_step_shifted);
                    b.start_address_ = static_cast<uint16_t>(b.start_address_ + k_step_shifted);
                    dsa_ptx::tcgen05_mma_f8f6f4_ss(
                        tmem_addr, a, b, instr, (k == 0) ? 0u : 1u);
                }
                dsa_ptx::tcgen05_commit_1sm(dsa_ptx::smem_ptr_to_uint(&smem_mbar));
            }
        }

        dsa_ptx::mbarrier_wait_parity(dsa_ptx::smem_ptr_to_uint(&smem_mbar), parity);
        parity ^= 1u;
        dsa_ptx::tcgen05_fence_after_thread_sync();

        uint32_t regs[kUMMA_N];
        dsa_ptx::tcgen05_ld_32x32b_x64_b32(tmem_addr, regs);
        dsa_ptx::tcgen05_wait_ld();
        (void)warp_id; (void)lane;

        if (tid < kUMMA_M) {
            const float* acc = reinterpret_cast<const float*>(regs);
            float acc_relu = 0.0f;
#pragma unroll
            for (int h = 0; h < kNumHeads; ++h) {
                const float r = acc[h] > 0.0f ? acc[h] : 0.0f;
                acc_relu = fmaf(r, smem_w[h], acc_relu);
            }
            const int   kv_abs = kv_base + tid;
            const float scaled = acc_relu * smem_kscale[buf][tid];
            logits_b[kv_abs] = (kv_abs < seq_len) ? __float2half_rn(scaled) : kHalfNegInf;
        }
    }

    __syncthreads();
    if (warp_id == 0) {
        dsa_ptx::tcgen05_fence_before_thread_sync();
        dsa_ptx::tcgen05_dealloc_1sm(tmem_addr, kTmemCols);
    }
}


// ============================================================================
// Stage 1 (persistent, warp-specialised path — Rank 3).
//
// 256 threads / CTA = 2 warpgroups:
//
//   Math warpgroup (warps 0-3, 128 threads, HIGH regs):
//     - loads Q and weights once into SMEM (producer does not participate)
//     - per tile-pair: waits on K_ready[buf], issues UMMA, waits UMMA,
//       reads TMEM, computes softmax·ReLU·weighted sum, emits logits,
//       signals K_done[buf]
//     - owns TMEM alloc/dealloc
//
//   Producer warpgroup (warps 4-7, 128 threads, LOW regs):
//     - per tile-pair: waits K_done[buf] (from kKVStages iters ago) to
//       re-use the buffer, issues cp.async for K + kscale into
//       smem_k[buf], waits for completion, signals K_ready[buf]
//
// K_ready[s] / K_done[s] are per-stage mbarriers with count=1; the
// handoff uses parity-based wait_parity.  Math and producer run on
// opposite phases of the pipeline — the producer is filling stage s+1
// while the math group computes and emits from stage s.
//
// Register reconfig: math warpgroup requests 232 regs/thread, producer
// surrenders down to 40 regs/thread (setmaxnreg.inc/dec).
// ============================================================================

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
    if (tile_pair_begin >= max_kv_tile_pairs) return;

    __shared__ __align__(1024) __nv_fp8_e4m3 smem_q[kNumHeads * kHeadDim];
    __shared__ __align__(1024) __nv_fp8_e4m3 smem_k[kKVStages][kUMMA_M * kHeadDim];
    __shared__ float    smem_kscale[kKVStages][kUMMA_M];
    __shared__ float    smem_w[kNumHeads];
    __shared__ uint32_t smem_tmem_ptr;
    __shared__ __align__(8) uint64_t smem_k_ready[kKVStages];
    __shared__ __align__(8) uint64_t smem_k_done[kKVStages];
    __shared__ __align__(8) uint64_t smem_umma_done;

    auto kSwizzle16B = [] (int idx) {
        return idx ^ ((idx >> 3) & 7);
    };

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

    // One-shot setup: warp 0 allocs TMEM, inits mbarriers.
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

    // Math warpgroup loads Q and weights; producer does nothing here.
    if (is_math) {
        const uint4* src = reinterpret_cast<const uint4*>(
            q_fp8 + static_cast<size_t>(b) * kNumHeads * kHeadDim);
        uint4* dst = reinterpret_cast<uint4*>(smem_q);
        constexpr int kVec = (kNumHeads * kHeadDim) / 16;
#pragma unroll 4
        for (int i = math_tid; i < kVec; i += kMathThreads) {
            dst[kSwizzle16B(i)] = src[i];
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
                    const int swz = kSwizzle16B(i);
                    dsa_ptx::cp_async_16B(smem_k_base + swz * 16u, &src0[i]);
                }
                if (d.page1_valid) {
                    const uint4* src1 = reinterpret_cast<const uint4*>(d.page_ptr_1);
#pragma unroll 4
                    for (int i = prod_tid; i < kVecHalf; i += kProducerThreads) {
                        const int swz = kSwizzle16B(i + kVecHalf);
                        dsa_ptx::cp_async_16B(smem_k_base + swz * 16u, &src1[i]);
                    }
                } else {
                    const uint4 zero4 = make_uint4(0u, 0u, 0u, 0u);
                    uint4* k_dst = reinterpret_cast<uint4*>(&smem_k[buf][0]);
#pragma unroll 4
                    for (int i = prod_tid; i < (kVecTotal - kVecHalf); i += kProducerThreads)
                        k_dst[kSwizzle16B(i + kVecHalf)] = zero4;
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

            // Wait for math to finish using buf from its previous visit.
            // Skip for the first visit to each buffer (i < kKVStages).
            if (i >= kKVStages) {
                dsa_ptx::mbarrier_wait_parity(
                    dsa_ptx::smem_ptr_to_uint(&smem_k_done[buf]),
                    done_parity[buf]);
                done_parity[buf] ^= 1u;
            }

            prefetch_tile_producer(buf, tile_pair_begin + i);

            // Wait until THIS tile's cp.async has retired, then signal math.
            // cp.async.wait_group is per-thread, so we additionally sync the
            // producer warpgroup to make every thread's cp.async writes
            // visible before the single-thread mbarrier arrive.  Without
            // this, math may consume smem_k before threads >0's cp.asyncs
            // have retired to the coherence point observable by math.
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

        const auto make_kmajor_desc = [&](const void* smem_base) {
            dsa_umma::SmemDescriptor d{};
            d.desc_                = 0;
            d.version_             = 1;
            d.lbo_mode_            = 0;
            d.base_offset_         = 0;
            d.layout_type_         = static_cast<uint8_t>(dsa_umma::LayoutType::SWIZZLE_128B);
            d.leading_byte_offset_ = 0;
            d.stride_byte_offset_  = static_cast<uint16_t>(kSboBytes >> 4);
            d.start_address_       = static_cast<uint16_t>(
                dsa_ptx::smem_ptr_to_uint(smem_base) >> 4);
            return d;
        };
        dsa_umma::SmemDescriptor a_desc_base[kKVStages];
#pragma unroll
        for (int s = 0; s < kKVStages; ++s) {
            a_desc_base[s] = make_kmajor_desc(&smem_k[s][0]);
        }
        const auto b_desc_base = make_kmajor_desc(smem_q);

        uint32_t umma_parity = 0u;
        uint32_t ready_parity[kKVStages] = {};

        for (int i = 0; i < num_tiles; ++i) {
            const int tp      = tile_pair_begin + i;
            const int buf     = i % kKVStages;
            const int kv_base = tp * kUMMA_M;

            // Wait for producer to have K[i] ready in smem_k[buf].
            dsa_ptx::mbarrier_wait_parity(
                dsa_ptx::smem_ptr_to_uint(&smem_k_ready[buf]),
                ready_parity[buf]);
            ready_parity[buf] ^= 1u;

            if (kv_base >= seq_len) {
                for (int j = math_tid; j < kUMMA_M; j += kMathThreads)
                    logits_b[kv_base + j] = kHalfNegInf;
                // Math-warpgroup-only sync, then signal K_done.
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
                        dsa_umma::SmemDescriptor a = a_desc_base[buf];
                        dsa_umma::SmemDescriptor b2 = b_desc_base;
                        const uint16_t k_step_shifted =
                            static_cast<uint16_t>((k * kUMMA_K_B) >> 4);
                        a.start_address_  = static_cast<uint16_t>(a.start_address_ + k_step_shifted);
                        b2.start_address_ = static_cast<uint16_t>(b2.start_address_ + k_step_shifted);
                        dsa_ptx::tcgen05_mma_f8f6f4_ss(
                            tmem_addr, a, b2, instr, (k == 0) ? 0u : 1u);
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

            // Sync math warpgroup, then release buffer back to producer.
            dsa_ptx::named_barrier_sync(kMathBarrierId, kMathThreads);
            if (math_tid == 0) {
                (void)dsa_ptx::mbarrier_arrive(
                    dsa_ptx::smem_ptr_to_uint(&smem_k_done[buf]));
            }
        }

        // Epilogue: TMEM dealloc (math warp 0 only).
        dsa_ptx::named_barrier_sync(kMathBarrierId, kMathThreads);
        if (warp_id == 0) {
            dsa_ptx::tcgen05_fence_before_thread_sync();
            dsa_ptx::tcgen05_dealloc_1sm(tmem_addr, kTmemCols);
        }
    }
}


// ============================================================================
// Stage 2: Top-K (k=2048) + page table transform
// ============================================================================

__device__ __forceinline__ uint16_t HalfToOrderedU16(__half h) {
    const uint16_t bits = __half_as_ushort(h);
    return (bits & 0x8000u) ? static_cast<uint16_t>(~bits)
                            : static_cast<uint16_t>(bits ^ 0x8000u);
}

// Merged from v2 `fused-topk-v1` (Stage-2 ordered-key SMEM cache): when
// `seq_len <= kStage2MaxCachedLen` (16384 tokens, 32 KB), we populate
// `smem_ordered[i] = HalfToOrderedU16(row_logits[i])` on round 0 and read
// the cache on rounds 1+, the gt-count pass, and both emit passes.
// Saves 4x HBM reads of `row_logits` per element + 4x fp16->ordered-u16
// conversions for every workload whose Stage-1 output fits in 32 KB
// (basically all of them given kTopK=2048, since the persistent kernel's
// `max_kv_tile_pairs * 128` is almost always <= 16384 in this bench).
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
            // Round 0: convert and cache, unconditionally bucketize
            // (prefix_mask == 0, so the filter trivially accepts all).
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

// ============================================================================
// Stage 1+2 fast path: when max_num_pages * kPageSize <= kTopK, every
// seq_len in the batch is guaranteed <= kTopK, so the top-K output is
// simply the block-table-transformed indices [0, seq_len) padded with -1.
// The values do NOT depend on Q, K, or weights — we can skip Stage 1 and
// the full Stage 2 entirely and emit directly.
//
// Block-table slice (max_num_pages <= 32 here) is cached in SMEM once so
// the inner write loop is a pure gather without repeated gmem reads.
// ============================================================================

// Merged from v2 fused-topk-v1 (commit 65866ae, "Fast-path host-overhead
// cleanup").  The kernel itself moves from 256 threads / scalar int stores
// to 128 threads / 4 vectorised int4 stores per thread (512 int4 = 2048
// ints total = kTopK).  The 128-thread variant has lower launch overhead
// on B200 (half the warp-scheduler init) and the int4 stores halve HBM
// transactions on the inner loop.  Out-of-bounds block-table reads are
// avoided by zero-padding smem_bt[] up to kMaxPagesInFastPath (those
// lanes always emit -1 because their token index i >= max_num_pages *
// kPageSize >= seq_len).
__global__ __launch_bounds__(128)
void topk_fast_path_kernel(
    const int* __restrict__ seq_lens,
    const int* __restrict__ block_table,
    int max_num_pages,
    int /*top_k*/,                // kTopK is a compile-time constant
    int* __restrict__ out_indices)
{
    constexpr int kMaxPagesInFastPath = kTopK / kPageSize;   // 32
    constexpr int kInts4PerRow        = kTopK / 4;           // 512 int4 per row
    constexpr int kInt4PerThread      = kInts4PerRow / 128;  // 4 int4 per thread

    const int b   = blockIdx.x;
    const int tid = threadIdx.x;

    __shared__ int smem_bt[kMaxPagesInFastPath];
    if (tid < kMaxPagesInFastPath) {
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

}  // anonymous namespace


// ============================================================================
// Host dispatch (PyTorch extension entry point)
// ============================================================================

void dsa_topk_indexer_cuda(
    torch::Tensor q_index_fp8,
    torch::Tensor k_index_cache_fp8,
    torch::Tensor weights,
    torch::Tensor seq_lens,
    torch::Tensor block_table,
    torch::Tensor topk_indices)
{
    // Merged from v2 fused-topk-v1 (commit 65866ae, "Fast-path host-overhead
    // cleanup").  69/128 bench workloads hit the fast path; on those
    // workloads every nanosecond of host-side CPU overhead between
    // torch.cuda.Event(start).record() and the kernel launch is charged
    // to the measurement.  So we only run the TORCH_CHECKs strictly
    // needed for the fast path here and defer Q / K / weights validation
    // to the slow-path section below.  Overhead budget (pre-cleanup):
    //   cudaGetDeviceProperties: 2-4 us   -> behind std::call_once
    //   12 TORCH_CHECKs + 6 data_ptr: ~1 us -> 4 checks + 3 data_ptr here
    //   <<<>>> launch: ~2.5 us
    //   kernel exec: ~0.3-0.5 us
    TORCH_CHECK(block_table.is_cuda() && block_table.is_contiguous() &&
                    block_table.scalar_type() == torch::kInt32,
                "block_table must be int32 contiguous CUDA");
    TORCH_CHECK(seq_lens.is_cuda() && seq_lens.is_contiguous() &&
                    seq_lens.scalar_type() == torch::kInt32,
                "seq_lens must be int32 contiguous CUDA");
    TORCH_CHECK(topk_indices.is_cuda() && topk_indices.is_contiguous() &&
                    topk_indices.scalar_type() == torch::kInt32,
                "topk_indices must be int32 contiguous CUDA");

    const int B             = static_cast<int>(block_table.size(0));
    const int max_num_pages = static_cast<int>(block_table.size(1));
    TORCH_CHECK(topk_indices.size(0) == B && topk_indices.size(1) == kTopK,
                "topk_indices shape must be [B, 2048]");

    const c10::DeviceIndex dev_index = block_table.get_device();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(dev_index).stream();

    const int* sl_ptr  = seq_lens.data_ptr<int32_t>();
    const int* bt_ptr  = block_table.data_ptr<int32_t>();
    int*       out_ptr = topk_indices.data_ptr<int32_t>();

    // Fast path: when the full paged context fits in top-K, the output is
    // the block-table-transformed indices 0..seq_len-1 padded with -1 —
    // independent of Q, K, and weights.  Skip Stage 1 and the full Stage 2
    // entirely and launch only a tiny gather kernel.
    //
    // max_num_pages * kPageSize <= kTopK  <=>  max_num_pages <= kTopK/kPageSize.
    constexpr int kFastPathMaxPages = kTopK / kPageSize;   // 32
    if (max_num_pages <= kFastPathMaxPages) {
        dim3 grid(B);
        dim3 block(128);
        topk_fast_path_kernel<<<grid, block, 0, stream>>>(
            sl_ptr, bt_ptr, max_num_pages, kTopK, out_ptr);
        return;
    }

    // --- Slow-path validation (mnp > 32): only paid when we need UMMA.
    TORCH_CHECK(q_index_fp8.is_cuda() && q_index_fp8.is_contiguous(),
                "q_index_fp8 must be contiguous CUDA");
    TORCH_CHECK(k_index_cache_fp8.is_cuda() && k_index_cache_fp8.is_contiguous(),
                "k_index_cache_fp8 must be contiguous CUDA");
    TORCH_CHECK(weights.is_cuda() && weights.is_contiguous() &&
                    weights.scalar_type() == torch::kFloat32,
                "weights must be float32 contiguous CUDA");

    // Size-aware Stage 1 dispatch (Rank 1 improvement).
    //
    // Empirically, on B200 the submission-v8 "short" kernel wins on small
    // per-row workloads (max_num_pages <= ~32) where the persistent kernel's
    // one-shot setup (Q/TMEM/mbar + first K prefetch) is a big fraction of
    // total time and where the grid already saturates the SMs.  On larger
    // rows the persistent kernel wins because it (a) halves the UMMA count
    // (2 pages per UMMA), (b) eliminates redundant Q reads, and (c) overlaps
    // K fetches with UMMA via cp.async double-buffering.
    //
    // Threshold chosen on the conservative side of measured data — tunable.
    constexpr int kPersistentPageThreshold = 40;
    const bool use_persistent = (max_num_pages >= kPersistentPageThreshold);

    const int max_kv_tile_pairs =
        (max_num_pages + kPagesPerUMMA - 1) / kPagesPerUMMA;

    // Logits tensor layout must match whatever Stage 1 writes.  The
    // persistent kernel writes max_kv_tile_pairs * kUMMA_M = 2-page-rounded
    // columns per row; the short kernel writes exactly max_num_pages *
    // kPageSize.  The rounded form is always >= the exact form, so Stage 2
    // (which iterates up to seq_len anyway) works with either.
    const int max_len = use_persistent
        ? max_kv_tile_pairs * (kPagesPerUMMA * kBlockKv)
        : max_num_pages * kPageSize;

    // Merged from v2 fused-topk-v1 (commit aacdc8b, "Persistent Stage-1
    // logits scratch (phase-3)").  PyTorch's caching allocator usually
    // reuses the same block across identically-sized calls, but the
    // FlashInfer-Bench sweep issues many different (B, max_len) per
    // session, which forces the allocator to walk its free-list (and
    // sometimes cudaMalloc) on every transition.  Hoisting the Stage-1
    // output buffer to per-device function-static state with a
    // power-of-two grow policy pays torch::empty() once per size-class
    // and reuses the pointer for all subsequent calls.  Both Stage-1
    // kernels (short and persistent WS / non-WS) write into the same
    // workspace; they are serialised on the same CUDA stream so no
    // aliasing concern.  Keyed by device index so multi-GPU setups each
    // latch their own max-size block.
    __half* logits_ptr;
    {
        static std::array<torch::Tensor, 8> s_scratch;
        static std::array<size_t, 8>        s_scratch_bytes = {};
        static std::array<std::mutex, 8>    s_scratch_mu;
        const int idx = (dev_index >= 0 && dev_index < 8) ? int(dev_index) : 0;
        const size_t needed_bytes = static_cast<size_t>(B) *
                                    static_cast<size_t>(max_len) *
                                    sizeof(__half);
        std::lock_guard<std::mutex> g(s_scratch_mu[idx]);
        if (s_scratch_bytes[idx] < needed_bytes) {
            size_t new_bytes = s_scratch_bytes[idx] ? s_scratch_bytes[idx]
                                                    : needed_bytes;
            while (new_bytes < needed_bytes) new_bytes <<= 1;
            const int64_t new_elems =
                static_cast<int64_t>(new_bytes / sizeof(__half));
            s_scratch[idx] = torch::empty(
                {new_elems},
                torch::TensorOptions()
                    .dtype(torch::kFloat16)
                    .device(q_index_fp8.device()));
            s_scratch_bytes[idx] = new_bytes;
        }
        logits_ptr = reinterpret_cast<__half*>(
            s_scratch[idx].data_ptr<at::Half>());
    }

    if (use_persistent) {
        // Pick tiles_per_cta so (a) we have enough CTAs to keep ~132 SMs
        // busy and (b) each CTA amortises Q load over many tile-pairs.
        constexpr int kSmTarget = 132 * 4;
        constexpr int kMinTiles = 4;
        constexpr int kMaxTiles = 64;
        int num_splits = (kSmTarget + B - 1) / B;
        if (num_splits < 1) num_splits = 1;
        if (num_splits > max_kv_tile_pairs) num_splits = max_kv_tile_pairs;
        int tiles_per_cta = (max_kv_tile_pairs + num_splits - 1) / num_splits;
        if (tiles_per_cta < kMinTiles) tiles_per_cta = kMinTiles;
        if (tiles_per_cta > kMaxTiles) tiles_per_cta = kMaxTiles;
        num_splits = (max_kv_tile_pairs + tiles_per_cta - 1) / tiles_per_cta;

        dim3 grid(num_splits, B);

        // Rank 3: warp-specialised persistent kernel (256 threads/CTA) is
        // the default on the persistent path.  Set DSA_TOPK_DISABLE_WS=1
        // to fall back to the single-warpgroup persistent kernel (useful
        // for A/B profiling).
        static const bool disable_ws = [] {
            const char* env = std::getenv("DSA_TOPK_DISABLE_WS");
            return env && env[0] != '0' && env[0] != '\0';
        }();
        const bool use_ws = !disable_ws;

        const auto* q_ptr  = reinterpret_cast<const __nv_fp8_e4m3*>(q_index_fp8.data_ptr());
        const auto* kv_ptr = reinterpret_cast<const uint8_t*>(k_index_cache_fp8.data_ptr());
        const float* w_ptr = weights.data_ptr<float>();
        if (use_ws) {
            dim3 block(256);
            paged_mqa_logits_umma_kernel_persistent_ws<<<grid, block, 0, stream>>>(
                q_ptr, kv_ptr, w_ptr, sl_ptr, bt_ptr,
                max_num_pages, max_kv_tile_pairs, tiles_per_cta,
                logits_ptr);
        } else {
            dim3 block(128);
            paged_mqa_logits_umma_kernel_persistent<<<grid, block, 0, stream>>>(
                q_ptr, kv_ptr, w_ptr, sl_ptr, bt_ptr,
                max_num_pages, max_kv_tile_pairs, tiles_per_cta,
                logits_ptr);
        }
    } else {
        dim3 grid(max_num_pages, B);
        dim3 block(128);
        paged_mqa_logits_umma_kernel_short<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_fp8_e4m3*>(q_index_fp8.data_ptr()),
            reinterpret_cast<const uint8_t*>(k_index_cache_fp8.data_ptr()),
            weights.data_ptr<float>(),
            sl_ptr, bt_ptr,
            max_num_pages, max_num_pages,
            logits_ptr);
    }

    {
        dim3 grid(B);
        dim3 block(kStage2Threads);
        topk_page_table_transform_kernel<<<grid, block, 0, stream>>>(
            logits_ptr, sl_ptr, bt_ptr,
            max_len, max_num_pages, kTopK, out_ptr);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &dsa_topk_indexer_cuda, "DSA TopK indexer (SM100a UMMA)");
}
