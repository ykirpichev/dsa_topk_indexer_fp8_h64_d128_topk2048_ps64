/*
 * CUDA DSA TopK Indexer — SM100a (Blackwell) UMMA fast path.
 * PyTorch extension build: compiled via torch.utils.cpp_extension.load().
 *
 * Pipeline (batch row `b`, context position `t`, 0 <= t < seq_lens[b]):
 *   S[h]       = sum_d Q[b,h,d] * K[page,slot,d]          (FP8 x FP8 -> FP32)
 *   logit[b,t] = scale[page,slot] * sum_h ReLU(S[h]) * weights[b,h]
 *   topk[b,:]  = top_k (k=2048) argmax of logit[b, 0:seq_lens[b]]
 *   topk[b,j]  = block_table[b, topk[b,j]/64] * 64 + (topk[b,j] % 64)
 *
 * Opt 1b: 2 pages per UMMA.
 * kUMMA_M=128 = 2 * kBlockKv, so we pack two consecutive KV pages into each
 * UMMA call instead of zero-padding the upper 64 rows.
 *
 * Opt 1: Persistent per-batch CTA (Q and TMEM loaded/allocated once).
 * Instead of launching one CTA per 2-page tile, we launch `num_splits` CTAs
 * per batch row.  Each CTA owns a contiguous range of tile-pairs and loops
 * over them, loading Q exactly once and allocating TMEM exactly once.
 *   grid = (num_splits, B)
 * `num_splits` is chosen on the host so the grid is large enough to saturate
 * the B200's ~132 SMs for small batch sizes but each CTA still amortises the
 * Q load over many tile-pairs.  For typical num_pages=11923, B=1 this
 * eliminates 95 MB of redundant Q reads.
 *
 * Opt 3: Double-buffered K loads (cp.async pipelining).
 * smem_k is 2-way double-buffered; at iteration `i` UMMA consumes buf (i&1)
 * while a cp.async prefetch for tile `i+1` fills the OPPOSITE buffer
 * ((i+1)&1).  That keeps exactly one cp.async group in flight at a time
 * without the buffer-aliasing race that a 2-buffer / i+2 prefetch would
 * create (tile `i+2` and tile `i` share the same buffer).  kscale is
 * double-buffered alongside K.
 */

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

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

// kPagesPerUMMA: pages processed in a single UMMA call.
// kUMMA_M = kPagesPerUMMA * kBlockKv = 2 * 64 = 128 (unchanged).
// Previously only the first kBlockKv rows were real; the rest were zero-padded.
// Now both halves carry real KV data, doubling effective UMMA throughput.
constexpr int kPagesPerUMMA    = 2;


// ============================================================================
// Stage 1: Paged MQA logits via tcgen05.mma.kind::f8f6f4 (SM100a UMMA)
//
// Grid: (num_splits, B).  Each CTA owns tile-pairs
// [cta_x * tiles_per_cta, min((cta_x+1) * tiles_per_cta, max_kv_tile_pairs)).
// Q, weights, TMEM, and mbarrier are allocated/loaded once per CTA and
// reused across the inner tile-pair loop.
// ============================================================================

__global__ __launch_bounds__(128)
void paged_mqa_logits_umma_kernel(
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

    const int cta_x     = blockIdx.x;   // split index within this batch row
    const int b         = blockIdx.y;
    const int tid       = threadIdx.x;
    const int warp_id   = tid / 32;
    const int lane      = tid % 32;

    const int seq_len = seq_lens[b];

    const int tile_pair_begin = cta_x * tiles_per_cta;
    int tile_pair_end         = tile_pair_begin + tiles_per_cta;
    if (tile_pair_end > max_kv_tile_pairs) tile_pair_end = max_kv_tile_pairs;

    __half* logits_b = logits_out + static_cast<size_t>(b) *
                                    static_cast<size_t>(max_kv_tile_pairs * kUMMA_M);

    const __half kHalfNegInf = __ushort_as_half(static_cast<unsigned short>(0xFC00));

    // Entire split is beyond the block_table — nothing to emit.
    if (tile_pair_begin >= max_kv_tile_pairs) return;

    // Double-buffered smem_k / smem_kscale.  Buffer `b` holds the K/scale
    // data for tile-pair (tile_pair_begin + i) where (i & 1) == b.
    __shared__ __align__(1024) __nv_fp8_e4m3 smem_q[kNumHeads * kHeadDim];       // 8 KB
    __shared__ __align__(1024) __nv_fp8_e4m3 smem_k[2][kUMMA_M * kHeadDim];      // 32 KB
    __shared__ float    smem_kscale[2][kUMMA_M];                                  // 1 KB
    __shared__ float    smem_w[kNumHeads];
    __shared__ uint32_t smem_tmem_ptr;
    __shared__ __align__(8) uint64_t smem_mbar;

    auto kSwizzle16B = [] (int idx) {
        return idx ^ ((idx >> 3) & 7);
    };

    // Check whether a given tile-pair produces any valid tokens, and resolve
    // its page pointers.  Packaged so prefetch_tile can return empty groups
    // uniformly for out-of-range tiles.
    struct TileDesc {
        bool          any_valid;     // tile has at least one in-range token
        bool          page1_valid;   // second page is in-range
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

    // Async K prefetch for tile `tp` into double-buffer slot `buf`.
    // Always closes a commit group so the caller's wait_group accounting
    // stays simple, regardless of whether the tile is in range.
    auto prefetch_tile = [&](int buf, int tp) {
        const TileDesc d = describe_tile(tp);
        if (d.any_valid) {
            constexpr int kVecHalf  = (kBlockKv * kHeadDim) / 16;  // 512 uint4 per page
            constexpr int kVecTotal = (kUMMA_M  * kHeadDim) / 16;  // 1024 uint4 total
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
                // Zero-fill upper half via plain smem stores.  These are
                // thread-local writes; CTA-wide visibility is established by
                // the post-wait __syncthreads in the main loop.
                const uint4 zero4 = make_uint4(0u, 0u, 0u, 0u);
                uint4* k_dst = reinterpret_cast<uint4*>(&smem_k[buf][0]);
#pragma unroll 4
                for (int i = tid; i < (kVecTotal - kVecHalf); i += kThreads)
                    k_dst[kSwizzle16B(i + kVecHalf)] = zero4;
            }

            // kscale is small (128 floats) — load via plain stores; the
            // post-wait __syncthreads makes it visible to the emit step.
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

    // Pre-launch prefetch for the first tile-pair only.
    // With 2-way double-buffering, the opposite buffer is loaded inside
    // the loop body for tile i+1 while UMMA consumes buffer i; launching
    // tile 1 here would write the same buffer the loop would write to
    // again at iter 1 via the i+1 prefetch — harmless, but unnecessary.
    prefetch_tile(0, tile_pair_begin);

    __syncthreads();  // TMEM/mbar init visibility + Q visibility to UMMA

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
    dsa_umma::SmemDescriptor a_desc_base[2] = {
        make_kmajor_desc(&smem_k[0][0]),
        make_kmajor_desc(&smem_k[1][0]),
    };
    const auto b_desc_base = make_kmajor_desc(smem_q);

    uint32_t parity = 0u;  // toggled after each mbarrier wait

    const int num_tiles = tile_pair_end - tile_pair_begin;

    // -------- per-tile-pair loop --------
    for (int i = 0; i < num_tiles; ++i) {
        const int tp      = tile_pair_begin + i;
        const int buf     = i & 1;
        const int kv_base = tp * kUMMA_M;

        // Wait for this tile's K prefetch to complete.  Only one group is
        // in flight at a time (the prefetch for tile i, queued last iter).
        dsa_ptx::cp_async_wait_group<0>();
        __syncthreads();  // CTA-wide visibility of K + kscale + zero-fill stores

        // Kick off the prefetch for tile i+1 into the OPPOSITE buffer while
        // UMMA consumes smem_k[buf].  No buffer aliasing with the current
        // UMMA because (i+1)&1 != i&1.
        if (i + 1 < num_tiles) {
            prefetch_tile((i + 1) & 1, tp + 1);
        }

        // Out-of-seq tile — no UMMA, just fill -inf sentinels.
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

        {
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

    // -------- one-shot teardown --------
    __syncthreads();  // ensure TMEM/emit readers complete
    if (warp_id == 0) {
        dsa_ptx::tcgen05_fence_before_thread_sync();
        dsa_ptx::tcgen05_dealloc_1sm(tmem_addr, kTmemCols);
    }
}


// ============================================================================
// Stage 2: Top-K (k=2048) + page table transform
//
// Opt 4 changes:
//  - Histogram: warp-level `__match_any_sync` elects a single leader lane per
//    (warp, bucket) per iteration; that leader does one atomicAdd of
//    `__popc(same)` into a single shared histogram.  Reduces histogram
//    atomics by ~32× vs the per-element atomicAdd and removes the need for
//    two split warp-group histograms.
//  - Emit: warp ballot replaces per-element atomicAdd to smem_emit_counter
//    (2048 single-location atomics → 64 warp-level atomics).
// ============================================================================

__device__ __forceinline__ uint16_t HalfToOrderedU16(__half h) {
    const uint16_t bits = __half_as_ushort(h);
    return (bits & 0x8000u) ? static_cast<uint16_t>(~bits)
                            : static_cast<uint16_t>(bits ^ 0x8000u);
}

__global__ __launch_bounds__(kStage2Threads)
void topk_page_table_transform_kernel(
    const __half* __restrict__ logits,
    const int*    __restrict__ seq_lens,
    const int*    __restrict__ block_table,
    int max_len, int max_num_pages, int top_k,
    int* __restrict__ out_indices)
{
    const int b      = blockIdx.x;
    const int tid    = threadIdx.x;
    const int warp   = tid >> 5;           // warp index within block (0-31)
    const int lane   = tid & 31;

    const int seq_len    = seq_lens[b];
    const __half* row_lg = logits + static_cast<size_t>(b) * max_len;
    int*          row_out= out_indices + static_cast<size_t>(b) * top_k;
    const int*    row_bt = block_table + static_cast<size_t>(b) * max_num_pages;

    // Single shared histogram (1 KB).  Intra-warp contention eliminated via
    // __match_any_sync; inter-warp contention kept low by per-warp leader.
    __shared__ uint32_t smem_hist[kRadix];
    __shared__ uint32_t smem_suffix[kRadix];
    __shared__ uint16_t smem_prefix;
    __shared__ uint32_t smem_remaining;
    __shared__ uint32_t smem_found_bucket;
    __shared__ uint32_t smem_found_remaining;
    __shared__ int      smem_gt_count;
    __shared__ int      smem_emit_counter;

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

#pragma unroll
    for (int round = 0; round < kRadixRounds; ++round) {
        const int shift = kOrderedBits - 8 - round * 8;
        const uint16_t prefix_mask =
            (round == 0) ? uint16_t{0}
                         : static_cast<uint16_t>(0xFFFFu << (kOrderedBits - round * 8));

        // Zero the shared histogram.
        if (tid < kRadix) smem_hist[tid] = 0u;
        __syncthreads();

        // Warp-level match_any reduces intra-warp contention: threads in the
        // same warp that land on the same bucket elect a single leader lane
        // which issues one atomicAdd of __popc(same).
        //
        // Use a base-stride loop so every lane of every warp reaches the
        // warp-collective intrinsics (__ballot_sync / __match_any_sync).
        // Tail lanes for which `base + tid >= seq_len` mark themselves
        // inactive via `cond=false` and participate with the full mask.
        const uint16_t prefix = smem_prefix;
#pragma unroll 4
        for (int base = 0; base < seq_len; base += blockDim.x) {
            const int  i        = base + tid;
            const bool in_range = (i < seq_len);
            uint16_t   ordered  = 0;
            if (in_range) ordered = HalfToOrderedU16(row_lg[i]);
            const bool cond = in_range &&
                              ((uint16_t)(ordered & prefix_mask) == prefix);
            const uint32_t active = __ballot_sync(0xffffffffu, cond);
            if (cond) {
                const uint32_t bucket = (ordered >> shift) & 0xFFu;
                const uint32_t same   = __match_any_sync(active, bucket);
                if ((uint32_t)lane == (uint32_t)(__ffs((int)same) - 1))
                    atomicAdd(&smem_hist[bucket], __popc(same));
            }
        }
        __syncthreads();

        // Suffix sum (reverse scan: smem_suffix[k] = sum_{i>=k} smem_hist[i])
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

    // Count elements > pivot (already using warp reduction — kept as-is)
    {
        int local = 0;
        for (int i = tid; i < seq_len; i += blockDim.x)
            if (HalfToOrderedU16(row_lg[i]) > pivot) local++;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            local += __shfl_xor_sync(0xffffffffu, local, off);
        if (lane == 0) atomicAdd(&smem_gt_count, local);
    }
    __syncthreads();

    const int gt_total = smem_gt_count;

    // Round seq_len up to the nearest warp multiple so all 32 threads in a warp
    // always reach __ballot_sync / __shfl_sync together (avoids sync divergence
    // on the last loop iteration when seq_len is not warp-aligned).
    const int seq_len_ceil32 = (seq_len + 31) & ~31;

    // Emit > pivot — warp ballot: one atomicAdd per warp instead of per element
    for (int i = tid; i < seq_len_ceil32; i += blockDim.x) {
        const bool in_bounds = (i < seq_len);
        const bool cond = in_bounds && (HalfToOrderedU16(row_lg[i]) > pivot);
        const uint32_t ballot = __ballot_sync(0xffffffffu, cond);
        if (ballot) {
            const int warp_active = __popc(ballot);
            int warp_base;
            if (lane == 0) warp_base = atomicAdd(&smem_emit_counter, warp_active);
            warp_base = __shfl_sync(0xffffffffu, warp_base, 0);
            if (cond) {
                const int lane_offset = __popc(ballot & ((1u << lane) - 1u));
                const int pos = warp_base + lane_offset;
                if (pos < gt_total) row_out[pos] = i;
            }
        }
    }
    __syncthreads();

    // Emit == pivot — warp ballot
    for (int i = tid; i < seq_len_ceil32; i += blockDim.x) {
        const bool in_bounds = (i < seq_len);
        const bool cond = in_bounds && (HalfToOrderedU16(row_lg[i]) == pivot);
        const uint32_t ballot = __ballot_sync(0xffffffffu, cond);
        if (ballot) {
            const int warp_active = __popc(ballot);
            int warp_base;
            if (lane == 0) warp_base = atomicAdd(&smem_emit_counter, warp_active);
            warp_base = __shfl_sync(0xffffffffu, warp_base, 0);
            if (cond) {
                const int lane_offset = __popc(ballot & ((1u << lane) - 1u));
                const int pos = warp_base + lane_offset;
                if (pos < top_k) row_out[pos] = i;
            }
        }
    }
    __syncthreads();

    // Apply block_table transform in-place
    for (int i = tid; i < top_k; i += blockDim.x) {
        const int tok = row_out[i];
        if (tok >= 0 && tok < seq_len)
            row_out[i] = row_bt[tok / kPageSize] * kPageSize + (tok % kPageSize);
        else
            row_out[i] = -1;
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
    TORCH_CHECK(q_index_fp8.is_cuda() && q_index_fp8.is_contiguous());
    TORCH_CHECK(k_index_cache_fp8.is_cuda() && k_index_cache_fp8.is_contiguous());
    TORCH_CHECK(weights.is_cuda() && weights.is_contiguous() && weights.dtype() == torch::kFloat32);
    TORCH_CHECK(seq_lens.is_cuda() && seq_lens.is_contiguous() && seq_lens.dtype() == torch::kInt32);
    TORCH_CHECK(block_table.is_cuda() && block_table.is_contiguous() && block_table.dtype() == torch::kInt32);
    TORCH_CHECK(topk_indices.is_cuda() && topk_indices.is_contiguous() && topk_indices.dtype() == torch::kInt32);

    const int B            = (int)q_index_fp8.size(0);
    const int max_num_pages= (int)block_table.size(1);
    // max_kv_tile_pairs: number of 2-page tiles per batch row
    const int max_kv_tiles      = max_num_pages;
    const int max_kv_tile_pairs = (max_kv_tiles + kPagesPerUMMA - 1) / kPagesPerUMMA;
    // logit buffer sized to accommodate complete tile pairs
    const int max_len      = max_kv_tile_pairs * (kPagesPerUMMA * kBlockKv);

    auto logits = torch::empty({B, max_len},
        torch::TensorOptions().dtype(torch::kFloat16).device(q_index_fp8.device()));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    {
        // Pick `tiles_per_cta` so that (a) we have enough CTAs to keep the
        // B200's ~132 SMs busy and (b) each CTA amortises the Q load over
        // many tile-pairs.  Target ~4 waves of ~132-wide CTAs across B rows.
        constexpr int kSmTarget    = 132 * 4;   // desired total CTAs
        constexpr int kMinTiles    = 4;         // don't let CTA work shrink too much
        constexpr int kMaxTiles    = 64;        // cap per-CTA serial work
        int num_splits = (kSmTarget + B - 1) / B;
        if (num_splits < 1) num_splits = 1;
        if (num_splits > max_kv_tile_pairs) num_splits = max_kv_tile_pairs;
        int tiles_per_cta = (max_kv_tile_pairs + num_splits - 1) / num_splits;
        if (tiles_per_cta < kMinTiles) tiles_per_cta = kMinTiles;
        if (tiles_per_cta > kMaxTiles) tiles_per_cta = kMaxTiles;
        num_splits = (max_kv_tile_pairs + tiles_per_cta - 1) / tiles_per_cta;

        dim3 grid(num_splits, B);
        dim3 block(128);
        paged_mqa_logits_umma_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_fp8_e4m3*>(q_index_fp8.data_ptr()),
            reinterpret_cast<const uint8_t*>(k_index_cache_fp8.data_ptr()),
            weights.data_ptr<float>(),
            seq_lens.data_ptr<int>(),
            block_table.data_ptr<int>(),
            max_num_pages, max_kv_tile_pairs, tiles_per_cta,
            reinterpret_cast<__half*>(logits.data_ptr()));
    }

    {
        dim3 grid(B);
        dim3 block(kStage2Threads);
        topk_page_table_transform_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(logits.data_ptr()),
            seq_lens.data_ptr<int>(),
            block_table.data_ptr<int>(),
            max_len, max_num_pages, kTopK,
            topk_indices.data_ptr<int>());
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &dsa_topk_indexer_cuda, "DSA TopK indexer (SM100a UMMA, 2 pages/MMA)");
}
