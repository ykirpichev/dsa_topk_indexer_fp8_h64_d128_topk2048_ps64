/*
 * FP8 Paged MQA Logits + Radix Top-K — B200 (sm_100), TVM-FFI CUDA entry.
 *
 * Single kernel symbol `kernel` drives the end-to-end pipeline:
 *
 *   1. Persistent-queue FP8-MMA logits kernel
 *        [B, H=64, D=128] Q  ×  paged FP8 K-cache  →  [B, max_context_len] f32
 *
 *      - Q · K via `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` PTX.
 *      - One CTA = 128 threads = 4 warps; each warp owns 16 rows of H.
 *        TPB_MMA=16 tokens per CTA (2 N-tiles of MMA_N=8), so each CTA
 *        emits a [H=64, TPB_MMA=16] f32 partial result tile.
 *      - Persistent scheduler: grid.x = sm_count * CTAS_PER_SM (default 8).
 *        Each CTA pulls a contiguous chunk of (b, tile) tasks from the flat
 *        task space (total = B * m_tiles). Tasks that share `b` reuse the
 *        in-shmem Q cache — Q is reloaded only when the row index changes.
 *        This removes the grid's wave tail and amortises the 8 KB Q load
 *        across many tiles of the same row.
 *      - Stage 4 per-token reduction: logit[t] = Σ_h ReLU(dot[h,t]*scale[t]) * w[h],
 *        in the same FP32 accumulation order used by the non-persistent MMA
 *        kernel. abs_err=0 / rel_err=0 vs. the scalar FP8 reference on the
 *        128-workload B200 sweep.
 *
 *   2. Single-CTA radix top-k kernel
 *        [B, max_context_len] f32 logits  →  [B, K_TOPK=2048] i32 indices.
 *
 *      - Phase 1: 4-pass 8-bit radix histogram over f32 sort-keys with prefix
 *        filtering — narrows `s_pivot` (sort_key of the K-th element) in
 *        log2(2^32)/8 = 4 passes. Privatized histogram infrastructure kept
 *        but NUM_HIST_BANKS=1 after sweep (atomic contention is not the
 *        hot path at these shapes).
 *      - Phase 2: one fused pass that collects keys < pivot via
 *        cub::BlockScan::ExclusiveSum on float4-grouped predicates, and
 *        interleaves the tie resolution (key == pivot) via atomicAdd.
 *      - Phase 3: cub::BlockRadixSort over the K_TOPK collected pairs.
 *      - Output index = block_table[b, t/PS] * PS + (t % PS) — the flat
 *        "physical KV slot" the harness expects.
 *      - TPB=256 by default; TPB=512 for the small-B / long-context tail
 *        (M>=8192 or (B<=4 and M>=4096)) where extra parallelism inside
 *        the single CTA pays for itself.
 *
 * Inputs are validated via TVM_FFI_ICHECK_*. Outputs are destination-passing
 * style: the harness pre-allocates `topk_indices` [B, K_TOPK] i32 and this
 * kernel writes directly into it. The intermediate [B, max_context_len] f32
 * logits buffer is held in a translation-unit-local cudaMalloc scratch that
 * grows monotonically; this avoids per-call allocator churn and matches the
 * torch-extension baseline's behaviour.
 */

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_radix_sort.cuh>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include <cstdint>
#include <cstdlib>
#include <mutex>

namespace {

/* ------------------------------------------------------------------ */
/* Compile-time constants shared by both kernels.                     */
/* ------------------------------------------------------------------ */
constexpr int H  = 64;    /* index heads   */
constexpr int D  = 128;   /* head dim      */
constexpr int PS = 64;    /* page size     */

constexpr int PAGE_FP8_BYTES   = PS * D;                    /* 8192 */
constexpr int PAGE_SCALE_BYTES = PS * 4;                    /*  256 */
constexpr int PAGE_BYTES       = PAGE_FP8_BYTES + PAGE_SCALE_BYTES;  /* 8448 */

constexpr int K_TOPK    = 2048;
constexpr int HIST_BITS = 8;
constexpr int HIST_SIZE = 1 << HIST_BITS;                   /*  256 */

constexpr int TPB_MMA   = 16;   /* tokens per MMA CTA (2 MMA_N=8 tiles) */

/* ------------------------------------------------------------------ */
/* FP8 m16n8k32 Tensor-Core MMA (Blackwell, sm_100).                  */
/*                                                                    */
/* C[16,8]_f32 += A[16,32]_e4m3 × B[32,8]_e4m3                        */
/*                                                                    */
/* PTX thread-data layout (standard m16n8k32):                        */
/*   group = lane/4 (0..7), pos = lane%4 (0..3).                      */
/*   A: 4 × u32 per lane, 16 FP8 values total.                        */
/*       a0: A[group    , 4*pos +  0..3 ]                             */
/*       a1: A[group + 8, 4*pos +  0..3 ]                             */
/*       a2: A[group    , 4*pos + 16..19]                             */
/*       a3: A[group + 8, 4*pos + 16..19]                             */
/*   B: 2 × u32 per lane (col-major B = row-major B^T):               */
/*       b0: B^T[group, 4*pos +  0..3 ]                               */
/*       b1: B^T[group, 4*pos + 16..19]                               */
/*   C: 4 × f32 per lane, row-major:                                  */
/*       c0: C[group    , 2*pos    ]                                  */
/*       c1: C[group    , 2*pos + 1]                                  */
/*       c2: C[group + 8, 2*pos    ]                                  */
/*       c3: C[group + 8, 2*pos + 1]                                  */
/* ------------------------------------------------------------------ */
__device__ __forceinline__ void mma_m16n8k32_e4m3_e4m3_f32(
    float& c0, float& c1, float& c2, float& c3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, "
        "{%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1));
}

/* ------------------------------------------------------------------ */
/* f32 → unsigned sort-key with descending ordering (IEEE-sign-safe). */
/* Largest f32 becomes the smallest key, so ascending radix of the    */
/* key space is equivalent to descending order of the original f32.   */
/* ------------------------------------------------------------------ */
__device__ __forceinline__ uint32_t f32_to_sort_key(float f) {
    uint32_t u    = __float_as_uint(f);
    uint32_t mask = (uint32_t)(-(int32_t)(u >> 31)) | 0x80000000u;
    uint32_t asc  = u ^ mask;
    return ~asc;
}

/* ================================================================== */
/* FP8 MMA persistent-queue logits kernel.                            */
/*                                                                    */
/* Fixed grid of `gridDim.x == num_ctas` CTAs; each CTA processes a   */
/* contiguous chunk of the flat task space                            */
/*   total_tasks = B * m_tiles,  m_tiles = ceil(max_context_len / TPB_MMA)
/* Task index = b * m_tiles + tile (row-major). Contiguous chunking   */
/* means most tasks processed by one CTA share the same `b`, so       */
/* Q[b, :, :] is cached in shmem and reloaded only on row boundaries. */
/* ================================================================== */
__global__ void __launch_bounds__(128)
fp8_paged_mqa_logits_mma_pq_kernel(
    const __nv_fp8_e4m3* __restrict__ q,             /* [B, H, D]            */
    const uint8_t*       __restrict__ k_cache,       /* raw page bytes       */
    const float*         __restrict__ weights,       /* [B, H]               */
    const int*           __restrict__ seq_lens,      /* [B]                  */
    const int*           __restrict__ block_table,   /* [B, max_num_pages]   */
    float*               __restrict__ logits,        /* [B, max_context_len] */
    int B,
    int max_context_len,
    int max_num_pages,
    int m_tiles                                      /* ceil(M / TPB_MMA)    */
) {
    constexpr int MMA_M        = 16;
    constexpr int MMA_N        = 8;
    constexpr int MMA_K        = 32;
    constexpr int BLOCK_TH     = 128;
    constexpr int NUM_WARPS    = BLOCK_TH / 32;              /* 4 */
    static_assert(H == NUM_WARPS * MMA_M, "H must equal NUM_WARPS * MMA_M");
    static_assert(TPB_MMA % MMA_N == 0,   "TPB_MMA must be a multiple of MMA_N");
    static_assert(TPB_MMA >= MMA_N,       "TPB_MMA must be >= MMA_N");
    constexpr int K_ITERS      = D / MMA_K;                  /* 4 */
    constexpr int N_ITERS      = TPB_MMA / MMA_N;            /* 2 */
    constexpr int INT4_PER_ROW = D / 16;                     /* 8 */
    constexpr int K_TILE_INT4  = TPB_MMA * INT4_PER_ROW;     /* 128 */

    /*  Shmem layout (TPB_MMA=16):
     *    q_shmem      : [H=64,  D=128]       FP8   = 8192 B
     *    k_shmem      : [TPB_MMA, D]         FP8   = 2048 B
     *    scale_shmem  : [TPB_MMA]            FP32  =   64 B
     *    weight_shmem : [H]                  FP32  =  256 B
     *    result_shmem : [H, TPB_MMA]         FP32  = 4096 B
     *  Total = 14656 B. Well below the 228 KB B200 cap. */
    extern __shared__ __align__(16) char raw_shmem[];
    auto* q_shmem      = reinterpret_cast<__nv_fp8_e4m3*>(raw_shmem);
    auto* k_shmem      = reinterpret_cast<__nv_fp8_e4m3*>(raw_shmem + H * D);
    auto* scale_shmem  = reinterpret_cast<float*>(
                             raw_shmem + H * D + TPB_MMA * D);
    auto* weight_shmem = reinterpret_cast<float*>(
                             raw_shmem + H * D + TPB_MMA * D + TPB_MMA * 4);
    auto* result_shmem = reinterpret_cast<float*>(
                             raw_shmem + H * D + TPB_MMA * D
                             + TPB_MMA * 4 + H * 4);

    const int tid      = threadIdx.x;
    const int warp_id  = tid / 32;
    const int lane     = tid & 31;
    const int group    = lane >> 2;      /* 0..7 : MMA row pair */
    const int pos      = lane & 3;       /* 0..3 : K/N sub-index */

    /* Contiguous task assignment: CTA i processes tasks [cta_lo, cta_hi). */
    const int total_tasks = B * m_tiles;
    const int num_ctas    = gridDim.x;
    const int chunk_sz    = (total_tasks + num_ctas - 1) / num_ctas;
    const int cta_lo      = blockIdx.x * chunk_sz;
    const int cta_hi      = min(cta_lo + chunk_sz, total_tasks);
    if (cta_lo >= cta_hi) return;

    /* Per-CTA Q cache: `cur_b` = row currently held in shmem. */
    int cur_b       = -1;
    int cur_seq_len = 0;

    /* Per-thread MMA addressing constants. */
    const int a_row0      = warp_id * MMA_M + group;   /* 0..63 */
    const int a_row1      = a_row0 + 8;                /* 0..63 */
    const int b_row_local = group;                     /* 0..7 inside one N-tile */

    for (int task = cta_lo; task < cta_hi; ++task) {
        const int b        = task / m_tiles;
        const int tile     = task - b * m_tiles;
        const int tok_base = tile * TPB_MMA;

        /* Q cache: reload only on row change. */
        if (b != cur_b) {
            cur_b       = b;
            cur_seq_len = __ldg(seq_lens + b);

            /* Q[b, :, :] = 8 KB → 64 B / thread = 4 × int4 loads. */
            const auto* q_src = reinterpret_cast<const int4*>(q + (size_t)b * H * D);
            auto*       q_dst = reinterpret_cast<int4*>(q_shmem);
            #pragma unroll
            for (int i = tid; i < (H * D) / 16; i += BLOCK_TH)
                q_dst[i] = __ldg(q_src + i);
            if (tid < H) weight_shmem[tid] = __ldg(weights + (size_t)b * H + tid);
            __syncthreads();   /* Q + weights ready before any MMA read */
        }

        /* Early exit: all tokens in this tile are past seq_len. Still emit
         * zeros for the padding slots in [tok_base, tok_base+TPB_MMA) that
         * fall inside max_context_len — the top-k kernel only reads the
         * first seq_len columns so the value doesn't matter for
         * correctness, but the buffer is owned by us and must be written
         * in a deterministic state. */
        if (tok_base >= cur_seq_len) {
            if (tid < TPB_MMA) {
                const int token = tok_base + tid;
                if (token < max_context_len) {
                    logits[(size_t)b * max_context_len + token] = 0.0f;
                }
            }
            continue;
        }

        /* Stage 2: load K_tile[TPB_MMA, D] + scales[TPB_MMA] from paged cache. */
        {
            #pragma unroll
            for (int i = tid; i < K_TILE_INT4; i += BLOCK_TH) {
                const int t_local = i / INT4_PER_ROW;
                const int chunk   = i % INT4_PER_ROW;
                const int token   = tok_base + t_local;
                int4 val = {0, 0, 0, 0};
                if (token < cur_seq_len) {
                    const int page_id     = block_table[(size_t)b * max_num_pages + token / PS];
                    const int off_in_page = token % PS;
                    const uint8_t* page_ptr = k_cache + (size_t)page_id * PAGE_BYTES;
                    const auto* k_src = reinterpret_cast<const int4*>(
                        page_ptr + (size_t)off_in_page * D);
                    val = __ldg(k_src + chunk);
                }
                reinterpret_cast<int4*>(k_shmem + t_local * D)[chunk] = val;
            }

            if (tid < TPB_MMA) {
                const int token = tok_base + tid;
                float s = 0.0f;
                if (token < cur_seq_len) {
                    const int page_id     = block_table[(size_t)b * max_num_pages + token / PS];
                    const int off_in_page = token % PS;
                    const uint8_t* page_ptr = k_cache + (size_t)page_id * PAGE_BYTES;
                    const float* scale_ptr = reinterpret_cast<const float*>(
                        page_ptr + PAGE_FP8_BYTES + (size_t)off_in_page * 4);
                    s = __ldg(scale_ptr);
                }
                scale_shmem[tid] = s;
            }
        }
        __syncthreads();   /* K + scales ready before Stage 3 */

        /* Stage 3: Q @ K_tile^T → result_shmem[H, TPB_MMA]. */
        #pragma unroll
        for (int n = 0; n < N_ITERS; ++n) {
            float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
            const int n_base = n * MMA_N;
            const int b_row  = n_base + b_row_local;

            #pragma unroll
            for (int kc = 0; kc < K_ITERS; ++kc) {
                const int k_base = kc * MMA_K;
                const uint32_t* qa0 = reinterpret_cast<const uint32_t*>(
                    q_shmem + a_row0 * D + k_base + 4 * pos);
                const uint32_t* qa1 = reinterpret_cast<const uint32_t*>(
                    q_shmem + a_row1 * D + k_base + 4 * pos);
                const uint32_t a0 = qa0[0];
                const uint32_t a1 = qa1[0];
                const uint32_t a2 = qa0[4];     /* +16 B = +4 uint32s */
                const uint32_t a3 = qa1[4];

                const uint32_t* qb = reinterpret_cast<const uint32_t*>(
                    k_shmem + b_row * D + k_base + 4 * pos);
                const uint32_t b0 = qb[0];
                const uint32_t b1 = qb[4];

                mma_m16n8k32_e4m3_e4m3_f32(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
            }

            const int out_r0 = warp_id * MMA_M + group;
            const int out_r1 = out_r0 + 8;
            const int col0   = n_base + 2 * pos;
            result_shmem[out_r0 * TPB_MMA + col0    ] = c0;
            result_shmem[out_r0 * TPB_MMA + col0 + 1] = c1;
            result_shmem[out_r1 * TPB_MMA + col0    ] = c2;
            result_shmem[out_r1 * TPB_MMA + col0 + 1] = c3;
        }
        __syncthreads();   /* result_shmem writers -> Stage 4 readers */

        /* Stage 4: per-token logit = Σ_h ReLU(dot[h,t] * scale[t]) * w[h]. */
        if (tid < TPB_MMA) {
            const int t_local = tid;
            const int token   = tok_base + t_local;
            float logit       = 0.0f;
            if (token < cur_seq_len) {
                const float scale = scale_shmem[t_local];
                #pragma unroll
                for (int h = 0; h < H; ++h) {
                    float dot = result_shmem[h * TPB_MMA + t_local] * scale;
                    if (dot > 0.0f) logit += dot * weight_shmem[h];
                }
            }
            if (token < max_context_len) {
                logits[(size_t)b * max_context_len + token] = logit;
            }
        }
        __syncthreads();   /* Stage 4 done -> shmem reusable next iter */
    }
}

/* ================================================================== */
/* Single-CTA radix top-K kernel.                                     */
/*                                                                    */
/* Each block handles one row (b). Four 8-bit radix passes on the     */
/* f32 sort-keys narrow the K-th-largest pivot; a fused Phase-2 pass  */
/* collects (key, phys_id) pairs either above the pivot (via CUB      */
/* prefix-scan) or at the pivot (via atomicAdd tie buffer). Phase 3   */
/* sorts the collected pairs with CUB BlockRadixSort and emits indices*/
/* (b, 0..K_TOPK-1) into `out`.                                       */
/*                                                                    */
/* TPB is templated so the host can pick 256 (default) or 512 (small  */
/* B / long context).                                                 */
/* ================================================================== */
template<int TPB>
__global__ void __launch_bounds__(TPB)
topk_radix_kernel_t(
    const float* __restrict__ logits,       /* [B, M]          */
    const int*   __restrict__ block_table,  /* [B, max_pages]  */
    const int*   __restrict__ seq_lens,     /* [B]             */
    int*         __restrict__ out,          /* [B, K_TOPK]     */
    int M, int max_pages
) {
    static_assert(K_TOPK % TPB == 0, "K_TOPK must be divisible by TPB");
    static_assert(TPB % 32 == 0,     "TPB must be a multiple of 32");
    static_assert(TPB >= HIST_SIZE,  "TPB must be >= HIST_SIZE (256)");

    constexpr int IPT = K_TOPK / TPB;                  /* 8 @ TPB=256, 4 @ TPB=512 */

    using ScanT = cub::BlockScan<int, TPB>;
    using SortT = cub::BlockRadixSort<uint32_t, TPB, IPT, int>;

    __shared__ uint32_t s_hist[HIST_SIZE];
    __shared__ uint32_t s_ckeys[K_TOPK];
    __shared__ int      s_cvals[K_TOPK];
    __shared__ int      s_cnt_above;
    __shared__ int      s_rem_k;
    __shared__ uint32_t s_pivot;
    __shared__ int      s_collect_b;
    __shared__ union {
        typename ScanT::TempStorage scan;
        typename SortT::TempStorage sort;
    } s_cub;

    const int b     = blockIdx.x;
    const int tid   = threadIdx.x;
    const int sl    = seq_lens[b];
    const int eff_k = min(sl, K_TOPK);
    const int sl4   = sl / 4;

    const float*  row    = logits      + (size_t)b * M;
    const int*    bt     = block_table + (size_t)b * max_pages;
    int*          outrow = out         + (size_t)b * K_TOPK;
    const float4* row4   = reinterpret_cast<const float4*>(row);

    /* Fast path for empty sequences. */
    if (sl == 0) {
        for (int i = tid; i < K_TOPK; i += TPB) outrow[i] = -1;
        return;
    }

    if (tid == 0) {
        s_pivot     = 0u;
        s_cnt_above = 0;
        s_rem_k     = eff_k;
        s_collect_b = 0;
    }
    __syncthreads();

    /* -------- Phase 1: 4-pass 8-bit radix histogram -> pivot sort_key. -------- */
    for (int pass = 0; pass < 4; ++pass) {
        if (s_rem_k == 0) break;

        const int      shift     = 24 - pass * 8;
        const uint32_t cur_pivot = s_pivot;

        for (int i = tid; i < HIST_SIZE; i += TPB) s_hist[i] = 0u;
        __syncthreads();

        if (pass == 0) {
            for (int t4 = tid; t4 < sl4; t4 += TPB) {
                float4 v = __ldg(row4 + t4);
                atomicAdd(&s_hist[(f32_to_sort_key(v.x) >> shift) & 255], 1u);
                atomicAdd(&s_hist[(f32_to_sort_key(v.y) >> shift) & 255], 1u);
                atomicAdd(&s_hist[(f32_to_sort_key(v.z) >> shift) & 255], 1u);
                atomicAdd(&s_hist[(f32_to_sort_key(v.w) >> shift) & 255], 1u);
            }
            for (int t = sl4 * 4 + tid; t < sl; t += TPB)
                atomicAdd(&s_hist[(f32_to_sort_key(__ldg(row + t)) >> shift) & 255], 1u);
        } else {
            const uint32_t prefix_ref = cur_pivot >> (shift + 8);
            for (int t4 = tid; t4 < sl4; t4 += TPB) {
                float4 v = __ldg(row4 + t4);
                uint32_t k0 = f32_to_sort_key(v.x), k1 = f32_to_sort_key(v.y);
                uint32_t k2 = f32_to_sort_key(v.z), k3 = f32_to_sort_key(v.w);
                if ((k0 >> (shift + 8)) == prefix_ref) atomicAdd(&s_hist[(k0 >> shift) & 255], 1u);
                if ((k1 >> (shift + 8)) == prefix_ref) atomicAdd(&s_hist[(k1 >> shift) & 255], 1u);
                if ((k2 >> (shift + 8)) == prefix_ref) atomicAdd(&s_hist[(k2 >> shift) & 255], 1u);
                if ((k3 >> (shift + 8)) == prefix_ref) atomicAdd(&s_hist[(k3 >> shift) & 255], 1u);
            }
            for (int t = sl4 * 4 + tid; t < sl; t += TPB) {
                uint32_t key = f32_to_sort_key(__ldg(row + t));
                if ((key >> (shift + 8)) == prefix_ref)
                    atomicAdd(&s_hist[(key >> shift) & 255], 1u);
            }
        }
        __syncthreads();

        if (tid == 0) {
            int cnt = 0, rem = s_rem_k;
            for (int i = 0; i < HIST_SIZE; ++i) {
                int h = (int)s_hist[i];
                if (cnt + h >= rem) {
                    uint32_t pmask = (shift + 8 >= 32) ? 0u : (~0u << (shift + 8));
                    s_pivot      = (cur_pivot & pmask) | ((uint32_t)i << shift);
                    s_cnt_above += cnt;
                    s_rem_k     -= cnt;
                    break;
                }
                cnt += h;
            }
        }
        __syncthreads();
    }

    const uint32_t pivot     = s_pivot;
    const int      cnt_above = s_cnt_above;
    const int      rem_k     = s_rem_k;

    /* -------- Phase 2: fused (2A: < pivot via scan) + (2B: == pivot via atomic). -------- */
    {
        int base = 0;   /* running write base for 2A (< pivot) elements */

        for (int t4_0 = 0; t4_0 < sl4; t4_0 += TPB) {
            const int t4 = t4_0 + tid;

            uint32_t keys[4] = { ~0u, ~0u, ~0u, ~0u };
            int      phys[4] = {  -1,  -1,  -1,  -1 };
            int      pred[4] = {   0,   0,   0,   0 };

            if (t4 < sl4) {
                float4 v = __ldg(row4 + t4);
                keys[0] = f32_to_sort_key(v.x); keys[1] = f32_to_sort_key(v.y);
                keys[2] = f32_to_sort_key(v.z); keys[3] = f32_to_sort_key(v.w);
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int t = t4 * 4 + i;
                    phys[i] = bt[t / PS] * PS + (t % PS);
                    pred[i] = (keys[i] < pivot) ? 1 : 0;
                }
            }

            int local_sum = pred[0] + pred[1] + pred[2] + pred[3];
            int rank, chunk_total;
            ScanT(s_cub.scan).ExclusiveSum(local_sum, rank, chunk_total);

            if (t4 < sl4) {
                int wp = base + rank;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    if (pred[i]) { s_ckeys[wp] = keys[i]; s_cvals[wp] = phys[i]; wp++; }
                    if (keys[i] == pivot) {
                        int pos = atomicAdd(&s_collect_b, 1);
                        if (pos < rem_k) {
                            s_ckeys[cnt_above + pos] = keys[i];
                            s_cvals[cnt_above + pos] = phys[i];
                        }
                    }
                }
            }
            base += chunk_total;
            __syncthreads();   /* ScanT temp storage reused next iteration */
        }

        /* Scalar tail: sl % 4 leftovers. */
        if (sl4 * 4 < sl) {
            const int  t     = sl4 * 4 + tid;
            const bool valid = (t < sl);
            uint32_t key = valid ? f32_to_sort_key(__ldg(row + t)) : ~0u;
            int pred = (valid && key < pivot) ? 1 : 0;
            int rank, chunk_total;
            ScanT(s_cub.scan).ExclusiveSum(pred, rank, chunk_total);
            if (valid) {
                if (pred) { s_ckeys[base + rank] = key; s_cvals[base + rank] = bt[t/PS]*PS + (t%PS); }
                if (key == pivot) {
                    int pos = atomicAdd(&s_collect_b, 1);
                    if (pos < rem_k) {
                        s_ckeys[cnt_above + pos] = key;
                        s_cvals[cnt_above + pos] = bt[t/PS]*PS + (t%PS);
                    }
                }
            }
            base += chunk_total;
            __syncthreads();
        }
    }

    /* Pad [eff_k .. K_TOPK-1] with sentinels that sort last. */
    for (int i = eff_k + tid; i < K_TOPK; i += TPB) {
        s_ckeys[i] = ~0u;
        s_cvals[i] = -1;
    }
    __syncthreads();

    /* Phase 3: CUB BlockRadixSort on K_TOPK (sort_key, phys_id) pairs. */
    uint32_t tkeys[IPT];
    int      tvals[IPT];

    #pragma unroll
    for (int i = 0; i < IPT; ++i) {
        tkeys[i] = s_ckeys[tid * IPT + i];
        tvals[i] = s_cvals[tid * IPT + i];
    }
    __syncthreads();   /* s_cub.sort aliases s_cub.scan; wait for readers */

    SortT(s_cub.sort).Sort(tkeys, tvals);

    #pragma unroll
    for (int i = 0; i < IPT; ++i)
        outrow[tid * IPT + i] = tvals[i];
}

/* ------------------------------------------------------------------ */
/* Device-wide SM count, queried lazily and cached.                   */
/* ------------------------------------------------------------------ */
int query_sm_count() {
    static int sm_count = []() {
        int dev = 0;
        cudaGetDevice(&dev);
        int n = 132;  /* B200 fallback */
        cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
        return n;
    }();
    return sm_count;
}

/* ------------------------------------------------------------------ */
/* Per-SM CTA multiplier for the persistent-queue logits kernel.      */
/*                                                                    */
/* Tuned on B200 (128-workload sweep, TPB_MMA=16):                    */
/*   cps=2 : mean 58 us, max 135 us  (regression vs single-tile)      */
/*   cps=4 : mean 47 us, max  96 us  (marginal win)                   */
/*   cps=6 : mean 44 us, max  77 us  (solid win)                      */
/*   cps=8 : mean 42 us, max  68 us  (plateau; default)               */
/*   cps=10: mean 41 us, max  68 us  (same perf, more shmem)          */
/* Gate on USE_MMA_PQ_CTAS_PER_SM for ablations.                      */
/* ------------------------------------------------------------------ */
int query_ctas_per_sm() {
    static int ctas_per_sm = []() {
        const char* e = std::getenv("USE_MMA_PQ_CTAS_PER_SM");
        int v = (e && e[0] != '\0') ? std::atoi(e) : 8;
        if (v < 1)  v = 1;
        if (v > 16) v = 16;
        return v;
    }();
    return ctas_per_sm;
}

/* ------------------------------------------------------------------ */
/* Translation-unit-local f32 logits scratch. Grown monotonically via */
/* cudaMalloc so we pay zero allocator overhead per call. Lifetime =  */
/* process; freed only on reset-to-larger.                            */
/* ------------------------------------------------------------------ */
float* get_logits_scratch(size_t elems) {
    static std::mutex       mu;
    static float*           ptr   = nullptr;
    static size_t           cap   = 0;

    std::lock_guard<std::mutex> lk(mu);
    if (elems > cap) {
        if (ptr) cudaFree(ptr);
        size_t bytes = elems * sizeof(float);
        cudaError_t err = cudaMalloc(&ptr, bytes);
        TVM_FFI_ICHECK_EQ(err, cudaSuccess)
            << "get_logits_scratch: cudaMalloc(" << bytes
            << ") failed: " << cudaGetErrorString(err);
        cap = elems;
    }
    return ptr;
}

/* ================================================================== */
/* TVM-FFI entry: `kernel`.                                           */
/*                                                                    */
/* Signature matches the flashinfer-bench track for                   */
/*   dsa_topk_indexer_fp8_h64_d128_topk2048_ps64                      */
/* with destination-passing-style output.                             */
/*                                                                    */
/* Inputs:                                                            */
/*   q_index_fp8        : [B, H=64, D=128]                fp8_e4m3    */
/*   k_index_cache_fp8  : [num_pages, PS=64, 1, D+4]      fp8_e4m3    */
/*                        (last dim = 128 FP8 bytes + 4 scale bytes / PS
/*                         = PS*(D+4) = 8448 B per page)              */
/*   weights            : [B, H=64]                       float32     */
/*   seq_lens           : [B]                             int32       */
/*   block_table        : [B, max_num_pages]              int32       */
/*                                                                    */
/* Outputs (DPS):                                                     */
/*   topk_indices       : [B, K_TOPK=2048]                int32       */
/* ================================================================== */
void kernel_fn(
    const tvm::ffi::Tensor& q_index_fp8,
    const tvm::ffi::Tensor& k_index_cache_fp8,
    const tvm::ffi::Tensor& weights,
    const tvm::ffi::Tensor& seq_lens,
    const tvm::ffi::Tensor& block_table,
    tvm::ffi::Tensor topk_indices
) {
    /* ---- dtype checks ---- */
    TVM_FFI_ICHECK_EQ(q_index_fp8.dtype().bits, 8)
        << "q_index_fp8 must be 8-bit (fp8_e4m3)";
    TVM_FFI_ICHECK_EQ(k_index_cache_fp8.dtype().bits, 8)
        << "k_index_cache_fp8 must be 8-bit (fp8_e4m3)";
    TVM_FFI_ICHECK_EQ(weights.dtype().code, kDLFloat) << "weights must be float32";
    TVM_FFI_ICHECK_EQ(weights.dtype().bits, 32);
    TVM_FFI_ICHECK_EQ(seq_lens.dtype().code, kDLInt) << "seq_lens must be int32";
    TVM_FFI_ICHECK_EQ(seq_lens.dtype().bits, 32);
    TVM_FFI_ICHECK_EQ(block_table.dtype().code, kDLInt) << "block_table must be int32";
    TVM_FFI_ICHECK_EQ(block_table.dtype().bits, 32);
    TVM_FFI_ICHECK_EQ(topk_indices.dtype().code, kDLInt) << "topk_indices must be int32";
    TVM_FFI_ICHECK_EQ(topk_indices.dtype().bits, 32);

    /* ---- shape checks ---- */
    TVM_FFI_ICHECK_EQ(q_index_fp8.ndim(), 3);
    TVM_FFI_ICHECK_EQ(q_index_fp8.size(1), H);
    TVM_FFI_ICHECK_EQ(q_index_fp8.size(2), D);

    TVM_FFI_ICHECK_EQ(k_index_cache_fp8.ndim(), 4);
    TVM_FFI_ICHECK_EQ(k_index_cache_fp8.size(1), PS);
    TVM_FFI_ICHECK_EQ(k_index_cache_fp8.size(2), 1);
    TVM_FFI_ICHECK_EQ(k_index_cache_fp8.size(3), D + 4)
        << "k_index_cache_fp8 last dim must be D+4 (FP8 tile + 4B scale)";

    const int64_t B_q        = q_index_fp8.size(0);
    const int64_t B_w        = weights.size(0);
    const int64_t B_sl       = seq_lens.size(0);
    const int64_t B_bt       = block_table.size(0);
    const int64_t B_out      = topk_indices.size(0);
    TVM_FFI_ICHECK_EQ(B_q, B_w)   << "batch mismatch: q vs weights";
    TVM_FFI_ICHECK_EQ(B_q, B_sl)  << "batch mismatch: q vs seq_lens";
    TVM_FFI_ICHECK_EQ(B_q, B_bt)  << "batch mismatch: q vs block_table";
    TVM_FFI_ICHECK_EQ(B_q, B_out) << "batch mismatch: q vs topk_indices";

    TVM_FFI_ICHECK_EQ(weights.ndim(), 2);
    TVM_FFI_ICHECK_EQ(weights.size(1), H);
    TVM_FFI_ICHECK_EQ(seq_lens.ndim(), 1);
    TVM_FFI_ICHECK_EQ(block_table.ndim(), 2);
    TVM_FFI_ICHECK_EQ(topk_indices.ndim(), 2);
    TVM_FFI_ICHECK_EQ(topk_indices.size(1), K_TOPK);

    const int B               = static_cast<int>(B_q);
    const int max_num_pages   = static_cast<int>(block_table.size(1));
    const int max_context_len = max_num_pages * PS;
    const int m_tiles         = (max_context_len + TPB_MMA - 1) / TPB_MMA;

    /* ---- stream ---- */
    const DLDevice dev = q_index_fp8.device();
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(dev.device_type, dev.device_id));

    /* ---- logits scratch (f32 [B, max_context_len]) ---- */
    const size_t logits_elems = static_cast<size_t>(B) * max_context_len;
    float* logits_ptr = get_logits_scratch(logits_elems);

    /* ---- input/output pointers ---- */
    const auto* q_ptr  = reinterpret_cast<const __nv_fp8_e4m3*>(q_index_fp8.data_ptr());
    const auto* kc_ptr = reinterpret_cast<const uint8_t*>(k_index_cache_fp8.data_ptr());
    const auto* w_ptr  = static_cast<const float*>(weights.data_ptr());
    const auto* sl_ptr = static_cast<const int*>(seq_lens.data_ptr());
    const auto* bt_ptr = static_cast<const int*>(block_table.data_ptr());
    int*        o_ptr  = static_cast<int*>(topk_indices.data_ptr());

    /* =========== Stage 1: FP8 MMA persistent-queue logits =========== */
    {
        const int total_tasks = B * m_tiles;
        int num_ctas = query_sm_count() * query_ctas_per_sm();
        if (num_ctas > total_tasks) num_ctas = total_tasks;
        if (num_ctas <= 0) num_ctas = 1;

        const dim3 grid(num_ctas);
        const dim3 block(128);
        const size_t shmem =
              static_cast<size_t>(H) * D * sizeof(__nv_fp8_e4m3)         /* q_shmem     */
            + static_cast<size_t>(TPB_MMA) * D * sizeof(__nv_fp8_e4m3)   /* k_shmem     */
            + static_cast<size_t>(TPB_MMA) * sizeof(float)               /* scale_shmem */
            + static_cast<size_t>(H) * sizeof(float)                     /* weight_shmem*/
            + static_cast<size_t>(H) * TPB_MMA * sizeof(float);          /* result_shmem*/

        fp8_paged_mqa_logits_mma_pq_kernel<<<grid, block, shmem, stream>>>(
            q_ptr, kc_ptr, w_ptr, sl_ptr, bt_ptr, logits_ptr,
            B, max_context_len, max_num_pages, m_tiles);

        const cudaError_t err = cudaGetLastError();
        TVM_FFI_ICHECK_EQ(err, cudaSuccess)
            << "fp8_paged_mqa_logits_mma_pq_kernel: " << cudaGetErrorString(err);
    }

    /* =========== Stage 2: single-CTA radix top-K =========== */
    {
        /* Heuristic: wider TPB=512 for large M / small-B long ctx. */
        const int M         = max_context_len;
        const int max_pages = max_num_pages;
        if (M >= 8192 || (B <= 4 && M >= 4096)) {
            topk_radix_kernel_t<512><<<B, 512, 0, stream>>>(
                logits_ptr, bt_ptr, sl_ptr, o_ptr, M, max_pages);
        } else {
            topk_radix_kernel_t<256><<<B, 256, 0, stream>>>(
                logits_ptr, bt_ptr, sl_ptr, o_ptr, M, max_pages);
        }

        const cudaError_t err = cudaGetLastError();
        TVM_FFI_ICHECK_EQ(err, cudaSuccess)
            << "topk_radix_kernel_t: " << cudaGetErrorString(err);
    }
}

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel, kernel_fn);
