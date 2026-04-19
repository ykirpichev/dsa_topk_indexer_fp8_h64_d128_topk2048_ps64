/*
 * FP8 Paged MQA Logits — CUDA kernel for DSA top-K indexer.
 *
 * For each (batch b, token t):
 *   logit[b,t] = Σ_h  ReLU( k_scale[t] * Σ_d  q_fp8[b,h,d] * k_fp8[t,d] ) * weights[b,h]
 *
 * Constants (compile-time): H=64 heads, D=128 head-dim, PS=64 page-size.
 * K cache layout per token: 128 fp8_e4m3 bytes followed by 1 float32 scale.
 *
 * Thread mapping
 *   Grid  : (B,  ceil(max_context_len / TOKENS_PER_BLOCK))
 *   Block : (H * TOKENS_PER_BLOCK,)   — H=64 threads per token, each owns one head
 *
 * Each token needs a 2-warp head reduction (H=64 = 2 warps of 32).
 * K bytes are loaded cooperatively into shared memory (2 bytes per thread).
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>

/* ------------------------------------------------------------------ */
/* Compile-time constants                                               */
/* ------------------------------------------------------------------ */
static constexpr int H  = 64;    /* index heads                        */
static constexpr int D  = 128;   /* head dim                           */
static constexpr int PS = 64;    /* page size                          */

/* K-cache page layout (deep_gemm format, CONCATENATED per page):
 *   bytes [0           .. PS*D-1 ]   = FP8 data for all PS tokens
 *   bytes [PS*D        .. PS*D+PS*4-1] = float32 scales for all PS tokens
 *   Total per page = PS*D + PS*4 = PS*(D+4) bytes (= 8448 for PS=64,D=128)
 *
 * The torch shape is [num_pages, PS, 1, D+4] so index [p, t, 0, :] spans
 * bytes [p_base + t*(D+4) .. p_base + (t+1)*(D+4)-1], but that DOES NOT
 * correspond to token `t`'s data. We use raw byte offsets instead.
 */
static constexpr int PAGE_FP8_BYTES   = PS * D;          /* 8192 */
static constexpr int PAGE_SCALE_BYTES = PS * 4;          /* 256  */
static constexpr int PAGE_BYTES       = PAGE_FP8_BYTES + PAGE_SCALE_BYTES; /* 8448 */
static constexpr int TOKENS_PER_BLOCK = 8;

/* ------------------------------------------------------------------ */
/* Kernel                                                               */
/* ------------------------------------------------------------------ */
__global__ void fp8_paged_mqa_logits_kernel(
    const __nv_fp8_e4m3* __restrict__ q,          /* [B, H, D]              */
    const uint8_t*        __restrict__ k_cache,   /* [num_pages*PS, D4]     */
    const float*          __restrict__ weights,   /* [B, H]                 */
    const int*            __restrict__ seq_lens,  /* [B]                    */
    const int*            __restrict__ block_table, /* [B, max_num_pages]   */
    float*                __restrict__ logits,    /* [B, max_context_len]   */
    int max_context_len,
    int max_num_pages
) {
    /*
     * Shared memory layout:
     *   k_shmem   [TOKENS_PER_BLOCK * D]  (__nv_fp8_e4m3, 1 byte each)
     *   scale_shmem [TOKENS_PER_BLOCK]    (float)
     *   warp_sums [TOKENS_PER_BLOCK * 2]  (float, one slot per warp-per-token)
     */
    extern __shared__ char raw_shmem[];
    auto* k_shmem     = reinterpret_cast<__nv_fp8_e4m3*>(raw_shmem);
    auto* scale_shmem = reinterpret_cast<float*>(raw_shmem + TOKENS_PER_BLOCK * D);
    float* warp_sums  = scale_shmem + TOKENS_PER_BLOCK;

    const int b           = blockIdx.x;
    const int tok_base    = blockIdx.y * TOKENS_PER_BLOCK;
    const int tid         = threadIdx.x;
    const int token_local = tid / H;   /* 0 .. TOKENS_PER_BLOCK-1 */
    const int head        = tid % H;   /* 0 .. H-1                */
    const int token       = tok_base + token_local;
    const int seq_len     = seq_lens[b];

    /* -------------------------------------------------------------- */
    /* 1. Cooperative K load into shared memory                        */
    /*    Each of H threads loads D/H = 2 FP8 bytes for its token.    */
    /* -------------------------------------------------------------- */
    if (token < seq_len) {
        const int page_id     = block_table[b * max_num_pages + token / PS];
        const int off_in_page = token % PS;
        /* Base pointer to this page (PAGE_BYTES per page). */
        const uint8_t* page_ptr = k_cache + (size_t)page_id * PAGE_BYTES;
        /* FP8 data for this token: PAGE_FP8_BYTES region, stride D per token. */
        const auto* k_ptr = reinterpret_cast<const __nv_fp8_e4m3*>(
            page_ptr + (size_t)off_in_page * D);
        /* Scale for this token: PAGE_SCALE_BYTES region, stride 4 per token. */
        const float* scale_ptr = reinterpret_cast<const float*>(
            page_ptr + PAGE_FP8_BYTES + (size_t)off_in_page * 4);

        const int d_off = head * (D / H);   /* 2 bytes per thread */
        k_shmem[token_local * D + d_off    ] = k_ptr[d_off    ];
        k_shmem[token_local * D + d_off + 1] = k_ptr[d_off + 1];

        if (head == 0)
            scale_shmem[token_local] = *scale_ptr;
    }
    __syncthreads();

    /* -------------------------------------------------------------- */
    /* 2. Dot product Q[head] · K[token] then ReLU * weight[head]     */
    /* -------------------------------------------------------------- */
    float score = 0.0f;
    if (token < seq_len) {
        const auto* q_head = q + ((size_t)b * H + head) * D;
        const auto* k_tok  = k_shmem + token_local * D;
        const float k_scale = scale_shmem[token_local];

        float dot = 0.0f;
#pragma unroll 16
        for (int d = 0; d < D; ++d)
            dot += static_cast<float>(q_head[d]) * static_cast<float>(k_tok[d]);

        dot *= k_scale;
        score = (dot > 0.0f) ? (dot * weights[b * H + head]) : 0.0f;
    }

    /* -------------------------------------------------------------- */
    /* 3. Warp reduction: sum scores over H=64 heads (2 warps/token)  */
    /* -------------------------------------------------------------- */
    const int lane        = tid % 32;
    const int warp_id     = tid / 32;
    const int warp_in_tok = warp_id % 2;   /* 0 → heads 0-31, 1 → heads 32-63 */

#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        score += __shfl_down_sync(0xffffffffu, score, off);

    if (lane == 0)
        warp_sums[token_local * 2 + warp_in_tok] = score;
    __syncthreads();

    /* -------------------------------------------------------------- */
    /* 4. Write logit (thread 0 of each token in the block)           */
    /* -------------------------------------------------------------- */
    if (head == 0 && token < max_context_len) {
        const float logit = (token < seq_len)
            ? warp_sums[token_local * 2] + warp_sums[token_local * 2 + 1]
            : 0.0f;
        logits[(size_t)b * max_context_len + token] = logit;
    }
}

/* ------------------------------------------------------------------ */
/* Host launcher (called from Python via PyBind11)                     */
/* ------------------------------------------------------------------ */
void fp8_paged_mqa_logits(
    torch::Tensor q,            /* [B, H, D]              float8_e4m3fn */
    torch::Tensor k_cache,      /* [num_pages, PS, 1, D4] uint8         */
    torch::Tensor weights,      /* [B, H]                 float32        */
    torch::Tensor seq_lens,     /* [B]                    int32          */
    torch::Tensor block_table,  /* [B, max_num_pages]     int32          */
    torch::Tensor logits,       /* [B, max_context_len]   float32 (out)  */
    int max_context_len,
    int max_num_pages
) {
    const int B = static_cast<int>(q.size(0));
    const dim3 grid(B, (max_context_len + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK);
    const dim3 block(H * TOKENS_PER_BLOCK);   /* 512 threads */

    const size_t shmem =
          (size_t)TOKENS_PER_BLOCK * D * sizeof(__nv_fp8_e4m3)  /* k_shmem     */
        + (size_t)TOKENS_PER_BLOCK     * sizeof(float)           /* scale_shmem */
        + (size_t)TOKENS_PER_BLOCK * 2 * sizeof(float);          /* warp_sums   */

    fp8_paged_mqa_logits_kernel<<<grid, block, shmem>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(q.data_ptr()),
        reinterpret_cast<const uint8_t*>(k_cache.data_ptr()),
        weights.data_ptr<float>(),
        seq_lens.data_ptr<int>(),
        block_table.data_ptr<int>(),
        logits.data_ptr<float>(),
        max_context_len,
        max_num_pages
    );
    auto _err = cudaGetLastError();
    TORCH_CHECK(_err == cudaSuccess, "fp8_paged_mqa_logits kernel error: ", cudaGetErrorString(_err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fp8_paged_mqa_logits", &fp8_paged_mqa_logits,
          "FP8 paged MQA logits (custom CUDA)");
}
