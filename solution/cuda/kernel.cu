/*
 * Optimized CUDA Kernel for DSA TopK Indexer — B200 (sm_100)
 *
 * Pipeline:
 *   1. Fused FP8 page gather + dequant  (vec4 loads when D=128; else 1 thread/dim)
 *   2. Batched GEMM: q @ K^T            (torch::bmm, float32)
 *   3. In-place relu_ + mul_(weights) on logits (bit-exact vs relu*weights)
 *   4. Invalid K rows zeroed in gather; batched sum over heads → [B,S]; topk
 *      (optional -DFIB_TOPK_FP16: scores cast to fp16 before topk, FlashInfer baseline style)
 *   5. physical_flat[b,s]=page*PS+offset then lookup for top-k local idx (FlashInfer baseline style)
 *
 * NaN handling: FP8 E4M3 NaN bytes propagate as float NaN through K_batched
 * → logit NaN → relu(NaN) = 0 (CUDA fmaxf(0, NaN) = 0).  Matches reference.
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <limits>
#include <vector>

// ============================================================================
// Kernel: fused page gather + FP8 dequant + scale → K_batched [B, S, D] f32
//
// Cache layout per page (PS*HDS bytes total):
//   [PS * D  fp8 bytes]  K[t, d] at offset t*D + d within page
//   [PS * 4  float bytes] scale[t] at offset PS*D + t*4 within page
//
// Block:  D threads (one per dimension)
// Grid:   B*S blocks (one per token)
// ============================================================================
__global__ void gather_dequant_kernel(
        const uint8_t* __restrict__ cache,
        const int32_t* __restrict__ block_table,
        const int32_t* __restrict__ seq_lens,
        float*         __restrict__ K_batched,
        int B, int S, int D, int P, int PS, int HDS, int actual_pages)
{
    const int token_idx = blockIdx.x;
    if (token_idx >= B * S) return;
    const int b = token_idx / S;
    const int s = token_idx % S;
    const int d = threadIdx.x;
    if (d >= D) return;

    // Invalid tail tokens: K=0 → q@K^T gives 0 logits (same as post-bmm mask; avoids extra pass)
    if (s >= seq_lens[b]) {
        K_batched[token_idx * D + d] = 0.f;
        return;
    }

    const int page_id  = block_table[b * actual_pages + s / PS];
    const int t        = s % PS;
    const size_t page_byte  = (size_t)page_id * PS * HDS;
    const size_t fp8_byte   = page_byte + (size_t)t * D + d;
    const size_t scale_byte = page_byte + (size_t)PS * D + (size_t)t * 4;

    const uint8_t fp8_u8 = cache[fp8_byte];
    __nv_fp8_e4m3 fp8_val;
    *reinterpret_cast<uint8_t*>(&fp8_val) = fp8_u8;
    const float scale = *reinterpret_cast<const float*>(cache + scale_byte);

    // NaN FP8 → float NaN → NaN * scale = NaN → relu(NaN) = 0 later.
    K_batched[token_idx * D + d] = static_cast<float>(fp8_val) * scale;
}

// Vectorized gather: D=128 → 32 threads × 4 FP8/uint32 load; one float scale in shared mem.
// Same numerics as gather_dequant_kernel (per-byte FP8 → f32 × scale).
// ============================================================================
__global__ void gather_dequant_vec4_kernel(
        const uint8_t* __restrict__ cache,
        const int32_t* __restrict__ block_table,
        const int32_t* __restrict__ seq_lens,
        float* __restrict__ K_batched,
        int B, int S, int D, int P, int PS, int HDS, int actual_pages)
{
    const int token_idx = blockIdx.x;
    if (token_idx >= B * S) return;
    const int b = token_idx / S;
    const int s = token_idx % S;
    const int tid = threadIdx.x;
    const int base_d = tid * 4;

    float* row_out = K_batched + (long long)token_idx * D;

    if (s >= seq_lens[b]) {
#pragma unroll
        for (int i = 0; i < 4 && base_d + i < D; ++i) {
            row_out[base_d + i] = 0.f;
        }
        return;
    }

    const int page_id = block_table[b * actual_pages + s / PS];
    const int t = s % PS;
    const size_t page_byte = (size_t)page_id * PS * HDS;
    const size_t scale_byte = page_byte + (size_t)PS * D + (size_t)t * 4;

    __shared__ float s_scale;
    if (tid == 0) {
        s_scale = *reinterpret_cast<const float*>(cache + scale_byte);
    }
    __syncthreads();

    if (base_d >= D) return;

    const size_t fp8_quad = page_byte + (size_t)t * D + (size_t)base_d;
    const uint32_t w = *reinterpret_cast<const uint32_t*>(cache + fp8_quad);
    const float sc = s_scale;

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint8_t u8 = (uint8_t)((w >> (i * 8)) & 0xffu);
        __nv_fp8_e4m3 fp8_val;
        *reinterpret_cast<uint8_t*>(&fp8_val) = u8;
        row_out[base_d + i] = static_cast<float>(fp8_val) * sc;
    }
}

// For batched topk: scores[b,s] = -inf for s >= seq_lens[b] (invalid positions).
// scores layout: row-major [B, row_stride] with valid columns 0..max_seq_len-1.
// ============================================================================
__global__ void mask_scores_past_seq_len_kernel(
        float* __restrict__ scores,
        const int32_t* __restrict__ seq_lens,
        int B,
        long long row_stride,
        int max_seq_len)
{
    const int b = blockIdx.x;
    if (b >= B) return;
    const int sl = seq_lens[b];
    float* row = scores + (long long)b * row_stride;
    const float neg_inf = __uint_as_float(0xff800000u);
    for (int s = (int)threadIdx.x + sl; s < max_seq_len; s += (int)blockDim.x) {
        row[s] = neg_inf;
    }
}

// FlashInfer baseline-style: physical_flat[b,s] = page_id*PS + (s%PS) for flattened token s.
// ============================================================================
__global__ void build_physical_flat_kernel(
        const int32_t* __restrict__ block_table,
        int32_t* __restrict__ physical_flat,
        int B,
        int S,
        int actual_pages,
        int PS)
{
    const long long idx =
            (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)B * S;
    if (idx >= total) return;
    const int b = (int)(idx / S);
    const int s = (int)(idx % S);
    const int page_idx = s / PS;
    const int t = s % PS;
    const int pid = block_table[(long long)b * actual_pages + page_idx];
    physical_flat[idx] = pid * PS + t;
}

// Map local top-k index -> global token id using precomputed physical_flat[b, :].
// ============================================================================
__global__ void page_transform_lookup_kernel(
        const int64_t* __restrict__ local_long,
        const int32_t* __restrict__ seq_lens,
        const int32_t* __restrict__ physical_flat,
        int S,
        int32_t* __restrict__ out_packed,
        int K_topk)
{
    const int b = blockIdx.x;
    const int sl = seq_lens[b];
    if (sl <= 0) return;
    const int k = (K_topk < sl) ? K_topk : sl;

    const int64_t* loc = local_long + (long long)b * K_topk;
    const int32_t* phys_row = physical_flat + (long long)b * S;
    int32_t* o = out_packed + (long long)b * K_topk;

    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        const int local = (int)loc[i];
        o[i] = phys_row[local];
    }
}

// ============================================================================
// Entry point
// ============================================================================
void run(
        torch::Tensor q_index_fp8,
        torch::Tensor k_index_cache_fp8,
        torch::Tensor weights,
        torch::Tensor seq_lens,
        torch::Tensor block_table,
        torch::Tensor topk_indices)
{
    const int B      = (int)q_index_fp8.size(0);
    const int H      = (int)q_index_fp8.size(1);
    const int D      = (int)q_index_fp8.size(2);
    const int P      = (int)k_index_cache_fp8.size(0);
    const int PS     = (int)k_index_cache_fp8.size(1);
    const int HDS    = (int)k_index_cache_fp8.size(3);
    const int K_topk = (int)topk_indices.size(1);

    topk_indices.fill_(-1);

    torch::Tensor cache_u8 = k_index_cache_fp8.scalar_type() == torch::kUInt8
            ? k_index_cache_fp8 : k_index_cache_fp8.view(torch::kUInt8);

    auto sl_cpu = seq_lens.cpu();
    std::vector<int> sl_vec(B);
    if (sl_cpu.scalar_type() == torch::kInt32) {
        auto p = sl_cpu.data_ptr<int32_t>();
        for (int b = 0; b < B; ++b) sl_vec[b] = p[b];
    } else {
        auto p = sl_cpu.data_ptr<int64_t>();
        for (int b = 0; b < B; ++b) sl_vec[b] = (int)p[b];
    }
    int max_seq_len = *std::max_element(sl_vec.begin(), sl_vec.end());
    if (max_seq_len == 0) return;

    int max_pages_needed = (max_seq_len + PS - 1) / PS;
    int actual_pages     = std::min(max_pages_needed, (int)block_table.size(1));
    int S                = actual_pages * PS;

    auto bt_i32 = block_table.slice(1, 0, actual_pages)
                             .to(torch::kInt32).clamp(0, P - 1).contiguous();
    int32_t*     out_ptr = topk_indices.data_ptr<int32_t>();
    cudaStream_t stream  = at::cuda::getCurrentCUDAStream();
    torch::Tensor seq_lens_dev = seq_lens.to(q_index_fp8.device()).to(torch::kInt32).contiguous();

    // -------------------------------------------------------------------------
    // Phase 1 — fused gather + FP8 dequant + scale → K_batched [B, S, D]
    // (invalid s >= seq_len[b] written as 0 — no separate logits mask pass)
    // -------------------------------------------------------------------------
    const int num_tokens = B * S;
    torch::Tensor K_batched = torch::empty({B, S, D},
            q_index_fp8.options().dtype(torch::kFloat32));

    if (D == 128 && (D % 4) == 0) {
        constexpr int kGatherVecThreads = 32;
        gather_dequant_vec4_kernel<<<num_tokens, kGatherVecThreads, 0, stream>>>(
                cache_u8.data_ptr<uint8_t>(),
                bt_i32.data_ptr<int32_t>(),
                seq_lens_dev.data_ptr<int32_t>(),
                K_batched.data_ptr<float>(),
                B, S, D, P, PS, HDS, actual_pages);
    } else {
        gather_dequant_kernel<<<num_tokens, D, 0, stream>>>(
                cache_u8.data_ptr<uint8_t>(),
                bt_i32.data_ptr<int32_t>(),
                seq_lens_dev.data_ptr<int32_t>(),
                K_batched.data_ptr<float>(),
                B, S, D, P, PS, HDS, actual_pages);
    }

    // -------------------------------------------------------------------------
    // Phase 2 — batched GEMM: logits [B, H, S] = q @ K^T
    // -------------------------------------------------------------------------
    auto q_float = q_index_fp8.to(torch::kFloat32);  // [B, H, D]
    auto logits  = torch::bmm(q_float, K_batched.transpose(1, 2)).contiguous();  // [B, H, S]

    // -------------------------------------------------------------------------
    // Phase 3 — in-place relu then mul (same as relu()*w broadcast; one fewer [B,H,S] temp)
    // Phase 4 — topk
    // Phase 5 — page-table transform → global token indices
    // -------------------------------------------------------------------------
    constexpr int BLOCK_T = 128;
    auto w_bcast = weights.contiguous().unsqueeze(2);
    logits.relu_();
    logits.mul_(w_bcast);

    // One batched reduction over heads (replaces B× sum_out on [H,sl] slices in profiler).
    auto scores_2d = logits.sum(/*dim=*/1);

    auto opts_dev = q_index_fp8.options();
    torch::Tensor local_topk_long = torch::empty({B, K_topk}, opts_dev.dtype(torch::kInt64));
#ifdef FIB_TOPK_FP16
    auto vals_opts = logits.options().dtype(torch::kFloat16);
#else
    auto vals_opts = logits.options();
#endif
    torch::Tensor topk_vals_buf = torch::empty({B, K_topk}, vals_opts);

    int min_pos_sl = max_seq_len;
    for (int b = 0; b < B; ++b) {
        const int sl = sl_vec[b];
        if (sl > 0) min_pos_sl = std::min(min_pos_sl, sl);
    }
    const bool batched_topk = (min_pos_sl >= K_topk);

    constexpr bool kTopkSorted = false;

    if (batched_topk) {
        torch::Tensor scores_for_topk = scores_2d.narrow(1, 0, max_seq_len);
        const int64_t row_stride = scores_for_topk.stride(0);
        mask_scores_past_seq_len_kernel<<<B, 256, 0, stream>>>(
                scores_for_topk.data_ptr<float>(),
                seq_lens_dev.data_ptr<int32_t>(),
                B,
                (long long)row_stride,
                max_seq_len);
#ifdef FIB_TOPK_FP16
        torch::Tensor scores_h = scores_for_topk.to(torch::kFloat16);
#else
        torch::Tensor& scores_h = scores_for_topk;
#endif
        at::topk_out(
                topk_vals_buf,
                local_topk_long,
                scores_h,
                K_topk,
                /*dim=*/-1,
                /*largest=*/true,
                kTopkSorted);
    } else {
        for (int b = 0; b < B; ++b) {
            const int sl = sl_vec[b];
            if (sl == 0) continue;
            const int k = std::min(K_topk, sl);
            auto row_scores = scores_2d.select(0, b).narrow(0, 0, sl);
#ifdef FIB_TOPK_FP16
            torch::Tensor row_h = row_scores.to(torch::kFloat16);
#else
            torch::Tensor row_h = row_scores;
#endif
            auto topk_vals_slice = topk_vals_buf.select(0, b).narrow(0, 0, k);
            auto topk_idx_slice = local_topk_long.select(0, b).narrow(0, 0, k);
            at::topk_out(
                    topk_vals_slice,
                    topk_idx_slice,
                    row_h,
                    k,
                    /*dim=*/-1,
                    /*largest=*/true,
                    kTopkSorted);
        }
    }

    // Baseline-style flattened page table: one int32 per (b,s) → global physical token index
    torch::Tensor physical_flat = torch::empty({B, S}, torch::dtype(torch::kInt32).device(q_index_fp8.device()));
    {
        const int threads = 256;
        const int blocks = (int)(((long long)B * S + threads - 1) / threads);
        build_physical_flat_kernel<<<blocks, threads, 0, stream>>>(
                bt_i32.data_ptr<int32_t>(),
                physical_flat.data_ptr<int32_t>(),
                B,
                S,
                actual_pages,
                PS);
    }

    page_transform_lookup_kernel<<<B, BLOCK_T, 0, stream>>>(
            local_topk_long.data_ptr<int64_t>(),
            seq_lens_dev.data_ptr<int32_t>(),
            physical_flat.data_ptr<int32_t>(),
            S,
            out_ptr,
            K_topk);
}

// ============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "DSA TopK Indexer: fused FP8-gather + bmm + in-place relu/mul + topK + batched page-table transform",
          py::arg("q_index_fp8"), py::arg("k_index_cache_fp8"), py::arg("weights"),
          py::arg("seq_lens"),    py::arg("block_table"),       py::arg("topk_indices"));
}
