/*
 * Optimized CUDA Kernel for DSA TopK Indexer — B200 (sm_100)
 *
 * Pipeline:
 *   1. Fused FP8 page gather + dequant  (custom CUDA kernel)
 *   2. Batched GEMM: q @ K^T            (torch::bmm, float32)
 *   3. In-place relu_ + mul_(weights) on logits (bit-exact vs relu*weights)
 *   4. Mask invalid [s >= seq_len] to 0, batched sum over heads → [B,S]; topk
 *   5. Batched page_transform: int64 top-k indices + device seq_lens → global int32
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
        float*         __restrict__ K_batched,
        int B, int S, int D, int P, int PS, int HDS, int actual_pages)
{
    const int token_idx = blockIdx.x;
    if (token_idx >= B * S) return;
    const int b = token_idx / S;
    const int s = token_idx % S;
    const int d = threadIdx.x;
    if (d >= D) return;

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

// Zero logits[b,h,s] for s >= seq_lens[b] so sum(dim=1) matches narrow(1,0,sl).sum(0).
// logits: [B, H, S] contiguous float32.
// ============================================================================
__global__ void mask_logits_past_seq_len_kernel(
        float* __restrict__ logits,
        const int32_t* __restrict__ seq_lens,
        int B, int H, int S)
{
    const int bh = blockIdx.x;
    const int b = bh / H;
    const int h = bh % H;
    if (b >= B || h >= H) return;
    const int sl = seq_lens[b];
    const long long base = ((long long)b * H + h) * S;
    for (int s = (int)threadIdx.x + sl; s < S; s += (int)blockDim.x) {
        logits[base + s] = 0.f;
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

// ============================================================================
// Batched page-table transform: one block per batch row b.
// local_long[b * K_topk + i] = top-k local indices (int64 from at::topk_out);
// k = min(K_topk, seq_lens[b])
// ============================================================================
__global__ void page_transform_batched_kernel(
        const int64_t* __restrict__ local_long,
        const int32_t* __restrict__ seq_lens,
        const int32_t* __restrict__ block_table,
        int32_t*       __restrict__ out_packed,
        int actual_pages,
        int K_topk,
        int PS)
{
    const int b = blockIdx.x;
    const int sl = seq_lens[b];
    if (sl <= 0) return;
    const int k = (K_topk < sl) ? K_topk : sl;

    const int64_t* loc = local_long + (long long)b * K_topk;
    const int32_t* bt  = block_table + (long long)b * actual_pages;
    int32_t* o = out_packed + (long long)b * K_topk;

    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        int local = (int)loc[i];
        o[i] = bt[local / PS] * PS + (local % PS);
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

    // -------------------------------------------------------------------------
    // Phase 1 — fused gather + FP8 dequant + scale → K_batched [B, S, D]
    // -------------------------------------------------------------------------
    const int num_tokens = B * S;
    torch::Tensor K_batched = torch::empty({B, S, D},
            q_index_fp8.options().dtype(torch::kFloat32));

    gather_dequant_kernel<<<num_tokens, D, 0, stream>>>(
            cache_u8.data_ptr<uint8_t>(),
            bt_i32.data_ptr<int32_t>(),
            K_batched.data_ptr<float>(),
            B, S, D, P, PS, HDS, actual_pages);

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

    torch::Tensor seq_lens_dev = seq_lens.to(logits.device()).to(torch::kInt32).contiguous();
    const int n_bh = B * H;
    mask_logits_past_seq_len_kernel<<<n_bh, 256, 0, stream>>>(
            logits.data_ptr<float>(),
            seq_lens_dev.data_ptr<int32_t>(),
            B, H, S);

    // One batched reduction over heads (replaces B× sum_out on [H,sl] slices in profiler).
    auto scores_2d = logits.sum(/*dim=*/1);

    auto opts_dev = q_index_fp8.options();
    torch::Tensor local_topk_long = torch::empty({B, K_topk}, opts_dev.dtype(torch::kInt64));
    torch::Tensor topk_vals_buf = torch::empty({B, K_topk}, logits.options());

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
        at::topk_out(
                topk_vals_buf,
                local_topk_long,
                scores_for_topk,
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
            auto topk_vals_slice = topk_vals_buf.select(0, b).narrow(0, 0, k);
            auto topk_idx_slice = local_topk_long.select(0, b).narrow(0, 0, k);
            at::topk_out(
                    topk_vals_slice,
                    topk_idx_slice,
                    row_scores,
                    k,
                    /*dim=*/-1,
                    /*largest=*/true,
                    kTopkSorted);
        }
    }

    page_transform_batched_kernel<<<B, BLOCK_T, 0, stream>>>(
            local_topk_long.data_ptr<int64_t>(),
            seq_lens_dev.data_ptr<int32_t>(),
            bt_i32.data_ptr<int32_t>(),
            out_ptr,
            actual_pages,
            K_topk,
            PS);
}

// ============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "DSA TopK Indexer: fused FP8-gather + bmm + in-place relu/mul + topK + batched page-table transform",
          py::arg("q_index_fp8"), py::arg("k_index_cache_fp8"), py::arg("weights"),
          py::arg("seq_lens"),    py::arg("block_table"),       py::arg("topk_indices"));
}
