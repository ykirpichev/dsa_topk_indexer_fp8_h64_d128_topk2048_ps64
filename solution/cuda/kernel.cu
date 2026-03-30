/*
 * Optimized CUDA Kernel for DSA TopK Indexer — B200 (sm_100)
 *
 * Pipeline:
 *   1. Fused FP8 page gather + dequant  (custom CUDA kernel)
 *   2. Batched GEMM: q @ K^T            (torch::bmm, float32) → logits [B,H,S]
 *   3. Per batch row: scores = (logits[b, :, 0:sl].relu() * weights[b].unsqueeze(1)).sum(0)
 *      — same reduction as the old full [B,H,S] weighted tensor, but peak memory is
 *      O(H*sl) per batch instead of O(B*H*S). torch::topk unchanged.
 *   4. page_transform kernel            (local → global)
 *
 * NaN: PyTorch relu on logits (same as reference).
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>

// ============================================================================
// Kernel: fused page gather + FP8 dequant + scale → K_batched [B, S, D] f32
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

    K_batched[token_idx * D + d] = static_cast<float>(fp8_val) * scale;
}

// ============================================================================
// Kernel: page-table transform local → global token indices
// ============================================================================
__global__ void page_transform_kernel(
        const int* __restrict__ local_indices,
        const int* __restrict__ block_table_b,
        int*       __restrict__ out,
        int actual_topk, int PS)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= actual_topk) return;
    int local = local_indices[i];
    out[i] = block_table_b[local / PS] * PS + (local % PS);
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
    const int*   bt_ptr  = bt_i32.data_ptr<int32_t>();
    int32_t*     out_ptr = topk_indices.data_ptr<int32_t>();
    cudaStream_t stream  = at::cuda::getCurrentCUDAStream();

    const int num_tokens = B * S;
    torch::Tensor K_batched = torch::empty({B, S, D},
            q_index_fp8.options().dtype(torch::kFloat32));

    gather_dequant_kernel<<<num_tokens, D, 0, stream>>>(
            cache_u8.data_ptr<uint8_t>(),
            bt_i32.data_ptr<int32_t>(),
            K_batched.data_ptr<float>(),
            B, S, D, P, PS, HDS, actual_pages);

    auto q_float = q_index_fp8.to(torch::kFloat32);
    auto logits  = torch::bmm(q_float, K_batched.transpose(1, 2)).contiguous();

    constexpr int BLOCK_T = 256;

    for (int b = 0; b < B; ++b) {
        const int sl = sl_vec[b];
        if (sl == 0) continue;
        const int actual_topk = std::min(K_topk, sl);

        auto row_logits = logits[b].narrow(1, 0, sl);
        auto row_w = weights[b].unsqueeze(1);
        auto scores = (row_logits.relu() * row_w).sum(0);
        auto topk_out = scores.topk(actual_topk, -1, true, true);
        auto topk_idxs = std::get<1>(topk_out).to(torch::kInt32).contiguous();

        int grid_t = (actual_topk + BLOCK_T - 1) / BLOCK_T;
        page_transform_kernel<<<grid_t, BLOCK_T, 0, stream>>>(
                topk_idxs.data_ptr<int32_t>(),
                bt_ptr + (long long)b * actual_pages,
                out_ptr + (long long)b * K_topk,
                actual_topk, PS);
    }
}

// ============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "DSA TopK Indexer: fused FP8-gather + bmm + batched row scores + topk + page transform",
          py::arg("q_index_fp8"), py::arg("k_index_cache_fp8"), py::arg("weights"),
          py::arg("seq_lens"),    py::arg("block_table"),       py::arg("topk_indices"));
}
