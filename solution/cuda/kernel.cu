/*
 * Optimized CUDA Kernel for DSA TopK Indexer — B200 (sm_100)
 *
 * Pipeline:
 *   1. Fused gather + q·K dot → [B,H,S] buffer (no K_batched).
 *   2. In-place aten::relu_ then mul_(weights.unsqueeze(2)) — custom relu*mul in CUDA
 *      diverges from ATen on many workloads (bitwise top-k).
 *   3. topk_out + batched page_transform
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <vector>

__global__ void fused_gather_qk_dot_kernel(
        const uint8_t* __restrict__ cache,
        const int32_t* __restrict__ block_table,
        const float* __restrict__ q_f32,
        float* __restrict__ logits_out,
        int B, int S, int H, int D, int P, int PS, int HDS, int actual_pages)
{
    const int token_idx = blockIdx.x;
    if (token_idx >= B * S) return;
    const int b = token_idx / S;
    const int s = token_idx % S;

    extern __shared__ float smem[];
    float* krow = smem;

    const int tid = threadIdx.x;

    if (tid < D) {
        const int page_id = block_table[b * actual_pages + s / PS];
        const int t = s % PS;
        const size_t page_byte = (size_t)page_id * PS * HDS;
        const size_t fp8_byte = page_byte + (size_t)t * D + tid;
        const size_t scale_byte = page_byte + (size_t)PS * D + (size_t)t * 4;

        const uint8_t fp8_u8 = cache[fp8_byte];
        __nv_fp8_e4m3 fp8_val;
        *reinterpret_cast<uint8_t*>(&fp8_val) = fp8_u8;
        const float scale = *reinterpret_cast<const float*>(cache + scale_byte);
        krow[tid] = static_cast<float>(fp8_val) * scale;
    }
    __syncthreads();

    const int64_t q_batch = (int64_t)b * H * D;

    if (tid == 0) {
        for (int h = 0; h < H; ++h) {
            float acc = 0.f;
            const float* qh = q_f32 + q_batch + (int64_t)h * D;
            for (int d = 0; d < D; ++d)
                acc = fmaf(qh[d], krow[d], acc);
            logits_out[(int64_t)b * H * S + (int64_t)h * S + s] = acc;
        }
    }
}

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

    auto q_float = q_index_fp8.to(torch::kFloat32).contiguous();
    auto w_cont = weights.contiguous();
    torch::Tensor weighted = torch::empty({B, H, S}, q_float.options());

    const int num_tokens = B * S;
    const size_t shmem = (size_t)D * sizeof(float);
    fused_gather_qk_dot_kernel<<<num_tokens, 128, shmem, stream>>>(
            cache_u8.data_ptr<uint8_t>(),
            bt_i32.data_ptr<int32_t>(),
            q_float.data_ptr<float>(),
            weighted.data_ptr<float>(),
            B, S, H, D, P, PS, HDS, actual_pages);
    weighted.relu_();
    weighted.mul_(w_cont.unsqueeze(2));

    constexpr int BLOCK_T = 256;

    auto opts_dev = q_index_fp8.options();
    torch::Tensor local_topk_long = torch::empty({B, K_topk}, opts_dev.dtype(torch::kInt64));
    torch::Tensor seq_lens_dev = seq_lens.to(weighted.device()).to(torch::kInt32).contiguous();

    for (int b = 0; b < B; ++b) {
        const int sl = sl_vec[b];
        if (sl == 0) continue;
        const int k = std::min(K_topk, sl);
        auto scores = weighted[b].narrow(1, 0, sl).sum(0);
        auto topk_vals = torch::empty({k}, scores.options());
        auto topk_idx_slice = local_topk_long.select(0, b).narrow(0, 0, k);
        at::topk_out(topk_vals, topk_idx_slice, scores, k, /*dim=*/-1, /*largest=*/true, /*sorted=*/true);
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "DSA TopK: fused gather+qk + in-place relu_/mul_ + topk + page transform",
          py::arg("q_index_fp8"), py::arg("k_index_cache_fp8"), py::arg("weights"),
          py::arg("seq_lens"),    py::arg("block_table"),       py::arg("topk_indices"));
}
