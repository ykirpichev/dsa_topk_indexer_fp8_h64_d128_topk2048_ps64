// Host entry point: picks one of the three plans from the workload shape and
// runs the resulting launch sequence through the graph cache. Unlike the
// kernel headers this one lives at global scope; it defines the exported
// symbol.

#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <array>
#include <mutex>

#include "cuda_utils.cuh"
#include "dsa_config.cuh"
#include "fast_path.cuh"
#include "graph_cache.cuh"
#include "stage1_persistent_ws.cuh"
#include "stage1_short.cuh"
#include "stage2_topk.cuh"

void dsa_topk_indexer_cuda(
    torch::Tensor q_index_fp8,
    torch::Tensor k_index_cache_fp8,
    torch::Tensor weights,
    torch::Tensor seq_lens,
    torch::Tensor block_table,
    torch::Tensor topk_indices)
{
    // Fast-path workloads run a ~2 us kernel, so host-side CPU work is a
    // measurable share of the reported latency: only the checks the fast path
    // needs run here, Q/K/weights validation is deferred below.
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

    const Plan plan = select_plan(max_num_pages);

    int                  max_kv_tile_pairs = 0;
    int                  tiles_per_cta     = 0;
    int                  num_splits        = 0;
    const int            max_len           = stage2_max_len(plan, max_num_pages);
    __half*              logits_ptr        = nullptr;
    const __nv_fp8_e4m3* q_ptr             = nullptr;
    const uint8_t*       kv_ptr            = nullptr;
    const float*         w_ptr             = nullptr;

    if (plan != Plan::FastPath) {
        // Validation only the UMMA paths need, paid only when we take them.
        TORCH_CHECK(q_index_fp8.is_cuda() && q_index_fp8.is_contiguous(),
                    "q_index_fp8 must be contiguous CUDA");
        TORCH_CHECK(k_index_cache_fp8.is_cuda() && k_index_cache_fp8.is_contiguous(),
                    "k_index_cache_fp8 must be contiguous CUDA");
        TORCH_CHECK(weights.is_cuda() && weights.is_contiguous() &&
                        weights.scalar_type() == torch::kFloat32,
                    "weights must be float32 contiguous CUDA");

        // Stage-1 logits scratch, one geometrically grown buffer per device.
        // `torch::empty` is not graph-safe, so this stays outside capture; the
        // pointer only moves when B*max_len outgrows the allocation, and the
        // cache entry (keyed by shape) misses at that point anyway.
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

        q_ptr  = reinterpret_cast<const __nv_fp8_e4m3*>(q_index_fp8.data_ptr());
        kv_ptr = reinterpret_cast<const uint8_t*>(k_index_cache_fp8.data_ptr());
        w_ptr  = weights.data_ptr<float>();

        if (plan == Plan::PersistentWs) {
            const PersistentGrid grid = compute_persistent_grid(B, max_num_pages);
            max_kv_tile_pairs = grid.max_kv_tile_pairs;
            tiles_per_cta     = grid.tiles_per_cta;
            num_splits        = grid.num_splits;
        }
    }

    // Parameterised on the stream so capture can run it on a private one.
    auto dispatch = [&](cudaStream_t s) {
        if (plan == Plan::FastPath) {
            dim3 grid(B);
            dim3 block(128);
            topk_fast_path_kernel<<<grid, block, 0, s>>>(
                sl_ptr, bt_ptr, max_num_pages, out_ptr);
            return;
        }
        // Stage 1.
        if (plan == Plan::PersistentWs) {
            dim3 grid(num_splits, B);
            dim3 block(256);
            paged_mqa_logits_umma_kernel_persistent_ws<<<grid, block, 0, s>>>(
                q_ptr, kv_ptr, w_ptr, sl_ptr, bt_ptr,
                max_num_pages, max_kv_tile_pairs, tiles_per_cta,
                logits_ptr);
        } else {
            // The page count is passed twice: block-table stride, then logits
            // row stride, which for this plan is max_num_pages * kPageSize.
            dim3 grid(max_num_pages, B);
            dim3 block(128);
            paged_mqa_logits_umma_kernel_short<<<grid, block, 0, s>>>(
                q_ptr, kv_ptr, w_ptr, sl_ptr, bt_ptr,
                max_num_pages, max_num_pages,
                logits_ptr);
        }
        // Stage 2.
        dim3 grid2(B);
        dim3 block2(kStage2Threads);
        topk_page_table_transform_kernel<<<grid2, block2, 0, s>>>(
            logits_ptr, sl_ptr, bt_ptr,
            max_len, max_num_pages, kTopK, out_ptr);
    };

    g_graph_cache.replay(
        GraphKey::make(stream, B, max_num_pages),
        capture_stream(),
        stream,
        dispatch);
}
