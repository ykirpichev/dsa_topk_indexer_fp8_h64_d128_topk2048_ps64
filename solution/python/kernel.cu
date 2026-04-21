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


// ============================================================================
// Stage 1: Paged MQA logits via tcgen05.mma.kind::f8f6f4 (SM100a UMMA)
// ============================================================================

__global__ __launch_bounds__(128)
void paged_mqa_logits_umma_kernel(
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
// Stage 2: Top-K (k=2048) + page table transform
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

        for (int i = tid; i < kRadix; i += blockDim.x) smem_hist[i] = 0u;
        __syncthreads();

        const uint16_t prefix = smem_prefix;
#pragma unroll 4
        for (int i = tid; i < seq_len; i += blockDim.x) {
            const uint16_t ordered = HalfToOrderedU16(row_lg[i]);
            if ((uint16_t)(ordered & prefix_mask) == prefix)
                atomicAdd(&smem_hist[(ordered >> shift) & 0xFFu], 1u);
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

    {
        int local = 0;
        for (int i = tid; i < seq_len; i += blockDim.x)
            if (HalfToOrderedU16(row_lg[i]) > pivot) local++;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            local += __shfl_xor_sync(0xffffffffu, local, off);
        if ((tid & 31) == 0) atomicAdd(&smem_gt_count, local);
    }
    __syncthreads();

    const int gt_total = smem_gt_count;

    for (int i = tid; i < seq_len; i += blockDim.x) {
        if (HalfToOrderedU16(row_lg[i]) > pivot) {
            const int pos = atomicAdd(&smem_emit_counter, 1);
            if (pos < gt_total) row_out[pos] = i;
        }
    }
    __syncthreads();

    for (int i = tid; i < seq_len; i += blockDim.x) {
        if (HalfToOrderedU16(row_lg[i]) == pivot) {
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
    const int max_kv_tiles = max_num_pages;
    const int max_len      = max_kv_tiles * kPageSize;

    auto logits = torch::empty({B, max_len},
        torch::TensorOptions().dtype(torch::kFloat16).device(q_index_fp8.device()));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    {
        dim3 grid(max_kv_tiles, B);
        dim3 block(128);
        paged_mqa_logits_umma_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_fp8_e4m3*>(q_index_fp8.data_ptr()),
            reinterpret_cast<const uint8_t*>(k_index_cache_fp8.data_ptr()),
            weights.data_ptr<float>(),
            seq_lens.data_ptr<int>(),
            block_table.data_ptr<int>(),
            max_num_pages, max_kv_tiles,
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
    m.def("run", &dsa_topk_indexer_cuda, "DSA TopK indexer (SM100a UMMA)");
}
