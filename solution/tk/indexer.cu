// DSA FP8 top-K indexer on ThunderKittens.
//
// Stage 1 is the part ThunderKittens is for: one CTA scores two KV pages with
// kittens::mm_ABt, the Blackwell tcgen05 MMA (FP8 E4M3 in, FP32 tensor-memory
// accumulate). K is paged, so each 16-byte chunk is loaded with an ordinary
// vector load and stored at the swizzled shared index the MMA descriptor
// reads. Stage 2 is the same two-round 8-bit radix select as the CUDA
// submission; ThunderKittens has no select primitive.

#include "kittens.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <stdexcept>
#include <string>

using namespace kittens;

namespace {

constexpr int kHeads = 64;
constexpr int kDim = 128;
constexpr int kPage = 64;
constexpr int kTopK = 2048;
constexpr int kPageBytes = 64 * 132;
constexpr int kTile = 128;
constexpr int kStage1Threads = 128;

// 128-byte swizzle, as a 16-byte vector index. Matches st_fp8e4m3's default
// layout when a row is 128 bytes: vector i moves to i ^ ((i >> 3) & 7).
__device__ __forceinline__ int swizzle_16b(int idx) {
    return idx ^ ((idx >> 3) & 7);
}

__global__ void fast_path(const int* __restrict__ seq_lens,
                          const int* __restrict__ block_table,
                          int* __restrict__ out,
                          int max_pages) {
    const int b = blockIdx.x;
    const int seq = seq_lens[b];
    for (int n = threadIdx.x; n < kTopK; n += blockDim.x) {
        const int page = block_table[b * max_pages + min(n / kPage, max_pages - 1)];
        out[b * kTopK + n] = (n < seq) ? page * kPage + (n % kPage) : -1;
    }
}

// One CTA, one M=128 tile of one row. 128 threads is one warpgroup, which is
// what mm_ABt and the tensor-memory load expect.
__global__ void stage1(const uint8_t* __restrict__ q,
                       const uint8_t* __restrict__ k,
                       const float* __restrict__ kscale,
                       const float* __restrict__ weights,
                       const int* __restrict__ seq_lens,
                       const int* __restrict__ block_table,
                       __half* __restrict__ logits,
                       int max_pages,
                       int row_stride) {
    const int tile = blockIdx.x;
    const int b = blockIdx.y;
    const int tx = threadIdx.x;
    const int seq = seq_lens[b];
    const int last = (seq > kTopK) ? (seq + kTile - 1) / kTile : 0;
    if (tile >= last) return;

    extern __shared__ alignment_dummy __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    auto& k_smem = al.allocate<st_fp8e4m3<kTile, kDim>>();
    auto& q_smem = al.allocate<st_fp8e4m3<kHeads, kDim>>();
    auto& c_smem = al.allocate<st_fl<kTile, kHeads>>();

    __shared__ float w_s[kHeads];
    __shared__ float scale_s[kTile];
    __shared__ semaphore mma_done;
    __shared__ uint32_t tmem_addr;

    const int last_page = max((seq - 1) / kPage, 0);
    const int* row_bt = block_table + b * max_pages;

    // One 16-byte store per vector, at the swizzled index the MMA descriptor
    // reads. A byte-wise store through operator[] issues 16 shared stores
    // for the same 16 bytes.
    {
        uint4* dst = reinterpret_cast<uint4*>(q_smem.data);
        #pragma unroll
        for (int j = 0; j < (kHeads * kDim) / 16 / kStage1Threads; ++j) {
            const int v = tx + j * kStage1Threads;
            const int h = v >> 3;
            const int col = (v & 7) << 4;
            const uint8_t* src = q + (static_cast<int64_t>(b) * kHeads + h) * kDim + col;
            dst[swizzle_16b(v)] = *reinterpret_cast<const uint4*>(src);
        }
    }
    {
        uint4* dst = reinterpret_cast<uint4*>(k_smem.data);
        uint4 regs[(kTile * kDim) / 16 / kStage1Threads];
        int dest[(kTile * kDim) / 16 / kStage1Threads];
        #pragma unroll
        for (int j = 0; j < (kTile * kDim) / 16 / kStage1Threads; ++j) {
            const int v = tx + j * kStage1Threads;
            const int i = v >> 3;
            const int col = (v & 7) << 4;
            const int tok = tile * kTile + i;
            const int slot = min(tok / kPage, last_page);
            const int page = row_bt[slot];
            const int row = tok % kPage;
            const int64_t addr = static_cast<int64_t>(page) * kPageBytes + row * kDim + col;
            regs[j] = *reinterpret_cast<const uint4*>(k + addr);
            dest[j] = swizzle_16b(v);
        }
        #pragma unroll
        for (int j = 0; j < (kTile * kDim) / 16 / kStage1Threads; ++j)
            dst[dest[j]] = regs[j];
    }
    if (tx < kHeads) w_s[tx] = weights[b * kHeads + tx];
    if (tx < kTile) {
        const int tok = tile * kTile + tx;
        const int slot = min(tok / kPage, last_page);
        const int page = row_bt[slot];
        const int row = tok % kPage;
        scale_s[tx] = kscale[static_cast<int64_t>(page) * (kPageBytes / 4) + (kDim * kPage) / 4 + row];
    }
    if (tx == 0) init_semaphore(mma_done, 0, 1);
    __syncthreads();

    // tcgen05.alloc is sync.aligned: the whole first warp has to execute it.
    tensor_allocator<1, 1, false> tm{};
    if (tx < 32) tm.provision(tmem_addr);
    __syncthreads();
    tm.set_addr(tmem_addr);
    tt<float, kTile, kHeads> acc = tm.allocate<tt<float, kTile, kHeads>>(0);

    // One thread issues the MMA. mm_ABt clears the accumulator, steps K by 32,
    // and commits mma_done when the last step lands.
    if (tx == 0) mm_ABt(acc, k_smem, q_smem, mma_done);
    wait(mma_done, 0);

    // A warpgroup holds the 128x64 accumulator as a 32x64 register tile.
    rt_fl<32, kHeads> regs;
    warpgroup::load_async(regs, acc);
    tensor_load_wait();
    warpgroup::sync(warpgroup::groupid() + 1);
    warpgroup::store(c_smem, regs);
    __syncthreads();

    // Thread i owns token i of this tile and reduces the 64 heads.
    float sum = 0.f;
    #pragma unroll
    for (int h = 0; h < kHeads; ++h) sum += fmaxf(c_smem[{tx, h}], 0.f) * w_s[h];
    const int tok = tile * kTile + tx;
    if (tok < seq) logits[b * row_stride + tok] = __float2half(sum * scale_s[tx]);

    __syncthreads();
    if (tx < 32) tm.deprovision();
}

__device__ __forceinline__ uint16_t half_to_ordered(const __half* row, int i) {
    const uint16_t bits = __half_as_ushort(row[i]);
    return (bits & 0x8000u) ? static_cast<uint16_t>(~bits)
                            : static_cast<uint16_t>(bits ^ 0x8000u);
}

// Two-round 8-bit radix select, 1024 threads, ordered keys cached in shared
// memory. Same structure as the CUDA submission's stage 2.
__global__ void __launch_bounds__(1024)
stage2(const __half* __restrict__ logits,
       const int* __restrict__ seq_lens,
       const int* __restrict__ block_table,
       int* __restrict__ out,
       int max_pages,
       int stride) {
    constexpr int kThreads = 1024;
    constexpr int kBins = 256;
    constexpr int kCache = 16384;
    const int b = blockIdx.x;
    const int tx = threadIdx.x;
    const int seq = seq_lens[b];
    const int* row_bt = block_table + b * max_pages;
    int* row_out = out + b * kTopK;
    const __half* row = logits + static_cast<int64_t>(b) * stride;

    for (int n = tx; n < kTopK; n += kThreads) {
        if (n < seq) {
            const int page = row_bt[min(n / kPage, max_pages - 1)];
            row_out[n] = page * kPage + (n % kPage);
        } else {
            row_out[n] = -1;
        }
    }
    if (seq <= kTopK) return;

    __shared__ uint32_t hist[kBins];
    __shared__ uint32_t suffix[kBins];
    __shared__ uint16_t prefix, ordered[kCache];
    __shared__ uint32_t remaining, found_bin, found_remaining;
    __shared__ int gt_count, emit;

    if (tx == 0) {
        prefix = 0;
        remaining = kTopK;
    }
    __syncthreads();

    #pragma unroll
    for (int round = 0; round < 2; ++round) {
        const int shift = 8 - round * 8;
        const uint16_t prefix_mask = (round == 0) ? uint16_t{0}
                                                   : static_cast<uint16_t>(0xFFFFu << (16 - round * 8));
        for (int i = tx; i < kBins; i += kThreads) hist[i] = 0;
        __syncthreads();
        const uint16_t cur = prefix;
        for (int i = tx; i < seq; i += kThreads) {
            uint16_t key;
            if (round == 0) {
                key = half_to_ordered(row, i);
                ordered[i] = key;
            } else {
                key = ordered[i];
            }
            if ((uint16_t)(key & prefix_mask) == cur)
                atomicAdd(hist + ((key >> shift) & 0xFFu), 1u);
        }
        __syncthreads();
        if (tx < kBins) suffix[tx] = hist[tx];
        __syncthreads();
        for (int stride_b = 1; stride_b < kBins; stride_b <<= 1) {
            uint32_t v = 0;
            if (tx < kBins) {
                v = suffix[tx];
                if (tx + stride_b < kBins) v += suffix[tx + stride_b];
            }
            __syncthreads();
            if (tx < kBins) suffix[tx] = v;
            __syncthreads();
        }
        if (tx == 0) {
            found_bin = 0;
            found_remaining = remaining;
        }
        __syncthreads();
        if (tx < kBins) {
            const uint32_t cge = suffix[tx];
            const uint32_t cgt = (tx + 1 < kBins) ? suffix[tx + 1] : 0u;
            if (cge >= remaining && cgt < remaining) {
                found_bin = tx;
                found_remaining = remaining - cgt;
            }
        }
        __syncthreads();
        if (tx == 0) {
            prefix = static_cast<uint16_t>(prefix | (found_bin << shift));
            remaining = found_remaining;
        }
        __syncthreads();
    }

    const uint16_t pivot = prefix;
    if (tx == 0) {
        gt_count = 0;
        emit = 0;
    }
    __syncthreads();
    int local = 0;
    for (int i = tx; i < seq; i += kThreads)
        local += ordered[i] > pivot;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        local += __shfl_xor_sync(0xffffffffu, local, off);
    if ((tx & 31) == 0) atomicAdd(&gt_count, local);
    __syncthreads();

    for (int i = tx; i < seq; i += kThreads) {
        if (ordered[i] > pivot) {
            const int pos = atomicAdd(&emit, 1);
            if (pos < gt_count) row_out[pos] = i;
        }
    }
    __syncthreads();
    for (int i = tx; i < seq; i += kThreads) {
        if (ordered[i] == pivot) {
            const int pos = atomicAdd(&emit, 1);
            if (pos < kTopK) row_out[pos] = i;
        }
    }
    __syncthreads();
    for (int i = tx; i < kTopK; i += kThreads) {
        const int tok = row_out[i];
        if (tok >= 0 && tok < seq)
            row_out[i] = row_bt[tok / kPage] * kPage + (tok % kPage);
        else
            row_out[i] = -1;
    }
}

int stage1_smem() {
    // The swizzle allocator rounds each tile up to 1024 bytes, so the raw
    // sizeof sum is not enough on its own.
    return static_cast<int>(sizeof(st_fp8e4m3<kTile, kDim>) + sizeof(st_fp8e4m3<kHeads, kDim>) +
                            sizeof(st_fl<kTile, kHeads>) + 4096);
}

void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

}  // namespace

void run(torch::Tensor q, torch::Tensor k, torch::Tensor weights, torch::Tensor seq_lens,
         torch::Tensor block_table, torch::Tensor topk_indices) {
    const int batch = block_table.size(0);
    const int max_pages = block_table.size(1);
    auto stream = at::cuda::getCurrentCUDAStream();

    if (max_pages <= 32) {
        fast_path<<<batch, 256, 0, stream>>>(
            seq_lens.data_ptr<int>(), block_table.data_ptr<int>(),
            topk_indices.data_ptr<int>(), max_pages);
        check(cudaGetLastError(), "fast path");
        return;
    }

    const int tiles = (max_pages + 1) / 2;
    const int stride = tiles * kTile;
    // Reused across calls. Kernels on this stream are ordered, so the next
    // launch cannot read the buffer until the previous stage 2 has finished
    // with it. A shape change has to drain the stream before the old storage
    // is released.
    static torch::Tensor logits;
    static int cached_batch = -1, cached_stride = -1;
    if (cached_batch != batch || cached_stride != stride || !logits.defined()) {
        if (logits.defined()) check(cudaStreamSynchronize(stream), "scratch resize");
        logits = torch::empty({batch, stride}, q.options().dtype(torch::kFloat16));
        cached_batch = batch;
        cached_stride = stride;
    }

    auto* fn = stage1;
    const int smem = stage1_smem();
    static int configured_smem = -1;
    if (configured_smem != smem) {
        check(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem), "stage1 smem");
        configured_smem = smem;
    }
    fn<<<dim3(tiles, batch), kStage1Threads, smem, stream>>>(
        reinterpret_cast<const uint8_t*>(q.data_ptr()),
        reinterpret_cast<const uint8_t*>(k.data_ptr()),
        reinterpret_cast<const float*>(k.data_ptr()),
        weights.data_ptr<float>(),
        seq_lens.data_ptr<int>(),
        block_table.data_ptr<int>(),
        reinterpret_cast<__half*>(logits.data_ptr()),
        max_pages,
        stride);
    check(cudaGetLastError(), "stage1 launch");

    if (stride > 16384) {
        throw std::runtime_error("row longer than the ThunderKittens stage 2 cache");
    }
    stage2<<<batch, 1024, 0, stream>>>(
        reinterpret_cast<const __half*>(logits.data_ptr()),
        seq_lens.data_ptr<int>(),
        block_table.data_ptr<int>(),
        topk_indices.data_ptr<int>(),
        max_pages,
        stride);
    check(cudaGetLastError(), "stage2 launch");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run);
}
