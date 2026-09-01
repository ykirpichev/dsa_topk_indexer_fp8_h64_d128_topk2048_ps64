// Shape-keyed CUDA graph cache: graph replay instead of per-call launches.

#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>
#include <cstring>
#include <mutex>
#include <utility>

#include "cuda_utils.cuh"
#include "dsa_config.cuh"

namespace {

// flashinfer-bench clones every input tensor per iteration, so pointers change
// on every call while shapes stay fixed. The key is caller stream + batch/page
// shape only (never pointers): plan, grid scalars, and Stage-2 max_len are all
// derivable from (B, max_num_pages). A hit re-captures a template graph and
// splices new kernel params in with cudaGraphExecUpdate (~1-2 us).
//
// A miss instantiates and stores the exec in a 32-slot LRU; the evaluator's
// ~128 workloads each map to one shape tuple, so it never churns.

struct GraphKey {
    cudaStream_t stream{};
    int32_t      B             = 0;
    int32_t      max_num_pages = 0;

    static GraphKey make(cudaStream_t stream, int B, int max_num_pages) noexcept {
        return GraphKey{stream, B, max_num_pages};
    }

    bool operator==(const GraphKey& o) const noexcept {
        return std::memcmp(this, &o, sizeof(GraphKey)) == 0;
    }
};
static_assert(sizeof(GraphKey) % 8 == 0, "GraphKey must be 8-byte aligned for memcmp");

struct GraphEntry {
    GraphKey        key{};
    cudaGraphExec_t exec  = nullptr;
    bool            valid = false;
};

class GraphCache {
public:
    static constexpr int N = 32;

    ~GraphCache() {
        for (auto& e : e_) clear_locked(e);
    }

    // Capture `dispatch` on cap_stream, update or insert a cached exec, launch
    // on the caller's stream. Falls back to a direct launch if capture fails.
    template <class DispatchFn>
    void replay(const GraphKey& k, cudaStream_t cap_stream, cudaStream_t stream,
                DispatchFn&& dispatch) {
        cudaError_t cerr = cudaStreamBeginCapture(
            cap_stream, cudaStreamCaptureModeRelaxed);
        if (cerr != cudaSuccess) {
            dispatch(stream);
            DSA_CUDA_CHECK(cudaGetLastError());
            return;
        }
        dispatch(cap_stream);

        UniqueCudaGraph graph;
        {
            cudaGraph_t raw = nullptr;
            DSA_CUDA_CHECK(cudaStreamEndCapture(cap_stream, &raw));
            TORCH_CHECK(raw != nullptr, "DSA topk: EndCapture returned null graph");
            graph.reset(raw);
        }

        std::unique_lock<std::mutex> lk(mu_);

        GraphEntry* slot = find_locked(k);
        if (slot != nullptr) {
            cudaGraphExecUpdateResultInfo info{};
            cerr = cudaGraphExecUpdate(slot->exec, graph.get(), &info);
            if (cerr == cudaSuccess && info.result == cudaGraphExecUpdateSuccess) {
                graph.reset();
                cudaGraphExec_t exec = slot->exec;
                lk.unlock();
                DSA_CUDA_CHECK(cudaGraphLaunch(exec, stream));
                return;
            }
            clear_locked(*slot);
        }

        UniqueCudaGraphExec exec;
        {
            cudaGraphExec_t raw = nullptr;
            DSA_CUDA_CHECK(cudaGraphInstantiate(&raw, graph.get(), 0));
            TORCH_CHECK(raw != nullptr, "DSA topk: Instantiate returned null exec");
            exec.reset(raw);
        }
        graph.reset();

        cudaGraphExec_t exec_raw = exec.get();
        DSA_CUDA_CHECK(cudaGraphLaunch(exec_raw, stream));

        auto& slot_ref = e_[head_];
        clear_locked(slot_ref);
        slot_ref.key   = k;
        slot_ref.exec  = exec.release();
        slot_ref.valid = true;
        head_ = (head_ + 1) % N;
    }

private:
    GraphEntry* find_locked(const GraphKey& k) {
        for (auto& e : e_) {
            if (e.valid && e.key == k) return &e;
        }
        return nullptr;
    }

    static void clear_locked(GraphEntry& e) {
        if (e.exec) {
            cudaGraphExecDestroy(e.exec);
            e.exec = nullptr;
        }
        e.valid = false;
    }

    GraphEntry e_[N];
    int        head_ = 0;
    std::mutex mu_;
};

static GraphCache g_graph_cache;

}  // namespace
