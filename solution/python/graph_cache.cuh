// Shape-keyed CUDA graph cache: graph replay instead of per-call launches.

#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <mutex>

#include "dsa_config.cuh"

namespace {

// flashinfer-bench clones every input tensor per iteration, so pointers change
// on every call while shapes -- and therefore the whole dispatch plan -- stay
// fixed. Hence the key is shapes only (caller stream, plan, grid-defining
// scalars) and never pointers: a hit re-captures a template graph and splices
// the new kernel params into the cached exec with cudaGraphExecUpdate (~1-2 us,
// against hundreds for a re-instantiate). Measured worth ~3% overall and ~5%
// on the fast path, where a launch is a large share of a 2 us kernel.
//
// A miss instantiates and stores the exec in a 32-slot LRU; the evaluator's
// ~128 workloads each map to one (plan, scalars) tuple, so it never churns.

struct GraphKey {
    cudaStream_t stream;        // caller stream for which exec was instantiated
    Plan         plan;          // which Stage-1 plan was captured
    uint8_t      _pad0 = 0;
    uint8_t      _pad1 = 0;
    uint8_t      _pad2 = 0;
    int32_t      B;
    int32_t      max_num_pages;
    int32_t      max_kv_arg;    // short: max_num_pages; persistent: max_kv_tile_pairs
    int32_t      tiles_per_cta; // persistent only
    int32_t      num_splits;    // persistent only (grid.x)
    int32_t      max_len;       // Stage-2 stride (bytes-per-row)
    int32_t      _pad3 = 0;

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
        for (auto& e : e_) {
            if (e.exec) cudaGraphExecDestroy(e.exec);
        }
    }

    GraphEntry* find_slot(const GraphKey& k) {
        std::lock_guard<std::mutex> lk(mu_);
        for (auto& e : e_) {
            if (e.valid && e.key == k) return &e;
        }
        return nullptr;
    }

    void insert(const GraphKey& k, cudaGraphExec_t x) {
        std::lock_guard<std::mutex> lk(mu_);
        auto& slot = e_[head_];
        if (slot.valid && slot.exec) cudaGraphExecDestroy(slot.exec);
        slot.key   = k;
        slot.exec  = x;
        slot.valid = true;
        head_ = (head_ + 1) % N;
    }

    void invalidate(GraphEntry* slot) {
        std::lock_guard<std::mutex> lk(mu_);
        slot->valid = false;
        slot->exec  = nullptr;
    }

private:
    GraphEntry e_[N];
    int        head_ = 0;
    std::mutex mu_;
};

static GraphCache g_graph_cache;

}  // namespace
