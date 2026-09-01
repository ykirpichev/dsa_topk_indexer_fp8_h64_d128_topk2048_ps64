// Shared CUDA host helpers: RAII handles, error checks, capture stream.

#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>

#include <utility>

#define DSA_CUDA_CHECK(expr)                                                  \
    do {                                                                      \
        const cudaError_t _dsa_err = (expr);                                  \
        TORCH_CHECK(_dsa_err == cudaSuccess,                                  \
                    "DSA topk: ", #expr, ": ", cudaGetErrorString(_dsa_err)); \
    } while (0)

namespace {

// Move-only owner for opaque CUDA graph handles. Destroy is ignored for null.
template <class Handle, cudaError_t (*Destroy)(Handle)>
class UniqueCudaHandle {
public:
    UniqueCudaHandle() = default;
    explicit UniqueCudaHandle(Handle h) noexcept : h_(h) {}
    ~UniqueCudaHandle() { reset(); }

    UniqueCudaHandle(UniqueCudaHandle&& o) noexcept : h_(o.release()) {}
    UniqueCudaHandle& operator=(UniqueCudaHandle&& o) noexcept {
        if (this != &o) {
            reset();
            h_ = o.release();
        }
        return *this;
    }

    UniqueCudaHandle(const UniqueCudaHandle&)            = delete;
    UniqueCudaHandle& operator=(const UniqueCudaHandle&) = delete;

    Handle get() const noexcept { return h_; }
    Handle release() noexcept { return std::exchange(h_, Handle{}); }
    explicit operator bool() const noexcept { return h_ != Handle{}; }

    void reset(Handle h = Handle{}) noexcept {
        if (h_) Destroy(h_);
        h_ = h;
    }

private:
    Handle h_{};
};

using UniqueCudaGraph     = UniqueCudaHandle<cudaGraph_t, cudaGraphDestroy>;
using UniqueCudaGraphExec = UniqueCudaHandle<cudaGraphExec_t, cudaGraphExecDestroy>;

// Capture cannot use the caller's stream: under flashinfer-bench's isolated-
// runner worker getCurrentCUDAStream() returns the legacy NULL stream, where
// BeginCapture fails with cudaErrorIllegalState.
inline cudaStream_t capture_stream() {
    static thread_local cudaStream_t s = nullptr;
    if (!s) cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    return s;
}

}  // namespace
