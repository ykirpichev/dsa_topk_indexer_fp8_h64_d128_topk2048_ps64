"""
Bindings for cuBLAS-based DSA TopK Indexer.

Responsibilities of this file:
  - JIT-compile kernel.cu via torch.utils.cpp_extension.load (once per process).
  - Call the compiled topk_indexer C++ function for the compute-heavy loop:
      per-batch cuBLAS Sgemm + ReLU/weighted-sum + Thrust top-K + page transform.
"""

import os
from pathlib import Path

import torch
import torch.utils.cpp_extension

_THIS_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Lazy load / JIT-compile kernel.cu
# ---------------------------------------------------------------------------
_ext = None
_loaded_jit_name: str | None = None


def _extra_cuda_cflags() -> list:
    flags = [
        "-O3",
        "--expt-relaxed-constexpr",
        "--ftz=true",
        "--prec-div=false",
        "-Xptxas",
        "-O3",
    ]
    # FlashInfer baseline-style: fp16 scores into top-k (set in Modal / env before first load)
    if os.environ.get("FIB_TOPK_FP16", "").lower() in ("1", "true", "yes"):
        flags.append("-DFIB_TOPK_FP16")
    return flags


def _jit_name() -> str:
    base = "topk_cuda_cublas_reluwm_pfmax"
    if os.environ.get("FIB_TOPK_FP16", "").lower() in ("1", "true", "yes"):
        return base + "_fp16topk"
    return base


def _load_ext():
    global _ext, _loaded_jit_name
    jn = _jit_name()
    if _ext is not None and _loaded_jit_name == jn:
        return _ext
    _ext = torch.utils.cpp_extension.load(
        name=jn,
        sources=[str(_THIS_DIR / "kernel.cu")],
        extra_cuda_cflags=_extra_cuda_cflags(),
        extra_ldflags=[],
        verbose=False,
    )
    _loaded_jit_name = jn
    return _ext


# ---------------------------------------------------------------------------
# Kernel entry point
# ---------------------------------------------------------------------------

@torch.no_grad()
def kernel(
    q_index_fp8: torch.Tensor,        # [B, H, D]   float8_e4m3fn, CUDA
    k_index_cache_fp8: torch.Tensor,  # [P, PS, 1, HDS] int8 (uint8 layout), CUDA
    weights: torch.Tensor,            # [B, H]      float32, CUDA
    seq_lens: torch.Tensor,           # [B]         int32/int64, CUDA
    block_table: torch.Tensor,        # [B, max_pages] int32/int64, CUDA
    topk_indices: torch.Tensor,       # [B, K_topk] int32, CUDA (pre-allocated output)
) -> None:
    _load_ext().run(
        q_index_fp8,
        k_index_cache_fp8,
        weights,
        seq_lens,
        block_table,
        topk_indices,
    )
