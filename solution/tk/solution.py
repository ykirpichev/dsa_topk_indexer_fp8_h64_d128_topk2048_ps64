"""DSA FP8 top-K indexer, ThunderKittens implementation.

Stage 1 calls kittens::mm_ABt (Blackwell tcgen05, FP8 E4M3 to FP32). The KV
cache is paged, so the tiles are gathered with 16-byte loads and stored at
the swizzled shared index the MMA reads. Stage 2 is a two-round 8-bit radix
select in the same translation unit. config.toml still points at the CUDA
submission; this is a side-by-side port.

ThunderKittens is not part of the contest image. The extension looks for
headers in /opt/ThunderKittens and compiles for sm_100a.
"""

import os
from pathlib import Path

import torch

_EXT = None


def _load():
    global _EXT
    if _EXT is not None:
        return _EXT
    from torch.utils.cpp_extension import load

    here = Path(__file__).resolve().parent
    os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0a"
    _EXT = load(
        name="dsa_tk_indexer",
        sources=[str(here / "indexer.cu")],
        extra_cuda_cflags=[
            "-std=c++20",
            "-O3",
            "--use_fast_math",
            "-DKITTENS_SM100",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-I/opt/ThunderKittens/include",
            "-I/opt/ThunderKittens/prototype",
        ],
        verbose=False,
    )
    return _EXT


def kernel(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices):
    _load().run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices)
