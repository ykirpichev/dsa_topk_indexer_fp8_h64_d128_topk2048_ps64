from pathlib import Path
import torch
import torch.utils.cpp_extension

_THIS_DIR = Path(__file__).parent
_ext = None


def _load_ext():
    global _ext
    if _ext is None:
        _ext = torch.utils.cpp_extension.load(
            name="dsa_topk_indexer",
            sources=[str(_THIS_DIR / "kernel.cu")],
            extra_cuda_cflags=[
                "-O3", "--expt-relaxed-constexpr",
                "-gencode", "arch=compute_100a,code=sm_100a",
            ],
            extra_ldflags=[],
            verbose=True,
        )
    return _ext


def kernel(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices):
    _load_ext().run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices)
