import functools
from pathlib import Path

_THIS_DIR = Path(__file__).parent


@functools.lru_cache(maxsize=None)
def _get_kernel():
    import tvm_ffi
    import tvm_ffi.cpp as cpp_ext

    cuda_src = str(_THIS_DIR / "kernel.cu")
    print(f"[dsa] Compiling {cuda_src}", flush=True)

    mod = cpp_ext.load(
        name="dsa_topk_indexer",
        cuda_files=[cuda_src],
        extra_cuda_cflags=[
            "-gencode", "arch=compute_100a,code=sm_100a",
            "-O3",
        ],
        extra_include_paths=[str(_THIS_DIR)],
        build_directory="/tmp/dsa_kernel_build",
    )
    print("[dsa] Compiled OK", flush=True)

    fn = tvm_ffi.get_global_func("kernel")
    print(f"[dsa] Loaded kernel: {fn}", flush=True)
    return fn


def kernel(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices):
    _get_kernel()(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices)
