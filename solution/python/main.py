"""
DSA top-K indexer — custom CUDA FP8 paged MQA logits + custom CUDA top-K.

Pipeline (all CUDA, no FlashInfer at runtime):
  1. kernel.cu::fp8_paged_mqa_logits         — FP8 paged MQA logits
  2. kernel.cu::topk_page_table_transform    — CUB segmented radix top-K

K-cache layout per page (deep_gemm format):
  [page_size * 128 FP8 bytes] [page_size * 4 scale bytes] = 8448 bytes / page.
"""

import os

import torch
import torch.utils.cpp_extension as _ext

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULE = None


def _load_module():
    global _MODULE
    if _MODULE is None:
        _MODULE = _ext.load(
            name="cuda_fp8_mqa_logits",
            sources=[os.path.join(_SRC_DIR, "kernel.cu")],
            extra_cuda_cflags=["-O3", "-arch=sm_100"],
            verbose=False,
        )
    return _MODULE


@torch.no_grad()
def run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table):
    """
    Args:
        q_index_fp8        : [B, H=64, D=128]           float8_e4m3fn
        k_index_cache_fp8  : [num_pages, PS=64, 1, 132] float8_e4m3fn
        weights            : [B, H=64]                   float32
        seq_lens           : [B]                         int32
        block_table        : [B, max_num_pages]          int32
    Returns:
        (topk_indices,)    : [B, 2048]                   int32
    """
    batch_size      = q_index_fp8.shape[0]
    page_size       = k_index_cache_fp8.shape[1]   # 64
    topk            = 2048
    device          = q_index_fp8.device
    max_num_pages   = block_table.shape[1]
    max_context_len = max_num_pages * page_size

    mod = _load_module()

    logits = torch.empty(batch_size, max_context_len, dtype=torch.float32, device=device)

    mod.fp8_paged_mqa_logits(
        q_index_fp8,
        k_index_cache_fp8.view(torch.uint8),
        weights,
        seq_lens,
        block_table,
        logits,
        max_context_len,
        max_num_pages,
    )

    topk_indices = mod.topk_page_table_transform(
        logits,
        block_table,
        seq_lens,
        topk,
    )
    return (topk_indices,)
