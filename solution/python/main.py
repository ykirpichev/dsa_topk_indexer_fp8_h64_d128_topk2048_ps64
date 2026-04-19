"""
DSA top-K indexer — custom CUDA FP8 paged MQA logits + FlashInfer top-k.

Pipeline:
  1. kernel.cu::fp8_paged_mqa_logits  (compiled at first call via torch.cpp_extension)
  2. flashinfer.top_k_page_table_transform

K-cache layout per token: 128 fp8_e4m3 bytes + 1 float32 scale  (132 bytes total).
"""

import os

import flashinfer
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
    batch_size    = q_index_fp8.shape[0]
    page_size     = k_index_cache_fp8.shape[1]   # 64
    topk          = 2048
    device        = q_index_fp8.device
    max_num_pages = block_table.shape[1]
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

    # Build token-level page table for FlashInfer:
    #   token_page_table[b, t] = physical token index (page_id * PS + offset)
    offsets = torch.arange(page_size, device=device, dtype=torch.int32)
    physical = block_table.unsqueeze(-1) * page_size + offsets   # [B, max_pages, PS]
    physical_flat = physical.reshape(batch_size, -1)              # [B, max_context_len]
    token_indices = torch.arange(max_context_len, device=device)
    mask = token_indices.unsqueeze(0) < seq_lens.unsqueeze(1)
    token_page_table = torch.where(mask, physical_flat, torch.zeros_like(physical_flat))

    topk_indices = flashinfer.top_k_page_table_transform(
        input=logits,
        src_page_table=token_page_table,
        lengths=seq_lens,
        k=topk,
    )
    return (topk_indices,)
