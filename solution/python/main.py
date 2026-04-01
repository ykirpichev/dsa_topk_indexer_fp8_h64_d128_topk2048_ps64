"""
HF dataset PR #2 baseline: deep_gemm FP8 paged MQA logits + FlashInfer top-k.

See: https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest/discussions/2/files
"""

from __future__ import annotations

import torch
import deep_gemm
import flashinfer


@torch.no_grad()
def run(
    q_index_fp8: torch.Tensor,
    k_index_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
) -> tuple[torch.Tensor]:
    """
    DeepSeek sparse attention top-K indexer using deep_gemm FP8 kernel + FlashInfer.

    Pipeline: deep_gemm.fp8_paged_mqa_logits -> flashinfer.top_k_page_table_transform
    """
    batch_size, num_index_heads, index_head_dim = q_index_fp8.shape
    num_pages, page_size, _, _ = k_index_cache_fp8.shape
    topk = 2048

    assert num_index_heads == 64
    assert index_head_dim == 128
    assert page_size == 64

    device = q_index_fp8.device
    max_num_pages = block_table.shape[1]
    max_context_len = max_num_pages * page_size

    q_index_fp8_4d = q_index_fp8.unsqueeze(1)
    k_index_cache_uint8 = k_index_cache_fp8.view(torch.uint8)

    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    schedule_meta = deep_gemm.get_paged_mqa_logits_metadata(seq_lens, page_size, num_sms)

    logits = deep_gemm.fp8_paged_mqa_logits(
        q_index_fp8_4d,
        k_index_cache_uint8,
        weights,
        seq_lens,
        block_table,
        schedule_meta,
        max_context_len,
        clean_logits=False,
    )

    offsets = torch.arange(page_size, device=device, dtype=torch.int32)
    physical = block_table.unsqueeze(-1) * page_size + offsets
    physical_flat = physical.reshape(batch_size, -1)
    token_indices = torch.arange(max_num_pages * page_size, device=device)
    mask = token_indices.unsqueeze(0) < seq_lens.unsqueeze(1)
    token_page_table = torch.where(mask, physical_flat, torch.zeros_like(physical_flat))

    topk_indices = flashinfer.top_k_page_table_transform(
        input=logits.to(torch.float16),
        src_page_table=token_page_table,
        lengths=seq_lens,
        k=topk,
    )

    return (topk_indices,)


__all__ = ["run"]
