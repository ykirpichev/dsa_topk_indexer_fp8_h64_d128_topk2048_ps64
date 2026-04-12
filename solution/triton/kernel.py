"""
DSA Top-K indexer — pure PyTorch reference path using batched GEMM (torch.bmm).

Scores per head: bmm(q.unsqueeze(1), K_batched.transpose(1, 2)) in place of q @ K.T,
then ReLU, learned head weights, sum, top-k. Matches the contest reference semantics.
"""

import torch


def _dequant_fp8_kv_cache(k_index_cache_fp8: torch.Tensor) -> torch.Tensor:
    """Dequantize FP8 KV cache from deep_gemm layout to float32. See dataset definition."""
    k_uint8 = k_index_cache_fp8.view(torch.uint8)
    num_pages, page_size, _num_heads, head_dim_sf = k_uint8.shape
    head_dim = head_dim_sf - 4

    kv_flat = k_uint8.view(num_pages, page_size * head_dim_sf)
    fp8_bytes = kv_flat[:, : page_size * head_dim].contiguous()
    fp8_tensor = fp8_bytes.view(num_pages, page_size, head_dim).view(torch.float8_e4m3fn)
    fp8_float = fp8_tensor.to(torch.float32)

    scale_bytes = kv_flat[:, page_size * head_dim :].contiguous()
    scale = scale_bytes.view(num_pages, page_size, 4).view(torch.float32)

    return fp8_float * scale


@torch.no_grad()
def kernel(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table):
    batch_size, num_index_heads, index_head_dim = q_index_fp8.shape
    num_pages, page_size, _, _ = k_index_cache_fp8.shape
    topk = 2048

    assert num_index_heads == 64
    assert index_head_dim == 128
    assert page_size == 64

    device = q_index_fp8.device

    q = q_index_fp8.to(torch.float32)
    k_all = _dequant_fp8_kv_cache(k_index_cache_fp8)

    topk_indices = torch.full((batch_size, topk), -1, dtype=torch.int32, device=device)

    for b in range(batch_size):
        seq_len = int(seq_lens[b].item())
        if seq_len == 0:
            continue

        num_pages_for_seq = (seq_len + page_size - 1) // page_size
        page_indices = block_table[b, :num_pages_for_seq].to(torch.long)

        k_paged = k_all[page_indices]
        k = k_paged.reshape(-1, index_head_dim)[:seq_len]

        q_b = q[b]
        # [H, 1, D] bmm [H, D, T] -> [H, 1, T]; same math as q_b @ k.T but uses bmm
        k_b = k.unsqueeze(0).expand(num_index_heads, -1, -1)
        scores = torch.bmm(q_b.unsqueeze(1), k_b.transpose(1, 2)).squeeze(1)

        scores_relu = torch.relu(scores)
        w = weights[b]
        final_scores = (scores_relu * w[:, None]).sum(dim=0)

        actual_topk = min(topk, seq_len)
        _, topk_idx = torch.topk(final_scores, actual_topk)

        page_idx_per_token = topk_idx // page_size
        offset_per_token = topk_idx % page_size
        global_page_idx = page_indices[page_idx_per_token]
        topk_tokens = global_page_idx * page_size + offset_per_token

        topk_indices[b, :actual_topk] = topk_tokens.to(torch.int32)

    return (topk_indices,)
