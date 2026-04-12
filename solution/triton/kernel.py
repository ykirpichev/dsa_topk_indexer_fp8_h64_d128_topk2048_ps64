"""
DSA Top-K indexer — pure PyTorch, batched paged layout + masks (deep_gemm-style).

Flattens block_table to length max_num_pages * page_size, gathers K and global token ids,
masks invalid positions past seq_lens, then batched bmm for per-head scores.
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
    _num_pages, page_size, _, _ = k_index_cache_fp8.shape
    topk = 2048

    assert num_index_heads == 64
    assert index_head_dim == 128
    assert page_size == 64

    device = q_index_fp8.device
    bt = block_table.to(torch.long)

    q = q_index_fp8.to(torch.float32)
    k_all = _dequant_fp8_kv_cache(k_index_cache_fp8)

    _, max_num_pages = bt.shape
    t_flat = max_num_pages * page_size
    j = torch.arange(t_flat, device=device, dtype=torch.long)
    slot = j // page_size
    off = j % page_size

    phys_page = bt[:, slot]
    k_flat = k_all[phys_page, off]
    global_token = phys_page * page_size + off

    valid = j.unsqueeze(0) < seq_lens.to(device).unsqueeze(1)

    bh = batch_size * num_index_heads
    k_bh = k_flat.unsqueeze(1).expand(-1, num_index_heads, -1, -1).reshape(bh, t_flat, index_head_dim)
    scores = torch.bmm(
        q.reshape(bh, 1, index_head_dim),
        k_bh.transpose(1, 2),
    ).view(batch_size, num_index_heads, t_flat)

    scores_relu = torch.relu(scores)
    final_scores = (scores_relu * weights.unsqueeze(-1)).sum(dim=1)
    final_scores = final_scores.masked_fill(~valid, float("-inf"))

    k_take = min(topk, t_flat)
    values, topk_flat = torch.topk(final_scores, k=k_take, dim=-1)

    topk_global = global_token.gather(1, topk_flat)
    good = torch.isfinite(values) & valid.gather(1, topk_flat)

    topk_indices = torch.full((batch_size, topk), -1, dtype=torch.int32, device=device)
    topk_indices[:, :k_take] = torch.where(good, topk_global, torch.tensor(-1, device=device, dtype=torch.long)).to(
        torch.int32
    )

    empty = seq_lens == 0
    if empty.any():
        topk_indices[empty] = -1

    return (topk_indices,)
