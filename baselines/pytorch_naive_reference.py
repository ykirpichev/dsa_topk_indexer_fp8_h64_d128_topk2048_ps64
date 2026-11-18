"""Contest-faithful PyTorch baseline for the DSA FP8 Top-K indexer.

Definition reference from flashinfer-bench
(``dsa_topk_indexer_fp8_h64_d128_topk2048_ps64``), cleaned for reading.
Harness speedups are vs this naive Python path (not FlashInfer/DeepGEMM).

---------------------------------------------------------------------------
Notation used in comments below
---------------------------------------------------------------------------
  B   = batch_size
  H   = num_index_heads = 64
  D   = index_head_dim  = 128
  PS  = page_size       = 64
  K   = topk            = 2048
  P   = num_pages            # total pages in the KV pool
  M   = max_num_pages        # block_table width (pages reserved per row)
  S   = seq_lens[b]          # valid logical tokens in row b
  Np  = ceil(S / PS)         # pages touched by row b

---------------------------------------------------------------------------
Math (per batch row b, logical token t < S)
---------------------------------------------------------------------------
  page, slot   = block_table[b, t // PS],  t % PS
  S_score[h]   = sum_d Q[b,h,d] * K_dequant[page, slot, d]     # FP8→FP32
  logit[t]     = scale[page,slot] * sum_h ReLU(S_score[h]) * weights[b,h]
  topk_local   = argmax_K(logit[0:S])                           # logical ids
  topk_phys[j] = block_table[b, j//PS] * PS + (j % PS)          # physical ids
               # unused slots padded with -1

---------------------------------------------------------------------------
KV cache layout (deep_gemm FP8, one page = 8448 bytes)
---------------------------------------------------------------------------
  bytes [0, 8192)     : PS * D FP8 values     (64 tokens × 128 dims)
  bytes [8192, 8448)  : PS FP32 scales        (one scale per token)
  viewed as           : [P, PS, 1, 132] int8

---------------------------------------------------------------------------
Tensor dictionary
---------------------------------------------------------------------------
  q_index_fp8       [B, H, D]        = [B, 64, 128]        float8_e4m3fn
  k_index_cache_fp8 [P, PS, 1, 132]  = [P, 64, 1, 132]     int8 (as uint8/FP8)
  weights           [B, H]           = [B, 64]             float32
  seq_lens          [B]                                    int32
  block_table       [B, M]                                 int32
  -> topk_indices   [B, K]           = [B, 2048]           int32
"""

from __future__ import annotations

import torch

H = 64
D = 128
PAGE_SIZE = 64
TOPK = 2048
HEAD_DIM_WITH_SCALE = 132  # D + 4 scale bytes


def dequant_fp8_kv_cache(k_index_cache_fp8: torch.Tensor) -> torch.Tensor:
    """Dequantize deep_gemm FP8 pages to FP32.

    Input:  k_index_cache_fp8  [P, PS, 1, 132] int8  (byte-identical to uint8)
    Output: K_dequant          [P, PS, D]      float32
            (= fp8_value * per-token scale, scale broadcast over D)
    """
    # cache_u8: [P, PS, 1, 132] uint8
    cache_u8 = k_index_cache_fp8.view(torch.uint8)
    num_pages, page_size, _num_heads, head_dim_sf = cache_u8.shape
    assert head_dim_sf == HEAD_DIM_WITH_SCALE
    head_dim = head_dim_sf - 4  # 128

    # Per page the bytes are packed flat: [all FP8 | all scales], NOT
    # interleaved as [fp8||scale] per token. Flatten, then split.
    # kv_flat: [P, PS*132] = [P, 8448] uint8
    kv_flat = cache_u8.view(num_pages, page_size * head_dim_sf)

    # FP8 payload: first PS*D bytes per page
    # fp8_bytes: [P, PS*D] = [P, 8192] uint8
    # fp8:       [P, PS, D] float8_e4m3fn
    # fp8_f32:   [P, PS, D] float32
    fp8_bytes = kv_flat[:, : page_size * head_dim].contiguous()
    fp8 = fp8_bytes.view(num_pages, page_size, head_dim).view(torch.float8_e4m3fn)
    fp8_f32 = fp8.to(torch.float32)

    # Scales: last PS*4 bytes per page -> PS float32s
    # scale_bytes: [P, PS*4] = [P, 256] uint8
    # scale:       [P, PS, 1] float32
    scale_bytes = kv_flat[:, page_size * head_dim :].contiguous()
    scale = scale_bytes.view(torch.float32).view(num_pages, page_size, 1)

    # Broadcast multiply: [P, PS, D] * [P, PS, 1] -> [P, PS, D]
    return fp8_f32 * scale


@torch.no_grad()
def run(
    q_index_fp8: torch.Tensor,       # [B, H, D]
    k_index_cache_fp8: torch.Tensor, # [P, PS, 1, 132]
    weights: torch.Tensor,           # [B, H]
    seq_lens: torch.Tensor,          # [B]
    block_table: torch.Tensor,       # [B, M]
) -> tuple[torch.Tensor]:
    """Official-style reference: returns ``(topk_indices [B, K],)``."""
    # q_index_fp8: [B, H, D]
    batch_size, num_heads, head_dim = q_index_fp8.shape
    # k_index_cache_fp8: [P, PS, 1, 132]
    _num_pages, page_size, _, _ = k_index_cache_fp8.shape
    assert num_heads == H and head_dim == D and page_size == PAGE_SIZE

    device = q_index_fp8.device

    # q:     [B, H, D]  fp32
    # K_all: [P, PS, D] fp32   (dequantized pool; gathered per row below)
    q = q_index_fp8.to(torch.float32)
    K_all = dequant_fp8_kv_cache(k_index_cache_fp8)

    # topk_indices: [B, K] int32, pre-filled with -1 padding
    topk_indices = torch.full(
        (batch_size, TOPK), -1, dtype=torch.int32, device=device
    )

    for b in range(batch_size):
        # S = number of valid logical tokens in this row
        seq_len = int(seq_lens[b].item())
        if seq_len == 0:
            continue

        # Logical tokens [0, S) live in the first Np = ceil(S / PS) pages.
        # page_indices: [Np]  — physical page ids from the block table
        num_pages_for_seq = (seq_len + page_size - 1) // page_size
        page_indices = block_table[b, :num_pages_for_seq].to(torch.long)  # [Np]

        # Gather those pages, flatten tokens, trim to S:
        #   K_all[page_indices]: [Np, PS, D]
        #   .reshape(-1, D):     [Np*PS, D]
        #   [:seq_len]:          [S, D]
        K = K_all[page_indices].reshape(-1, head_dim)[:seq_len]  # [S, D]

        # ---- score ------------------------------------------------------------
        # q[b]:      [H, D]
        # K.T:       [D, S]
        # q[b] @ K.T -> [H, S]   raw head scores
        # ReLU + per-head weight broadcast:
        #   weights[b]:          [H]
        #   weights[b][:, None]: [H, 1]
        #   scores:              [H, S]
        scores = torch.relu(q[b] @ K.T) * weights[b][:, None]  # [H, S]

        # Sum across heads -> one logit per logical token
        # logits: [S]
        logits = scores.sum(dim=0)

        # ---- top-K over logical tokens ----------------------------------------
        # topk_local: [actual_topk]  logical ids in [0, S)
        actual_topk = min(TOPK, seq_len)
        _, topk_local = torch.topk(logits, actual_topk)  # values unused

        # ---- logical -> physical ----------------------------------------------
        # page_col: [actual_topk]  which block-table column (logical page)
        # slot:     [actual_topk]  offset within the page
        # physical: [actual_topk]  page_id * PS + slot
        page_col = topk_local // page_size
        slot = topk_local % page_size
        physical = page_indices[page_col] * page_size + slot

        # Write into the output row; remaining K - actual_topk entries stay -1
        topk_indices[b, :actual_topk] = physical.to(torch.int32)

    # topk_indices: [B, K] int32
    return (topk_indices,)


@torch.no_grad()
def kernel(
    q_index_fp8: torch.Tensor,       # [B, H, D]
    k_index_cache_fp8: torch.Tensor, # [P, PS, 1, 132]
    weights: torch.Tensor,           # [B, H]
    seq_lens: torch.Tensor,          # [B]
    block_table: torch.Tensor,       # [B, M]
    topk_indices: torch.Tensor,      # [B, K] destination
) -> None:
    """Destination-passing wrapper matching ``solution/python/solution.py``."""
    out, = run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table)
    topk_indices.copy_(out)  # [B, K]


def _demo_cpu() -> None:
    """Tiny synthetic example so you can refresh the math without a GPU."""
    torch.manual_seed(0)
    B, P, max_pages = 2, 8, 4
    # seq_lens: [B] — row0 has 100 tokens, row1 has 50
    seq_lens = torch.tensor([100, 50], dtype=torch.int32)
    # block_table: [B, M]
    block_table = torch.arange(B * max_pages, dtype=torch.int32).view(B, max_pages) % P

    # q: [B, H, D] float8
    q = torch.randn(B, H, D).to(torch.float8_e4m3fn)

    # Build a deep_gemm-style cache: per page, FP8 bytes then FP32 scales.
    # Resulting cache: [P, PS, 1, 132] int8
    pages = []
    for _ in range(P):
        fp8 = torch.randn(PAGE_SIZE, D).to(torch.float8_e4m3fn).view(torch.uint8)  # [PS, D]
        scales = torch.ones(PAGE_SIZE, dtype=torch.float32)  # [PS]
        page = torch.empty(PAGE_SIZE * HEAD_DIM_WITH_SCALE, dtype=torch.uint8)  # [8448]
        page[: PAGE_SIZE * D] = fp8.reshape(-1)                 # [8192]
        page[PAGE_SIZE * D :] = scales.view(torch.uint8).reshape(-1)  # [256]
        pages.append(page.view(PAGE_SIZE, 1, HEAD_DIM_WITH_SCALE))  # [PS, 1, 132]
    cache = torch.stack(pages, dim=0).view(torch.int8)  # [P, PS, 1, 132]

    # weights: [B, H]
    weights = torch.ones(B, H, dtype=torch.float32)
    out, = run(q, cache, weights, seq_lens, block_table)  # [B, K]

    print("topk_indices shape:", tuple(out.shape))
    print("row0 (first 8):", out[0, :8].tolist())
    print("row0 padding (-1 count):", int((out[0] < 0).sum()))
    print("row1 padding (-1 count):", int((out[1] < 0).sum()))
    # seq_len=50 < 2048 => 2048 - 50 = 1998 pads on row1
    assert int((out[1] < 0).sum()) == TOPK - 50
    print("demo OK")


if __name__ == "__main__":
    _demo_cpu()
