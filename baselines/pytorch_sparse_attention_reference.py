"""Contest-faithful PyTorch baseline for DSA sparse attention (MLA-style).

Definition reference from flashinfer-bench:
``dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64``, cleaned for reading.

This is the second half of DeepSeek Sparse Attention decoding: the indexer
(see ``pytorch_naive_reference.py``) picks top-K physical KV tokens; this
kernel runs full attention **only** on those tokens.

---------------------------------------------------------------------------
Notation used in comments below
---------------------------------------------------------------------------
  T   = num_tokens          # decode: batch size; prefill: total Q tokens
  H   = num_qo_heads = 16   # Q heads after TP=8 split (128/8)
  Dc  = head_dim_ckv = 512  # compressed KV / value dim
  Dp  = head_dim_kpe = 64   # key positional-encoding dim
  PS  = page_size    = 64
  K   = topk         = 2048 # max sparse KV slots per query token
  P   = num_pages
  Nkv = P * PS              # flattened KV token capacity
  V   = # valid indices for one query token  (0 <= V <= K; -1 padded)

---------------------------------------------------------------------------
Math (per query token t, head h)
---------------------------------------------------------------------------
  idx[j]       = sparse_indices[t, j]          # physical id, or -1
  logits[h,j]  = sm_scale * ( q_nope[t,h]·ckv[idx[j]]
                            + q_pe[t,h]  ·kpe[idx[j]] )     # j valid only
  attn[h,:]    = softmax_j(logits[h,:])
  out[t,h,:]   = sum_j attn[h,j] * ckv[idx[j]]              # values = CKV
  lse[t,h]     = log2( sum_j exp(logits[h,j]) )             # base-2 LSE

sm_scale is typically 1/sqrt(128+64) = 1/sqrt(192) (MLA pre-absorption QK dim).

---------------------------------------------------------------------------
Tensor dictionary
---------------------------------------------------------------------------
  q_nope          [T, H, Dc]     = [T, 16, 512]   bfloat16
  q_pe            [T, H, Dp]     = [T, 16,  64]   bfloat16
  ckv_cache       [P, PS, Dc]    = [P, 64, 512]   bfloat16
  kpe_cache       [P, PS, Dp]    = [P, 64,  64]   bfloat16
  sparse_indices  [T, K]         = [T, 2048]      int32   (page*64+slot | -1)
  sm_scale        scalar         float32
  -> output       [T, H, Dc]     = [T, 16, 512]   bfloat16
  -> lse          [T, H]         = [T, 16]        float32
"""

from __future__ import annotations

import math

import torch

NUM_QO_HEADS = 16
HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64
PAGE_SIZE = 64
TOPK = 2048


@torch.no_grad()
def run(
    q_nope: torch.Tensor,       # [T, H, Dc]
    q_pe: torch.Tensor,         # [T, H, Dp]
    ckv_cache: torch.Tensor,    # [P, PS, Dc]
    kpe_cache: torch.Tensor,    # [P, PS, Dp]
    sparse_indices: torch.Tensor,  # [T, K]
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Official-style reference: returns ``(output [T,H,Dc], lse [T,H])``."""
    # ---- unpack / check shapes ------------------------------------------------
    # q_nope: [T, H, Dc]
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    # q_pe:   [T, H, Dp]
    head_dim_kpe = q_pe.shape[-1]
    # ckv_cache: [P, PS, Dc]
    _num_pages, page_size, _ = ckv_cache.shape
    # sparse_indices: [T, K]
    topk = sparse_indices.shape[-1]

    assert num_qo_heads == NUM_QO_HEADS          # H  == 16
    assert head_dim_ckv == HEAD_DIM_CKV          # Dc == 512
    assert head_dim_kpe == HEAD_DIM_KPE          # Dp == 64
    assert page_size == PAGE_SIZE                # PS == 64
    assert topk == TOPK                          # K  == 2048
    assert sparse_indices.shape[0] == num_tokens
    assert q_pe.shape[:2] == (num_tokens, num_qo_heads)
    assert kpe_cache.shape[:2] == ckv_cache.shape[:2]

    device = q_nope.device

    # ---- flatten paged KV -> token-addressable rows ---------------------------
    # Indexer physical ids already encode  page * PS + slot, so a gather is
    # just advanced indexing into these flat tables.
    #
    # ckv_cache: [P, PS, Dc]  ->  Kc_all: [Nkv, Dc]   Nkv = P*PS
    # kpe_cache: [P, PS, Dp]  ->  Kp_all: [Nkv, Dp]
    Kc_all = ckv_cache.reshape(-1, head_dim_ckv).to(torch.float32)  # [Nkv, Dc]
    Kp_all = kpe_cache.reshape(-1, head_dim_kpe).to(torch.float32)  # [Nkv, Dp]

    # ---- allocate outputs -----------------------------------------------------
    # output: [T, H, Dc]  bf16   (zeros = "no valid KV" default)
    output = torch.zeros(
        (num_tokens, num_qo_heads, head_dim_ckv),
        dtype=torch.bfloat16,
        device=device,
    )
    # lse: [T, H]  fp32   (-inf = empty softmax normalizer)
    lse = torch.full(
        (num_tokens, num_qo_heads),
        -float("inf"),
        dtype=torch.float32,
        device=device,
    )

    log2 = math.log(2.0)  # for base-2 LSE: log2(x) = ln(x) / ln(2)

    # ---- per-query-token attention --------------------------------------------
    for t in range(num_tokens):
        # indices: [K] int32 — physical KV ids; -1 = padding
        indices = sparse_indices[t]                 # [K]
        valid = indices != -1                       # [K] bool
        tok_idx = indices[valid].to(torch.long)     # [V]  V = valid.count
        if tok_idx.numel() == 0:
            # leave output[t] = 0, lse[t] = -inf
            continue

        # Gather selected KV rows:
        #   Kc: [V, Dc]   compressed keys (= values for the output matmul)
        #   Kp: [V, Dp]   positional keys (score only; not used as values)
        Kc = Kc_all[tok_idx]                        # [V, Dc]
        Kp = Kp_all[tok_idx]                        # [V, Dp]

        # Queries for this token (promote to fp32 for the matmuls):
        #   qn: [H, Dc]   content / "nope" query
        #   qp: [H, Dp]   positional query
        qn = q_nope[t].to(torch.float32)            # [H, Dc]
        qp = q_pe[t].to(torch.float32)              # [H, Dp]

        # ---- scores -----------------------------------------------------------
        # Content term:   qn @ Kc.T  ->  [H, Dc] @ [Dc, V]  = [H, V]
        # Positional:     qp @ Kp.T  ->  [H, Dp] @ [Dp, V]  = [H, V]
        # MLA logits = sm_scale * (content + positional)
        logits_ckv = qn @ Kc.T                      # [H, V]
        logits_kpe = qp @ Kp.T                      # [H, V]
        logits = (logits_ckv + logits_kpe) * sm_scale  # [H, V]

        # ---- LSE (base 2) -----------------------------------------------------
        # logsumexp over the V sparse slots, per head:
        #   lse[t]: [H] = log2(sum_j exp(logits[h,j]))
        #         = ln(sum_j exp(...)) / ln(2)
        lse[t] = torch.logsumexp(logits, dim=-1) / log2  # [H]

        # ---- softmax + value accumulate ---------------------------------------
        # attn: [H, V]  — probability over the V selected KV tokens
        attn = torch.softmax(logits, dim=-1)        # [H, V]

        # Values are CKV only (KPE is score-side):
        #   attn @ Kc  ->  [H, V] @ [V, Dc]  = [H, Dc]
        out_t = attn @ Kc                           # [H, Dc]
        output[t] = out_t.to(torch.bfloat16)        # store [H, Dc] as bf16

    # output: [T, H, Dc]  bf16
    # lse:    [T, H]      fp32
    return output, lse


@torch.no_grad()
def kernel(
    q_nope: torch.Tensor,          # [T, H, Dc]
    q_pe: torch.Tensor,            # [T, H, Dp]
    ckv_cache: torch.Tensor,       # [P, PS, Dc]
    kpe_cache: torch.Tensor,       # [P, PS, Dp]
    sparse_indices: torch.Tensor,  # [T, K]
    sm_scale: float,
    output: torch.Tensor,          # [T, H, Dc]  (destination)
    lse: torch.Tensor,             # [T, H]      (destination)
) -> None:
    """Destination-passing wrapper (contest solutions are often DPS)."""
    out, lse_out = run(
        q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale
    )
    output.copy_(out)   # [T, H, Dc]
    lse.copy_(lse_out)  # [T, H]


def _demo_cpu() -> None:
    """Tiny synthetic example — no GPU required."""
    torch.manual_seed(0)
    T, P = 2, 4
    # MLA pre-absorption scale: 1/sqrt(d_qk + d_kpe) with d_qk=128 before absorb
    sm_scale = 1.0 / math.sqrt(128 + 64)

    q_nope = torch.randn(T, NUM_QO_HEADS, HEAD_DIM_CKV, dtype=torch.bfloat16)  # [2,16,512]
    q_pe = torch.randn(T, NUM_QO_HEADS, HEAD_DIM_KPE, dtype=torch.bfloat16)    # [2,16,64]
    ckv = torch.randn(P, PAGE_SIZE, HEAD_DIM_CKV, dtype=torch.bfloat16)        # [4,64,512]
    kpe = torch.randn(P, PAGE_SIZE, HEAD_DIM_KPE, dtype=torch.bfloat16)        # [4,64,64]

    # sparse_indices: [T, K] physical ids (page*64 + slot); pad with -1.
    # Token 0 attends to 5 KV slots; token 1 to 3.
    sparse = torch.full((T, TOPK), -1, dtype=torch.int32)  # [2, 2048]
    sparse[0, :5] = torch.tensor([0, 1, 2, 64, 65], dtype=torch.int32)
    sparse[1, :3] = torch.tensor([10, 11, 12], dtype=torch.int32)

    out, lse = run(q_nope, q_pe, ckv, kpe, sparse, sm_scale)
    # out: [2, 16, 512] bf16
    # lse: [2, 16]      fp32

    print("output shape:", tuple(out.shape), out.dtype)
    print("lse shape:   ", tuple(lse.shape), lse.dtype)
    print("lse[0,:4]:   ", [round(float(x), 4) for x in lse[0, :4]])
    print("lse[1,:4]:   ", [round(float(x), 4) for x in lse[1, :4]])
    assert out.shape == (T, NUM_QO_HEADS, HEAD_DIM_CKV)
    assert torch.isfinite(lse).all()
    print("demo OK")


if __name__ == "__main__":
    _demo_cpu()
