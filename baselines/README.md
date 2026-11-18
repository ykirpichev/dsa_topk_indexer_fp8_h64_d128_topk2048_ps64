# Baselines

Contest **definition references** from flashinfer-bench, cleaned for reading.
These are the naive Python paths the harness times against (not FlashInfer /
DeepGEMM).

## DSA pipeline (how the two kernels fit)

```
Q, K_fp8, weights, block_table
        │
        ▼
┌─────────────────────────┐
│  Top-K Indexer          │  baselines/pytorch_naive_reference.py
│  score + topk=2048      │  → sparse_indices [T, 2048]  (phys ids / -1)
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Sparse Attention       │  baselines/pytorch_sparse_attention_reference.py
│  MLA on selected KV     │  → output [T, 16, 512], lse [T, 16]
└─────────────────────────┘
```

## Files

| file | definition |
|---|---|
| `pytorch_naive_reference.py` | `dsa_topk_indexer_fp8_h64_d128_topk2048_ps64` |
| `pytorch_sparse_attention_reference.py` | `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64` |

```bash
# CPU smoke / math refresh (no GPU needed)
python baselines/pytorch_naive_reference.py
python baselines/pytorch_sparse_attention_reference.py
```

Each module exposes:

- `run(...)` — official reference return style
- `kernel(...)` — destination-passing wrapper
- `_demo_cpu()` — tiny synthetic self-check under `__main__`
