# Optimization learnings (DSA TopK indexer CUDA)

**Policy:** Land **git commits / submission tags only for measured improvements** (Modal smoke or full 128 as appropriate). Experiments that do not help stay documented here only (no extra submission tags).

## Log

| Date (UTC) | Change | Result | Notes |
|------------|--------|--------|--------|
| 2026-03-31 | Reuse single `topk_vals` buffer `[K_topk]` across batch `topk_out` | **Reverted** | No clear win vs prior; noisy Modal timings. Prefer fresh alloc per row or revisit with profiler. |
| 2026-03-31 | `sum_out` into reused `scores_buf[max_seq_len]` instead of `sum(0)` temp | **Not landed** | Smoke geomean ~6.94× vs ~7.25× baseline on same Modal run — no improvement. |
| 2026-03-31 | `bmm_out` into pre-allocated `logits` + `K^T` `.contiguous()` | **Not landed** | Smoke geomean ~6.67× vs ~7.25× — extra transpose contiguity cost; no win. |
| 2026-03-30 | Fused `relu*weight` inside CUDA with `fmaxf` / ternary | **Reverted** | Breaks bitwise top-k vs `aten::relu` + `mul` on many workloads. |
| 2026-03-30 | On-the-fly `q·K` dot (no `K_batched`, no `bmm`) | **Reverted** | Parallel tree reduction ≠ `bmm` numerics; serial `fmaf` matched but much slower. |
| 2026-03-30 | `_nested_from_padded_tensor` + `sum` for scores | **Aborted** | `RUNTIME_ERROR` on Modal; API/layout mismatch for `[B,H,S]`. |
| 2026-03-29 | In-place `logits.relu_(); logits.mul_(w_bcast)` vs `relu()*w` | **Kept** | Drops one `[B,H,S]` temp; 128/128 + smoke pass. Tagged `submission-v3`. |

## Ideas not yet tried (or worth revisiting)

- Batched / fused top-k with exact ATen tie semantics.
- `seq_lens` on device only (avoid `.cpu()` each call) if benchmark allows.
- cuBLASLt for batched `(H×D)·(D×S)` if numerics can be proven to match `bmm`.
