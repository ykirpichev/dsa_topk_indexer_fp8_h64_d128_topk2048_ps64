# Performance Notes — CUDA top-k page table transform

Scratch pad for follow-up work on `solution/python/kernel.cu::topk_page_table_transform`.
Resume when ready.

## Current state (commit e1fc8f0, branch `cuda_flash_infer_style_logits_and_fi_topk`)

Fully custom CUDA pipeline (no `flashinfer` at runtime):

1. `fp8_paged_mqa_logits_kernel`    — FP8 paged MQA logits.
2. `topk_prepare_kernel`            — build `(key=logit, value=physical_token_id)` pairs, `(-inf, -1)` for `t >= seq_lens[b]`.
3. `cub::DeviceSegmentedRadixSort::SortPairsDescending` — per-batch full sort.
4. `topk_gather_kernel`             — copy first `k` values, pad with `-1` when `k > M`.

All 128 workloads PASS with `abs_err=0`, `rel_err=0` on Modal B200.

## Measured performance (128 workloads, Modal B200)

| Version | min | max | mean | mean latency |
|---|---|---|---|---|
| FlashInfer top-k       | 8.01x  | 168.26x | 17.81x | 0.220 ms |
| Our CUB sort top-k     | 6.55x  |  68.60x | 12.05x | 0.262 ms |

Head-to-head `CUB / FI` latency ratio: min 1.07x, max 2.23x, mean 1.30x.
Worst cases (2.2x) are tiny workloads (30 us → 67 us), dominated by fixed per-call costs.
Best cases (~1.08x) are large workloads where the logit kernel dominates.

## Why we lag FlashInfer

1. **Full sort vs. radix select** — CUB sorts every element (`O(N · passes)`). FlashInfer runs a dedicated radix top-k that finds the `k`-th pivot and only partitions around it (`O(N)` with small constant).
2. **Per-call scratch allocation** — we allocate `keys_in/out`, `values_in/out`, offsets, and CUB `temp_storage` on every call. Dominates the 30 us baseline.
3. **Segmented-sort overhead for small segments** — short `max_context_len` rows don't amortize CUB's per-segment setup. FlashInfer keeps the work inside a single CTA per row.
4. Logit kernel itself is ~matched; the delta is almost entirely in the top-k path.

## Ideas to close the gap (in order of expected ROI)

### [A] Cache scratch buffers on the module
Move the five `torch::empty(...)` allocations (`keys_in/out`, `values_in/out`, `offsets`, `temp_storage`) into a module-level state object, grown on demand to the largest `(B, max_context_len)` seen.
- Expected impact: removes ~30 us fixed overhead → tiny workloads drop from 67 us to ~35 us (parity with FI), mean ratio moves from 1.30x → ~1.10x.
- Low risk, small code change.

### [B] Single-pass block-per-row radix select (the real fix)
Replace the CUB sort with a custom kernel: one CTA per `(b)`:
  1. Cooperatively histogram the top 11 bits of the float key (with proper sign-flip encoding so descending order works on the `uint32` representation).
  2. Prefix-sum the histogram to find the bucket containing the `k`-th largest.
  3. Mask the kept prefix, recurse on the pivot bucket for the remaining `k' = k - kept` elements (2–3 iterations is usually enough for 32-bit floats).
  4. Write the top-k `(physical_token_id)` values directly to `out[b]`, padding with `-1`.
- This is what FlashInfer's `radix_topk_page_table_transform` does.
- Expected impact: matches or beats FlashInfer on large rows; big win on mid-size rows.
- Medium effort. Reuse existing `topk_prepare_kernel` as a fused input preparation step (or inline it).

### [C] Fuse prepare → select → gather
Never materialize `[B, max_context_len]` scratch. Read `logits` / `block_table` / `seq_lens` directly inside the radix-select kernel and write `output[B, k]` in one pass.
- Saves global-memory bandwidth (no 2x roundtrip through scratch).
- Natural follow-up to [B].

### [D] (Nice-to-have) Eliminate the `offsets` tensor
For the current CUB path, we build `offsets = arange(0, (B+1)*M, M)` on every call. Could be a single one-element fill or a cached tensor. Minor; subsumed by [A] or by dropping CUB entirely in [B].

## Suggested next-session order

1. Apply [A] first (quick, unblocks clean measurements of the top-k cost).
2. Remeasure: if small-workload ratio is already near 1.0 and large-workload ratio is ~1.05x, decide whether [B] is worth pursuing for mean-speedup wins.
3. Implement [B] + [C] together if [A] isn't enough.

## Useful commands

```bash
# Smoke test:
modal run scripts/run_modal.py --smoke

# 20 workloads:
modal run scripts/run_modal.py --n-workloads 20

# Full 128 workloads:
modal run scripts/run_modal.py

# Compare per-workload latency across two runs (see chat; parse PASSED lines):
#   ms/speedup pattern: "0.262 ms | 12.05x speedup"
```
