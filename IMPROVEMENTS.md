# Kernel Performance Improvements

Current result: ~2.2× vs FlashInfer (full benchmark).
Target: ≥ 3–4× (matches "Agent-Assisted" baseline of ~11.3× DSA overall).

---

## Root-Cause Analysis

### Stage 1 — `paged_mqa_logits_umma_kernel`

**Problem: Q is reloaded once per page block.**
Grid is `(max_kv_tiles, B)` — for `num_pages=11923` and `batch=1` that is 11 923 blocks.
Every block loads the full Q tensor (64 heads × 128 dim × 1 byte = **8 KB**) from global
memory, even though it is identical for all blocks in the same batch row.

`11 923 × 8 KB = 95 MB` of redundant Q reads.
At B200 HBM bandwidth ~8 TB/s that alone costs **~12 µs** out of a ~16 µs total budget.

**Secondary problems:**
- `tcgen05_alloc / tcgen05_dealloc` called 11 923 times (non-trivial HW overhead per call).
- Only 64 of 128 threads do useful work (50 % waste in the logit-accumulation step).
- Only `warp_id==0, lane==0` issues UMMA — rest of warp stalls.

### Intermediate logit buffer

`B × max_len × 2 bytes` written to global memory by stage 1, fully re-read by stage 2.
For the test case: `1 × 763 552 × 2 = 1.5 MB` — two extra HBM round-trips.

### Stage 2 — `topk_page_table_transform_kernel`

- 3 passes over the row (histogram build, gt-emit, eq-emit), each reading 763 K FP16 values.
- `atomicAdd` to shared memory for emit — 1 024 threads contending on 1 counter.
- Radix histogram uses `atomicAdd` per element instead of warp-local accumulators.

---

## Improvement Plan

### Opt 1 — Persistent per-batch CTA (Q loaded once)  ★ highest impact

**Idea:** Launch `B` CTAs (one per batch row) instead of `B × max_kv_tiles`.
Each CTA:
1. Loads Q once into shared memory (`smem_q[kNumHeads * kHeadDim]`).
2. Loads weights once into `smem_w[kNumHeads]`.
3. Allocates TMEM **once** before the page loop.
4. Iterates over pages `[0, num_pages)`:
   - Loads K tile into smem.
   - Issues UMMA (reuses same `tmem_addr`).
   - Waits for result; 64 threads accumulate logit into global logit buffer.
5. Deallocates TMEM once after the loop.

**Effect:** Eliminates `(num_pages-1) × 8 KB = ~95 MB` of Q reads.
Expected speedup of Stage 1: ~1.8–2×.

**Changes:** `paged_mqa_logits_umma_kernel` grid becomes `dim3(B)`;
kernel gets a `num_pages_per_batch` loop; block_table row is `block_table + b*max_num_pages`.

---

### Opt 2 — Fuse top-K into the page loop (no intermediate logit buffer)

**Idea:** Instead of writing `max_len` logits to global memory and re-reading in a separate
kernel, maintain a **streaming top-K** in shared memory while iterating over pages.

Algorithm per CTA:
```
smem candidate buffer: uint16_t keys[2*K], int indices[2*K]   // 4096 × 6 B = ~24 KB
current_min = -inf

for each page:
    compute 64 logits (as before)
    for each of 64 tokens:
        if logit > current_min:
            append (ordered_key, global_token_idx) to buffer
    if buffer.size >= 2*K:
        warp-parallel partial sort → keep top-K, update current_min

emit top-K indices → apply block_table transform → write output
```

**Effect:** Eliminates the `~1.5 MB` logit buffer and the entire stage-2 kernel launch.
Reduces total HBM traffic by ~3 MB per batch row.

**Complexity:** Medium — requires warp-parallel in-smem selection (can use a simple
merge of sorted warp-level lists, or reuse the existing radix logic on a 4096-element set).

---

### Opt 3 — Double-buffer K cache loads (overlap memory with compute)

**Idea:** Use `__pipeline_memcpy_async` / `cp.async` to prefetch K for page `i+1`
while UMMA runs on K for page `i`.

```cuda
// Ping-pong between smem_k[0] and smem_k[1]
cp_async_load(smem_k[next], page_ptr_next);          // start prefetch
...
tcgen05_mma(tmem, smem_k[curr], smem_q);             // compute current
...
pipeline_wait();                                       // wait prefetch
swap(curr, next);
```

**Effect:** Hides K-load latency (~1–2 µs per page at HBM bandwidth) behind UMMA.
Expected benefit: 10–20 % on top of Opt 1+2.

**Constraint:** Doubles smem_k from 16 KB to 32 KB; total smem still well within B200's
~228 KB per SM.

---

### Opt 4 — Warp-local histograms in stage-2 radix top-K

*Applicable only if Opt 2 is not implemented (separate stage 2 still exists).*

**Idea:** Replace per-element `atomicAdd(&smem_hist[bucket], 1)` with per-warp
register-level accumulation:
```cuda
uint32_t local_hist[kRadix] = {0};          // in registers
for (int i = tid; i < seq_len; i += blockDim.x)
    local_hist[(ordered >> shift) & 0xFF]++;
// single reduction pass: one atomicAdd per (warp, bucket) instead of per element
for (int r = 0; r < kRadix; r++)
    if (local_hist[r]) atomicAdd(&smem_hist[r], local_hist[r]);
```

**Effect:** Reduces `atomicAdd` count from `seq_len` (763 K) to `kRadix × (blockDim/32)` = 8 K.
Eliminates most shared-memory contention in the histogram step.

---

## Iteration Protocol

Each optimization is implemented, smoke-tested, and full-benchmarked independently
before moving to the next, so regressions are caught early.

```
modal run scripts/run_modal_compare.py --smoke --compare-fi    # smoke
modal run scripts/run_modal_compare.py --compare-fi            # full
```
