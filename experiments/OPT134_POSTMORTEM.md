# Opt 1 / 3 / 4 experiment — post-mortem and future plan

## TL;DR

- Current `solution/python/kernel.cu` (= tag `submission-v8`, commit `4ad39ad`)
  is **faster overall** than the experimental variant that applied Opt 1
  (persistent per-batch CTA), Opt 3 (cp.async double-buffered K), and Opt 4
  (warp-local radix histograms).
- The experimental kernel is saved as
  `experiments/kernel_opt134_persistent_cta_dbuf_warp_hist.cu` for reference.
- Keep `submission-v8` as the active submission; future work should revisit
  Opt 1 / Opt 3 with a different launch/pipeline shape (see "Future work").

## What was tried

| Opt | Idea | Where |
|---|---|---|
| 1 | Persistent per-batch CTA: amortise Q-load + TMEM alloc across many tile-pairs in one CTA; host picks `num_splits` / `tiles_per_cta` to keep ~528 CTAs resident. | Stage 1 |
| 3 | Double-buffered `smem_k` + `smem_kscale` with `cp.async`; prefetch tile `i+1` into the opposite buffer while UMMA consumes buffer `i`. | Stage 1 |
| 4 | Warp-local radix histogram in Stage 2 using `__ballot_sync` + `__match_any_sync`: one `atomicAdd(__popc(same))` per (warp,bucket). | Stage 2 |

Opt 2 (fuse top-K into the page loop) was intentionally skipped — it conflicts
with the multi-CTA-per-batch split introduced by Opt 1 and would require a
cross-CTA top-K merge.

A subtle bug was found and fixed during development: the original Opt 3 had
`smem_k` aliasing — at iter `i` UMMA reads `smem_k[i&1]` while a prefetch for
tile `i+2` was being kicked off into `smem_k[(i+2)&1] == smem_k[i&1]`, racing
on the same buffer. The fix prefetches tile `i+1` into the opposite buffer and
keeps exactly one `cp.async` group in flight.

## Benchmark on B200 (128 dataset workloads, `run_modal_compare.py --compare-fi`)

| metric (µs) | submission-v8 | Opt 1/3/4 exp. | delta |
|---|---|---|---|
| mean | **16.0** | 19.3 | +21 % slower |
| p50 | **14.4** | 15.8 | +10 % slower |
| p95 | 30.6 | **29.2** | −5 % (win on tail) |
| min | **9.5** | 11.1 | +17 % slower |
| max | 30.9 | **29.6** | −4 % |
| head-to-head vs FlashInfer-DeepGEMM | 128/0/0 | 128/0/0 | both sweep |

Correctness: both pass all 128 workloads under the dataset's tolerance
(abs_err up to a few × 10⁻² on a handful of workloads, well inside rtol/atol).

## Why the experiment lost on mean

1. **Persistent-CTA tax on tiny workloads.** Most of the dataset is
   `batch_size ≤ 32`, `max_num_pages ≤ 32` → only 1–16 tile-pairs per row.
   With `num_splits ≥ kSmTarget/B`, the host launches many CTAs whose
   `tile_pair_begin` range only covers 1–4 tiles, so the one-time costs
   (Q load, TMEM alloc/dealloc, mbar init) are no longer amortised — they're
   just repeated slightly differently. The original kernel's 1-CTA-per-tile
   launch already hides Q-load behind the compiler/L2 and is fine for these
   sizes.
2. **Opt 3 forces one extra `__syncthreads` per tile** (to make cp.async
   landings visible) that the submission avoids via a simpler direct-load
   path. For small `num_tiles`, that sync dominates the claimed overlap.
3. **Opt 4 helps Stage 2, but Stage 2 is already ≤ 2 µs** on most workloads
   in the current kernel — the mean saving is small and is eaten by the
   extra `__syncthreads` reorder introduced with the warp-collective loop.
4. **p95 / max win is real** — Opt 1/3 does pay off once a single batch row
   has ≳ 82 pages (`num_tiles ≥ 41`). That's where HBM latency actually
   starts to dominate. The dataset only has 20/128 workloads in that
   regime, so the mean is dragged down by the small-workload regressions.

## What to keep from the experiment

- **Correctness fix for Opt 3 aliasing** (prefetch `i+1`, not `i+2`) is a
  generally useful pattern; preserved in the experimental file.
- **Host-side `num_splits`/`tiles_per_cta` heuristic** is a reusable
  building block if we ever go back to persistent CTAs.
- **Opt 4 histogram via `__match_any_sync`** is simpler than the old
  split warp-group accumulator and a good drop-in whenever Stage 2
  becomes a bottleneck again.

## Future improvements to consider

### A. Size-aware launch (fix the regression, keep the p95 win)

Pick the persistent / non-persistent path at launch time:

```cpp
const int total_tiles = max_kv_tile_pairs;
const bool use_persistent = total_tiles >= 32;  // tune on real data
if (use_persistent) {
    // Opt 1 path with split heuristic
} else {
    // submission-v8 path: grid(max_kv_tile_pairs, B)
}
```

Expected outcome: match submission-v8 on the small/medium regime
(≤ 20 µs) and pick up the 1–3 µs p95/max improvement on large workloads.
This is the highest ROI next step.

### B. 3-way buffered K pipeline (Opt 3+)

With 3 buffers we can keep 2 `cp.async` groups in flight (`wait_group<1>`
instead of `<0>`). Buffer aliasing vanishes because `(i) & 3`, `(i+1) & 3`,
`(i+2) & 3` are all distinct. On long-sequence workloads (82–91 pages/row)
this should turn ~30 µs → ~25 µs by fully hiding HBM latency behind UMMA.

Cost: +16 KB smem (K) + 256 B (kscale). Still well within SM limits.

### C. Keep Stage 1 layout, but fuse a warp-group top-K (Opt 2, redone)

Two-pass version: within each CTA compute a per-tile partial top-K into a
small shared buffer (k' = 256), then do a CTA-wide top-256 merge at the
end. Skips the global logits tensor entirely for the common case and
removes Stage 2 for all but the largest `max_num_pages`. Implementation
cost is high; best explored only if Stage 2 re-emerges as a bottleneck.

### D. Warp-specialisation inside the current 128-thread CTA

Dedicate warp 0 to issuing `cp.async`s + UMMA, warps 1–3 to softmax/emit.
Would let us overlap the emit path with the next tile's K fetch without
changing the grid shape. Lower impact than (A)/(B) but cheap to try.

### E. Precompute a scalar `seq_block_count[b]` on host

Avoid repeating the `kv_base >= seq_len` check per tile per thread; let
the host pass `ceil(seq_lens[b] / kUMMA_M)` and let each CTA early-exit
past that bound. Saves a few ns per tile, not a big win but trivial.

## File map

- `solution/python/kernel.cu` — active submission (`submission-v8`, 3.5× FI
  in the smoke report, 128/0/0 head-to-head, 16.0 µs mean).
- `experiments/kernel_opt134_persistent_cta_dbuf_warp_hist.cu` — Opt 1+3+4
  experiment; correct but slower overall. Good starting point if we tackle
  (A) or (B) above.

## Reproducing these results

```bash
# Active submission (fast path)
modal run scripts/run_modal_compare.py --compare-fi

# Experiment
cp experiments/kernel_opt134_persistent_cta_dbuf_warp_hist.cu solution/python/kernel.cu
modal run scripts/run_modal_compare.py --compare-fi
git checkout solution/python/kernel.cu    # restore submission-v8
```

Always cross-check against `modal run scripts/run_modal.py` too —
`run_modal_compare.py` still reports a latency for `INCORRECT_NUMERICAL`
workloads, which is how the aliasing bug initially went undetected.
