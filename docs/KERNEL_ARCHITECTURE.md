# DSA FP8 Top-K Indexer — architecture

B200 (sm_100a) kernel for the `dsa_topk_indexer_fp8_h64_d128_topk2048_ps64`
definition of the MLSys 2026 FlashInfer contest, sparse-attention track.
Sources live in [`solution/python/`](../solution/python).

## What it computes

For each batch row `b` and context position `t` in `[0, seq_lens[b])`:

```
S[h]       = sum_d Q[b,h,d] * K[page,slot,d]          # FP8 x FP8 -> FP32
logit[b,t] = scale[page,slot] * sum_h ReLU(S[h]) * weights[b,h]
topk[b,:]  = indices of the k = 2048 largest logits in row b
topk[b,j]  = block_table[b, topk[b,j]/64] * 64 + (topk[b,j] % 64)
```

with `h < 64` heads, `d < 128` head dim, page size 64, `k = 2048`. The paged KV
cache stores each slot as 128 FP8 values followed by a per-slot FP32 scale at
byte offset 8192 within the 8448-byte page.

Output rows are padded with `-1` beyond `seq_lens[b]`.

## Measured results

128 contest workloads on B200, all `PASSED`, timed with CUPTI:

| plan | workloads | mean latency |
|---|---|---|
| fast path | 69 | 2.26 µs |
| short | 39 | 12.65 µs |
| persistent (warp-specialised) | 20 | 17.42 µs |
| all | 128 | 7.80 µs |

Per-kernel resource usage (`cuobjdump -res-usage`):

| kernel | registers | shared memory | stack |
|---|---|---|---|
| `topk_fast_path_kernel` | 31 | 1 152 B | 0 |
| `paged_mqa_logits_umma_kernel_short` | 78 | 26 128 B | 0 |
| `paged_mqa_logits_umma_kernel_persistent_ws` | 98 | 60 224 B | 40 B |
| `topk_page_table_transform_kernel` | 26 | 35 864 B | 0 |

## Module map

One translation unit: `kernel.cu` includes each header exactly once, and all
device code sits in an anonymous namespace.

| file | contents |
|---|---|
| `kernel.cu` | file-level docs, includes, pybind module |
| `dsa_config.cuh` | problem geometry, tuned constants, dispatch thresholds, `Plan` enum |
| `tcgen05_ptx.h` | inline PTX: `cp.async`, mbarriers, named barriers, `tcgen05.*`, `setmaxnreg` |
| `umma_desc.h` | UMMA shared-memory and instruction descriptor bit layouts |
| `stage1_common.cuh` | SMEM swizzle and K-major descriptor helpers shared by both Stage-1 kernels |
| `fast_path.cuh` | Stage 1+2 bypass |
| `stage1_short.cuh` | Stage 1 for mid-size contexts |
| `stage1_persistent_ws.cuh` | Stage 1 for large contexts |
| `stage2_topk.cuh` | radix top-K and block-table transform |
| `graph_cache.cuh` | shape-keyed CUDA graph capture and replay |
| `dispatch.cuh` | host entry point, plan selection, launches |

## Dispatch

`max_num_pages` (the block table's second dimension) selects the plan. The
thresholds are in `dsa_config.cuh`.

| `max_num_pages` | plan | grid | block | kernels launched |
|---|---|---|---|---|
| ≤ 32 | `Plan::FastPath` | `(B)` | 128 | fast path only |
| 33 … 63 | `Plan::Short` | `(max_num_pages, B)` | 128 | short Stage 1 + Stage 2 |
| ≥ 64 | `Plan::PersistentWs` | `(num_splits, B)` | 256 | persistent Stage 1 + Stage 2 |

The fast path is exact, not an approximation: `max_num_pages * 64 ≤ 2048`
implies every `seq_lens[b] ≤ 2048 = k`, so the top-K set is all of
`[0, seq_len)` and the answer does not depend on Q, K or weights.

For the persistent plan the grid targets ~4 CTAs per SM across 132 SMs, then
clamps per-CTA tile-pairs to `[4, 64]` and recomputes the split count.

## Stage 1

Both kernels compute the same logits; they differ in how much setup they
amortise.

Shared structure: Q (64×128 FP8, 8 KB) and one or two K pages are staged in
SMEM under a 128-byte XOR swizzle, `tcgen05.mma.kind::f8f6f4` accumulates
`K · Qᵀ` into TMEM, `tcgen05.ld.32x32b.x64.b32` reads 64 FP32 accumulators per
lane back, then each thread owning a KV slot does ReLU, the weighted sum over
64 heads, and one FP16 store.

**Short** (`stage1_short.cuh`): one page per UMMA (`kUMMA_M = 128` with the
upper half zero-padded), one tile per CTA, 128 threads. Q, TMEM and the
mbarrier are set up per CTA, which is only worth it while the flat
`(max_num_pages, B)` grid still fills the GPU.

**Persistent, warp-specialised** (`stage1_persistent_ws.cuh`): two pages per
UMMA (`kUMMA_M = 128`, both halves real), 256 threads as two warpgroups:

- math warpgroup (warps 0–3, `setmaxnreg` up to 232 regs/thread) loads Q and
  weights once, owns TMEM, and per tile-pair issues UMMA, reads TMEM, computes
  and emits logits;
- producer warpgroup (warps 4–7, down to 40 regs/thread) streams K and the
  per-slot scales into a `kKVStages = 3` deep `cp.async` buffer.

Handoff uses per-stage `K_ready` / `K_done` mbarriers with parity waits, so the
producer fills stage `s+1` while the math group consumes stage `s`.

## Stage 2

`topk_page_table_transform_kernel`, one CTA of 1024 threads per row.

1. FP16 logits are mapped to comparison-ordered `uint16` (flip the sign bit for
   positives, invert everything for negatives), so an unsigned radix select
   sorts them correctly.
2. Two radix rounds of 8 bits each narrow down the k-th largest value (the
   pivot): histogram, suffix scan, pick the bucket containing rank `k`.
3. Count values strictly greater than the pivot, emit those indices, then emit
   indices equal to the pivot until the row is full.
4. Transform token indices to paged slots via the block table; anything out of
   range becomes `-1`.

When `seq_len ≤ 16384` the ordered keys are cached in shared memory (32 KB) on
round 0, which removes four re-reads of the logits row from HBM and four
conversion passes. Essentially all contest workloads fit.

## Graph cache

The harness re-clones input tensors on every timed iteration, so pointers move
but shapes do not, which makes the launch sequence replayable.

Removing the cache and launching directly was measured on B200 against a
noise floor of ±1.7% (two runs of identical code): fast path 2.17 → 2.30 µs
(+5.8%), short 11.18 → 11.32 µs (+1.3%), persistent 17.10 → 17.80 µs (+4.1%),
+3.2% over the sampled workloads with no workload improving. Worth keeping, but
note the effect is far smaller than the ~2.5 µs of host launch overhead
suggests: most of that is hidden from the reported device time, and the
"6.5 → 2.3 µs" figure in earlier write-ups was mostly the CUPTI timing fix.

`graph_cache.cuh` keys a 32-slot LRU on shape only: caller stream, plan, and
the scalars that define the grid and the logits stride. On a hit, the launch
sequence is re-captured on a dedicated thread-local stream and spliced into the
cached executable with `cudaGraphExecUpdate` (~1–2 µs for a 2-node graph)
rather than re-instantiated; the executable is then launched on the caller's
stream. Capture uses `cudaStreamCaptureModeRelaxed` on our own stream because
the harness hands us the legacy NULL stream, where `cudaStreamBeginCapture`
fails with `cudaErrorIllegalState`. Any capture failure falls back to a direct
launch.

Scratch allocation and anything else that is not graph-safe runs outside the
capture region.

## Invariants to preserve

Things that are correct but non-obvious, and that a future change can silently
break:

1. **Stage-1 skip must match Stage-2's row fast path.** Both Stage-1 kernels
   return early when `seq_len <= kTopK`, because Stage 2 emits those rows
   directly from `seq_lens` and `block_table` without reading logits. Changing
   one side without the other reads uninitialised scratch.
2. **Logits row stride must agree between plans.** The short plan writes stride
   `max_num_pages * 64`; the persistent plan writes
   `max_kv_tile_pairs * 128`. The host passes the matching value to Stage 2 as
   `max_len`, and that value is part of the graph key.
3. **Producer visibility before signalling.** `cp.async.wait_group` is
   per-thread, so the producer syncs its whole warpgroup on a named barrier
   *before* the single-thread `mbarrier.arrive` on `K_ready`. Without it the
   math group can read K bytes whose copies have not retired.
4. **`fence_barrier_init()` after `mbarrier.init`.** Without the fence,
   `tcgen05.commit` can arrive on an uninitialised barrier and the kernel traps
   (Xid 43).
5. **Swizzle and descriptor must agree.** The SMEM fill and the UMMA descriptor
   both derive from `stage1_common.cuh`; changing one without the other
   silently corrupts operands.
6. **Padding must be neutral.** Unused K halves are zero-filled and their
   scales set to 0; logits past `seq_len` are written as FP16 `-inf` (`0xFC00`)
   so radix select can never choose them.
7. **Fast-path SMEM padding.** `smem_bt[]` is zero-padded to
   `kFastPathMaxPages` so out-of-range block-table reads never happen; those
   lanes emit `-1` because their token index is already ≥ `seq_len`.
8. **Tie emission order.** Stage 2 emits all indices strictly greater than the
   pivot before any index equal to it. Reordering these passes changes which
   tied tokens land in the output.
9. **The redundant bounds in `describe_tile` are load-bearing.** Removing the
   `tp < tile_pair_begin || tp >= tile_pair_end` check — unreachable, since the
   producer only ever asks for tiles it owns — costs 11.6% on the persistent
   kernel on its own: the compiler drops to 96 registers and reschedules the
   producer/consumer pipeline. The `tile_pair_begin >= max_kv_tile_pairs` early
   return next to it measured within noise but is kept for the same reason, as
   is the equivalent bound in the short kernel (also within noise, −0.9%). None
   of the three are worth removing for the one line each saves.
10. **Stage 2's non-cached path is a correctness path.** No contest workload
   exceeds `kStage2MaxCachedLen`, but rows longer than 16384 need it; without
   it they would read cache entries that were never written.

## Measurement

Timing must go through CUPTI. `flashinfer.testing.bench_gpu_time_with_cupti`
falls back to CUDA events with only a `UserWarning` if `cupti-python` is
missing, too old (< 13.0.0), or records no activity — and on this kernel that
fallback adds about 5 µs to a 2 µs measurement, which is larger than most of
the optimisations here.

```bash
modal run scripts/check_cupti.py          # verify CUPTI is active, not falling back
```

The check compares CUPTI against CUDA events on a ~2 µs kernel and an ~85 µs
kernel: the event overhead should appear as a roughly constant few-microsecond
offset on both, and no fallback warning should be raised.

## Verifying a change

```bash
# 1. device code: diff SASS against git HEAD (no GPU needed, ~2 min)
modal run scripts/check_sass.py

# 2. functional + rough timing: 4 workloads per plan, ~1 min
modal run scripts/run_modal_compare.py --quick --no-compare-fi | tee quick.log
python scripts/compare_bench_logs.py baseline.log quick.log

# 3. only when a change can plausibly move latency: all 128 workloads, ~20 min
modal run scripts/run_modal_compare.py --no-compare-fi | tee new.log
python scripts/compare_bench_logs.py baseline.log new.log
```

`check_sass.py` compiles two revisions with identical flags and diffs the SASS
per kernel. For a refactor meant to be performance-neutral on the device side,
every live kernel should come back `IDENTICAL`, which leaves only the host
changes for the benchmark to cover — that is what makes step 2 sufficient for
cleanup work.

`compare_bench_logs.py` joins two logs by workload uuid and reports per-plan
means. Judge on those means, not on individual workloads: latencies are
reported to 0.1 µs, so on the ~2 µs fast-path bucket a single reporting step is
already ~4.5%.
