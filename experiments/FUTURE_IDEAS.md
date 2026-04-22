# Future optimisation ideas — informed by FA4, CUTLASS, DeepGEMM, TRT-LLM

Tagged references at end of document.

## Where we are today

`submission-v8` on B200, 128 dataset workloads:

- mean 16.0 µs, p50 14.4 µs, p95 30.6 µs, min 9.5 µs, max 30.9 µs
- 2.3× vs FlashInfer-DeepGEMM mean, 128/0/0 head-to-head
- Kernel structure: 1 CTA = 1 warpgroup (128 threads), all threads do every
  step (Q load, K `cp.async`, UMMA issue, TMEM readout, softmax ReLU,
  weighted sum, emit). No warp specialisation, no TMA, no persistent
  scheduling, no multi-CTA cluster.

DeepGEMM's `sm100_fp8_paged_mqa_logits` (what FlashInfer actually wraps)
uses warp specialisation, TMA pipelines, register reconfig, persistent
scheduling, 2-CTA UMMA. We beat it today largely because of kernel-launch
overhead — FI spends ~15 µs in Python/host glue for tiny workloads — but
on medium/large workloads we're closer to even. Most of the "headroom"
below is in beating DeepGEMM on the large regime.

## Proposed ideas, ranked by (expected gain × implementation cost)

### Rank 1 — Size-aware launch (minimal, biggest win)

Keep submission-v8's simple grid for small workloads and only switch to a
persistent scheduler for large ones.

```cpp
const bool persistent =
    (max_kv_tile_pairs >= 24) || (B == 1 && max_kv_tile_pairs >= 8);
if (persistent) {
    // Opt-1 path with tiles_per_cta heuristic (from experiment)
} else {
    // submission-v8 path: grid(max_kv_tile_pairs, B)
}
```

- Expected gain: recover submission-v8's 16 µs mean, pick up 1–3 µs at p95
  (net +5–10 % on mean, ~10 % on p95 for long-seq workloads).
- Effort: small (≈ 50 LOC host side, kernel already has `tiles_per_cta`).
- Risk: very low. Both paths are already validated.

### Rank 2 — 2-CTA cluster UMMA (the big structural win)

Blackwell supports `tcgen05.mma.cta_group::2`: two adjacent CTAs in the
same cluster collaboratively execute one MMA. A is duplicated, B and D
are sharded across the two SMs' TMEM, so **per-CTA SMEM bandwidth
requirements halve** and per-CTA K-cache reads halve. [1][4]

Concrete shape for our kernel (kNumHeads=64, kHeadDim=128, BLOCK_KV=64):

- 2-CTA tile: M=128 (= kUMMA_M), N=128, K=32.
- A (Q) is replicated in both CTAs' SMEM (identical Q block per 2-CTA).
- B (K) is split: CTA0 loads 64 pages × 64 dim slice, CTA1 loads the
  remaining 64 dim slice. The two CTAs share via DSMEM / cluster fence.
- Launch `cluster_size=(2, 1, 1)`; each batch gets half as many cluster
  launches as we have tile pairs.

Why this works here: our kernel is memory-bound on K loads (`kv_cache` at
~100 MB/s per B200 SM). Halving per-SM K traffic directly halves the
memory-bound workloads' latency. For `max_num_pages ≥ 82` workloads
(currently ~30 µs, p95/max), expected floor is ~18–22 µs.

- Expected gain: 20–30 % on the p95/max workloads; neutral or small win
  on small workloads (cluster launch overhead).
- Effort: medium-large. Requires `cta_group::2` descriptors, DSMEM setup
  via `mapa.shared::cluster`, cluster-wide mbarrier. Reference: CUTLASS
  Blackwell 2SM GEMM example in `examples/blackwell/`. [1][4]
- Risk: medium. 2-CTA MMA has stricter alignment and requires the two
  CTAs to stay alive simultaneously — no early exit without explicit
  coordination.

### Rank 3 — Warp specialisation à la FA4 / DeepGEMM

Split the 128-thread CTA (or grow to 256) into producer/consumer warps
with register reconfig:

| role | warps | registers | work |
|---|---|---|---|
| TMA/cp.async producer | 1 | low (≤ 40) | issue K/kscale loads, arrive on `full_kv_barrier` |
| UMMA issuer | 1 | low (≤ 40) | `elect_one` + `tcgen05.mma` + `tcgen05.commit` |
| Math / emit consumer | 2–6 | high (≥ 200) | TMEM readout, ReLU·w sum, emit half logits |

DeepGEMM uses this exact layout with `kNumSpecializedThreads=128` +
`kNumMathThreads=256` and `warpgroup_reg_dealloc/alloc`. [3]

- Expected gain: hides HBM latency behind UMMA without needing 2-CTA;
  improves p95/max by ~10–15 % on its own.
- Effort: medium. Needs `warpgroup_reg_alloc<N>` intrinsics, a named
  barrier pipeline (`cutlass::arch::NamedBarrier` or hand-coded
  mbarriers), and splitting our single-loop kernel into phases.
- Risk: medium. Register-count tuning is fiddly on 128-thread CTAs
  (currently 128/CTA, 16 regs/thread headroom). Would likely grow CTA
  to 256 threads.

### Rank 4 — True TMA for Q and K, not cp.async

Switch from `cp.async.shared.global` to `cp.async.bulk.tensor` (TMA)
with `TmaDescriptor`s built on the host:

- Q: one TMA load per batch row, no indirection.
- K: 2-D TMA with block-table indirection is *not* supported — we'd need
  a separate TMA descriptor per page (too many) or keep cp.async for K
  and TMA only for Q + weights. DeepGEMM uses TMA for Q and an indirect
  paged loader for K.

- Expected gain: small on its own (few hundred ns per launch), but unlocks
  multi-stage pipelining and cluster-level multicast for Q.
- Effort: medium. Need `cute::TmaDescriptor` setup, `mbarrier_arrive_expect_tx`
  transaction barriers, cluster-dim tag.
- Risk: low once scaffolding exists.

### Rank 5 — Deeper K pipeline (3 or 4 stages)

Keep cp.async, but go from 2 buffers to 3–4 for `smem_k`:

```cpp
__shared__ __align__(1024) __nv_fp8_e4m3 smem_k[kKVStages][kUMMA_M * kHeadDim];
```

With `kKVStages=3` and `cp_async_wait_group<2>()`, we have 2 in-flight
loads while UMMA runs on a third — fully hides HBM latency for large
`num_tiles`. Fixes the aliasing hazard cleanly (i, i+1, i+2 land in
distinct buffers).

Cost: +16 KB smem per extra stage. B200 SMs have 228 KB smem, we
currently use ~40 KB → plenty of room.

- Expected gain: 10–15 % on the `max_num_pages ≥ 82` workloads, ~0 on
  small ones.
- Effort: small (~30 LOC edit), build on Opt 3 experiment.
- Risk: low. Purely a performance change.

### Rank 6 — Precompute scheduling metadata on host

Inspired by TRT-LLM PR #12198 and SGLang's `get_paged_mqa_logits_metadata`
at init time:

- Precompute `tiles_per_row_b[b] = ceil_div(seq_lens[b], kUMMA_M)` so
  every CTA can early-exit beyond its row's real tiles without per-tile
  `kv_base >= seq_len` checks.
- Precompute flat base pointers into `kv_cache` for each `(b, page_idx)`
  pair, folding the `block_table` load + `* stride` multiply.

- Expected gain: ~200–500 ns per kernel launch (saves a handful of per-tile
  divmods and an `__ldg`); visible on the smallest workloads where 500 ns
  is 5 % of total.
- Effort: small. Host-side precompute + new kernel args.
- Risk: low.

### Rank 7 — Fuse Stage 2 top-K per CTA into Stage 1

When `max_num_pages ≤ ~40` the full logits tensor fits comfortably in
SMEM per CTA (256 tiles × 128 half = 64 KB). Let each persistent CTA run
its own radix top-K on its partial logits before writing; then Stage 2
merges per-CTA candidates (at most `kSmTarget × 2048 = ~1 M` candidates)
in a thin global pass — or skip Stage 2 entirely when there's 1 CTA per
batch. This is the "Opt 2" idea reframed around persistent scheduling.

- Expected gain: saves ~1.5–2 µs Stage-2 overhead on small workloads
  (currently Stage 2 ≈ 20 % of total at 9.5 µs min).
- Effort: large. Needs cross-CTA merge and top-K semantics carefully
  preserved across splits.
- Risk: medium. Numerical ties and mask handling are easy to get wrong.

### Rank 8 — Block-scaled UMMA (mxf8 / UE8M0)

`tcgen05.mma.kind::mxf8f6f4.block_scale` folds FP8 per-block scales into
the MMA itself, eliminating the explicit `kscale` multiply on the
accumulator. Works with UE8M0 (exponent-only) scales — exactly what
DSA's FP8 packed KV cache already uses (head_dim_with_scale=132 = 128 FP8
+ 4 bytes of scale = UE8M0 per 128-dim block). [2][5]

- Expected gain: 1–2 µs by removing one TMEM readout + scale multiply
  pass per tile; cleaner code.
- Effort: medium. Requires scale-vector descriptors and matching layout.
- Risk: medium. Must verify our `kv_cache` scale layout matches what
  `mxf8f6f4.block_scale` expects.

### Rank 9 — Tile scheduler with load balancing

When `seq_lens` has high variance (most of our failing-edge workloads
have `batch_size=4–8`, `max_num_pages=30–90`), statically splitting
per batch row leaves SMs idle. A global work queue over
`(b, tile_pair)` pairs, consumed via `atomicAdd` on a counter, keeps all
SMs busy. FA4 uses this exact pattern for variable-length sequences. [6]

- Expected gain: 5–10 % on the workloads with max_num_pages > 40 (tail of
  distribution).
- Effort: medium. New scheduler with global counter, kernel loop over
  `next_task()`.
- Risk: low–medium. Atomic contention on counter if too many SMs; mitigate
  with striping.

### Rank 10 — Micro-optimisations on the hot path

Smaller individual wins; good polish after structural work:

a. **ReLU + weighted sum in TMEM**: read S out once, apply ReLU and
   per-head weight in the TMEM→registers copy rather than in a second
   smem pass. Saves one smem round-trip per tile.

b. **FP16 weights baked into Q-descriptor scale**: multiply `weights[b,h]`
   into Q once after load instead of per-tile on the accumulator.

c. **Warp-specialised emit**: one warp writes logits to HBM while the
   other three compute the next tile's softmax → hides `st.global` behind
   math.

d. **`__pipeline_memcpy_async` prefetch of `block_table[b,:]`** into
   SMEM at CTA entry; avoids per-tile `__ldg` on the block table.

e. **Opt 4 histogram** (saved in experiment) is correct and useful if
   Stage 2 becomes a bottleneck again.

## Staging recommendation

1. Ship **Rank 1** (size-aware launch) first — it recovers submission-v8
   on small workloads while giving the Opt 1/3 persistent path on large
   ones. Low risk, already 80 % coded.
2. Add **Rank 5** (3-stage K pipeline) alongside — tiny edit, synergistic.
3. Implement **Rank 3** (warp specialisation) — unlocks the real FA4-style
   pipeline and prepares the structure for Rank 2.
4. Pursue **Rank 2** (2-CTA cluster UMMA) — the single biggest ceiling lift
   for memory-bound large-seq workloads.
5. Revisit **Rank 7 / 8 / 9** once (1)–(4) land; decide based on where the
   bottleneck has shifted.

Expected cumulative mean after (1)+(2)+(3)+(5): ~11–13 µs (vs current
16 µs), with p95/max around 18–22 µs (vs current 30 µs).

## Shipped post-v8 (current branch state)

| # | Idea                                       | Result      |
|---|--------------------------------------------|-------------|
| B | Stage-1+2 static fast path (mnp ≤ 32)      | **shipped** |
| A1| Stage-1 per-CTA early-return on `seq_len ≤ kTopK` | **shipped** |
| A2| Host-side `seq_lens.max()` sync to extend fast path | **rejected** |
| R3| Warp specialisation — producer + math warpgroups   | **shipped** |
|TMB| TMEM double-buffering (prev-iter deferred processing) | **rejected** (no MMA compute to hide) |
|R5d| Deeper K pipeline: kKVStages 3 → 4         | **rejected** (producer not the bottleneck) |
|AC4| 4-way accumulator split in the 64-FMA reduction | **rejected** (no change; FMA chain not critical) |

Current state (after R3):
- mean 12.4 µs, p50 6.6 µs, p95 22.4 µs, min 6.3 µs, max 23.1 µs
- 128/0/0 vs FlashInfer
- Per bucket: mnp≤32 = 6.5 µs; 33–39 = 16.6 µs; 40–63 = 19.3 µs; ≥64 = 22.2 µs
- `DSA_TOPK_DISABLE_WS=1` falls back to the single-warpgroup kernel (A/B toggle).

### R3 what landed

- `paged_mqa_logits_umma_kernel_persistent_ws` — 256 threads / CTA, split
  into a math warpgroup (warps 0-3) and a producer warpgroup (warps 4-7).
- Producer owns `cp.async` for K + kscale; math owns Q load, UMMA issue,
  TMEM readout, ReLU·weighted sum, and emit.
- Per-stage `K_ready` and `K_done` mbarriers drive the handoff; the
  two warpgroups run on opposite phases of the kKVStages=3 pipeline.
- Register reconfig: math claims 232 regs/thread (`setmaxnreg.inc`),
  producer surrenders down to 40 regs/thread (`setmaxnreg.dec`).
- Critical correctness fix during bring-up: `cp.async.wait_group` is
  per-thread, so a producer-warpgroup `bar.sync` is needed BEFORE the
  single-thread `mbarrier.arrive` on `K_ready[buf]` — otherwise math
  can consume `smem_k` before threads 1-127's cp.asyncs are visible
  from thread 0's release fence.  Without this sync we hit 24/128
  INCORRECT_NUMERICAL; with it, 128/128 PASS.

### R3 measured gains (vs A1 baseline, same Modal session)

| bucket          | A1 baseline | +R3   | Δ       |
|-----------------|-------------|-------|---------|
| mnp ≤ 32 (69)   |  6.44 µs    | 6.48  |  ~=     |
| mnp 33–39 (22)  | 16.71 µs    |16.55  | −1 %    |
| mnp 40–63 (17)  | 20.69 µs    |19.27  | **−6.9 %** |
| mnp ≥ 64  (20)  | 23.52 µs    |22.18  | **−5.7 %** |

Mean −3.1 %, p95 −4.7 %, max −2.1 %.  The gain is concentrated where
Stage 1 actually dominates (mnp ≥ 40); small buckets are unaffected
because they take the static fast path.

### R3 what we did NOT ship (and why)

- **TMEM double-buffering — tried, rejected.**  Allocated 2 × kUMMA_N
  TMEM cols and restructured the math loop to issue `UMMA[i]` early,
  then process `iter (i-1)` (wait `UMMA_done[(i-1)%2]`, TMEM_ld, ReLU
  · weighted sum, emit, arrive `K_done[(i-1)%kKVStages]`).  Expected
  gain: overlap MMA compute with TMEM_ld + HBM emit.  Actual: 128/128
  still correct, but mnp 40-63 = 19.27 → 19.62 (+1.8 %) and
  mnp ≥ 64 = 22.18 → 22.59 (+1.8 %).  Original diagnosis: two causes —
  (1) `tcgen05.mma` issue is nearly free, so hiding it behind emit was
  a small win, and (2) delayed `K_done` release starves the producer
  pipeline.

  **Updated diagnosis after kKVStages=4 A/B (see below):** cause (2)
  is wrong — producer is not starved.  Cause (1) is the real story:
  there is no useful MMA compute to hide behind emit, because the math
  path (TMEM_ld + per-head ReLU·w sum + emit) *is* what's keeping us
  above ~20 µs on mnp ≥ 40, not MMA compute.  Triple-buffering would
  hit the same wall.

- **kKVStages 3 → 4 (deeper K pipeline) — tried, rejected.**  Hypothesis
  was that bumping to 4 stages would give the producer more prefetch
  headroom and unlock TMEM double-buffering as a follow-on.  Bench
  (vs R3 baseline in same Modal session):

  | bucket        | n  | R3 (3 stages) | 4 stages | Δ       |
  |---------------|----|---------------|----------|---------|
  | mnp ≤ 32      | 69 |  6.44 µs      | 6.29 µs  | −2.3 %  |
  | mnp 33–39     | 22 | 16.71 µs      | 16.70 µs | ≈       |
  | mnp 40–63     | 17 | 19.27 µs      | **19.45 µs** | **+0.9 %** |
  | mnp ≥ 64      | 20 | 22.18 µs      | 22.11 µs | −0.3 %  |
  | overall mean  |128 | 12.4  µs      | 12.30 µs | −0.8 %  |
  | overall p95   |    | 22.4  µs      | 22.20 µs | −0.9 %  |

  The 40-63 bucket — the one we expected to benefit if producer was
  the limiter — flat-to-slightly-worse.  Overall 0.8 % reduction is
  within run-to-run noise (±1-2 %).  Conclusion: **producer is not
  currently the bottleneck on mnp ≥ 40**; 16 KB extra SMEM buys no
  real headroom.  Stays at 3 stages.

  This reshapes the next-lever picture — see below.

### Where the math path actually spends its time (current hypothesis)

After iter i's `wait umma_done`, every math thread does:
1. `tcgen05_ld_32x32b_x64_b32` — 64 regs (256 B per thread) from TMEM.
2. ReLU + per-head weighted sum — 64 FMAs, needs `smem_w` and 64
   accumulator regs in flight.
3. `st.global` — 1 FP16 value (2 B/thread, 128 threads = 256 B total
   per tile).
4. `named_barrier_sync` on the math warpgroup.
5. Thread 0 arrives `K_done[buf]`.

The emit itself (step 3) is tiny — just 256 B to HBM per tile, fully
pipelined behind step 4/5.  Steps 1 + 2 are the bulk: TMEM_ld is
roughly 200 cycles of memory latency + 64 FMAs at ~1 cycle each but
all dependent through `acc_relu`, so compute is ~64 serial cycles.

### What we tried to verify this, and what's left

1. **4-way accumulator split in the ReLU·w sum loop — tried, rejected.**
   Original loop is a 64-FMA serial dep chain through `acc_relu`.
   Hypothesis: at ~4-cycle FMA latency → ~256 cycle critical path.
   Split into 4 independent accumulators (16 FMAs each) + a 3-add
   reduction at the end.

   Bench (same Modal session, vs R3 baseline):

   | bucket        | n  | R3    | 4-way split | Δ      |
   |---------------|----|-------|-------------|--------|
   | mnp ≤ 32      | 69 |  6.44 | 6.33        | −1.7 % |
   | mnp 33–39     | 22 | 16.71 | 16.77       | +0.4 % |
   | mnp 40–63     | 17 | 19.27 | 19.48       | +1.1 % |
   | mnp ≥ 64      | 20 | 22.18 | 22.27       | +0.4 % |
   | overall mean  |128 | 12.4  | 12.36       | −0.3 % |

   All within run-to-run noise (~±1-2 %).  Either ptxas already
   schedules the FMAs with sufficient ILP, or (more likely) the FMA
   chain is already hidden behind `tcgen05_wait_ld` latency.

   **Combined evidence from kKVStages=4 + 4-way-split — both no-ops**:
   neither producer prefetch depth nor ReLU-sum critical path moves
   the needle.  The per-tile bottleneck must be further upstream.

### The actual per-tile dominator (current best guess)

`tcgen05_ld_32x32b_x64_b32` transfers 64 × 4 B = 256 B per thread from
TMEM to registers, and `tcgen05_wait_ld` is a hard per-thread stall
until the entire load retires.  TMEM→register latency on Blackwell
is in the ~150-250 cycle range; because our consumers (ReLU, FMAs)
start immediately after `wait_ld`, the load latency is on the
critical path and is NOT overlapped with anything useful — the MMA
for the next tile can't start because we only have single-buffer
TMEM, and the load from this tile must finish before we overwrite
TMEM.  The 64-FMA chain runs *after* `wait_ld`, never during it.

That reframes the shortlist:

1. **TMEM double-buffer + warp-specialised MMA issue**.  This IS the
   fix — but the naïve double-buffer (tried, rejected) didn't pay
   off because MMA issue+commit is fast enough that freeing TMEM a
   tile earlier didn't translate to real overlap.  The version that
   *would* pay off: keep MMA issue in warp 0 as today, but start
   `UMMA[i+1]` into `TMEM[(i+1)%2]` the instant iter i's
   `tcgen05_wait_ld` returns (not after emit).  That overlaps MMA
   compute for iter i+1 with ReLU·sum + emit of iter i.  Earlier
   attempt deferred the *post-processing*, not the *MMA issue* — so
   we hid the wrong thing.

2. **2-CTA cluster UMMA (Rank 2).**  Halves per-SM K-cache reads
   *and* halves the TMEM footprint per SM — so `tcgen05_ld`
   transfers half as many bytes per thread, shortening the per-tile
   critical path directly.  Biggest structural win for mnp ≥ 64.

3. **Block-scaled UMMA (Rank 8).**  Folds `smem_kscale` into the MMA
   itself; eliminates the `* smem_kscale[...]` multiply on the
   accumulator and one smem load per tile — a few cycles per tile,
   but it stacks with (1).

De-prioritised:

- **Warp-specialised emit.**  Emit is 256 B/tile, already hidden
  behind named-barrier → isolation into a dedicated warp cannot save
  more than a handful of cycles per tile.  Not worth the complexity.
- **Deeper K pipeline (kKVStages > 3).**  A/B said no, and makes the
  hot loop bigger in SMEM without touching the real critical path.

### A2 rejection — why host-side sync is too expensive

Tried two implementations, both regressed the 33–39 bucket from 16.6 µs
to ~36 µs (+20 µs overhead per call):

1. `seq_lens.max().item<int>()` — PyTorch launches a reduction kernel,
   then does a blocking D→H copy. Cost: ~20 µs.
2. `cudaMemcpyAsync` of the full `seq_lens` vector + `cudaStreamSynchronize`
   + CPU-side max. Cost: ~20 µs — same as (1), so the overhead is
   dominated by the stream sync itself, not by the PyTorch reduction.

On B200 in this benchmark harness, any `cudaStreamSynchronize` (or
equivalent event wait) on the submission stream is ~15–20 µs of
wall-clock time — likely because the stream has in-flight bookkeeping
from prior iterations. That's larger than the ~12 µs saving the fast
path would unlock, so A2 is net-negative on every qualifying workload.

**When A2 would actually win**:
- If the sync could be hidden behind the launch (it can't — we need the
  answer *before* deciding which kernel to launch).
- If we built CUDA-graph conditional nodes (CUDA 12.4+ feature): launch
  both fast path and full pipeline into a graph, let the GPU pick based
  on a device-resident flag written by a tiny reduction kernel. Skips
  host sync entirely. **Effort: high (graph capture infra).**
- If we accepted a speculative "assume fast path works" execution with
  a device-resident validation kernel and a fallback recompute. Too
  complex and correctness-hostile for this benchmark.

Conclusion: A2 is archived. The only practical way to extend the fast
path beyond the static `mnp ≤ 32` threshold is **device-side conditional
execution** — future work, Rank 3 (warp specialisation) unlocks enough
structure to make that feasible without a graph.

## References

1. FlashAttention-4 paper (Dao et al., arXiv 2603.05451, Mar 2026) and
   Tri Dao's blog post `tridao.me/blog/2026/flash4/` — asymmetric scaling,
   2-CTA MMA, TMEM intermediates, warp specialisation.
2. NVIDIA CUTLASS Blackwell docs — `tcgen05.mma.kind::f8f6f4`,
   `mxf8f6f4.block_scale`, 128×256 tiles, cta_group=2.
3. DeepGEMM `sm100_fp8_paged_mqa_logits.cuh` (branch d30fc36c) —
   reference production kernel: warp specialisation, multi-stage Q/KV
   pipelines, TMA, persistent scheduling, register reconfig.
4. Colfax Research — "Writing GEMM Kernels Using Tensor Memory For
   Blackwell GPUs" — practical TMEM + UMMA walkthrough.
5. SGLang roadmap #15025 — `fp8_paged_mqa_logits` with `nextn=2/4`,
   UE8M0 scaling, dual-stream indexer.
6. Flash-Attention SM100 tile scheduler
   (`flash_attn/cute/tile_scheduler.py`) — persistent work-stealing over
   `(batch, head, tile)` triples.
7. TensorRT-LLM PR #12198 — fused cat+FP8+scatter and `indexerKCacheGather`
   for DSA; validates host-side pointer precomputation as a win.
