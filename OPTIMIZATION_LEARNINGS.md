# Optimization learnings (DSA TopK indexer CUDA)

**Policy:** Commit / tag **only** when a change shows a **clear win** on Modal smoke **or** full 128 (same harness config). Everything else lives here.

**Baseline (reference kernel):** `gather` + `bmm` + `.contiguous()` + in-place `relu_` / `mul_` + `topk_out` + batched `page_transform`. Smoke geomean **varies run-to-run** on Modal (e.g. **~6.85×–7.5×** seen); treat **≥0.15×** relative gain as “maybe real” only if repeated.

## Log (chronological)

| # | Date (UTC) | Idea | Smoke geomean (17 wl) | Verdict |
|---|------------|------|------------------------|---------|
| — | 2026-03-31 | **Baseline** (current `kernel.cu` on branch) | **~6.85×** | Same run as below |
| 1 | 2026-03-31 | Drop `.contiguous()` after `bmm` (in-place `relu_` needs row-major; often still OK) | **~6.89×**, 17/17 | **No commit** — within noise vs 6.85×; reverted to `.contiguous()` for safety |
| 2 | 2026-03-31 | `q_float = to(f32).contiguous()` before `bmm` | **~7.20×** then **~7.08×** on repeat | **No commit** — inconsistent; likely Modal noise |
| 3 | 2026-03-31 | Reuse single `topk_vals` buffer `[K_topk]` | (prior run) | **Reverted** — see git `d49792f` |
| 4 | 2026-03-31 | `sum_out` into `scores_buf[max_seq_len]` | **~6.94×** vs ~7.25× ref | **Not landed** |
| 5 | 2026-03-31 | `bmm_out` + explicit `K^T.contiguous()` | **~6.67×** | **Not landed** |
| 6 | 2026-03-30 | Fused `relu*weight` in CUDA (`fmaxf` / ternary) | FAIL numerical | **Reverted** |
| 7 | 2026-03-30 | Fused on-the-fly `q·K` (no `K_batched`) | match + slow OR fail | **Reverted** |
| 8 | 2026-03-30 | `_nested_from_padded_tensor` + `sum` | RUNTIME_ERROR | **Aborted** |
| 9 | 2026-03-29 | In-place `relu_` + `mul_` vs `relu()*w` | 128/128 + smoke OK | **Kept** — `submission-v3` @ `96da9a2` |
| 10 | 2026-03-30 | **Combined:** `sum_out` → reused `scores_buf[max_seq_len]` + `topk_out` → reused `topk_vals_buf[B,K_topk]` (slices per batch row) | **7.39×**, 17/17 PASSED (`FIB_MODAL_SMOKE=1`, Modal B200) | **Kept** — `submission-v4` @ `9b29407` (after `python3 scripts/pack_solution.py`; `solution.json` unchanged). JIT `topk_cuda_cublas_buf`. |
| 11 | 2026-03-30 | **Batched** `logits.sum(dim=1)` → `[B,S]`, mask tail with `-inf`, per-row `topk` on `narrow(0,0,sl)` | **9.46×** geomean but **7/17 INCORRECT_NUMERICAL** | **Reverted** — `sum` over full padded `S` changes FP reduction order vs reference `narrow(...,sl).sum(0)` on `[H,sl]`; dataset relaxes tolerance somewhat but **still** fails top-k index checks on large-T rows. |
| 12 | 2026-03-30 | **Top-k:** `sorted=false` + when `min(seq_len) ≥ K_topk`, one batched `topk` on `[B, max_seq_len]` (padded `-inf`); else per-row `topk` | **8.12×**, 17/17 PASSED (`FIB_MODAL_SMOKE=1`, `FIB_RTOL=265` `FIB_ATOL=17500`) | **Kept** — `run_modal.py` forwards rtol/atol; JIT `topk_cuda_cublas_topkopt`. Match column not 1.0 when ties reorder; strict contest may need `sorted=true` or full-128 check. |
| 13 | 2026-03-30 | **Head reduction (profiler #1):** `mask_logits_past_seq_len_kernel` zeros `[b,h,s]` for `s≥sl`, then **one** `logits.sum(dim=1)` → `[B,S]`; topk unchanged | **8.50×**, 17/17 PASSED (same Modal rtol/atol) | **Kept** — replaces B× `sum_out` on slices (~45% CUDA at idx 127); JIT `topk_cuda_cublas_bsum2`. |

## Next ideas (not run — billing / time)

10. **`torch::matmul`** instead of `bmm` for `[B,H,D]@[B,D,S]` — often same backend; verify once.
11. **Precompute `K_contig = K_batched.transpose(1,2).contiguous()` once** and `bmm(q, K_contig)` — measure vs single transpose (may match current).
12. **`weights` already contiguous:** `unsqueeze_` in-place if API allows, or cache `w_bcast` across calls if extension stays loaded.
13. **Vectorized `uint4` loads** in `gather_dequant_kernel` — more code; profile first on B200.
14. **`page_transform` block size 128 vs 256 vs 512** — micro; only if Nsight shows it hot.
15. **`seq_lens` on GPU** + skip host `sl_vec` where harness guarantees sync — needs careful correctness for `narrow`/`topk`.
16. **cuBLASLt batched GEMM** — only if bit-exact vs `bmm` proven.
17. **CUDA graph** capture of steady-state launch sequence — high effort; eval environment may not replay.

## Reproduce smoke (when Modal works)

```bash
FIB_MODAL_SMOKE=1 modal run scripts/run_modal.py
# Compare "Geomean: …" line; run 2–3× before trusting deltas.
```

## Batch ablations (automated)

| Batch | Idea | Geomean | Status |
|---|---|---|---|
| 0 | baseline before batch | 6.70× | OK |
| 18 | at::matmul instead of torch::bmm for logits | 6.25× | OK (Δ-0.45 vs baseline) |
| 19 | page_transform BLOCK_T 128 | 7.18× | OK (Δ+0.48 vs baseline) |
| 20 | page_transform BLOCK_T 512 | 6.84× | OK (Δ+0.14 vs baseline) |
| 21 | weights.unsqueeze(2) without .contiguous() on weights | 7.17× | OK (Δ+0.47 vs baseline) |
| 22 | q_float .to(f32).contiguous() before bmm | 6.91× | OK (Δ+0.21 vs baseline) |
| 23 | Explicit K_T = transpose(1,2).contiguous() then bmm(q, K_T) | 6.56× | OK (Δ-0.14 vs baseline) |
| 24 | Drop .contiguous() after bmm (keep relu_ in-place) | 7.66× | OK (Δ+0.96 vs baseline) |
| 25 | Reuse topk_vals buffer [K_topk] across batch loop | 6.66× | OK (Δ-0.04 vs baseline) |
| 26 | extra_cuda_cflags: add -Xptxas -O3 | 6.17× | OK (Δ-0.53 vs baseline) |
| 27 | extra_cuda_cflags: -O2 instead of -O3 | 7.76× | OK (Δ+1.06 vs baseline) |
| 28 | sum_out into scores_buf[max_seq_len] | 7.42× | OK (Δ+0.72 vs baseline) |
| 29 | bmm_out into pre-allocated logits, q and K_T contiguous | 5.62× | OK (Δ-1.08 vs baseline) |
| 30 | JIT name bump to force clean rebuild (same flags) | 6.97× | OK (Δ+0.27 vs baseline) |
| 31 | permute(0,2,1) instead of transpose(1,2) for K side of bmm | 6.95× | OK (Δ+0.25 vs baseline) |
| 32 | K_batched.contiguous() immediately after gather kernel | 6.80× | OK (Δ+0.10 vs baseline) |

**Interpretation (2026-03-31 batch):** One smoke per variant vs a **single** baseline run (**6.70×**). Modal variance is large — e.g. **#24** and **#27** look like big wins but earlier sessions showed **dropping `.contiguous()`** and **`-O2`** as neutral or worse. **Do not land** from this table alone; re-run top candidates **3×** and check **128 workloads** before any commit/tag.

**Rerun:** `python3 scripts/opt_ablations.py` (restores `kernel.cu` / `binding.py` after each experiment).

## Combined “winners” stack (not landed)

Merged in one build: `K_batched.contiguous()` after gather; `q_float` contiguous; `bmm` on `permute({0,2,1})` **without** post-`.contiguous()`; `BLOCK_T=128`; `weights.unsqueeze(2)` only; `sum_out` + `scores_buf`; reused `topk_vals_buf`; `binding` **`-O2`** + JIT name `topk_cuda_cublas_combo`.

| Run | Geomean (17 wl) | Verdict |
|-----|-----------------|--------|
| Modal smoke 2026-03-31 | **7.06×**, 17/17 PASSED | **Reverted** — not clearly better than prior **~7.2–7.5×** single-variant runs; interactions + noise. |

Lesson: **positive single ablations do not compose** reliably; validate combos on **multiple** smokes + **128** full before shipping.

### Batch 2 (ids 33–47)

| Id | Idea | Geomean | Status |
|---|---|---|---|
| 0 | baseline before batch | 6.46× | OK |
| 33 | seq_lens_dev created early (q device) before gather; remove duplicate before loop | 6.88× | OK (Δ+0.42 vs baseline) |
| 34 | page_transform BLOCK_T 192 | 7.02× | OK (Δ+0.56 vs baseline) |
| 35 | binding extra_cflags host -O3 | 6.98× | OK (Δ+0.52 vs baseline) |
| 36 | nvcc -gencode arch=compute_100,code=sm_100 | 6.95× | OK (Δ+0.49 vs baseline) |
| 37 | nvcc --ftz=true | 6.94× | OK (Δ+0.48 vs baseline) |
| 38 | nvcc --prec-div=false (faster approx div) | 6.91× | OK (Δ+0.45 vs baseline) |
| 39 | page_transform BLOCK_T 64 | 6.83× | OK (Δ+0.37 vs baseline) |
| 40 | page_transform BLOCK_T 384 | 7.20× | OK (Δ+0.74 vs baseline) |
| 41 | bmm uses transpose().clone() before matmul | 6.34× | OK (Δ-0.12 vs baseline) |
| 42 | JIT name topk_cuda_cublas_b42 (rebuild) | 6.95× | OK (Δ+0.49 vs baseline) |
| 43 | Use at::relu_(logits) instead of logits.relu_() | 6.98× | OK (Δ+0.52 vs baseline) |
| 44 | Use at::mul_(logits, w_bcast) instead of logits.mul_(w_bcast) | — | FAIL_OR_ERROR |
| 45 | K_batched empty -> zeros (same gather overwrite) | 6.55× | OK (Δ+0.09 vs baseline) |
| 46 | bt_i32 without .contiguous() after clamp | 7.08× | OK (Δ+0.62 vs baseline) |
| 47 | nvcc -Xptxas -O2 (PTX assembler opt) | 6.49× | OK (Δ+0.03 vs baseline) |

**Batch 2 notes:** Baseline **6.46×** (single run). **#44** `at::mul_(logits, w_bcast)` → **FAIL_OR_ERROR** (likely API/signature — use `logits.mul_(w_bcast)`). **#36–38, 47** change FP32 semantics risk for contest — do not ship without correctness audit. **#40** (BLOCK 384) and **#46** (no `bt` contiguous) look best on this noisy run — re-validate.

**Rerun batch 2:** `python3 scripts/opt_ablations.py --batch2`

### `page_transform` `BLOCK_T` resweep (same session, sequential smokes)

Repeated Modal smoke (**17 wl**) with only `constexpr int BLOCK_T` changed (defaults back to **256** in repo).

| `BLOCK_T` | Geomean | Notes |
|-----------|---------|--------|
| 64 | **6.95×** | 17/17 |
| 128 | **7.09×** | 17/17 |
| 192 | **6.91×** | 17/17 |
| **256** (default) | **6.80×** | 17/17 |
| 384 | **6.54×** | 17/17 — *contradicts batch-2’s 7.20× for 384 → noise* |
| 512 | **7.09×** | 17/17 |

**Conclusion:** **Not safe to pick a “winner” from one pass.** Best *this* sweep: **128 / 512** (~7.09×); worst: **384** (6.54×). Earlier batch had **384** best. **Recommendation:** run **3× median** per candidate on smoke, then **full 128** for finalists **128 vs 256 vs 512** only. **Do not ship 384** without confirming batch-2 result was fluke.

**Reproduce sweep (bash):**

```bash
cp solution/cuda/kernel.cu /tmp/k.bak
for BT in 64 128 192 256 384 512; do
  sed -i "s/constexpr int BLOCK_T = [0-9]*;/constexpr int BLOCK_T = ${BT};/" solution/cuda/kernel.cu
  echo "=== BLOCK_T=$BT ==="
  FIB_MODAL_SMOKE=1 modal run scripts/run_modal.py 2>&1 | grep Geomean
  cp /tmp/k.bak solution/cuda/kernel.cu
done
```

### More ideas (page_transform & neighbors)

1. **Warp-specialized transform:** one warp per batch row `b` (up to 32 threads) when `k = min(K_topk, sl) ≤ 32` often — else fall back to current loop (reduces launch overhead for small `k`).
2. **Fuse last steps:** single kernel reading `local_topk_long` + `seq_lens` + `block_table` + writing `out` with **I/O coalesced** (measure vs tiny kernel today).
3. **`__launch_bounds__`** on `page_transform_batched_kernel` to help occupancy for chosen `BLOCK_T`.
4. **Vectorize index gather:** load 2× `int64` or `int4` from `loc` when `k` and alignment allow.
5. **Dynamic shared memory:** stage `bt` row or `loc[0:k]` for `k` small (avoid repeated global reads).
6. **Profile Nsight** on B200: confirm `page_transform` % before investing — it may be **<5%** of end-to-end.
7. **Match `k` to warp multiple:** pad loop or use `k` rounded up to 32 for simpler divergence (only if padding indices never read).
8. **Second stream:** overlap `page_transform` with next workload’s host prep (only if harness allows async).
9. **Compile-time `BLOCK_T`** via `-DBLOCK_T=128` + small set of **prebuilt** extension names to A/B without `sed`.
10. **Grid-stride loop** with fixed **256 threads** and `for (i = tid; i < k; i += 256)` — sometimes better than `blockDim` tied to `BLOCK_T` for irregular `k`.

---

## Profiling: current bottleneck (Modal B200, `torch.profiler`)

**Method:** `FIB_MODAL_PROFILE=1 modal run scripts/modal_profile.py` with `acc_events=True`, **`sort_by=cuda_time_total`**, after JIT warmup. **`FIB_PROFILE_WORKLOAD_INDEX`** is forwarded from the local entrypoint to the Modal worker** (fixed 2026-03-30 — previously the remote defaulted to 16). Nsight Systems / NCU still unreliable on this Modal stack.

### Small / medium **T** (workload index **16**, B=4 repeats)

| Bucket | ~% Self CUDA | Notes |
|--------|----------------|--------|
| **`aten::topk`** | **~34%** | Many small batch rows |
| **`aten::sum`** | **~34%** | Head reduction |
| **`aten::bmm`** | **~7%** | |
| **`gather_dequant_kernel`** | **~5%** | |
| **`page_transform_batched_kernel`** | **~3%** | |
| **`aten::copy_`** / DtoH | **~7–10%** combined | Host sync path |

### Large **T** (workload index **127**, `FIB_PROFILE_REPEAT=6`)

| Bucket | ~% Self CUDA | Notes |
|--------|----------------|--------|
| **`aten::sum`** over heads | **~44.6%** | **Dominant** — 150 `sum` ops in trace (batched path: per-row `sum_out` into padded buffer) |
| **`aten::topk`** (`gatherTopK`) | **~34.6%** | 150 `topk` calls |
| **`aten::bmm`** / CUTLASS | **~7.0%** | GEMM finally non-trivial vs sum+topk |
| **`gather_dequant_kernel`** | **~7.0%** | Scales with tokens |
| **`aten::mul_`** (relu×weights epilogue) | **~2.2%** | |

**Conclusion:** At **large sequence length**, **head reduction (`sum`)** and **top-k** dominate (**~79%** combined). `page_transform` is still a small slice. **`bmm` and gather** are large enough that **Tensor Core / FP8 GEMM** and **fused gather→GEMM** are the main levers for **large jumps** in speedup—not micro-tuning `BLOCK_T`.

### On **~30×** speedup at large **T**

Contest **speedup** is vs the **Python reference**, not vs peak FLOPs. Hitting **30×** on the **largest-T** rows typically needs one or more of:

1. **Make GEMM the story** — e.g. **FP8/TF32 Tensor Core** batched matmul with acceptable error under relaxed `rtol/atol`, or a fused path that avoids an extra full **[B,H,S]** materialization.
2. **Cut top-k work** — **approximate top-k**, smaller effective **K** for hot paths, or a **fused** “score + select” kernel (only if eval allows).
3. **Fuse gather + dot** — remove **`K_batched`** read/write cost (currently ~7% at idx 127; can dominate if GEMM is accelerated).

None of these are “free”; each trades implementation cost and/or numerical policy vs the reference.

### What to optimize next (priority)

1. **Top-k path** — batched or fused selection, or fuse **score build + top-k** while keeping **exact** indices vs reference (hard).
2. **Head reduction** — fuse **weighted relu** with **sum over H** so you do not materialize full `[H, sl]` for the reduction (must match ATen reduction tree).
3. **GEMM** — becomes important when **S** is huge; at profiled shapes **`bmm` ~5%**; revisit after (1–2) or profile again on max-**T** workload only.
4. **Gather** — vectorized loads / coalescing; only ~4% here but simpler than top-k.
5. **`page_transform`** — low priority unless Nsight on **your** max workload shows otherwise.

**Re-run profile (large T):**

```bash
FIB_MODAL_PROFILE=1 FIB_PROFILE_WORKLOAD_INDEX=127 FIB_PROFILE_REPEAT=6 modal run scripts/modal_profile.py
```
