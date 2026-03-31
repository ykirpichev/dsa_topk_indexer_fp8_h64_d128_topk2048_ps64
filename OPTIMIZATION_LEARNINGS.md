# Optimization learnings (DSA TopK indexer CUDA)

**Policy:** Commit / tag **only** when a change shows a **clear win** on Modal smoke **or** full 128 (same harness config). Everything else lives here.

**Baseline (reference kernel):** `gather` + `bmm` + `.contiguous()` + in-place `relu_` / `mul_` + `topk_out` + batched `page_transform`. Smoke geomean **varies run-to-run** on Modal (e.g. **~6.85×–7.5×** seen); treat **≥0.15×** relative gain as “maybe real” only if repeated.

**Modal blocked:** 2026-03-31 — `workspace billing cycle spend limit reached`; further cloud smokes **not** run until billing resets.

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
