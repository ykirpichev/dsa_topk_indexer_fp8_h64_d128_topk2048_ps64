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
