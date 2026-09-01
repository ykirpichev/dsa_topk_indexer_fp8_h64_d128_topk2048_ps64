---
name: modal-dev-iteration
description: >-
  Iterate on the DSA FP8 top-K indexer kernel with Modal B200 validation:
  smoke-test changes on Modal, then run a full-scale benchmark via
  scripts/run_modal.py and validate results. Use when editing
  solution/python/, running Modal benches, smoke tests, full benchmarks,
  or when the user asks to validate kernel changes on B200.
---

# Modal B200 development iteration

Never validate on localhost. Always use Modal B200. Pack is automatic inside
the runners (`scripts/pack_solution.py`).

Active sources: `solution/python/` (`config.toml` → `language = "python"`).

## Protocol

Copy and track:

```
Iteration:
- [ ] Implement change in solution/python/
- [ ] Smoke on Modal — must PASS before continuing
- [ ] Finish remaining code/doc edits for this change
- [ ] Full-scale Modal run via scripts/run_modal.py
- [ ] Validate results; only then call the change done
```

### 1. Smoke test (gate every intermediate change)

After each meaningful edit (or small batch of related edits), ensure the
change is good with a smoke run on Modal:

```bash
mkdir -p artifacts/dev
modal run scripts/run_modal_compare.py --smoke --no-compare-fi \
  2>&1 | tee artifacts/dev/smoke_$(date +%Y%m%d_%H%M%S).log
```

- 1 workload, `warmup=1`, `iterations=1`, `trials=1` (fast compile + correctness).
- Pass criteria: our solution status is `PASSED`, no traceback, `abs_err` sane
  (typically `0` for this indexer).
- On failure: fix, re-smoke. Do not start a full run.

Optional mid-size gate (still not full):

```bash
modal run scripts/run_modal.py --max-workloads 8 \
  2>&1 | tee artifacts/dev/subset_$(date +%Y%m%d_%H%M%S).log
```

### 2. Full-scale run (when the change is complete)

After changes are done and smoke has passed:

```bash
modal run scripts/run_modal.py \
  2>&1 | tee artifacts/dev/full_$(date +%Y%m%d_%H%M%S).log
```

This packs `solution/python/`, runs **all** workloads for the definition on
one B200 with `warmup=3`, `iterations=100`, `num_trials=5`.

### 3. Validate full results

From the log (and the printed workload table):

| Check | Expectation |
|---|---|
| Status | Every workload `PASSED` (definition has 128 workloads) |
| Correctness | No `INCORRECT_*`; `abs_err` / `rel_err` within harness limits |
| Completeness | Result count matches all workloads (not a truncated `--max-workloads` run) |
| Sanity | Latencies in µs-scale for small pages; no compile/runtime crash |

Fail any row → treat as failed iteration: fix, smoke again, then re-run full.

Report to the user: pass/fail counts, any failing uuid + status/error, and
rough latency summary if present (mean / notable outliers).

## Rules

- Prefer smoke frequently; full runs are expensive (~tens of minutes).
- Do not skip smoke “because the change is small.”
- Do not declare success from smoke alone — full `scripts/run_modal.py` is
  required once the change set is finished.
- Do not edit the Modal image / runner scripts unless the task is about them.
- For A/B vs FlashInfer or refactor neutrality, optionally also run
  `modal run scripts/run_modal_compare.py --compare-fi` and/or
  `python scripts/compare_bench_logs.py BASELINE_LOG NEW_LOG` — that does
  **not** replace the full `scripts/run_modal.py` gate above.

## Prerequisites (one-time)

```bash
modal setup
modal volume create flashinfer-trace   # then put the contest trace set into it
```

If Modal auth or the `flashinfer-trace` volume is missing, stop and tell the
user; do not fall back to local GPU.
