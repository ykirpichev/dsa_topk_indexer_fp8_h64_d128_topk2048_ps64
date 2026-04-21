"""
Side-by-side Modal B200 benchmark: OUR clean_baseline kernel vs any other
solutions registered in the trace set (typically the FlashInfer +
DeepGEMM reference).

Usage:
    modal run scripts/run_modal_compare.py --compare-fi
    modal run scripts/run_modal_compare.py --smoke --compare-fi
    modal run scripts/run_modal_compare.py --n-workloads 20 --compare-fi

The image matches the fi branch's canonical bench image: FlashInfer CI
CUDA 13.2 + FlashInfer head + flashinfer-bench head + DeepGEMM pinned
to the Sep 29, 2025 SM100 commit (API-compatible with the shipped
FlashInfer solution's 1D seq_lens convention).

Does NOT modify scripts/run_modal.py or any files inside solution/.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet

app = modal.App("flashinfer-bench-compare")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

# Same image the upstream starter kit uses + DeepGEMM pinned to the
# Sep 29 2025 SM100 commit (59f2c07cf2). That commit is the last one
# API-compatible with the shipped FlashInfer solution's 1D seq_lens
# convention. (Head commits enforce 2D context_lens.)
image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:latest")
    .apt_install("git")
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .pip_install("wheel", "setuptools")
    .run_commands(
        "git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git /tmp/deep-gemm",
        "cd /tmp/deep-gemm && git checkout 59f2c07cf2 && "
        "git submodule update --init --recursive",
        "pip install --no-build-isolation /tmp/deep-gemm || pip install /tmp/deep-gemm",
        "git clone --recursive --depth 1 "
        "https://github.com/flashinfer-ai/flashinfer.git /tmp/flashinfer",
        "pip install --no-build-isolation /tmp/flashinfer",
        "git clone --depth 1 "
        "https://github.com/flashinfer-ai/flashinfer-bench.git /tmp/flashinfer-bench",
        "pip install /tmp/flashinfer-bench",
    )
)


def _worker_benchmark_config(smoke: bool = False) -> BenchmarkConfig:
    """Build BenchmarkConfig on the worker; honour flashinfer_bench defaults."""
    import os
    defaults = BenchmarkConfig()
    if smoke:
        return BenchmarkConfig(
            warmup_runs=1, iterations=1, num_trials=1,
            rtol=defaults.rtol, atol=defaults.atol,
        )
    return BenchmarkConfig(
        warmup_runs=int(os.environ.get("FIB_WARMUP_RUNS", str(defaults.warmup_runs))),
        iterations=int(os.environ.get("FIB_ITERATIONS", str(defaults.iterations))),
        num_trials=int(os.environ.get("FIB_NUM_TRIALS", str(defaults.num_trials))),
        rtol=float(os.environ.get("FIB_RTOL", str(defaults.rtol))),
        atol=float(os.environ.get("FIB_ATOL", str(defaults.atol))),
    )


def _collect_results(result_trace_set, definition_name: str) -> dict:
    """Flatten evaluation traces into {solution_name: {workload_uuid: entry}}."""
    out: dict = {}
    for trace in result_trace_set.traces.get(definition_name, []):
        if not trace.evaluation:
            continue
        entry = {
            "status": trace.evaluation.status.value,
            "axes": getattr(trace.workload, "axes", {}),
        }
        if trace.evaluation.performance:
            entry["latency_ms"] = trace.evaluation.performance.latency_ms
            entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
            entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
        if trace.evaluation.correctness:
            entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
            entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
        if trace.evaluation.status.value != "PASSED":
            try:
                ev_dict = trace.evaluation.model_dump()
                diag = {k: str(v)[:2000] for k, v in ev_dict.items()
                        if v and k not in ("performance", "correctness", "status")}
                if diag:
                    entry["diag"] = diag
            except Exception:
                pass
        sol_name = getattr(trace.solution, "name", str(trace.solution))
        out.setdefault(sol_name, {})[trace.workload.uuid] = entry
    return out


@app.function(image=image, gpu="B200:1", timeout=3600,
              volumes={TRACE_SET_PATH: trace_volume})
def run_benchmark(solution: Solution, smoke: bool = False,
                  n_workloads: int = 0, compare_fi: bool = False) -> dict:
    """Run ours (+ optionally every other registered solution) on B200."""
    config = _worker_benchmark_config(smoke=smoke)

    trace_set = TraceSet.from_path(TRACE_SET_PATH)

    if solution.definition not in trace_set.definitions:
        raise ValueError(
            f"Definition '{solution.definition}' not in trace set. "
            f"Known: {list(trace_set.definitions)}"
        )

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])
    if not workloads:
        raise ValueError(f"No workloads for '{solution.definition}'")

    if smoke:
        workloads = workloads[:1]
    elif n_workloads > 0:
        workloads = workloads[:n_workloads]

    solutions = [solution]
    extra_names = []
    if compare_fi:
        in_trace = trace_set.solutions.get(definition.name, [])
        print(f"[compare-fi] trace set has {len(in_trace)} solution(s):")
        for s in in_trace:
            print(f"  - {s.name} (lang={getattr(s, 'language', '?')})")
        for other in in_trace:
            if other.name != solution.name:
                solutions.append(other)
                extra_names.append(other.name)

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: solutions},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=True)

    results_by_sol = _collect_results(result_trace_set, definition.name)
    return {
        "definition": definition.name,
        "our_solution": solution.name,
        "extra_solutions": extra_names,
        "results": results_by_sol,
    }


def _print_report(payload: dict):
    import statistics as _stats

    def_name = payload.get("definition", "<unknown>")
    our_name = payload.get("our_solution", "")
    results_by_sol = payload.get("results", {})

    for sol_name, per_wl in results_by_sol.items():
        if sol_name == our_name:
            continue
        bad = [(u, r) for u, r in per_wl.items() if r.get("status") != "PASSED"]
        if bad:
            print(f"\n[diag] `{sol_name}` non-PASSED on {len(bad)}/{len(per_wl)}. First:")
            u, r = bad[0]
            print(f"  workload {u[:8]}: status={r['status']}")
            for k, v in (r.get("diag") or {}).items():
                print(f"    {k}: {str(v)[:1200]}")

    our_results = results_by_sol.get(our_name, {})
    print(f"\n{def_name} — solution `{our_name}`:")
    for uuid, r in our_results.items():
        status = r.get("status")
        line = f"  {uuid[:8]}...: {status}"
        if r.get("latency_ms") is not None:
            line += f" | {r['latency_ms']*1000:.1f} us"
        if r.get("speedup_factor") is not None:
            line += f" | {r['speedup_factor']:.2f}x vs naive ref"
        if r.get("max_abs_error") is not None:
            line += (f" | abs_err={r['max_abs_error']:.2e},"
                     f" rel_err={r.get('max_rel_error', 0):.2e}")
        axes = r.get("axes", {})
        if axes:
            line += "  [" + ", ".join(f"{k}={v}" for k, v in axes.items()) + "]"
        print(line)

    other_sols = [s for s in results_by_sol if s != our_name]
    if other_sols:
        print("\nPer-workload side-by-side (us):")
        header = (f"{'#':>3}  {'uuid8':<10}  {'axes':<34}  "
                  f"{'ours_us':>8}  " +
                  "  ".join(f"{s[:16]+'_us':>20}" for s in other_sols) +
                  f"  {'FI/ours':>8}")
        print(header)
        print("-" * len(header))
        rows_sorted = sorted(
            our_results.items(),
            key=lambda kv: (kv[1].get("latency_ms") or 0.0),
        )
        ours_faster = 0
        fi_faster = 0
        ties = 0
        for i, (uuid, r) in enumerate(rows_sorted):
            ours_us = (r.get("latency_ms") or 0.0) * 1000.0
            axes = r.get("axes", {})
            axes_s = ", ".join(f"{k}={v}" for k, v in axes.items())
            cells = []
            fi_ratio = None
            for s in other_sols:
                o = results_by_sol.get(s, {}).get(uuid)
                if o and o.get("latency_ms") is not None:
                    v = o["latency_ms"] * 1000.0
                    cells.append(f"{v:>20.1f}")
                    if fi_ratio is None and ours_us > 0:
                        fi_ratio = v / ours_us
                else:
                    cells.append(f"{'—':>20}")
            ratio_s = f"{fi_ratio:>7.2f}x" if fi_ratio is not None else "       —"
            print(f"{i+1:>3}  {uuid[:10]:<10}  {axes_s:<34}  "
                  f"{ours_us:>8.1f}  " + "  ".join(cells) + f"  {ratio_s}")
            if fi_ratio is not None:
                if fi_ratio > 1.05:   ours_faster += 1
                elif fi_ratio < 0.95: fi_faster += 1
                else:                 ties += 1

        print("\nAggregate latency (us):")
        def _stats_row(d):
            vals = sorted(v["latency_ms"] * 1000.0 for v in d.values()
                          if v.get("latency_ms") is not None)
            if not vals:
                return (float("nan"),) * 5
            mean = _stats.mean(vals)
            n = len(vals)
            p50 = vals[n // 2]
            p95 = vals[min(n - 1, int(0.95 * n))]
            return (mean, p50, p95, vals[0], vals[-1])
        all_sols = [our_name] + other_sols
        print(f"  {'solution':<44}  {'mean':>7}  {'p50':>7}  "
              f"{'p95':>7}  {'min':>7}  {'max':>7}")
        for s in all_sols:
            m, p50, p95, mn, mx = _stats_row(results_by_sol.get(s, {}))
            print(f"  {s:<44}  {m:>7.1f}  {p50:>7.1f}  {p95:>7.1f}  "
                  f"{mn:>7.1f}  {mx:>7.1f}")

        if other_sols:
            first = other_sols[0]
            print(f"\nHead-to-head vs `{first}` (>5% margin = win):")
            print(f"  ours wins:  {ours_faster}")
            print(f"  FI wins:    {fi_faster}")
            print(f"  ties:       {ties}")


@app.local_entrypoint()
def main(smoke: bool = False, n_workloads: int = 0, compare_fi: bool = True):
    """Pack the solution from solution/ and run on Modal B200.

    Defaults to --compare-fi=True. Pass --compare-fi False to run ours only.
    """
    from scripts.pack_solution import pack_solution

    print("Packing solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")
    if compare_fi:
        print("compare-fi: will include every other solution in the trace set.")
    if smoke:
        print("Smoke mode: 1 workload, warmup=1, iterations=1, trials=1.")
    elif n_workloads > 0:
        print(f"Subset mode: {n_workloads} workloads.")
    else:
        print("Full mode: all workloads in the trace set.")

    payload = run_benchmark.remote(
        solution, smoke=smoke, n_workloads=n_workloads, compare_fi=compare_fi
    )
    if not payload:
        print("No results returned!")
        return
    _print_report(payload)
