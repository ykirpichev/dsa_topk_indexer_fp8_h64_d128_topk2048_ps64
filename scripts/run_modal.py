"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks
on NVIDIA B200 GPUs via Modal.

Smoke run (few workloads, shorter timing — still measures speedup vs reference)::

    FIB_MODAL_SMOKE=1 modal run scripts/run_modal.py

Cap workloads on a full-style benchmark (default timing config)::

    FIB_MODAL_MAX_WORKLOADS=16 modal run scripts/run_modal.py

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/
"""

import os
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet

app = modal.App("flashinfer-bench")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:latest", add_python="3.12")
    .pip_install("flashinfer-bench")
)


@app.function(image=image, gpu="B200:1", timeout=7200, volumes={TRACE_SET_PATH: trace_volume})
def run_benchmark(
    solution: Solution, smoke: bool = False, max_workloads: int | None = None
) -> dict:
    """Run benchmark on Modal B200 and return results."""
    if smoke:
        # Fewer workloads / trials than production, but profile_baseline=True is required
        # for reference_latency_ms and speedup_factor.
        config = BenchmarkConfig(
            warmup_runs=2,
            iterations=40,
            num_trials=3,
            timeout_seconds=1800,
            profile_baseline=True,
        )
        workload_limit = 8
    else:
        config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)
        workload_limit = None

    trace_set = TraceSet.from_path(TRACE_SET_PATH)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])

    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    if smoke:
        workloads = workloads[:workload_limit]
    elif max_workloads is not None and max_workloads > 0:
        workloads = workloads[:max_workloads]

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=True)

    traces = result_trace_set.traces.get(definition.name, [])
    results = {definition.name: {}}

    for trace in traces:
        if trace.evaluation:
            entry = {
                "status": trace.evaluation.status.value,
                "solution": trace.solution,
            }
            if trace.evaluation.performance:
                entry["latency_ms"] = trace.evaluation.performance.latency_ms
                entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
                entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
            if trace.evaluation.correctness:
                entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
                entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
            results[definition.name][trace.workload.uuid] = entry

    return results


def print_results(results: dict):
    """Print benchmark results in a formatted way."""
    for def_name, traces in results.items():
        print(f"\n{def_name}:")
        speedups: list[float] = []
        for workload_uuid, result in traces.items():
            status = result.get("status")
            print(f"  Workload {workload_uuid[:8]}...: {status}", end="")

            if result.get("latency_ms") is not None:
                print(f" | sol {result['latency_ms']:.3f} ms", end="")

            if result.get("reference_latency_ms") is not None:
                print(f" | ref {result['reference_latency_ms']:.3f} ms", end="")

            if result.get("speedup_factor") is not None:
                sp = result["speedup_factor"]
                speedups.append(sp)
                print(f" | {sp:.2f}x speedup", end="")

            if result.get("max_abs_error") is not None:
                abs_err = result["max_abs_error"]
                rel_err = result.get("max_rel_error", 0)
                print(f" | abs_err={abs_err:.2e}, rel_err={rel_err:.2e}", end="")

            print()

        if speedups:
            mean_sp = sum(speedups) / len(speedups)
            print(
                f"\n  --- Aggregate ({len(speedups)} workloads): "
                f"mean speedup {mean_sp:.2f}x "
                f"(min {min(speedups):.2f}x, max {max(speedups):.2f}x) ---"
            )


@app.local_entrypoint()
def main():
    """Pack solution and run benchmark on Modal."""
    smoke = os.environ.get("FIB_MODAL_SMOKE", "").lower() in ("1", "true", "yes")
    cap_raw = os.environ.get("FIB_MODAL_MAX_WORKLOADS", "").strip()
    max_workloads: int | None = None
    if cap_raw:
        try:
            max_workloads = int(cap_raw)
        except ValueError:
            max_workloads = None

    from scripts.pack_solution import pack_solution

    print("Packing solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")

    if smoke:
        print("\nRunning smoke benchmark on Modal B200 (8 workloads, timed vs reference)...")
    elif max_workloads:
        print(f"\nRunning benchmark on Modal B200 (first {max_workloads} workloads)...")
    else:
        print("\nRunning full benchmark on Modal B200...")
    results = run_benchmark.remote(solution, smoke=smoke, max_workloads=max_workloads)

    if not results:
        print("No results returned!")
        return

    print_results(results)
