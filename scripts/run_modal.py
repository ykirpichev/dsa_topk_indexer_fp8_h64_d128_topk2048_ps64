"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks
on NVIDIA B200 GPUs via Modal.

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/

Troubleshooting:
- safetensors "header too large" on the reference run: trace blobs are likely Git
  LFS pointers. Run git lfs pull locally, then scripts/refresh_contest_dataset_modal.sh.

The remote image is flashinfer/flashinfer-ci-cu132 (CUDA 13.2 + PyTorch). DeepGEMM,
FlashInfer, and flashinfer-bench are installed from GitHub (see image build below).
Set CUDA_HOME for extension builds.

Correctness uses BenchmarkConfig rtol/atol (element-wise; see flashinfer_bench bench/utils).
Defaults come from scripts.bench_config (same as installed flashinfer_bench; dataset JSON
has no per-definition rtol/atol).
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Benchmark, Solution, TraceSet

from scripts.bench_config import default_benchmark_config

app = modal.App("flashinfer-bench")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:latest")
    .apt_install("git")
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .pip_install("wheel", "setuptools")
    .run_commands(
        "git clone --recursive --depth 1 https://github.com/deepseek-ai/DeepGEMM.git /tmp/DeepGEMM",
        # PEP517 isolated build has no torch; DeepGEMM setup.py imports torch
        "pip install --no-build-isolation /tmp/DeepGEMM",
        "git clone --recursive --depth 1 https://github.com/flashinfer-ai/flashinfer.git /tmp/flashinfer",
        "pip install --no-build-isolation /tmp/flashinfer",
        "git clone --depth 1 https://github.com/flashinfer-ai/flashinfer-bench.git /tmp/flashinfer-bench",
        "pip install /tmp/flashinfer-bench",
    )
)


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_benchmark(solution: Solution) -> dict:
    """Run benchmark on Modal B200 and return results.

    BenchmarkConfig is created in the worker so it matches the image's flashinfer-bench
    (avoid pickling a client BenchmarkConfig across different package versions).
    """
    config = default_benchmark_config()

    trace_set = TraceSet.from_path(TRACE_SET_PATH)

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])

    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

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
        for workload_uuid, result in traces.items():
            status = result.get("status")
            print(f"  Workload {workload_uuid[:8]}...: {status}", end="")

            if result.get("latency_ms") is not None:
                print(f" | {result['latency_ms']:.3f} ms", end="")

            if result.get("speedup_factor") is not None:
                print(f" | {result['speedup_factor']:.2f}x speedup", end="")

            if result.get("max_abs_error") is not None:
                abs_err = result["max_abs_error"]
                rel_err = result.get("max_rel_error", 0)
                print(f" | abs_err={abs_err:.2e}, rel_err={rel_err:.2e}", end="")

            print()


@app.local_entrypoint()
def main():
    """Pack solution and run benchmark on Modal."""
    from scripts.pack_solution import pack_solution

    bench_cfg = default_benchmark_config()
    print(
        f"Benchmark config (local display): warmup={bench_cfg.warmup_runs} "
        f"iters={bench_cfg.iterations} trials={bench_cfg.num_trials} "
        f"rtol={bench_cfg.rtol:g} atol={bench_cfg.atol:g}"
    )
    print(
        "Worker uses scripts.bench_config.default_benchmark_config() "
        "(flashinfer_bench defaults unless FIB_* overrides)."
    )

    print("Packing solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution...")
    solution = Solution.model_validate_json(solution_path.read_text())
    print(f"Loaded: {solution.name} ({solution.definition})")

    print("\nRunning benchmark on Modal B200...")
    results = run_benchmark.remote(solution)

    if not results:
        print("No results returned!")
        return

    print_results(results)
