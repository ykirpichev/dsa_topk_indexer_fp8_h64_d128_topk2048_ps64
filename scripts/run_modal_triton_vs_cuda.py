"""
Side-by-side Modal B200 benchmark of the two implementations in this repo:

  solution/python   CUDA kernels built at runtime (the contest submission)
  solution/triton   the same algorithm written in Triton

Both are packed from source and run through flashinfer-bench on the same B200
worker against the same workloads, then reported per dispatch plan.

Usage:
    modal run scripts/run_modal_triton_vs_cuda.py --smoke
    modal run scripts/run_modal_triton_vs_cuda.py --quick
    modal run scripts/run_modal_triton_vs_cuda.py

    # Per-kernel GPU time (Stage 1 / Stage 2 split) instead of the benchmark:
    modal run scripts/run_modal_triton_vs_cuda.py --kernels

    # Same runs with a newer Triton than the contest image ships (3.7.0):
    DSA_TRITON_VERSION=3.9.0 modal run scripts/run_modal_triton_vs_cuda.py

Add --json-out PATH to also save the raw results as JSON.

Does not touch config.toml, solution.json or the other runner scripts.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Benchmark, BenchmarkConfig, BuildSpec, Solution, TraceSet

app = modal.App("flashinfer-bench-triton-vs-cuda")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

TRITON_SOLUTION_NAME = "dsa_topk_indexer_fp8_b200_triton"

# Image definition copied from scripts/run_modal_compare.py so that Modal
# reuses the cached build.
image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:latest")
    .apt_install("git")
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .pip_install("wheel", "setuptools")
    # cupti-python is in the evaluation environment (see EVALUATION.md).
    # Without it, flashinfer-bench falls back to CUDA events which adds
    # ~3-4 us of event-sync overhead per measurement.
    .pip_install("cupti-python")
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

# Optional extra layer: a different Triton release than the image ships.
TRITON_VERSION = os.environ.get("DSA_TRITON_VERSION", "")
if TRITON_VERSION:
    image = image.pip_install(f"triton=={TRITON_VERSION}").env(
        {"DSA_TRITON_VERSION": TRITON_VERSION})

PLANS = ("fast path", "short", "persistent")


def _plan(max_num_pages: int) -> str:
    """Dispatch plan for a workload; thresholds are the same in both solutions."""
    if max_num_pages <= 32:
        return "fast path"
    return "short" if max_num_pages < 64 else "persistent"


def _axes(entry) -> dict:
    """Axes of a workload entry (a Workload, or a Trace wrapping one)."""
    axes = getattr(entry, "axes", None)
    if axes is None:
        axes = getattr(getattr(entry, "workload", None), "axes", None)
    return axes or {}


def _first_per_plan(workloads: list, limit: int) -> list:
    """The first `limit` workloads of each dispatch plan, so that a short run
    still exercises every kernel."""
    taken = dict.fromkeys(PLANS, 0)
    picked = []
    for w in workloads:
        pages = _axes(w).get("max_num_pages")
        if pages is None:
            continue
        plan = _plan(pages)
        if taken[plan] < limit:
            taken[plan] += 1
            picked.append(w)
    return picked


def _versions() -> dict:
    import torch
    import triton

    return {"triton": triton.__version__, "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0)}


def _pack_solutions() -> list:
    """[cuda, triton] Solution objects packed from the source trees."""
    from flashinfer_bench.agents import pack_solution_from_files
    from scripts.pack_solution import load_config

    config = load_config()
    meta, build = config["solution"], config["build"]

    cuda = pack_solution_from_files(
        path=str(PROJECT_ROOT / "solution" / build["language"]),
        spec=BuildSpec(
            language=build["language"],
            target_hardware=["cuda"],
            entry_point=build["entry_point"],
            destination_passing_style=build.get("destination_passing_style", True),
        ),
        name=meta["name"],
        definition=meta["definition"],
        author=meta["author"],
    )
    triton = pack_solution_from_files(
        path=str(PROJECT_ROOT / "solution" / "triton"),
        spec=BuildSpec(
            language="triton",
            target_hardware=["cuda"],
            entry_point="solution.py::kernel",
            destination_passing_style=True,
        ),
        name=TRITON_SOLUTION_NAME,
        definition=meta["definition"],
        author=meta["author"],
    )
    return [cuda, triton]


@app.function(image=image, gpu="B200:1", timeout=3600,
              volumes={TRACE_SET_PATH: trace_volume})
def run_benchmark(solutions: list, smoke: bool = False, per_plan: int = 0,
                  warmup: int = 3, iterations: int = 100, trials: int = 5) -> dict:
    """Benchmark every solution on the same workloads; returns flat results."""
    if smoke:
        config = BenchmarkConfig(warmup_runs=1, iterations=1, num_trials=1)
    else:
        config = BenchmarkConfig(warmup_runs=warmup, iterations=iterations,
                                 num_trials=trials)

    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    definition = trace_set.definitions[solutions[0].definition]
    workloads = trace_set.workloads.get(definition.name, [])
    if not workloads:
        raise ValueError(f"No workloads for '{definition.name}'")

    if smoke or per_plan > 0:
        workloads = _first_per_plan(workloads, 1 if smoke else per_plan)

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: solutions},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )
    result = Benchmark(bench_trace_set, config).run_all(dump_traces=True)

    results: dict = {}
    for trace in result.traces.get(definition.name, []):
        ev = trace.evaluation
        if not ev:
            continue
        entry = {"status": ev.status.value, "axes": dict(_axes(trace))}
        if ev.performance:
            entry["latency_us"] = ev.performance.latency_ms * 1000.0
        if ev.correctness:
            entry["max_abs_error"] = ev.correctness.max_absolute_error
            entry["max_rel_error"] = ev.correctness.max_relative_error
        if ev.status.value != "PASSED":
            entry["log"] = str(getattr(ev, "log", ""))[-1500:]
        name = getattr(trace.solution, "name", str(trace.solution))
        results.setdefault(name, {})[trace.workload.uuid] = entry

    return {
        "definition": definition.name,
        "solutions": [s.name for s in solutions],
        "results": results,
        "versions": _versions(),
        "config": {"warmup": config.warmup_runs, "iterations": config.iterations,
                   "trials": config.num_trials},
    }


def _trace_kernels(fn, args: list, warmup: int, iterations: int) -> list:
    """GPU activities of `iterations` calls of fn(*args), one list per call.

    Follows flashinfer.testing.bench_gpu_time_with_cupti, which the benchmark
    uses for timing (cold L2 before every call, CUPTI activity records matched
    to the call by correlation id), but keeps every activity instead of only
    the first-start-to-last-end span. Each activity is (name, start_ns, end_ns).
    """
    import bisect

    import torch
    from cupti import cupti

    megabyte = 1024 * 1024
    try:
        from flashinfer.testing.utils import get_l2_cache_size
        scrub_bytes = 2 * get_l2_cache_size("cuda")
    except ImportError:
        scrub_bytes = 256 * megabyte
    # Overwriting a buffer twice the size of L2 evicts the inputs from it.
    flush = torch.empty(scrub_bytes, dtype=torch.int8, device="cuda").zero_
    for _ in range(warmup):
        flush()
        fn(*args)
    torch.cuda.synchronize()

    kinds = cupti.ActivityKind
    gpu_kinds = {kinds.CONCURRENT_KERNEL: None, kinds.MEMCPY: "MEMCPY", kinds.MEMSET: "MEMSET"}
    api_kinds = (kinds.RUNTIME, kinds.DRIVER)
    launches, activities = [], []       # (start, correlation id); (correlation id, record)

    def buffer_requested():
        return 8 * megabyte, 0

    def buffer_completed(records):
        for r in records:
            if r.kind in gpu_kinds:
                name = gpu_kinds[r.kind] or r.name
                activities.append((r.correlation_id, (name, r.start, r.end)))
            elif r.kind in api_kinds:
                launches.append((r.start, r.correlation_id))

    for kind in (*api_kinds, *gpu_kinds):
        cupti.activity_enable(kind)
    cupti.activity_register_callbacks(buffer_requested, buffer_completed)
    calls = []
    for _ in range(iterations):
        flush()
        torch.cuda.synchronize()
        begin = cupti.get_timestamp()
        fn(*args)
        end = cupti.get_timestamp()
        torch.cuda.synchronize()
        calls.append((begin, end))
    cupti.activity_flush_all(0)
    for kind in (*api_kinds, *gpu_kinds):
        cupti.activity_disable(kind)
    cupti.finalize()

    launches.sort()
    starts = [start for start, _ in launches]
    by_id: dict = {}
    for correlation_id, record in activities:
        by_id.setdefault(correlation_id, []).append(record)
    per_call = []
    for begin, end in calls:
        ids = {launches[i][1] for i in range(bisect.bisect_left(starts, begin),
                                             bisect.bisect_right(starts, end))}
        records = [r for i in ids for r in by_id.get(i, [])]
        per_call.append(sorted(records, key=lambda r: r[1]))
    return per_call


@app.function(image=image, gpu="B200:1", timeout=3600,
              volumes={TRACE_SET_PATH: trace_volume})
def run_kernel_profile(solutions: list, per_plan: int = 0, warmup: int = 3,
                       iterations: int = 50) -> dict:
    """Median GPU time of every kernel each solution launches, per workload."""
    import statistics

    from flashinfer_bench.bench.evaluators.utils import allocate_outputs
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors
    from flashinfer_bench.compile import BuilderRegistry

    device = "cuda:0"
    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    definition = trace_set.definitions[solutions[0].definition]
    workloads = [getattr(w, "workload", w)
                 for w in trace_set.workloads.get(definition.name, [])]
    if per_plan > 0:
        workloads = _first_per_plan(workloads, per_plan)

    registry = BuilderRegistry.get_instance()
    runnables = [(s.name, registry.build(definition, s)) for s in solutions]

    results: dict = {name: {} for name, _ in runnables}
    for workload in workloads:
        tensors = load_safetensors(definition, workload, trace_set.root)
        inputs = gen_inputs(definition, workload, device=device, safe_tensors=tensors)
        for name, runnable in runnables:
            args = list(inputs) + allocate_outputs(definition, inputs, device)
            calls = _trace_kernels(runnable, args, warmup, iterations)
            # Calls with the usual number of launches (all of them, in practice).
            count = statistics.mode(len(c) for c in calls)
            calls = [c for c in calls if len(c) == count and count > 0]
            if not calls:
                continue
            kernels = []
            for i in range(count):
                kernels.append({
                    "name": calls[0][i][0],
                    # Offsets are relative to the first kernel's start.
                    "start_us": statistics.median((c[i][1] - c[0][1]) / 1e3 for c in calls),
                    "duration_us": statistics.median((c[i][2] - c[i][1]) / 1e3 for c in calls),
                })
            span = statistics.median(
                (max(r[2] for r in c) - c[0][1]) / 1e3 for c in calls)
            results[name][workload.uuid] = {
                "axes": dict(workload.axes), "span_us": span, "kernels": kernels}

    return {
        "definition": definition.name,
        "solutions": [s.name for s in solutions],
        "results": results,
        "versions": _versions(),
        "config": {"warmup": warmup, "iterations": iterations},
    }


def _print_report(payload: dict):
    import statistics

    cuda_name, triton_name = payload["solutions"]
    cuda, triton = payload["results"].get(cuda_name, {}), payload["results"].get(triton_name, {})
    v, c = payload["versions"], payload["config"]

    print(f"\n{payload['definition']}")
    print(f"  {v['gpu']}, torch {v['torch']}, triton {v['triton']}")
    print(f"  warmup={c['warmup']} iterations={c['iterations']} trials={c['trials']}")
    print(f"  cuda   = {cuda_name}")
    print(f"  triton = {triton_name}")

    for name, per_wl in ((cuda_name, cuda), (triton_name, triton)):
        for uuid, r in per_wl.items():
            if r["status"] != "PASSED":
                print(f"\n[not passed] {name} {uuid[:8]}: {r['status']}\n{r.get('log', '')}")

    def us(r):
        return r.get("latency_us") if r and r["status"] == "PASSED" else None

    def cell(x, width=9):
        return f"{x:>{width}.2f}" if x is not None else f"{'-':>{width}}"

    rows = []
    for uuid in cuda.keys() | triton.keys():
        axes = (cuda.get(uuid) or triton.get(uuid))["axes"]
        rows.append((axes.get("max_num_pages", 0), axes.get("batch_size", 0), uuid,
                     us(cuda.get(uuid)), us(triton.get(uuid))))
    rows.sort()

    print("\nPer workload (us):")
    print(f"  {'uuid8':<8}  {'batch':>5}  {'pages':>5}  {'plan':<10}  "
          f"{'cuda':>9}  {'triton':>9}  {'triton/cuda':>11}")
    for pages, batch, uuid, a, b in rows:
        ratio = b / a if a and b else None
        print(f"  {uuid[:8]:<8}  {batch:>5}  {pages:>5}  {_plan(pages):<10}  "
              f"{cell(a)}  {cell(b)}  {cell(ratio, 11)}")

    print("\nPer dispatch plan (mean latency, us):")
    print(f"  {'plan':<10}  {'n':>4}  {'cuda ok':>7}  {'triton ok':>9}  "
          f"{'cuda':>8}  {'triton':>8}  {'triton/cuda':>11}")
    for plan in PLANS + ("all",):
        group = [r for r in rows if plan == "all" or _plan(r[0]) == plan]
        if not group:
            continue
        both = [(a, b) for *_, a, b in group if a and b]
        mean_a = statistics.mean(a for a, _ in both) if both else None
        mean_b = statistics.mean(b for _, b in both) if both else None
        ratio = mean_b / mean_a if both else None
        passed_a = sum(a is not None for *_, a, _ in group)
        passed_b = sum(b is not None for *_, b in group)
        print(f"  {plan:<10}  {len(group):>4}  {passed_a:>7}  {passed_b:>9}  "
              f"{cell(mean_a, 8)}  {cell(mean_b, 8)}  {cell(ratio, 11)}")


def _print_kernel_profile(payload: dict):
    import statistics

    v, c = payload["versions"], payload["config"]
    print(f"\n{payload['definition']}: GPU time per kernel")
    print(f"  {v['gpu']}, torch {v['torch']}, triton {v['triton']}")
    print(f"  warmup={c['warmup']} iterations={c['iterations']}, medians in us")

    for name in payload["solutions"]:
        per_wl = payload["results"].get(name, {})
        print(f"\n{name}")
        print(f"  {'uuid8':<8}  {'batch':>5}  {'pages':>5}  {'span':>6}  kernels (start+duration)")
        rows = sorted(per_wl.items(), key=lambda kv: (kv[1]["axes"].get("max_num_pages", 0),
                                                      kv[1]["axes"].get("batch_size", 0)))
        for uuid, r in rows:
            axes = r["axes"]
            cells = "  ".join(f"{k['start_us']:.2f}+{k['duration_us']:.2f}" for k in r["kernels"])
            print(f"  {uuid[:8]:<8}  {axes.get('batch_size', 0):>5}  "
                  f"{axes.get('max_num_pages', 0):>5}  {r['span_us']:>6.2f}  {cells}")

        print(f"\n  {'plan':<10}  {'n':>4}  {'span':>6}  mean duration per kernel")
        for plan in PLANS:
            group = [r for r in per_wl.values() if _plan(r["axes"].get("max_num_pages", 0)) == plan]
            if not group:
                continue
            count = min(len(r["kernels"]) for r in group)
            means = [statistics.mean(r["kernels"][i]["duration_us"] for r in group)
                     for i in range(count)]
            span = statistics.mean(r["span_us"] for r in group)
            print(f"  {plan:<10}  {len(group):>4}  {span:>6.2f}  "
                  + "  ".join(f"{m:.2f}" for m in means))
        names = {k["name"] for r in per_wl.values() for k in r["kernels"]}
        print("  kernel names: " + ", ".join(sorted(n[:60] for n in names)))


@app.local_entrypoint()
def main(smoke: bool = False, quick: bool = False, per_plan: int = 0,
         warmup: int = 3, iterations: int = 100, trials: int = 5,
         kernels: bool = False, json_out: str = ""):
    """Pack both solutions and benchmark them side by side on Modal B200.

    --smoke     one workload per dispatch plan, one iteration (correctness only)
    --quick     4 workloads per plan at warmup=2 / iterations=25 / trials=2
    default     every workload at the contest settings
    --kernels   trace per-kernel GPU time instead of running the benchmark
    --json-out  also write the raw results to this file
    """
    import json
    if quick:
        per_plan = per_plan or 4
        warmup, iterations, trials = 2, 25, 2

    solutions = _pack_solutions()
    for s in solutions:
        print(f"Packed {s.name} ({s.spec.language}, {len(s.sources)} files)")
    if TRITON_VERSION:
        print(f"Triton override: {TRITON_VERSION}")

    if kernels:
        payload = run_kernel_profile.remote(
            solutions, per_plan=1 if smoke else per_plan, warmup=warmup,
            iterations=5 if smoke else iterations)
        _print_kernel_profile(payload)
    else:
        payload = run_benchmark.remote(solutions, smoke=smoke, per_plan=per_plan,
                                       warmup=warmup, iterations=iterations, trials=trials)
        _print_report(payload)
    if json_out:
        Path(json_out).write_text(json.dumps(payload, indent=1))
        print(f"\nSaved {json_out}")
