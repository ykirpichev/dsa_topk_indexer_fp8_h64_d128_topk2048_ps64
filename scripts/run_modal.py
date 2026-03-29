"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks
on NVIDIA B200 GPUs via Modal.

Smoke run (few workloads, shorter timing — still measures speedup vs reference)::

    FIB_MODAL_SMOKE=1 modal run scripts/run_modal.py

Cap workloads on a full-style benchmark (default timing config)::

    FIB_MODAL_MAX_WORKLOADS=16 modal run scripts/run_modal.py

CuTe DSL gather/page JIT for ``solution/python/kernel.py``::

    FIB_MODAL_DSA_CUTE_JIT=1 modal run scripts/run_modal.py

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

# Track-specific constants (dsa_topk_indexer_fp8_h64_d128_topk2048_ps64)
_H       = 64    # number of index heads
_D       = 128   # head dimension (FP8 bytes per head per token)
_PS      = 64    # page size (tokens per page)
_HDS     = 132   # bytes per token in KV cache (128 FP8 + 4 float32 scale)
_K_TOPK  = 2048  # topk
_B200_HBM_BW_TBS = 8.0  # B200 HBM3e peak bandwidth (TB/s)

image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:latest", add_python="3.12")
    .pip_install("flashinfer-bench", "nvidia-cutlass-dsl==4.4.2")
)


def _load_seq_lens(trace_root: str, workload) -> list[int] | None:
    """Load seq_lens tensor from the workload's safetensors file."""
    try:
        import safetensors.torch
        sl_spec = workload.inputs.get("seq_lens", {})
        if isinstance(sl_spec, dict) and sl_spec.get("type") == "safetensors":
            path = Path(trace_root) / sl_spec["path"]
            st = safetensors.torch.load_file(str(path))
            return st[sl_spec["tensor_key"]].tolist()
    except Exception:
        pass
    return None


def _padded_s(max_sl: int) -> int:
    """Padded token length along K (same as kernel): ceil(max_sl / PS) * PS."""
    if max_sl <= 0:
        return 0
    pages = (max_sl + _PS - 1) // _PS
    return pages * _PS


def _roofline_ms_logical_tokens(seq_lens: list[int]) -> float:
    """HBM lower bound assuming one K-cache read per *logical* token only (optimistic)."""
    B = len(seq_lens)
    T = sum(seq_lens)
    k_bytes = T * _HDS
    q_bytes = B * _H * _D
    w_bytes = B * _H * 4
    out_bytes = sum(min(_K_TOPK, sl) * 4 for sl in seq_lens)
    total = k_bytes + q_bytes + w_bytes + out_bytes
    return total / (_B200_HBM_BW_TBS * 1e12) * 1e3


def _roofline_ms_impl_staging(seq_lens: list[int]) -> float:
    """HBM-centric bound closer to this implementation: padded gather + K f32 staging + logits.

    Counts:
      - K cache: B * S_pad * _HDS (Triton touches every padded slot, not only sum(seq_lens))
      - K_batched f32: write + read for GEMM ≈ 2 * B * S_pad * _D * 4
      - Q (fp32 for bmm): B * _H * _D * 4
      - logits f32: B * _H * S_pad * 4 (bmm output; subsequent ops add more traffic)
      - weights + topk output as in logical model
    """
    B = len(seq_lens)
    if B == 0:
        return 0.0
    max_sl = max(seq_lens)
    S_pad = _padded_s(max_sl)
    q_bytes = B * _H * _D * 4
    w_bytes = B * _H * 4
    out_bytes = sum(min(_K_TOPK, sl) * 4 for sl in seq_lens)
    k_cache_bytes = B * S_pad * _HDS
    k_staging_bytes = 2 * B * S_pad * _D * 4
    logits_bytes = B * _H * S_pad * 4
    total = k_cache_bytes + k_staging_bytes + logits_bytes + q_bytes + w_bytes + out_bytes
    return total / (_B200_HBM_BW_TBS * 1e12) * 1e3


@app.function(image=image, gpu="B200:1", timeout=7200, volumes={TRACE_SET_PATH: trace_volume})
def run_benchmark(
    solution: Solution, smoke: bool = False, max_workloads: int | None = None
) -> dict:
    """Run benchmark on Modal B200 and return results."""
    # Forward to isolated solution workers (Python/CuTe path)
    if os.environ.get("FIB_MODAL_DSA_CUTE_JIT", "").lower() in ("1", "true", "yes"):
        os.environ["DSA_CUTE_JIT"] = "1"
    if os.environ.get("FIB_MODAL_DSA_FP8_TMMA_MM", "").lower() in ("1", "true", "yes"):
        os.environ["DSA_FP8_TMMA_MM"] = "1"
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

            # Workload metadata for roofline
            wl = trace.workload
            axes = wl.axes if hasattr(wl, "axes") else wl.get("axes", {})
            batch_size = axes.get("batch_size", 1) if isinstance(axes, dict) else getattr(axes, "batch_size", 1)
            seq_lens = _load_seq_lens(TRACE_SET_PATH, wl if hasattr(wl, "inputs") else type("W", (), {"inputs": wl.get("inputs", {})})())
            if seq_lens is not None:
                entry["seq_lens"] = seq_lens
            else:
                # Fallback: approximate from axes
                max_num_pages = axes.get("max_num_pages", 0) if isinstance(axes, dict) else getattr(axes, "max_num_pages", 0)
                entry["seq_lens"] = [max_num_pages * _PS] * batch_size

            results[definition.name][trace.workload.uuid] = entry

    return results


def print_results(results: dict):
    """Print benchmark results with roofline analysis table."""
    import math

    for def_name, traces in results.items():
        print(f"\n{def_name}:")

        # Collect rows for table
        rows = []
        for workload_uuid, result in traces.items():
            status = result.get("status", "UNKNOWN")
            latency = result.get("latency_ms")
            speedup = result.get("speedup_factor")
            abs_err = result.get("max_abs_error")
            seq_lens = result.get("seq_lens", [])

            T = sum(seq_lens) if seq_lens else None
            max_sl = max(seq_lens) if seq_lens else None
            S_pad = _padded_s(max_sl) if max_sl is not None else None
            rf_log = _roofline_ms_logical_tokens(seq_lens) if seq_lens else None
            rf_impl = _roofline_ms_impl_staging(seq_lens) if seq_lens else None
            pct_peak = (rf_impl / latency * 100) if (rf_impl and latency) else None
            # correctness match: fraction of topk indices in common with reference
            # abs_err=0 means exact match → 1.0; use directly
            match = (1.0 - abs_err) if abs_err is not None else None

            rows.append((T, S_pad, latency, speedup, match, rf_log, rf_impl, pct_peak, status))

        # Sort by T ascending
        rows.sort(key=lambda r: (r[0] is None, r[0]))

        # RF_log = optimistic HBM (logical tokens only); RF_impl = padded gather + K f32 + logits
        print(
            f"  {'T':>8} | {'S_pad':>6} | {'Lat(ms)':>7} | {'Spdup':>7} | {'Match':>6} | "
            f"{'RF_Lµs':>7} | {'RF_Iµs':>7} | {'%Pk_I':>8} | Status"
        )
        print(
            f"  {'-'*8}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}-+-"
            f"{'-'*7}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}"
        )

        speedups = []
        for T, S_pad, latency, speedup, match, rf_log, rf_impl, pct_peak, status in rows:
            t_str = f"{T:8d}" if T is not None else f"{'?':>8}"
            s_str = f"{S_pad:6d}" if S_pad is not None else f"{'?':>6}"
            lat_str = f"{latency:7.3f}" if latency is not None else f"{'?':>7}"
            sp_str = f"{speedup:6.2f}x" if speedup is not None else f"{'?':>7}"
            mt_str = f"{match:6.4f}" if match is not None else f"{'?':>6}"
            rl_us = rf_log * 1000 if rf_log is not None else None
            ri_us = rf_impl * 1000 if rf_impl is not None else None
            rl_str = f"{rl_us:7.3f}" if rl_us is not None else f"{'?':>7}"
            ri_str = f"{ri_us:7.3f}" if ri_us is not None else f"{'?':>7}"
            if pct_peak is None:
                pk_str = f"{'?':>8}"
            elif pct_peak < 0.1:
                pk_str = f"{pct_peak:7.4f}%"
            else:
                pk_str = f"{pct_peak:7.1f}%"
            print(
                f"  {t_str} | {s_str} | {lat_str} | {sp_str} | {mt_str} | "
                f"{rl_str} | {ri_str} | {pk_str} | {status}"
            )
            if speedup is not None:
                speedups.append(speedup)

        if speedups:
            geomean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
            passed = sum(1 for *_, st in rows if st == "PASSED")
            total = len(rows)
            print(f"\n  Geomean: {geomean:.2f}x ({passed}/{total} PASSED)")
            print(
                "\n  Roofline: RF_L = logical-token K reads only; RF_I = padded K-cache + "
                "2×K_f32 staging + logits (closer to this kernel). Low %Pk_I ⇒ compute-bound "
                "or unmodeled traffic (e.g. sort/topk, block_table, Python loop)."
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
