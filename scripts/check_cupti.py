"""
Verify that CUPTI timing is actually active on the Modal B200 worker.

`flashinfer.testing.bench_gpu_time_with_cupti` degrades silently: if
cupti-python is missing, too old, or the CUPTI activity buffers come back
empty, it emits a UserWarning and falls back to CUDA-event timing, which
adds ~3-5 us of event/launch overhead per measurement. On this kernel the
fast-path bucket is ~2.3 us, so a silent fallback roughly triples the
reported latency and makes A/B comparisons meaningless.

This script runs on the same image as scripts/run_modal_compare.py and
reports:
  1. cupti-python version + CUDA/driver versions,
  2. whether a real bench_gpu_time_with_cupti call falls back (warnings are
     captured, not printed to /dev/null),
  3. tiny-kernel timing: CUPTI should be well below CUDA events (proves the
     launch overhead is excluded),
  4. large-kernel timing: CUPTI should agree with CUDA events within a few
     percent (proves the CUPTI numbers are calibrated, not garbage),
  5. that flashinfer_bench's timing path is the CUPTI one.

Usage:
    modal run scripts/check_cupti.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

app = modal.App("flashinfer-bench-cupti-check")

# Image spec is byte-identical to scripts/run_modal_compare.py so Modal
# reuses the cached layers instead of rebuilding.
image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:latest")
    .apt_install("git")
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .pip_install("wheel", "setuptools")
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


@app.function(image=image, gpu="B200:1", timeout=1200)
def check() -> dict:
    import inspect
    import statistics
    import warnings

    import torch

    try:
        from flashinfer.testing import bench_gpu_time_with_cuda_event, bench_gpu_time_with_cupti
    except ImportError:
        from flashinfer.testing.utils import (
            bench_gpu_time_with_cuda_event,
            bench_gpu_time_with_cupti,
        )

    report: dict = {}

    # ---- 1. Environment ---------------------------------------------------
    try:
        from importlib.metadata import version as _version

        report["cupti_python_version"] = _version("cupti-python")
    except Exception as e:  # pragma: no cover - diagnostic path
        report["cupti_python_version"] = f"<missing: {e}>"

    try:
        from cupti import cupti as _cupti  # noqa: F401

        report["cupti_import"] = "ok"
    except Exception as e:  # pragma: no cover - diagnostic path
        report["cupti_import"] = f"FAILED: {e}"

    report["torch_version"] = torch.__version__
    report["torch_cuda"] = torch.version.cuda
    report["device_name"] = torch.cuda.get_device_name(0)
    report["capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    try:
        report["driver_version"] = torch.cuda.driver_version()
    except Exception:
        report["driver_version"] = "n/a"

    # ---- 2. flashinfer_bench timing path ----------------------------------
    try:
        from flashinfer_bench.bench import timing as fib_timing

        src = inspect.getsource(fib_timing)
        report["fib_uses_cupti"] = "bench_gpu_time_with_cupti" in src
        report["fib_timing_file"] = fib_timing.__file__
    except Exception as e:  # pragma: no cover - diagnostic path
        report["fib_uses_cupti"] = f"FAILED: {e}"

    # ---- 3+4. Measure tiny and large kernels both ways --------------------
    # Tiny: a small elementwise add, a few microseconds of device time. This
    # is the regime our fast path lives in, where event overhead dominates.
    tiny_a = torch.randn(1024, device="cuda", dtype=torch.float16)
    tiny_b = torch.randn(1024, device="cuda", dtype=torch.float16)
    tiny_out = torch.empty_like(tiny_a)

    def tiny_fn(a, b, out):
        torch.add(a, b, out=out)

    # Large: a matmul with milliseconds of device time, where CUPTI and CUDA
    # events must agree because launch overhead is negligible.
    big = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    big_out = torch.empty(4096, 4096, device="cuda", dtype=torch.bfloat16)

    def big_fn(x, out):
        torch.mm(x, x, out=out)

    def _timed(fn, args, use_cupti: bool, trials: int = 3):
        """Return (median_us_per_trial_list, fallback_warnings)."""
        medians = []
        fallbacks = []
        for _ in range(trials):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if use_cupti:
                    times = bench_gpu_time_with_cupti(
                        fn=fn,
                        dry_run_iters=3,
                        repeat_iters=100,
                        input_args=tuple(args),
                        cold_l2_cache=True,
                        use_cuda_graph=False,
                    )
                else:
                    times = bench_gpu_time_with_cuda_event(
                        fn=fn,
                        dry_run_iters=3,
                        repeat_iters=100,
                        input_args=tuple(args),
                        cold_l2_cache=True,
                    )
                for w in caught:
                    msg = str(w.message)
                    if "Falling back" in msg or "CUPTI" in msg:
                        fallbacks.append(msg)
            medians.append(statistics.median(times) * 1000.0)  # ms -> us
        return medians, fallbacks

    tiny_cupti, tiny_fallbacks = _timed(tiny_fn, (tiny_a, tiny_b, tiny_out), use_cupti=True)
    tiny_event, _ = _timed(tiny_fn, (tiny_a, tiny_b, tiny_out), use_cupti=False)
    big_cupti, big_fallbacks = _timed(big_fn, (big, big_out), use_cupti=True)
    big_event, _ = _timed(big_fn, (big, big_out), use_cupti=False)

    report["tiny_cupti_us"] = tiny_cupti
    report["tiny_event_us"] = tiny_event
    report["big_cupti_us"] = big_cupti
    report["big_event_us"] = big_event
    report["fallback_warnings"] = tiny_fallbacks + big_fallbacks

    return report


def _verdict(r: dict) -> tuple[bool, list[str]]:
    """Decide whether CUPTI timing is trustworthy. Returns (ok, notes)."""
    import statistics

    notes = []
    ok = True

    if r.get("cupti_import") != "ok":
        ok = False
        notes.append(f"cupti import failed: {r.get('cupti_import')}")

    major = str(r.get("cupti_python_version", "0")).split(".")[0]
    if not major.isdigit() or int(major) < 13:
        ok = False
        notes.append(f"cupti-python must be >= 13.0.0, got {r.get('cupti_python_version')}")

    if r.get("fallback_warnings"):
        ok = False
        notes.append(f"fell back to CUDA events: {r['fallback_warnings'][0]}")
    else:
        notes.append("no fallback warnings raised during timed calls")

    if r.get("fib_uses_cupti") is not True:
        ok = False
        notes.append(f"flashinfer_bench timing path does not use CUPTI: {r.get('fib_uses_cupti')}")
    else:
        notes.append("flashinfer_bench.bench.timing calls bench_gpu_time_with_cupti")

    tiny_c = statistics.median(r.get("tiny_cupti_us") or [0])
    tiny_e = statistics.median(r.get("tiny_event_us") or [0])
    big_c = statistics.median(r.get("big_cupti_us") or [0])
    big_e = statistics.median(r.get("big_event_us") or [0])

    # CUDA events measure device time plus a fixed per-iteration launch/event
    # overhead; CUPTI measures device time only. So the right check is not
    # "the two agree" but "events minus CUPTI is the same small constant on a
    # 2 us kernel and on an 85 us kernel". A CUPTI path that silently fell
    # back would show a ~0 us offset instead.
    if tiny_c and tiny_e and big_c and big_e:
        off_tiny = tiny_e - tiny_c
        off_big = big_e - big_c

        if off_tiny <= 0.5:
            ok = False
            notes.append(
                f"tiny kernel: CUPTI {tiny_c:.2f} us vs events {tiny_e:.2f} us — "
                "no measurable overhead difference, CUPTI is likely not in use"
            )
        else:
            notes.append(
                f"tiny kernel: CUPTI {tiny_c:.2f} us vs events {tiny_e:.2f} us "
                f"(overhead excluded: {off_tiny:.2f} us)"
            )
            notes.append(
                f"large kernel: CUPTI {big_c:.1f} us vs events {big_e:.1f} us "
                f"(overhead excluded: {off_big:.2f} us)"
            )
            if abs(off_big - off_tiny) <= 3.0:
                notes.append(
                    f"event overhead is a constant {off_tiny:.1f}-{off_big:.1f} us across a "
                    "2 us and an 85 us kernel — CUPTI device time is calibrated"
                )
            else:
                ok = False
                notes.append(
                    f"event overhead is not constant ({off_tiny:.2f} us vs {off_big:.2f} us) "
                    "— timing path is inconsistent"
                )

    # Run-to-run spread on the tiny kernel: high variance means the numbers
    # are not usable for A/B comparisons at the ~2 us scale.
    tiny_vals = r.get("tiny_cupti_us") or []
    if len(tiny_vals) > 1:
        spread = max(tiny_vals) - min(tiny_vals)
        notes.append(f"tiny kernel CUPTI spread across trials: {spread:.2f} us")

    return ok, notes


@app.local_entrypoint()
def main():
    r = check.remote()

    print("\n=== CUPTI environment (Modal B200 worker) ===")
    for key in (
        "device_name",
        "capability",
        "torch_version",
        "torch_cuda",
        "driver_version",
        "cupti_python_version",
        "cupti_import",
        "fib_timing_file",
        "fib_uses_cupti",
    ):
        print(f"  {key:<22} {r.get(key)}")

    print("\n=== Timing (median of 100 iters, 3 trials, us) ===")
    for key in ("tiny_cupti_us", "tiny_event_us", "big_cupti_us", "big_event_us"):
        vals = r.get(key) or []
        print("  {:<16} {}".format(key, ", ".join(f"{v:.2f}" for v in vals)))

    ok, notes = _verdict(r)
    print("\n=== Verdict ===")
    for n in notes:
        print(f"  - {n}")
    print(f"\n  CUPTI timing usable: {'YES' if ok else 'NO'}")
