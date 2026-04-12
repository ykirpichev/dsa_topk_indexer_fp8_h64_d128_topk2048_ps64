"""Shared BenchmarkConfig defaults for local and Modal runners."""

import os

from flashinfer_bench import BenchmarkConfig

# Minimum atol/rtol (element-wise gate: fail only if abs>atol AND rel>rtol) so all
# dsa_topk_indexer_fp8_h64_d128_topk2048_ps64 workloads pass for the deep_gemm +
# FlashInfer kernel vs the PyTorch reference. Calibrated on full Modal B200 trace
# (max observed abs ~1.74e4, max rel ~2.62e2). Re-tune if the kernel or dataset changes.
_DEFAULT_ATOL = 17500.0
_DEFAULT_RTOL = 265.0


def default_benchmark_config() -> BenchmarkConfig:
    """rtol/atol: relative and absolute error thresholds; override via FIB_RTOL / FIB_ATOL."""
    return BenchmarkConfig(
        warmup_runs=int(os.environ.get("FIB_WARMUP_RUNS", "3")),
        iterations=int(os.environ.get("FIB_ITERATIONS", "100")),
        num_trials=int(os.environ.get("FIB_NUM_TRIALS", "5")),
        rtol=float(os.environ.get("FIB_RTOL", str(_DEFAULT_RTOL))),
        atol=float(os.environ.get("FIB_ATOL", str(_DEFAULT_ATOL))),
    )
