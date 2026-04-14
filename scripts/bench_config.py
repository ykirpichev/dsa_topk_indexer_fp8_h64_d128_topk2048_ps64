"""Shared BenchmarkConfig defaults for local and Modal runners."""

import os

from flashinfer_bench import BenchmarkConfig

# Contest trace JSON (definitions / workloads) does not specify rtol/atol. The
# evaluation stack uses flashinfer_bench; its BenchmarkConfig defaults apply
# (see flashinfer_bench.bench.config: rtol=atol=1e-2, warmup=10, iters=50, trials=3).
_FIB = BenchmarkConfig()


def default_benchmark_config() -> BenchmarkConfig:
    """Defaults track installed flashinfer_bench; override any field via FIB_* env."""
    return BenchmarkConfig(
        warmup_runs=int(os.environ.get("FIB_WARMUP_RUNS", str(_FIB.warmup_runs))),
        iterations=int(os.environ.get("FIB_ITERATIONS", str(_FIB.iterations))),
        num_trials=int(os.environ.get("FIB_NUM_TRIALS", str(_FIB.num_trials))),
        rtol=float(os.environ.get("FIB_RTOL", str(_FIB.rtol))),
        atol=float(os.environ.get("FIB_ATOL", str(_FIB.atol))),
    )
