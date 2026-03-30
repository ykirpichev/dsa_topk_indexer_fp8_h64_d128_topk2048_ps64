"""
GPU profiling on Modal (B200) for the packed CUDA solution.

Uses ``torch.profiler`` (CUDA + CPU) around the same code path as the benchmark
after JIT warmup. Nsight Systems / Nsight Compute are attempted when present but
often fail on cloud GPUs (injection segfaults or incompatible profiler libraries).

Run from repo root (after modal setup + trace volume):

    FIB_MODAL_PROFILE=1 modal run scripts/modal_profile.py

    # Workload index in trace-set order (same order as smoke: first 17 → indices 0..16)
    FIB_MODAL_PROFILE=1 FIB_PROFILE_WORKLOAD_INDEX=16 modal run scripts/modal_profile.py
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
import torch
from flashinfer_bench import Solution, TraceSet
from flashinfer_bench.bench.evaluators.utils import allocate_outputs
from flashinfer_bench.bench.utils import gen_inputs, load_safetensors
from flashinfer_bench.compile import BuilderRegistry

app = modal.App("flashinfer-bench-profile")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:latest", add_python="3.12")
    .pip_install("flashinfer-bench")
)


def _resolve_workload_index(idx_raw: str, n_workloads: int) -> int:
    try:
        idx = int(idx_raw)
    except ValueError:
        idx = 16
    if idx < 0:
        idx = n_workloads + idx
    return max(0, min(idx, n_workloads - 1))


def _try_nsys_optional(tdir: Path, lines: list[str]) -> None:
    """Best-effort Nsight Systems; often fails on Modal (segfault under injection)."""
    nsys = shutil.which("nsys")
    if not nsys:
        lines.append("\n=== nsys: not in PATH ===\n")
        return
    runner = [
        sys.executable,
        "-u",
        "-m",
        "flashinfer_bench.agents._solution_runner",
        "--data-dir",
        str(tdir),
        "--device",
        "cuda:0",
        "--trace-set-path",
        TRACE_SET_PATH,
    ]
    with tempfile.TemporaryDirectory(prefix="fib_nsys_") as ntmp:
        rep = Path(ntmp) / "fib"
        cmd = [nsys, "profile", "--trace=cuda,nvtx", "-o", str(rep), "--force-overwrite", "true"] + runner
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            out = (r.stdout or "") + (r.stderr or "")
            lines.append("\n=== nsys (exit %d, tail) ===\n" % r.returncode)
            lines.append(out[-6000:] if len(out) > 6000 else out)
        except Exception as e:
            lines.append("\n=== nsys failed: %s ===\n" % e)


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def profile_gpu(solution: dict) -> str:
    sol = Solution.model_validate(solution)
    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    if sol.definition not in trace_set.definitions:
        return f"ERROR: definition {sol.definition!r} not in trace set"
    wlist = trace_set.workloads.get(sol.definition, [])
    if not wlist:
        return "ERROR: no workloads"

    idx_default = os.environ.get("FIB_PROFILE_WORKLOAD_INDEX", "16")
    idx = _resolve_workload_index(idx_default, len(wlist))
    workload = wlist[idx].workload
    definition = trace_set.definitions[sol.definition]

    lines: list[str] = []
    lines.append(
        f"Profile: definition={sol.definition} workload_index={idx}/{len(wlist)} uuid={workload.uuid}\n"
    )

    registry = BuilderRegistry.get_instance()
    runnable = registry.build(definition, sol)
    device = "cuda:0"

    safe_tensors = None
    if any(inp.type == "safetensors" for inp in workload.inputs.values()):
        safe_tensors = load_safetensors(definition, workload, Path(TRACE_SET_PATH))

    inputs = gen_inputs(definition, workload, device, safe_tensors)
    outputs = allocate_outputs(definition, inputs, device)

    # JIT + cache warmup (not included in profiler table)
    with torch.no_grad():
        runnable.call_destination_passing(*inputs, *outputs)
    torch.cuda.synchronize()

    # --- PyTorch profiler (reliable on Modal) ---
    repeats = int(os.environ.get("FIB_PROFILE_REPEAT", "4"))
    repeats = max(1, min(repeats, 32))

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=False,
        acc_events=True,
    ) as prof:
        with torch.no_grad():
            for _ in range(repeats):
                runnable.call_destination_passing(*inputs, *outputs)
        torch.cuda.synchronize()

    buf = io.StringIO()
    print(
        prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=40,
        ),
        file=buf,
    )
    lines.append("\n=== torch.profiler: sort_by=cuda_time_total ===\n")
    lines.append(buf.getvalue())

    buf2 = io.StringIO()
    print(
        prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=40,
        ),
        file=buf2,
    )
    lines.append("\n=== torch.profiler: sort_by=self_cuda_time_total ===\n")
    lines.append(buf2.getvalue())

    runnable.cleanup()

    # --- Optional Nsight Systems (often segfaults under injection on cloud GPUs) ---
    if os.environ.get("FIB_MODAL_TRY_NSYS", "").lower() in ("1", "true", "yes"):
        with tempfile.TemporaryDirectory(prefix="fib_prof_nsys_") as tmp:
            tdir = Path(tmp)
            (tdir / "definition.json").write_text(definition.model_dump_json())
            (tdir / "solution.json").write_text(sol.model_dump_json())
            (tdir / "workload.json").write_text(workload.model_dump_json())
            _try_nsys_optional(tdir, lines)
    else:
        lines.append(
            "\n=== nsys: skipped (set FIB_MODAL_TRY_NSYS=1 to attempt; torch.profiler above is primary on Modal) ===\n"
        )

    lines.append(
        "\n=== Nsight Compute (ncu): use a local machine with full CUDA toolkit + matching driver; "
        "Modal often returns LibraryNotLoaded for ncu ===\n"
    )

    return "".join(lines)


@app.local_entrypoint()
def main():
    from scripts.pack_solution import pack_solution

    if os.environ.get("FIB_MODAL_PROFILE", "").lower() not in ("1", "true", "yes"):
        print("Set FIB_MODAL_PROFILE=1 to run GPU profiling on Modal.", file=sys.stderr)
        sys.exit(1)

    path = pack_solution()
    solution = Solution.model_validate_json(path.read_text())
    print("Profiling on Modal B200 (torch.profiler)...")
    print(profile_gpu.remote(solution.model_dump(mode="json")))


if __name__ == "__main__":
    main()
