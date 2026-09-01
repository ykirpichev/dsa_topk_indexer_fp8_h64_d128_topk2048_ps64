"""
Run official `flashinfer-bench run` CLI on Modal B200 (isolated runner, same flags as EVALUATION.md).

Installs repo `solution.json` into the Modal `flashinfer-trace` volume under the standard
solutions/ tree, then runs:

  flashinfer-bench run --local /data ... --use-isolated-runner ...

Usage:
  python scripts/pack_solution.py   # refresh solution.json if needed
  modal run scripts/run_modal_fib_cli.py

Requires one-time dataset upload to the volume (see README):
  modal volume put flashinfer-trace /path/to/mlsys26-contest /data
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Solution
from flashinfer_bench.data import load_json_file
from flashinfer_bench.data.definition import Definition

app = modal.App("flashinfer-bench-cli-eval")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = Path("/data")

# Match scripts/run_modal.py / EVALUATION container (CUDA 13.2 + flashinfer-bench)
image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:latest")
    .apt_install("git")
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .pip_install("wheel", "setuptools")
    .run_commands(
        "git clone --recursive --depth 1 https://github.com/deepseek-ai/DeepGEMM.git /tmp/DeepGEMM",
        "pip install --no-build-isolation /tmp/DeepGEMM",
        "git clone --recursive --depth 1 https://github.com/flashinfer-ai/flashinfer.git /tmp/flashinfer",
        "pip install --no-build-isolation /tmp/flashinfer",
        "git clone --depth 1 https://github.com/flashinfer-ai/flashinfer-bench.git /tmp/flashinfer-bench",
        "pip install /tmp/flashinfer-bench",
    )
)


def _safe_segment(segment: str) -> str:
    if not segment or "/" in segment or "\\" in segment or segment in (".", ".."):
        raise ValueError(f"Invalid path segment: {segment!r}")
    return segment


def _solution_install_path(root: Path, solution: Solution, op_type: str) -> Path:
    return (
        root
        / "solutions"
        / _safe_segment(solution.author)
        / _safe_segment(op_type)
        / _safe_segment(solution.definition)
        / f"{_safe_segment(solution.name)}.json"
    )


def _remove_prior_solution_files(root: Path, solution_name: str) -> None:
    """Allow re-run: TraceSet rejects duplicate solution names across the tree."""
    d = root / "solutions"
    if not d.is_dir():
        return
    for p in d.rglob(f"{solution_name}.json"):
        try:
            p.unlink()
        except OSError:
            pass


@app.function(
    image=image,
    gpu="B200:1",
    timeout=86400,  # full `--fresh` runs can take many hours (1000+ workloads)
    volumes={str(TRACE_SET_PATH): trace_volume},
)
def run_flashinfer_bench_cli(solution_json: str, resume: bool = True) -> dict:
    """Inject solution.json into /data and run `flashinfer-bench run` (isolated runner)."""
    root = TRACE_SET_PATH
    solution = Solution.model_validate_json(solution_json)

    matches = sorted(root.glob(f"definitions/**/{solution.definition}.json"))
    if not matches:
        raise FileNotFoundError(
            f"No definitions/**/{solution.definition}.json under {root}. "
            "Upload the contest TraceSet to the Modal volume (see README)."
        )

    definition = load_json_file(Definition, matches[0])
    _remove_prior_solution_files(root, solution.name)

    out_path = _solution_install_path(root, solution, definition.op_type)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(solution.model_dump_json(indent=2), encoding="utf-8")

    trace_volume.commit()

    cmd = [
        "flashinfer-bench",
        "run",
        "--local",
        str(root),
        "--definitions",
        solution.definition,
        "--solutions",
        solution.name,
        "--save-results",
        "--use-isolated-runner",
        "--log-level",
        "INFO",
        "--timeout",
        "300",
    ]
    if resume:
        cmd.append("--resume")
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(root), env=env)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "solution_name": solution.name,
        "definition": solution.definition,
        "solution_path": str(out_path),
    }


@app.local_entrypoint()
def main(fresh: bool = False):
    """
    :param fresh: If True, omit `--resume` so workloads re-run (slow; use after code changes).
                  Default False matches EVALUATION.md (resume skips finished jobs).
    """
    solution_path = PROJECT_ROOT / "solution.json"
    if not solution_path.is_file():
        print(f"Missing {solution_path}. Run: python scripts/pack_solution.py", file=sys.stderr)
        raise SystemExit(1)

    raw = solution_path.read_text(encoding="utf-8")
    resume = not fresh
    print(
        f"Using solution.json ({len(raw)} bytes) -> Modal `flashinfer-bench run` "
        f"({'no --resume (full run)' if fresh else 'with --resume'})"
    )

    result = run_flashinfer_bench_cli.remote(raw, resume=resume)
    print("Command:", " ".join(result["cmd"]))
    print("Installed at:", result.get("solution_path"))
    print("exit:", result["returncode"])
    if result["stdout"]:
        print("--- stdout ---")
        print(result["stdout"])
    if result["stderr"]:
        print("--- stderr ---")
        print(result["stderr"])

    if result["returncode"] != 0:
        raise SystemExit(result["returncode"])
    print("Done.")


if __name__ == "__main__":
    main()
