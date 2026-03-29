"""Build a FlashInfer-Bench Solution for the Python CuTe DSL kernel under solution/python/."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from flashinfer_bench import BuildSpec, Solution, SourceFile

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    path = PROJECT_ROOT / "config.toml"
    with open(path, "rb") as f:
        return tomllib.load(f)


def build_python_solution() -> Solution:
    """Load solution/python/kernel.py and config.toml into a Solution object."""
    config = load_config()
    sol = config["solution"]
    build = config["build"]

    kernel_path = PROJECT_ROOT / "solution" / "python" / "kernel.py"
    if not kernel_path.is_file():
        raise FileNotFoundError(f"Missing Python kernel: {kernel_path}")

    spec = BuildSpec(
        language="python",
        target_hardware=["cuda"],
        entry_point=build["entry_point"],
        destination_passing_style=build.get("destination_passing_style", True),
    )

    return Solution(
        name=sol["name"],
        definition=sol["definition"],
        author=sol["author"],
        spec=spec,
        sources=[
            SourceFile(path="kernel.py", content=kernel_path.read_text(encoding="utf-8")),
        ],
    )


def write_solution_json(output_path: Path | None = None) -> Path:
    if output_path is None:
        output_path = PROJECT_ROOT / "solution.json"
    solution = build_python_solution()
    output_path.write_text(solution.model_dump_json(indent=2))
    return output_path


def main() -> None:
    out = write_solution_json()
    print(f"Wrote {out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
