# AGENTS.md

## Cursor Cloud specific instructions

### Project Overview

This is a FlashInfer AI Kernel Generation Contest starter kit (MLSys 2026). Participants implement GPU kernels (Triton or CUDA) and benchmark them using `flashinfer-bench`. See `README.md` for full documentation.

### Key Services

| Service | How to verify | Notes |
|---------|--------------|-------|
| `flashinfer-bench` Python API | `python3 -c "from flashinfer_bench import BuildSpec; print('OK')"` | Core dependency; installed via pip |
| `flashinfer-bench` CLI | `flashinfer-bench --help` | Requires `~/.local/bin` on PATH |
| `modal` CLI | `modal --version` | Requires `~/.local/bin` on PATH; needs `modal setup` for auth |
| Solution packing | `python3 scripts/pack_solution.py` | Requires correct `entry_point` format in `config.toml` (see below) |

### Non-obvious Caveats

- **PATH**: pip installs CLI tools to `~/.local/bin`. If `flashinfer-bench` CLI is not found, run `export PATH="$HOME/.local/bin:$PATH"`.
- **config.toml `entry_point` format**: The template ships with `entry_point = "kernel"`, but `flashinfer-bench` v0.1.2+ requires `<file_path>::<function_name>` format (e.g., `kernel.py::kernel`). Using the old format causes a Pydantic validation error. When using the Python API directly (as shown in README), specify the correct format.
- **No GPU on Cloud Agent VMs**: `run_local.py` and `run_modal.py` require CUDA GPUs. On CPU-only VMs, you can still validate the environment by packing solutions and using the `flashinfer_bench` Python API.
- **No linter/test suite**: This repo has no configured linter, test runner, or CI. Validation is done by running `pack_solution.py` and the benchmark scripts.
- **Python 3.12 required**: The `flashinfer-bench` package targets Python 3.12.
