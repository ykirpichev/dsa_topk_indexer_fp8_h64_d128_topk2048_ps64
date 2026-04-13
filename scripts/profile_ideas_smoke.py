#!/usr/bin/env python3
"""
Apply one of 10 kernel/binding variants from git HEAD, run Modal smoke, print geomean.

Usage:
  python3 scripts/profile_ideas_smoke.py --list
  python3 scripts/profile_ideas_smoke.py --id 1 --dry-run
  python3 scripts/profile_ideas_smoke.py --id 1   # applies + modal smoke (slow)
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
K = ROOT / "solution/cuda/kernel.cu"
B = ROOT / "solution/cuda/binding.py"


def _git_show(path: str) -> str:
    return subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=ROOT, text=True)


def _write(p: Path, s: str) -> None:
    p.write_text(s)


def _parse_geomean(log: str) -> str | None:
    for line in log.splitlines():
        if "Geomean:" in line:
            return line.strip()
    return None


IDEAS: dict[int, tuple[str, str]] = {
    1: ("ptx_o3", "-Xptxas -O3 in nvcc flags"),
    2: ("lb_page", "__launch_bounds__(128,4) on page_transform_batched_kernel"),
    3: ("lb_gather", "__launch_bounds__(128,2) on gather_dequant_kernel"),
    4: ("mask128", "mask_scores_past_seq_len_kernel <<<B, 128>>>"),
    5: ("block256", "BLOCK_T 256 for page_transform"),
    6: ("sum_out", "at::sum_out into empty scores_2d instead of logits.sum(1)"),
    7: ("no_ftz", "remove --ftz and --prec-div (baseline O3 only)"),
    8: ("O2", "compiler -O2 instead of -O3"),
    9: ("combo1", "ptx_o3 + lb_page + mask128"),
    10: ("tf32_bind", "torch.backends.cuda.matmul.allow_tf32 = False in _load_ext"),
}


def patch_id(n: int) -> tuple[str, str]:
    k = _git_show("solution/cuda/kernel.cu")
    b = _git_show("solution/cuda/binding.py")

    def set_jit(name: str) -> None:
        nonlocal b
        b = re.sub(r'name="topk_cuda_cublas[^"]*"', f'name="{name}"', b, count=1)

    def set_flags(flags_inner: str) -> None:
        nonlocal b
        b = re.sub(
            r"extra_cuda_cflags=\[[^\]]*\]",
            f"extra_cuda_cflags={flags_inner}",
            b,
            count=1,
        )

    if n == 1:
        set_flags("['-O3', '--expt-relaxed-constexpr', '--ftz=true', '--prec-div=false', '-Xptxas', '-O3']")
        set_jit("topk_cuda_idea01")
    elif n == 2:
        k = k.replace(
            "__global__ void page_transform_batched_kernel(",
            "__global__ void __launch_bounds__(128, 4) page_transform_batched_kernel(",
            1,
        )
        set_jit("topk_cuda_idea02")
    elif n == 3:
        k = k.replace(
            "__global__ void gather_dequant_kernel(",
            "__global__ void __launch_bounds__(128, 2) gather_dequant_kernel(",
            1,
        )
        set_jit("topk_cuda_idea03")
    elif n == 4:
        k = k.replace(
            "mask_scores_past_seq_len_kernel<<<B, 256, 0, stream>>>",
            "mask_scores_past_seq_len_kernel<<<B, 128, 0, stream>>>",
            1,
        )
        set_jit("topk_cuda_idea04")
    elif n == 5:
        k = re.sub(r"constexpr int BLOCK_T = \d+;", "constexpr int BLOCK_T = 256;", k, count=1)
        set_jit("topk_cuda_idea05")
    elif n == 6:
        old = """    // One batched reduction over heads (replaces B× sum_out on [H,sl] slices in profiler).
    auto scores_2d = logits.sum(/*dim=*/1);"""
        new = """    // One batched reduction over heads — sum_out into prealloc (avoids sum() temp)
    torch::Tensor scores_2d = torch::empty({B, S}, logits.options());
    at::sum_out(scores_2d, logits, /*dim=*/1, /*keepdim=*/false);"""
        if old not in k:
            raise RuntimeError("sum_out patch: pattern not found")
        k = k.replace(old, new, 1)
        set_jit("topk_cuda_idea06")
    elif n == 7:
        set_flags("['-O3', '--expt-relaxed-constexpr']")
        set_jit("topk_cuda_idea07")
    elif n == 8:
        set_flags("['-O2', '--expt-relaxed-constexpr', '--ftz=true', '--prec-div=false']")
        set_jit("topk_cuda_idea08")
    elif n == 9:
        set_flags("['-O3', '--expt-relaxed-constexpr', '--ftz=true', '--prec-div=false', '-Xptxas', '-O3']")
        k = k.replace(
            "__global__ void page_transform_batched_kernel(",
            "__global__ void __launch_bounds__(128, 4) page_transform_batched_kernel(",
            1,
        )
        k = k.replace(
            "mask_scores_past_seq_len_kernel<<<B, 256, 0, stream>>>",
            "mask_scores_past_seq_len_kernel<<<B, 128, 0, stream>>>",
            1,
        )
        set_jit("topk_cuda_idea09")
    elif n == 10:
        b = b.replace(
            "    if _ext is None:\n        _ext = torch.utils.cpp_extension.load(",
            "    if _ext is None:\n"
            "        import torch as _torch\n"
            "        _torch.backends.cuda.matmul.allow_tf32 = False\n"
            "        _torch.backends.cudnn.allow_tf32 = False\n"
            "        _ext = torch.utils.cpp_extension.load(",
            1,
        )
        set_jit("topk_cuda_idea10")
    else:
        raise ValueError(f"unknown id {n}")

    return k, b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, default=0)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--apply-only", action="store_true")
    args = ap.parse_args()

    if args.list:
        for i, (name, desc) in IDEAS.items():
            print(f"{i:2d}  {name:12s}  {desc}")
        return

    if args.id < 1 or args.id > 10:
        print("Use --id 1..10 or --list", file=sys.stderr)
        sys.exit(1)

    name, desc = IDEAS[args.id]
    print(f"=== Idea {args.id}: {name} — {desc} ===")

    k, b = patch_id(args.id)
    if args.dry_run:
        print(k[:800])
        print("...")
        print(b)
        return

    _write(K, k)
    _write(B, b)
    print(f"Wrote {K} and {B}")

    if args.apply_only:
        return

    print("Running FIB_MODAL_SMOKE=1 modal run scripts/run_modal.py ...")
    import os

    r = subprocess.run(
        ["modal", "run", "scripts/run_modal.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "FIB_MODAL_SMOKE": "1"},
    )
    log = (r.stdout or "") + (r.stderr or "")
    gm = _parse_geomean(log)
    print(gm or log[-2000:])
    if r.returncode != 0:
        print(log[-4000:], file=sys.stderr)
        sys.exit(r.returncode)


if __name__ == "__main__":
    main()
