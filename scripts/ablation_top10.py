#!/usr/bin/env python3
"""
Apply one of several small kernel/binding patches and optionally run Modal smoke.

Usage:
  python3 scripts/ablation_top10.py --id 0 --dry-run    # print patch only
  python3 scripts/ablation_top10.py --id 3              # apply patch 3 to repo

Patch IDs (see OPTIMIZATION_LEARNINGS.md "Ablation top-10"):
  0 = restore baseline (no extra flags, BLOCK_T=128)
  1 = nvcc -Xptxas -O3
  2 = page_transform BLOCK_T=256
  3 = page_transform BLOCK_T=512
  4 = mask_scores + mask gather threads = 512 (both kernels)
  5 = K_T contiguous before bmm (explicit)
  6 = extra_cuda: --ftz=true --prec-div=false
  7 = BLOCK_T=128 + PTX O3 (combo)
  8 = gather grid: use 256 threads (D=128, pad) — NOT IMPLEMENTED (needs kernel change)
  9 = JIT name bump only (force rebuild, no logic)
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KERNEL = ROOT / "solution" / "cuda" / "kernel.cu"
BINDING = ROOT / "solution" / "cuda" / "binding.py"


def _git_show(path_in_repo: str) -> str:
    return subprocess.check_output(
        ["git", "show", f"HEAD:{path_in_repo}"],
        cwd=ROOT,
        text=True,
    )


def read(p: Path) -> str:
    return p.read_text()


def write(p: Path, s: str) -> None:
    p.write_text(s)


def patch_block_t(text: str, val: int) -> str:
    return re.sub(
        r"constexpr int BLOCK_T = \d+;",
        f"constexpr int BLOCK_T = {val};",
        text,
        count=1,
    )


def patch_mask_threads(text: str, val: int) -> str:
    # mask_scores and gather launch: <<<B, 256 -> val
    t = re.sub(
        r"mask_scores_past_seq_len_kernel<<<B, \d+",
        f"mask_scores_past_seq_len_kernel<<<B, {val}",
        text,
        count=1,
    )
    return t


def patch_binding_flags(text: str, extra: list[str] | None, jit_name: str) -> str:
    if extra is None:
        flags = '["-O3", "--expt-relaxed-constexpr"]'
    else:
        inner = ", ".join(repr(x) for x in ["-O3", "--expt-relaxed-constexpr"] + extra)
        flags = f"[{inner}]"
    text = re.sub(
        r'extra_cuda_cflags=\[[^\]]*\]',
        f"extra_cuda_cflags={flags}",
        text,
        count=1,
    )
    text = re.sub(
        r'name="topk_cuda_cublas[^"]*"',
        f'name="{jit_name}"',
        text,
        count=1,
    )
    return text


def patch_k_t_contiguous(text: str) -> str:
    old = """    auto q_float = q_index_fp8.to(torch::kFloat32);  // [B, H, D]
    auto logits  = torch::bmm(q_float, K_batched.transpose(1, 2)).contiguous();  // [B, H, S]"""
    new = """    auto q_float = q_index_fp8.to(torch::kFloat32);  // [B, H, D]
    auto K_T = K_batched.transpose(1, 2).contiguous();
    auto logits  = torch::bmm(q_float, K_T).contiguous();  // [B, H, S]"""
    if old not in text:
        raise SystemExit("patch K_T: pattern not found")
    return text.replace(old, new, 1)


def apply_id(pid: int) -> tuple[str, str]:
    k = _git_show("solution/cuda/kernel.cu")
    b = _git_show("solution/cuda/binding.py")
    if pid == 0:
        k = patch_block_t(k, 128)
        b = patch_binding_flags(b, None, "topk_cuda_cublas_ab0")
    elif pid == 1:
        k = patch_block_t(k, 128)
        b = patch_binding_flags(b, ["-Xptxas", "-O3"], "topk_cuda_cublas_ab1")
    elif pid == 2:
        k = patch_block_t(k, 256)
        b = patch_binding_flags(b, None, "topk_cuda_cublas_ab2")
    elif pid == 3:
        k = patch_block_t(k, 512)
        b = patch_binding_flags(b, None, "topk_cuda_cublas_ab3")
    elif pid == 4:
        k = patch_block_t(k, 128)
        k = patch_mask_threads(k, 512)
        b = patch_binding_flags(b, None, "topk_cuda_cublas_ab4")
    elif pid == 5:
        k = patch_k_t_contiguous(k)
        k = patch_block_t(k, 128)
        b = patch_binding_flags(b, None, "topk_cuda_cublas_ab5")
    elif pid == 6:
        k = patch_block_t(k, 128)
        b = patch_binding_flags(
            b, ["--ftz=true", "--prec-div=false"], "topk_cuda_cublas_ab6"
        )
    elif pid == 7:
        k = patch_block_t(k, 256)
        b = patch_binding_flags(b, ["-Xptxas", "-O3"], "topk_cuda_cublas_ab7")
    elif pid == 8:
        raise SystemExit("id 8 not implemented (gather thread model)")
    elif pid == 9:
        k = patch_block_t(k, 128)
        b = patch_binding_flags(b, None, "topk_cuda_cublas_ab9_rebuild")
    else:
        raise SystemExit(f"unknown id {pid}")
    return k, b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    k, b = apply_id(args.id)
    if args.dry_run:
        print("--- kernel.cu (excerpt) ---")
        for line in k.splitlines():
            if "BLOCK_T" in line or "bmm" in line or "K_T" in line:
                print(line)
        print("--- binding ---")
        for line in b.splitlines():
            if "extra_cuda" in line or "name=" in line:
                print(line)
        return
    write(KERNEL, k)
    write(BINDING, b)
    print(f"Applied ablation id={args.id} to {KERNEL} and {BINDING}")


if __name__ == "__main__":
    main()
