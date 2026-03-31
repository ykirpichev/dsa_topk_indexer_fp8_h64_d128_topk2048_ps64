#!/usr/bin/env python3
"""
Run Modal smoke after applying each patch to solution/cuda/kernel.cu and/or binding.py.
Restores files after each run. Append-only log to OPTIMIZATION_LEARNINGS.md section.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KERNEL = ROOT / "solution" / "cuda" / "kernel.cu"
BINDING = ROOT / "solution" / "cuda" / "binding.py"
LOG = ROOT / "OPTIMIZATION_LEARNINGS.md"
BACKUP_K = ROOT / ".opt_ablation_kernel.cu.bak"
BACKUP_B = ROOT / ".opt_ablation_binding.py.bak"


def save_backups():
    shutil.copy2(KERNEL, BACKUP_K)
    shutil.copy2(BINDING, BACKUP_B)


def restore():
    shutil.copy2(BACKUP_K, KERNEL)
    shutil.copy2(BACKUP_B, BINDING)


def run_smoke() -> tuple[float | None, str]:
    r = subprocess.run(
        ["bash", "-lc", "cd /workspace && FIB_MODAL_SMOKE=1 modal run scripts/run_modal.py 2>&1"],
        capture_output=True,
        text=True,
        timeout=120000,
    )
    out = r.stdout + r.stderr
    if "INCORRECT" in out or "COMPILE_ERROR" in out or "RUNTIME_ERROR" in out:
        m = re.search(r"Geomean:\s*([\d.]+)x", out)
        g = float(m.group(1)) if m else None
        return g, "FAIL_OR_ERROR"
    m = re.search(r"Geomean:\s*([\d.]+)x", out)
    if not m:
        return None, "NO_GEOMEAN"
    return float(m.group(1)), "OK"


@dataclass
class Exp:
    id: int
    name: str
    kernel_replacements: list[tuple[str, str]]  # (old, new)
    binding_replacements: list[tuple[str, str]]


EXPERIMENTS: list[Exp] = [
    Exp(
        18,
        "at::matmul instead of torch::bmm for logits",
        [
            (
                "auto logits  = torch::bmm(q_float, K_batched.transpose(1, 2)).contiguous();  // [B, H, S]",
                "auto logits  = at::matmul(q_float, K_batched.transpose(1, 2)).contiguous();  // [B, H, S]",
            )
        ],
        [],
    ),
    Exp(
        19,
        "page_transform BLOCK_T 128",
        [("constexpr int BLOCK_T = 256;", "constexpr int BLOCK_T = 128;")],
        [],
    ),
    Exp(
        20,
        "page_transform BLOCK_T 512",
        [("constexpr int BLOCK_T = 256;", "constexpr int BLOCK_T = 512;")],
        [],
    ),
    Exp(
        21,
        "weights.unsqueeze(2) without .contiguous() on weights",
        [
            (
                "auto w_bcast = weights.contiguous().unsqueeze(2);",
                "auto w_bcast = weights.unsqueeze(2);  // ablation: assume already contiguous",
            )
        ],
        [],
    ),
    Exp(
        22,
        "q_float .to(f32).contiguous() before bmm",
        [
            (
                "auto q_float = q_index_fp8.to(torch::kFloat32);  // [B, H, D]",
                "auto q_float = q_index_fp8.to(torch::kFloat32).contiguous();  // [B, H, D]",
            )
        ],
        [],
    ),
    Exp(
        23,
        "Explicit K_T = transpose(1,2).contiguous() then bmm(q, K_T)",
        [
            (
                "    auto q_float = q_index_fp8.to(torch::kFloat32);  // [B, H, D]\n"
                "    auto logits  = torch::bmm(q_float, K_batched.transpose(1, 2)).contiguous();  // [B, H, S]",
                "    auto q_float = q_index_fp8.to(torch::kFloat32);  // [B, H, D]\n"
                "    auto K_T = K_batched.transpose(1, 2).contiguous();\n"
                "    auto logits  = torch::bmm(q_float, K_T).contiguous();  // [B, H, S]",
            )
        ],
        [],
    ),
    Exp(
        24,
        "Drop .contiguous() after bmm (keep relu_ in-place)",
        [
            (
                "auto logits  = torch::bmm(q_float, K_batched.transpose(1, 2)).contiguous();  // [B, H, S]",
                "auto logits  = torch::bmm(q_float, K_batched.transpose(1, 2));  // [B, H, S] ablation",
            )
        ],
        [],
    ),
    Exp(
        25,
        "Reuse topk_vals buffer [K_topk] across batch loop",
        [
            (
                "    torch::Tensor seq_lens_dev = seq_lens.to(logits.device()).to(torch::kInt32).contiguous();\n\n"
                "    for (int b = 0; b < B; ++b) {\n"
                "        const int sl = sl_vec[b];\n"
                "        if (sl == 0) continue;\n"
                "        const int k = std::min(K_topk, sl);\n"
                "        auto scores = logits[b].narrow(1, 0, sl).sum(0);\n"
                "        auto topk_vals = torch::empty({k}, scores.options());",
                "    torch::Tensor seq_lens_dev = seq_lens.to(logits.device()).to(torch::kInt32).contiguous();\n"
                "    torch::Tensor topk_vals_buf = torch::empty({K_topk}, logits.options());\n\n"
                "    for (int b = 0; b < B; ++b) {\n"
                "        const int sl = sl_vec[b];\n"
                "        if (sl == 0) continue;\n"
                "        const int k = std::min(K_topk, sl);\n"
                "        auto scores = logits[b].narrow(1, 0, sl).sum(0);\n"
                "        auto topk_vals = topk_vals_buf.narrow(0, 0, k);",
            )
        ],
        [],
    ),
    Exp(
        26,
        "extra_cuda_cflags: add -Xptxas -O3",
        [],
        [
            (
                'extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],',
                'extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-Xptxas", "-O3"],',
            )
        ],
    ),
    Exp(
        27,
        "extra_cuda_cflags: -O2 instead of -O3",
        [],
        [
            (
                'extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],',
                'extra_cuda_cflags=["-O2", "--expt-relaxed-constexpr"],',
            )
        ],
    ),
    Exp(
        28,
        "sum_out into scores_buf[max_seq_len]",
        [
            (
                "    torch::Tensor seq_lens_dev = seq_lens.to(logits.device()).to(torch::kInt32).contiguous();\n\n"
                "    for (int b = 0; b < B; ++b) {\n"
                "        const int sl = sl_vec[b];\n"
                "        if (sl == 0) continue;\n"
                "        const int k = std::min(K_topk, sl);\n"
                "        auto scores = logits[b].narrow(1, 0, sl).sum(0);",
                "    torch::Tensor seq_lens_dev = seq_lens.to(logits.device()).to(torch::kInt32).contiguous();\n"
                "    torch::Tensor scores_buf = torch::empty({max_seq_len}, logits.options());\n\n"
                "    for (int b = 0; b < B; ++b) {\n"
                "        const int sl = sl_vec[b];\n"
                "        if (sl == 0) continue;\n"
                "        const int k = std::min(K_topk, sl);\n"
                "        auto row = logits[b].narrow(1, 0, sl);\n"
                "        auto scores = scores_buf.narrow(0, 0, sl);\n"
                "        at::sum_out(scores, row, /*dim=*/0, /*keepdim=*/false);",
            )
        ],
        [],
    ),
    Exp(
        29,
        "bmm_out into pre-allocated logits, q and K_T contiguous",
        [
            (
                "    auto q_float = q_index_fp8.to(torch::kFloat32);  // [B, H, D]\n"
                "    auto logits  = torch::bmm(q_float, K_batched.transpose(1, 2)).contiguous();  // [B, H, S]",
                "    auto q_float = q_index_fp8.to(torch::kFloat32).contiguous();\n"
                "    auto K_T = K_batched.transpose(1, 2).contiguous();\n"
                "    torch::Tensor logits = torch::empty({B, H, S}, q_float.options());\n"
                "    at::bmm_out(logits, q_float, K_T);",
            )
        ],
        [],
    ),
    Exp(
        30,
        "JIT name bump to force clean rebuild (same flags)",
        [],
        [
            (
                'name="topk_cuda_cublas",',
                'name="topk_cuda_cublas_ab30",',
            )
        ],
    ),
    Exp(
        31,
        "permute(0,2,1) instead of transpose(1,2) for K side of bmm",
        [
            (
                "torch::bmm(q_float, K_batched.transpose(1, 2)).contiguous();",
                "torch::bmm(q_float, K_batched.permute({0, 2, 1})).contiguous();",
            )
        ],
        [],
    ),
    Exp(
        32,
        "K_batched.contiguous() immediately after gather kernel",
        [
            (
                "            B, S, D, P, PS, HDS, actual_pages);\n\n    // -------------------------------------------------------------------------\n    // Phase 2",
                "            B, S, D, P, PS, HDS, actual_pages);\n    K_batched = K_batched.contiguous();\n\n    // -------------------------------------------------------------------------\n    // Phase 2",
            )
        ],
        [],
    ),
]


def apply_exp(exp: Exp) -> None:
    restore()
    ktxt = KERNEL.read_text()
    for old, new in exp.kernel_replacements:
        if old not in ktxt:
            raise RuntimeError(f"kernel missing pattern for exp {exp.id}: {old[:60]}...")
        ktxt = ktxt.replace(old, new, 1)
    KERNEL.write_text(ktxt)
    btxt = BINDING.read_text()
    for old, new in exp.binding_replacements:
        if old not in btxt:
            raise RuntimeError(f"binding missing pattern for exp {exp.id}")
        btxt = btxt.replace(old, new, 1)
    BINDING.write_text(btxt)


def append_log(rows: list[tuple[int, str, float | None, str]]):
    block = "\n| " + " | ".join(["Batch", "Idea", "Geomean", "Status"]) + " |\n"
    block += "|" + "|".join(["---"] * 4) + "|\n"
    for eid, name, g, st in rows:
        gv = f"{g:.2f}×" if g is not None else "—"
        block += f"| {eid} | {name} | {gv} | {st} |\n"
    text = LOG.read_text()
    marker = "## Batch ablations"
    if marker not in text:
        text = text.rstrip() + f"\n\n{marker} (automated)\n"
    else:
        text = text.rstrip() + "\n"
    text += block
    LOG.write_text(text)


def main():
    save_backups()
    restore()
    base_g, base_st = run_smoke()
    print(f"BASELINE geomean={base_g} status={base_st}")
    rows: list[tuple[int, str, float | None, str]] = []
    rows.append((0, "baseline before batch", base_g, base_st))

    for exp in EXPERIMENTS:
        try:
            apply_exp(exp)
        except Exception as e:
            rows.append((exp.id, exp.name, None, f"PATCH_FAIL: {e}"))
            restore()
            continue
        g, st = run_smoke()
        delta = (g - base_g) if (g is not None and base_g is not None) else None
        note = st
        if delta is not None:
            note = f"{st} (Δ{delta:+.2f} vs baseline)"
        rows.append((exp.id, exp.name, g, note))
        print(f"Exp {exp.id}: geomean={g} {note}")
        restore()

    append_log(rows)
    print("Appended to OPTIMIZATION_LEARNINGS.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
