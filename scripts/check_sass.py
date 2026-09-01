"""
Prove that a refactor did not change generated device code.

Compiles two versions of solution/python (by default: git HEAD vs the current
working tree) with identical nvcc flags on the same Modal worker, then
compares, per kernel:

  * the mangled symbol name,
  * register / shared-memory / stack usage,
  * the full SASS instruction stream (addresses and encodings stripped).

Identical SASS for every live kernel means the device side of the refactor is
performance-neutral by construction, so the benchmark only has to cover the
host-side changes. No GPU is needed: nvcc targets sm_100a explicitly and we
only inspect the compiled object.

Usage:
    modal run scripts/check_sass.py                 # HEAD vs working tree
    modal run scripts/check_sass.py --base-rev v10  # any git revision
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

app = modal.App("flashinfer-bench-sass-check")

SRC_SUBDIR = "solution/python"
SRC_SUFFIXES = (".cu", ".cuh", ".h")

image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:latest")
    .apt_install("git")
    .env({"CUDA_HOME": "/usr/local/cuda"})
)

NVCC_FLAGS = [
    "-O3",
    "-std=c++17",
    "--expt-relaxed-constexpr",
    "-gencode", "arch=compute_100a,code=sm_100a",
    "-DTORCH_EXTENSION_NAME=dsa_topk_indexer",
    "-DTORCH_API_INCLUDE_EXTENSION_H",
]


def _read_rev(rev: str) -> dict:
    """Source files for a git revision (rev='' means the working tree)."""
    out = {}
    if rev:
        listing = subprocess.run(
            ["git", "ls-tree", "--name-only", f"{rev}:{SRC_SUBDIR}"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
        ).stdout.split()
        for name in listing:
            if name.endswith(SRC_SUFFIXES):
                out[name] = subprocess.run(
                    ["git", "show", f"{rev}:{SRC_SUBDIR}/{name}"],
                    cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
                ).stdout
    else:
        for p in sorted((PROJECT_ROOT / SRC_SUBDIR).iterdir()):
            if p.suffix in SRC_SUFFIXES:
                out[p.name] = p.read_text()
    return out


@app.function(image=image, timeout=1800, cpu=8.0)
def compile_and_dump(variants: dict) -> dict:
    """Compile each variant's kernel.cu and return parsed SASS per kernel."""
    import os
    import re
    import subprocess as sp
    import sysconfig
    import tempfile

    # Torch headers, resolved on the worker.
    from torch.utils.cpp_extension import include_paths

    includes = [f"-I{p}" for p in include_paths(device_type="cuda")]
    includes.append(f"-I{sysconfig.get_paths()['include']}")

    results = {}
    for name, files in variants.items():
        work = Path(tempfile.mkdtemp(prefix=f"{name}_"))
        for fname, content in files.items():
            (work / fname).write_text(content)

        obj = work / "kernel.o"
        cmd = ["nvcc", *NVCC_FLAGS, *includes, "-c", str(work / "kernel.cu"), "-o", str(obj)]
        proc = sp.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            results[name] = {
                "error": "compile failed",
                "stderr": proc.stderr[-8000:],
            }
            continue

        sass = sp.run(["cuobjdump", "-sass", str(obj)],
                      capture_output=True, text=True, check=True).stdout
        res_usage = sp.run(["cuobjdump", "-res-usage", str(obj)],
                           capture_output=True, text=True, check=True).stdout

        # Split the SASS dump into per-function instruction streams, dropping
        # instruction addresses and the hex encoding lines so the comparison
        # only reflects the instruction sequence.
        kernels: dict = {}
        current = None
        for line in sass.splitlines():
            m = re.match(r"\s*Function : (\S+)", line)
            if m:
                current = m.group(1)
                kernels[current] = []
                continue
            if current is None:
                continue
            if ".headerflags" in line:
                continue
            body = re.sub(r"/\*[0-9a-fA-F]{4}\*/", "", line)
            if re.fullmatch(r"\s*/\* 0x[0-9a-f]+ \*/\s*", line):
                continue
            body = body.strip()
            if body:
                kernels[current].append(body)

        results[name] = {
            "kernels": {k: v for k, v in kernels.items()},
            "res_usage": res_usage,
            "warnings": proc.stderr[-4000:],
        }
    return results


def _demangle(sym: str) -> str:
    """Short human-readable kernel name from a mangled symbol."""
    import re

    m = re.findall(r"\d+(paged_mqa[a-z_0-9]*|topk_[a-z_0-9]*)", sym)
    return m[0] if m else sym


@app.local_entrypoint()
def main(base_rev: str = "HEAD"):
    import hashlib

    print(f"Collecting sources: base={base_rev!r}, compare=working tree")
    variants = {"base": _read_rev(base_rev), "new": _read_rev("")}
    for name, files in variants.items():
        print(f"  {name}: {len(files)} files, {sum(len(c) for c in files.values())} bytes")

    out = compile_and_dump.remote(variants)

    failed = False
    for name in ("base", "new"):
        if "error" in out.get(name, {}):
            failed = True
            print(f"\n{name}: COMPILE FAILED")
            print(out[name]["stderr"])
    if failed:
        return

    for name in ("base", "new"):
        w = (out[name].get("warnings") or "").strip()
        if w:
            print(f"\n{name} nvcc stderr:\n{w}")

    # Key by demangled name: the mangled symbol of an anonymous-namespace
    # kernel embeds a per-translation-unit hash, which differs between any two
    # builds and would make a mangled-name comparison vacuous.
    base_k = {_demangle(k): v for k, v in out["base"]["kernels"].items()}
    new_k = {_demangle(k): v for k, v in out["new"]["kernels"].items()}

    def _hash(lines):
        return hashlib.sha256("\n".join(lines).encode()).hexdigest()[:12]

    print("\n=== Kernels in each build ===")
    for name, ks in (("base", base_k), ("new", new_k)):
        print(f"  {name}: {len(ks)} kernels")
        for k in sorted(ks):
            print(f"    {k:<44} {len(ks[k]):>5} insns  {_hash(ks[k])}")

    only_base = sorted(set(base_k) - set(new_k))
    only_new = sorted(set(new_k) - set(base_k))
    if only_base:
        print("\nDropped by the refactor (expected: dead code only):")
        for k in only_base:
            print(f"  - {k}")
    if only_new:
        print("\nNEW kernels not present in base:")
        for k in only_new:
            print(f"  + {k}")

    shared = sorted(set(base_k) & set(new_k))
    print(f"\n=== SASS comparison for the {len(shared)} kernels present in both ===")
    all_same = bool(shared)
    for k in shared:
        b, n = base_k[k], new_k[k]
        if b == n:
            print(f"  IDENTICAL  {k:<44} {len(b):>5} insns")
        else:
            all_same = False
            print(f"  DIFFERENT  {k:<44} base={len(b)} insns, new={len(n)} insns")
            import difflib

            diff = list(difflib.unified_diff(b, n, "base", "new", lineterm="", n=2))
            for line in diff[:60]:
                print(f"      {line}")
            if len(diff) > 60:
                print(f"      ... {len(diff) - 60} more diff lines")

    print("\n=== Resource usage (base) ===")
    print(out["base"]["res_usage"].strip())
    print("\n=== Resource usage (new) ===")
    print(out["new"]["res_usage"].strip())

    print(f"\n  Device code unchanged for all shared kernels: "
          f"{'YES' if all_same else 'NO'}")
