"""
Compare two run_modal_compare.py logs workload-by-workload.

Used to check that a refactor is performance-neutral: it joins the two runs on
workload uuid, groups by which kernel plan the workload dispatches to, and
reports the per-bucket and worst-case latency deltas.

Usage:
    python scripts/compare_bench_logs.py BASELINE_LOG NEW_LOG
"""

import re
import statistics
import sys
from pathlib import Path

# A record starts with the truncated uuid; modal's console wraps long lines,
# so continuation lines have to be folded back in before parsing.
RECORD_START = re.compile(r"^\s{2}([0-9a-f]{8})\.\.\.:")
FIELDS = re.compile(
    r"^\s{2}(?P<uuid>[0-9a-f]{8})\.\.\.:\s*(?P<status>\w+)"
    r"(?:\s*\|\s*(?P<latency>[\d.]+) us)?"
    r"(?:\s*\|\s*(?P<speedup>[\d.]+)x)?"
)
PAGES = re.compile(r"max_num_pages=(\d+)")

# Dispatch thresholds from solution/python/dsa_config.cuh.
FAST_PATH_MAX_PAGES = 32
PERSISTENT_PAGE_THRESHOLD = 64


def parse(path: Path) -> dict:
    text = path.read_text()

    folded = []
    for line in text.splitlines():
        if RECORD_START.match(line):
            folded.append(line)
        elif folded and line.strip() and not line.startswith(("Stopping", "✓", "Runner")):
            folded[-1] += " " + line.strip()

    out = {}
    for line in folded:
        m = FIELDS.match(line)
        if not m:
            continue
        rec = {
            "status": m.group("status"),
            "latency_us": float(m.group("latency")) if m.group("latency") else None,
            "speedup": float(m.group("speedup")) if m.group("speedup") else None,
            "pages": None,
        }
        pages = PAGES.search(line)
        if pages:
            rec["pages"] = int(pages.group(1))
        out[m.group("uuid")] = rec
    return out


def plan_of(pages) -> str:
    if pages is None:
        return "unknown"
    if pages <= FAST_PATH_MAX_PAGES:
        return "fast path"
    if pages < PERSISTENT_PAGE_THRESHOLD:
        return "short"
    return "persistent ws"


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2

    base_path, new_path = Path(sys.argv[1]), Path(sys.argv[2])
    base, new = parse(base_path), parse(new_path)

    print(f"baseline: {base_path}  ({len(base)} workloads)")
    print(f"new:      {new_path}  ({len(new)} workloads)")

    only_base = set(base) - set(new)
    only_new = set(new) - set(base)
    if only_base or only_new:
        print(f"\nWARNING: workload sets differ "
              f"(baseline-only={len(only_base)}, new-only={len(only_new)})")

    shared = sorted(set(base) & set(new))
    status_changes = [u for u in shared if base[u]["status"] != new[u]["status"]]
    if status_changes:
        print(f"\nSTATUS CHANGES on {len(status_changes)} workloads:")
        for u in status_changes:
            print(f"  {u}: {base[u]['status']} -> {new[u]['status']}")
    else:
        n_pass = sum(1 for u in shared if new[u]["status"] == "PASSED")
        print(f"\nstatus: {n_pass}/{len(shared)} PASSED in both runs")

    rows = []
    for u in shared:
        b, n = base[u]["latency_us"], new[u]["latency_us"]
        if not b or not n:
            continue
        rows.append({
            "uuid": u,
            "pages": base[u]["pages"],
            "plan": plan_of(base[u]["pages"]),
            "base": b,
            "new": n,
            "delta_pct": (n - b) / b * 100.0,
        })

    print(f"\n=== Per-plan latency (us), {len(rows)} workloads ===")
    header = (f"  {'plan':<15} {'n':>4} {'base mean':>10} {'new mean':>10} "
              f"{'delta':>8} {'worst':>8} {'best':>8}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for plan in ("fast path", "short", "persistent ws", "unknown"):
        grp = [r for r in rows if r["plan"] == plan]
        if not grp:
            continue
        bm = statistics.mean(r["base"] for r in grp)
        nm = statistics.mean(r["new"] for r in grp)
        deltas = [r["delta_pct"] for r in grp]
        print(f"  {plan:<15} {len(grp):>4} {bm:>10.2f} {nm:>10.2f} "
              f"{(nm - bm) / bm * 100:>7.2f}% {max(deltas):>7.2f}% {min(deltas):>7.2f}%")

    bm = statistics.mean(r["base"] for r in rows)
    nm = statistics.mean(r["new"] for r in rows)
    print(f"  {'ALL':<15} {len(rows):>4} {bm:>10.2f} {nm:>10.2f} "
          f"{(nm - bm) / bm * 100:>7.2f}%")

    base_sp = [base[u]["speedup"] for u in shared if base[u]["speedup"]]
    new_sp = [new[u]["speedup"] for u in shared if new[u]["speedup"]]
    if base_sp and new_sp:
        print(f"\nmean speedup vs naive ref: baseline {statistics.mean(base_sp):.1f}x "
              f"-> new {statistics.mean(new_sp):.1f}x")

    worst = sorted(rows, key=lambda r: -r["delta_pct"])[:8]
    print("\n=== Largest regressions ===")
    for r in worst:
        print(f"  {r['uuid']}  pages={r['pages']:<4} {r['plan']:<14} "
              f"{r['base']:>7.2f} -> {r['new']:>7.2f} us  ({r['delta_pct']:+.2f}%)")

    # The harness reports latency rounded to 0.1 us, so on the ~2 us fast-path
    # bucket one reporting step is already ~4.5% for a single workload. Judge
    # on per-plan means, and only treat slower-than-baseline as a problem.
    print("\n=== Verdict ===")
    ok = not status_changes
    for plan in ("fast path", "short", "persistent ws"):
        grp = [r for r in rows if r["plan"] == plan]
        if not grp:
            continue
        b = statistics.mean(r["base"] for r in grp)
        n = statistics.mean(r["new"] for r in grp)
        d = (n - b) / b * 100.0
        if d > 2.0:
            ok = False
            flag = "REGRESSION"
        elif d < -2.0:
            flag = "faster"
        else:
            flag = "ok"
        print(f"  {plan:<15} {d:+.2f}% mean  [{flag}]")
    print(f"\n  No regression: {'YES' if ok else 'NO'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
