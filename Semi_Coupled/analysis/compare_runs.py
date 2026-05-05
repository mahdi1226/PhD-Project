#!/usr/bin/env python3
"""compare_runs.py — Plan A regression check.

Compares two Semi_Coupled runs row-by-row. Reports rel-L2 AND abs-L2
diffs per column. PASS if EITHER (rel ≤ rel_tol) OR (abs ≤ abs_tol)
— the OR avoids spurious failures on near-zero quantities (e.g., U_max
during pre-ramp ferrofluid setups, where any tiny perturbation gives
huge relative difference but trivially small absolute).

Defaults: rel_tol=1e-3, abs_tol=1e-5. Iterative solver tolerance is
~1e-8 per step; over 200 steps the worst-case accumulated drift is
~2e-6 absolute on solution quantities.

Usage:
  python3 analysis/compare_runs.py <run_a_dir> <run_b_dir>
                                   [--rel-tol 1e-3] [--abs-tol 1e-5]
"""
import csv
import sys
from pathlib import Path

import numpy as np


def load(path: Path) -> tuple[list[str], np.ndarray]:
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = []
        for row in reader:
            if not row or row[0].lstrip().startswith("#"):
                continue
            try:
                int(row[0])
            except ValueError:
                continue
            rows.append(row)
    return header, np.array(rows, dtype=float)


def main():
    args = sys.argv[1:]
    rel_tol = 1e-3
    abs_tol = 1e-5
    if "--rel-tol" in args:
        i = args.index("--rel-tol")
        rel_tol = float(args[i + 1])
        args = args[:i] + args[i + 2:]
    if "--abs-tol" in args:
        i = args.index("--abs-tol")
        abs_tol = float(args[i + 1])
        args = args[:i] + args[i + 2:]
    if len(args) != 2:
        sys.exit(f"Usage: {sys.argv[0]} <run_a_dir> <run_b_dir> "
                 "[--rel-tol 1e-3] [--abs-tol 1e-5]")

    a_dir = Path(args[0])
    b_dir = Path(args[1])
    a_csv = a_dir / "diagnostics.csv"
    b_csv = b_dir / "diagnostics.csv"

    if not a_csv.exists():
        sys.exit(f"Missing: {a_csv}")
    if not b_csv.exists():
        sys.exit(f"Missing: {b_csv}")

    h_a, A = load(a_csv)
    h_b, B = load(b_csv)

    n = min(len(A), len(B))
    if n == 0:
        sys.exit("No data rows in one or both files.")
    print(f"  rows: a={len(A)}, b={len(B)}, comparing first {n}")

    # Solution columns to check (existence-checked first)
    primary = [
        "mass", "theta_min", "theta_max",
        "M_max", "H_max",
        "U_max", "ux_max", "uy_max",
        "interface_y_max", "interface_y_mean",
        "E_internal", "E_total",
    ]

    print(f"\n  column          | rel-L2     | abs-L2     | PASS (rel≤{rel_tol:.0e} OR abs≤{abs_tol:.0e})")
    print("  ----------------+------------+------------+--------------------------------")
    fails = []
    for name in primary:
        if name not in h_a or name not in h_b:
            continue
        ia = h_a.index(name)
        ib = h_b.index(name)
        a = A[:n, ia]
        b = B[:n, ib]
        absd = np.linalg.norm(a - b)
        rel = absd / (np.linalg.norm(a) + 1e-30)
        ok = (rel <= rel_tol) or (absd <= abs_tol)
        marker = "PASS" if ok else "FAIL"
        print(f"  {name:15s} | {rel:.4e} | {absd:.4e} | {marker}")
        if not ok:
            fails.append((name, rel, absd))

    print("\n  Summary:")
    if fails:
        print(f"  ❌ {len(fails)} column(s) failed:")
        for name, rel, absd in fails:
            print(f"     {name}: rel={rel:.3e}, abs={absd:.3e}")
        sys.exit(1)
    else:
        print(f"  ✅ All checked columns match (rel≤{rel_tol:.0e} OR abs≤{abs_tol:.0e}).")


if __name__ == "__main__":
    main()
