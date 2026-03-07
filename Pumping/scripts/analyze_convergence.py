#!/usr/bin/env python3
"""
Analyze convergence study results for the droplet deformation benchmark.

Two convergence modes:
  1. Epsilon convergence (--type eps):
     Fixed H0 and refinement, varies epsilon.
     Expects directories: *droplet_deformation_r{REF}
     Reads epsilon from the run's parameters (logged in diagnostics).

  2. Mesh convergence (--type mesh):
     Fixed H0 and epsilon, varies refinement.
     Expects directories: *droplet_deformation_r{5,6,7}

Auto-detection:
  Scans Results/ for droplet_deformation directories and groups them by
  refinement level and epsilon value (inferred from directory structure
  and CSV diagnostics).

Usage:
  python3 scripts/analyze_convergence.py --type eps
  python3 scripts/analyze_convergence.py --type mesh
  python3 scripts/analyze_convergence.py --type both
"""

import os
import sys
import glob
import csv
import argparse
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = "/Users/mahdi/Projects/git/PhD-Project/Pumping/Results"
CHI = 1.19
R_DROP = 0.2
SIGMA = 1.0
MU_0 = 1.0


def read_diagnostics(csv_path):
    """Read diagnostics CSV, return dict of numpy arrays."""
    data = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        for h in header:
            data[h] = []
        for row in reader:
            if len(row) < len(header):
                continue
            for i, h in enumerate(header):
                try:
                    data[h].append(float(row[i]))
                except ValueError:
                    data[h].append(0.0)
    return {k: np.array(v) for k, v in data.items()}


def find_equilibrium_ar(data, ramp_time=0.5, window=50):
    """Find equilibrium aspect ratio from post-ramp data."""
    ar = data.get('aspect_ratio', np.array([1.0]))
    t = data.get('time', np.array([0.0]))

    # Post-ramp data
    mask = t > ramp_time + 0.1
    if mask.sum() < 10:
        ar_post = ar[-min(window, len(ar)):]
    else:
        ar_post = ar[mask]

    ar_tail = ar_post[-min(window, len(ar_post)):]
    ar_eq = np.mean(ar_tail)

    u_max = data.get('U_max', np.array([0.0]))
    u_final = u_max[-1] if len(u_max) > 0 else 0

    if len(ar_tail) > 5:
        rel_var = np.std(ar_tail) / max(abs(ar_eq), 1e-12)
        converged = rel_var < 0.01 and u_final < 0.01
    else:
        converged = False

    return ar_eq, converged, u_final


def extract_ref_from_dirname(dirname):
    """Extract refinement level from directory name like ..._r6."""
    match = re.search(r'_r(\d+)$', dirname)
    return int(match.group(1)) if match else None


def read_params_file(dirpath):
    """Read key=value params file, return dict."""
    params = {}
    for fname in ['params.txt', 'parameters.txt']:
        p = os.path.join(dirpath, fname)
        if os.path.exists(p):
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if '=' in line:
                        key, val = line.split('=', 1)
                        params[key.strip()] = val.strip()
            break
    return params


def extract_epsilon_from_dir(dirpath):
    """Extract epsilon from params.txt or diagnostics CSV."""
    # Primary: params.txt (key=value format)
    pdict = read_params_file(dirpath)
    if 'epsilon' in pdict:
        try:
            return float(pdict['epsilon'])
        except ValueError:
            pass

    # Fallback: diagnostics CSV epsilon column
    csv_path = os.path.join(dirpath, 'diagnostics.csv')
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            header = f.readline().strip().split(',')
            if 'epsilon' in [h.strip() for h in header]:
                first_row = f.readline().strip().split(',')
                idx = [h.strip() for h in header].index('epsilon')
                return float(first_row[idx])

    return None


def extract_h0_from_dir(dirpath):
    """Extract H0 (field strength) from params.txt."""
    pdict = read_params_file(dirpath)
    if 'H_max' in pdict:
        try:
            return float(pdict['H_max'])
        except ValueError:
            pass
    return None


def scan_results(target_ref=None, target_eps=None, h0_target=None):
    """Scan Results/ for droplet deformation runs and extract metadata."""
    pattern = os.path.join(RESULTS_DIR, "*droplet_deformation_r*")
    dirs = sorted(glob.glob(pattern))

    results = []
    for d in dirs:
        dirname = os.path.basename(d)
        ref = extract_ref_from_dirname(dirname)
        if ref is None:
            continue
        if target_ref is not None and ref != target_ref:
            continue

        csv_path = os.path.join(d, "diagnostics.csv")
        if not os.path.exists(csv_path):
            continue

        data = read_diagnostics(csv_path)
        if len(data.get('time', [])) < 10:
            continue

        eps = extract_epsilon_from_dir(d)
        h0 = extract_h0_from_dir(d)

        ar_eq, converged, u_final = find_equilibrium_ar(data)
        D = (ar_eq - 1.0) / (ar_eq + 1.0)

        h = 1.0 / (2 ** ref)
        Bo_m = MU_0 * CHI * h0**2 * R_DROP / SIGMA if h0 else None

        mass = data.get('phi_mass', np.array([0.0]))
        mass_change = (abs(mass[-1] - mass[0]) / max(abs(mass[0]), 1e-12)
                       if len(mass) > 1 else 0.0)

        results.append({
            'dir': dirname,
            'ref': ref,
            'h': h,
            'eps': eps,
            'eps_over_h': eps / h if eps else None,
            'H0': h0,
            'Bo_m': Bo_m,
            'AR': ar_eq,
            'D': D,
            'converged': converged,
            'u_final': u_final,
            'mass_pct': mass_change * 100,
            'n_steps': len(data['time']),
            't_final': data['time'][-1],
        })

    return results


def analyze_eps_convergence(results):
    """Analyze and plot epsilon convergence (D vs eps at fixed ref)."""
    # Group by ref level
    refs = sorted(set(r['ref'] for r in results))

    print("\n" + "=" * 90)
    print("EPSILON CONVERGENCE: D(eps) at fixed mesh")
    print("=" * 90)

    for ref in refs:
        runs = sorted([r for r in results if r['ref'] == ref
                       and r['eps'] is not None],
                      key=lambda x: x['eps'])
        if len(runs) < 2:
            continue

        print(f"\n--- Refinement level {ref} (h = 1/{2**ref} = {1.0/(2**ref):.5f}) ---")
        print(f"{'eps':>10s}  {'eps/h':>8s}  {'AR':>8s}  {'D':>9s}  "
              f"{'U_fin':>9s}  {'Conv':>5s}  {'mass%':>8s}  Directory")
        print("-" * 100)

        for r in runs:
            conv = "YES" if r['converged'] else "no"
            print(f"{r['eps']:10.4f}  {r['eps_over_h']:8.2f}  {r['AR']:8.4f}  "
                  f"{r['D']:9.6f}  {r['u_final']:9.2e}  {conv:>5s}  "
                  f"{r['mass_pct']:8.4f}  {r['dir']}")

        # Rate estimation: if D converges, |D(eps) - D_ref| ~ eps^p
        if len(runs) >= 3:
            eps_arr = np.array([r['eps'] for r in runs])
            D_arr = np.array([r['D'] for r in runs])
            D_ref = D_arr[-1]  # finest eps as reference
            dD = np.abs(D_arr[:-1] - D_ref)
            eps_c = eps_arr[:-1]
            if np.all(dD > 0) and np.all(eps_c > 0):
                # log-log slope
                p, _ = np.polyfit(np.log(eps_c), np.log(dD), 1)
                print(f"\n  Convergence rate: |D(eps) - D_finest| ~ eps^{p:.2f}")
                print(f"  (Using finest eps={eps_arr[-1]:.4f} as reference)")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(refs)))

    for idx, ref in enumerate(refs):
        runs = sorted([r for r in results if r['ref'] == ref
                       and r['eps'] is not None],
                      key=lambda x: x['eps'])
        if len(runs) < 2:
            continue
        eps_arr = np.array([r['eps'] for r in runs])
        D_arr = np.array([r['D'] for r in runs])
        ax.plot(eps_arr, D_arr, 'o-', color=colors[idx], ms=8, lw=2,
                label=f'ref {ref} (h=1/{2**ref})')

    ax.set_xlabel('epsilon (interface width parameter)', fontsize=12)
    ax.set_ylabel('D = (AR-1)/(AR+1)', fontsize=12)
    ax.set_title('Epsilon Convergence: D vs eps at fixed mesh', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "convergence_eps.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot: {plot_path}")

    # Summary CSV
    out_csv = os.path.join(RESULTS_DIR, "convergence_eps_summary.csv")
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ref', 'h', 'epsilon', 'eps_over_h', 'AR', 'D',
                         'converged', 'U_final', 'mass_change_pct', 'directory'])
        for r in sorted(results, key=lambda x: (x['ref'], x['eps'] or 0)):
            if r['eps'] is not None:
                writer.writerow([r['ref'], f"{r['h']:.6f}", f"{r['eps']:.4f}",
                                 f"{r['eps_over_h']:.2f}", f"{r['AR']:.4f}",
                                 f"{r['D']:.6f}", r['converged'],
                                 f"{r['u_final']:.2e}", f"{r['mass_pct']:.4f}",
                                 r['dir']])
    print(f"Saved: {out_csv}")


def analyze_mesh_convergence(results):
    """Analyze and plot mesh convergence (D vs h at fixed eps)."""
    # Group by epsilon
    eps_vals = sorted(set(r['eps'] for r in results if r['eps'] is not None))

    print("\n" + "=" * 90)
    print("MESH CONVERGENCE: D(h) at fixed epsilon")
    print("=" * 90)

    for eps in eps_vals:
        runs = sorted([r for r in results if r['eps'] == eps],
                      key=lambda x: x['h'])
        if len(runs) < 2:
            continue

        print(f"\n--- epsilon = {eps:.4f} ---")
        print(f"{'ref':>5s}  {'h':>10s}  {'eps/h':>8s}  {'AR':>8s}  {'D':>9s}  "
              f"{'U_fin':>9s}  {'Conv':>5s}  {'mass%':>8s}  Directory")
        print("-" * 100)

        for r in runs:
            conv = "YES" if r['converged'] else "no"
            print(f"{r['ref']:5d}  {r['h']:10.5f}  {r['eps_over_h']:8.2f}  "
                  f"{r['AR']:8.4f}  {r['D']:9.6f}  {r['u_final']:9.2e}  "
                  f"{conv:>5s}  {r['mass_pct']:8.4f}  {r['dir']}")

        # Rate estimation: |D(h) - D_ref| ~ h^p
        if len(runs) >= 3:
            h_arr = np.array([r['h'] for r in runs])
            D_arr = np.array([r['D'] for r in runs])
            D_ref = D_arr[0]  # finest mesh (smallest h) as reference
            dD = np.abs(D_arr[1:] - D_ref)
            h_c = h_arr[1:]
            if np.all(dD > 0) and np.all(h_c > 0):
                p, _ = np.polyfit(np.log(h_c), np.log(dD), 1)
                print(f"\n  Convergence rate: |D(h) - D_finest| ~ h^{p:.2f}")
                print(f"  (Using finest h={h_arr[0]:.5f} (ref {runs[0]['ref']}) "
                      f"as reference)")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(eps_vals), 1)))

    for idx, eps in enumerate(eps_vals):
        runs = sorted([r for r in results if r['eps'] == eps],
                      key=lambda x: x['h'])
        if len(runs) < 2:
            continue
        h_arr = np.array([r['h'] for r in runs])
        D_arr = np.array([r['D'] for r in runs])
        ref_labels = [str(r['ref']) for r in runs]

        ax.plot(h_arr, D_arr, 'o-', color=colors[idx], ms=8, lw=2,
                label=f'eps={eps:.3f}')
        for h, D, rl in zip(h_arr, D_arr, ref_labels):
            ax.annotate(f'r{rl}', (h, D), textcoords="offset points",
                        xytext=(5, 5), fontsize=8, color='gray')

    ax.set_xlabel('h = 1/N (mesh size)', fontsize=12)
    ax.set_ylabel('D = (AR-1)/(AR+1)', fontsize=12)
    ax.set_title('Mesh Convergence: D vs h at fixed epsilon', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.invert_xaxis()  # Finer mesh (smaller h) to the right

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "convergence_mesh.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot: {plot_path}")

    # Summary CSV
    out_csv = os.path.join(RESULTS_DIR, "convergence_mesh_summary.csv")
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epsilon', 'ref', 'h', 'eps_over_h', 'AR', 'D',
                         'converged', 'U_final', 'mass_change_pct', 'directory'])
        for r in sorted(results, key=lambda x: (x['eps'] or 0, x['h'])):
            if r['eps'] is not None:
                writer.writerow([f"{r['eps']:.4f}", r['ref'], f"{r['h']:.6f}",
                                 f"{r['eps_over_h']:.2f}", f"{r['AR']:.4f}",
                                 f"{r['D']:.6f}", r['converged'],
                                 f"{r['u_final']:.2e}", f"{r['mass_pct']:.4f}",
                                 r['dir']])
    print(f"Saved: {out_csv}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze convergence studies for droplet deformation')
    parser.add_argument('--type', choices=['eps', 'mesh', 'both'],
                        default='both',
                        help='Convergence study type')
    args = parser.parse_args()

    # Scan all droplet deformation results
    results = scan_results()

    if not results:
        print("No droplet_deformation results found in Results/")
        print("Run the sweep scripts first:")
        print("  bash scripts/eps_convergence_sweep.sh")
        print("  bash scripts/mesh_convergence_sweep.sh")
        sys.exit(1)

    # Filter to runs that have epsilon info
    with_eps = [r for r in results if r['eps'] is not None]
    without_eps = [r for r in results if r['eps'] is None]

    print(f"Found {len(results)} total runs, "
          f"{len(with_eps)} with epsilon info, "
          f"{len(without_eps)} without")

    if without_eps:
        print("\nRuns without epsilon info (need manual H0/eps mapping):")
        for r in without_eps:
            print(f"  {r['dir']}  ref={r['ref']}  D={r['D']:.6f}")

    # Theory values for reference
    theory_coeff_2D = CHI / (3.0 * (2.0 + CHI)**2)
    mu_r = 1.0 + CHI
    theory_coeff_3D = (9.0 / 16.0) * (mu_r - 1.0) / (mu_r + 0.5)
    print(f"\n2D theory: D = {theory_coeff_2D:.6f} * Bo_m")
    print(f"3D theory: D = {theory_coeff_3D:.6f} * Bo_m")

    if args.type in ('eps', 'both') and len(with_eps) >= 2:
        analyze_eps_convergence(with_eps)

    if args.type in ('mesh', 'both') and len(with_eps) >= 2:
        analyze_mesh_convergence(with_eps)

    if len(with_eps) < 2:
        print("\nInsufficient runs with epsilon info for convergence analysis.")
        print("The sweep scripts need to log epsilon to diagnostics.csv or")
        print("save a params file for auto-detection to work.")
        print("\nTo enable auto-detection, add epsilon logging to fhd_ch_driver.cc")
        print("(see --save-params flag suggestion below).")


if __name__ == '__main__':
    main()
