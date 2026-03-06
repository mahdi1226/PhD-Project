#!/usr/bin/env python3
"""
Analyze deformation sweep results: D vs Bo_m.

Finds droplet_deformation_r6 directories, reads diagnostics.csv,
extracts equilibrium aspect ratio, computes D.

Compares with:
  - 2D small-deformation theory: D = Bo_m * chi / (2*(2+chi))
  - Afkhami 2010 (3D axisymmetric): D = (9/16) * Bo_m * f(mu_r)
    where f(mu_r) = (mu_r - 1) / (mu_r + 1/2) for prolate spheroid

Generates D vs Bo_m comparison plot.
"""

import os
import glob
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = "/Users/mahdi/Projects/git/PhD-Project/Pumping/Results"
CHI = 1.19
R_DROP = 0.2
SIGMA = 1.0
MU_0 = 1.0

# Known H0 values from the sweep script, mapped by directory timestamp
# These are the ACTUAL values passed to --field-strength
H0_MAP = {
    # Batch 1 (sub-cell tracking, t_final=3.0)
    '030526_215237_droplet_deformation_r6': 1.0,
    '030526_215240_droplet_deformation_r6': 1.5,
    '030526_215243_droplet_deformation_r6': 2.0,
    # Batch 2 (sub-cell tracking, t_final=3.0)
    '030626_053200_droplet_deformation_r6': 3.0,
    '030626_053202_droplet_deformation_r6': 4.0,
    '030626_053205_droplet_deformation_r6': 5.0,
}


def read_diagnostics(csv_path):
    """Read diagnostics CSV, return dict of arrays."""
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

    # Check convergence: is AR changing?
    u_max = data.get('U_max', np.array([0.0]))
    u_final = u_max[-1] if len(u_max) > 0 else 0

    if len(ar_tail) > 5:
        rel_var = np.std(ar_tail) / max(abs(ar_eq), 1e-12)
        converged = rel_var < 0.01 and u_final < 0.01
    else:
        converged = False

    return ar_eq, converged, u_final


def main():
    # Find all droplet deformation runs at ref 6
    pattern = os.path.join(RESULTS_DIR, "*droplet_deformation_r6")
    all_dirs = sorted(glob.glob(pattern))

    if not all_dirs:
        print("No droplet_deformation_r6 directories found!")
        return

    print("=" * 90)
    print("Deformation Sweep Analysis: D vs Bo_m")
    print(f"Chi = {CHI}, R = {R_DROP}, sigma = {SIGMA}")
    print(f"Bo_m = mu_0 * chi * H0^2 * R / sigma = {MU_0*CHI*R_DROP/SIGMA:.4f} * H0^2")
    print("=" * 90)

    # 2D small-deformation theory: pressure balance on circular interface
    # Magnetic normal stress: f_n = (mu_0/2)*M_n^2 with M_n = chi*2H0/(2+chi)*cos(theta)
    # Curvature perturbation: Delta_kappa = 3*delta*cos(2theta)/R^2
    # D = Bo_m * chi / (3*(2+chi)^2)
    theory_coeff_2D = CHI / (3.0 * (2.0 + CHI)**2)

    # 3D Afkhami theory: D = (9/16) * Bo_m * (mu_r - 1)/(mu_r + 1/2)
    # mu_r = 1 + chi (relative permeability)
    mu_r = 1.0 + CHI
    f_mu_r_3D = (mu_r - 1.0) / (mu_r + 0.5)
    theory_coeff_3D = (9.0 / 16.0) * f_mu_r_3D

    results = []
    for d in all_dirs:
        dirname = os.path.basename(d)

        # Skip if not in our known map (e.g., test runs)
        if dirname not in H0_MAP:
            continue

        csv_path = os.path.join(d, "diagnostics.csv")
        if not os.path.exists(csv_path):
            continue

        data = read_diagnostics(csv_path)
        if len(data.get('time', [])) < 10:
            continue

        H0 = H0_MAP[dirname]
        Bo_m = MU_0 * CHI * H0**2 * R_DROP / SIGMA

        ar_eq, converged, u_final = find_equilibrium_ar(data)
        D = (ar_eq - 1.0) / (ar_eq + 1.0)

        # Mass conservation
        mass = data.get('phi_mass', np.array([0.0]))
        mass_change = abs(mass[-1] - mass[0]) / max(abs(mass[0]), 1e-12) if len(mass) > 1 else 0.0

        results.append({
            'H0': H0,
            'Bo_m': Bo_m,
            'AR': ar_eq,
            'D': D,
            'D_theory_2D': theory_coeff_2D * Bo_m,
            'D_theory_3D': theory_coeff_3D * Bo_m,
            'converged': converged,
            'u_final': u_final,
            'mass_pct': mass_change * 100,
            'n_steps': len(data['time']),
            't_final': data['time'][-1],
            'dir': dirname,
        })

    if not results:
        print("\nNo valid results found!")
        return

    results.sort(key=lambda x: x['Bo_m'])

    # Print table
    print(f"\n{'H0':>6s}  {'Bo_m':>8s}  {'AR':>8s}  {'D_num':>9s}  "
          f"{'D_th(2D)':>9s}  {'D_th(3D)':>9s}  {'U_fin':>9s}  "
          f"{'Conv':>5s}  {'dM%':>7s}  Directory")
    print("-" * 120)

    for r in results:
        conv = "YES" if r['converged'] else "no"
        print(f"{r['H0']:6.1f}  {r['Bo_m']:8.4f}  {r['AR']:8.4f}  {r['D']:9.6f}  "
              f"{r['D_theory_2D']:9.6f}  {r['D_theory_3D']:9.6f}  {r['u_final']:9.2e}  "
              f"{conv:>5s}  {r['mass_pct']:7.4f}  {r['dir']}")

    print(f"\n2D small-deformation theory: D = {theory_coeff_2D:.6f} * Bo_m")
    print(f"3D small-deformation theory: D = {theory_coeff_3D:.6f} * Bo_m")
    print(f"Linear regime valid for D < ~0.05")

    # Save summary CSV
    out_csv = os.path.join(RESULTS_DIR, "deformation_sweep_summary.csv")
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['H0', 'Bo_m', 'AR_eq', 'D_numerical',
                         'D_theory_2D', 'D_theory_3D',
                         'converged', 'U_final', 'mass_change_pct', 'directory'])
        for r in results:
            writer.writerow([f"{r['H0']:.1f}", f"{r['Bo_m']:.4f}", f"{r['AR']:.4f}",
                             f"{r['D']:.6f}", f"{r['D_theory_2D']:.6f}",
                             f"{r['D_theory_3D']:.6f}",
                             r['converged'], f"{r['u_final']:.2e}",
                             f"{r['mass_pct']:.4f}", r['dir']])
    print(f"\nSaved: {out_csv}")

    # ====================================================================
    # Generate plot: D vs Bo_m
    # ====================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    Bo_arr = np.array([r['Bo_m'] for r in results])
    D_arr = np.array([r['D'] for r in results])
    conv_arr = np.array([r['converged'] for r in results])

    # Theory curves
    Bo_theory = np.linspace(0, max(Bo_arr) * 1.1, 200)
    D_theory_2D = theory_coeff_2D * Bo_theory
    D_theory_3D = theory_coeff_3D * Bo_theory

    # --- Left panel: full range ---
    ax1.plot(Bo_theory, D_theory_2D, 'b--', lw=2,
             label=f'2D theory: D = {theory_coeff_2D:.4f}·Bo_m')
    ax1.plot(Bo_theory, D_theory_3D, 'r--', lw=2,
             label=f'3D theory: D = {theory_coeff_3D:.4f}·Bo_m')

    # Numerical: equilibrated vs not
    eq_mask = conv_arr
    ax1.plot(Bo_arr[eq_mask], D_arr[eq_mask], 'ko', ms=10, mfc='green',
             label='Numerical (equilibrated)')
    if (~eq_mask).any():
        ax1.plot(Bo_arr[~eq_mask], D_arr[~eq_mask], 'k^', ms=10, mfc='orange',
                 label='Numerical (not equilibrated)')

    # Annotate H0 values
    for r in results:
        ax1.annotate(f'H₀={r["H0"]:.0f}',
                     (r['Bo_m'], r['D']),
                     textcoords="offset points",
                     xytext=(8, 5), fontsize=8, color='gray')

    ax1.set_xlabel('Bo_m = μ₀χH₀²R/σ', fontsize=12)
    ax1.set_ylabel('D = (AR−1)/(AR+1)', fontsize=12)
    ax1.set_title('Droplet Deformation vs Magnetic Bond Number', fontsize=13)
    ax1.legend(fontsize=9, loc='upper left')
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)

    # --- Right panel: zoom on low Bo_m (linear regime) ---
    Bo_low = np.linspace(0, 1.5, 100)
    D_theory_2D_low = theory_coeff_2D * Bo_low
    D_theory_3D_low = theory_coeff_3D * Bo_low

    ax2.plot(Bo_low, D_theory_2D_low, 'b--', lw=2, label='2D theory')
    ax2.plot(Bo_low, D_theory_3D_low, 'r--', lw=2, label='3D theory')

    # Only plot low Bo_m data
    low_mask = Bo_arr < 1.5
    if low_mask.any():
        ax2.plot(Bo_arr[low_mask], D_arr[low_mask], 'ko', ms=10, mfc='green')
        for r in [r for r in results if r['Bo_m'] < 1.5]:
            ax2.annotate(f'H₀={r["H0"]:.0f}',
                         (r['Bo_m'], r['D']),
                         textcoords="offset points",
                         xytext=(8, 5), fontsize=9, color='gray')

    ax2.set_xlabel('Bo_m', fontsize=12)
    ax2.set_ylabel('D', fontsize=12)
    ax2.set_title('Linear Regime (low Bo_m)', fontsize=13)
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, 1.5)
    ax2.set_ylim(0, 0.15)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.05, color='gray', ls=':', lw=1, alpha=0.5)
    ax2.text(1.35, 0.052, 'D=0.05\n(linear limit)', fontsize=7,
             ha='right', va='bottom', color='gray')

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "deformation_sweep_D_vs_Bom.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {plot_path}")

    # Also plot AR vs Bo_m
    fig2, ax3 = plt.subplots(figsize=(8, 6))
    ax3.plot(Bo_arr[eq_mask], np.array([r['AR'] for r in results])[eq_mask],
             'ko-', ms=10, mfc='green', label='Numerical (equilibrated)')
    if (~eq_mask).any():
        ax3.plot(Bo_arr[~eq_mask],
                 np.array([r['AR'] for r in results])[~eq_mask],
                 'k^--', ms=10, mfc='orange', label='Numerical (not equilibrated)')
    for r in results:
        ax3.annotate(f'H₀={r["H0"]:.0f}',
                     (r['Bo_m'], r['AR']),
                     textcoords="offset points",
                     xytext=(8, 5), fontsize=9, color='gray')
    ax3.set_xlabel('Bo_m', fontsize=12)
    ax3.set_ylabel('Aspect Ratio (y_span / x_span)', fontsize=12)
    ax3.set_title(f'Droplet Aspect Ratio vs Bo_m (χ={CHI}, R={R_DROP})', fontsize=13)
    ax3.legend(fontsize=10)
    ax3.set_xlim(left=0)
    ax3.set_ylim(bottom=0.95)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=1.0, color='gray', ls=':', lw=1)

    plt.tight_layout()
    plot_path2 = os.path.join(RESULTS_DIR, "deformation_sweep_AR_vs_Bom.png")
    plt.savefig(plot_path2, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {plot_path2}")


if __name__ == '__main__':
    main()
