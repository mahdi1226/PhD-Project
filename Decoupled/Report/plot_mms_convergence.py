#!/usr/bin/env python3
"""
Plot MMS convergence rates for all subsystems.
Reads CSV results from each subsystem's MMS test output.
Generates publication-quality convergence plots with reference slopes.

Usage:
    python3 plot_mms_convergence.py
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

# ============================================================================
# Configuration
# ============================================================================
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "Report", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# Find the LATEST CSV for each subsystem
def latest_csv(pattern):
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None

POISSON_CSV = latest_csv(os.path.join(BASE, "poisson/poisson_results/mms/*_poisson_convergence.csv"))
CH_CSV = latest_csv(os.path.join(BASE, "cahn_hilliard/cahn_hilliard_results/mms/*_ch_mms_convergence.csv"))
MAG_CSV = latest_csv(os.path.join(BASE, "magnetization/magnetization_results/mms/*_magnetization_convergence.csv"))
NS_CSV_A = latest_csv(os.path.join(BASE, "navier_stokes/navier_stokes_results/mms/*_ns_mms_2d_phase_a.csv"))
NS_CSV_B = latest_csv(os.path.join(BASE, "navier_stokes/navier_stokes_results/mms/*_ns_mms_2d_phase_b.csv"))
NS_CSV_C = latest_csv(os.path.join(BASE, "navier_stokes/navier_stokes_results/mms/*_ns_mms_2d_phase_c.csv"))
NS_CSV_D = latest_csv(os.path.join(BASE, "navier_stokes/navier_stokes_results/mms/*_ns_mms_2d_phase_d.csv"))
POISSON_MAG_CSV = latest_csv(os.path.join(BASE, "mms_tests/cmake-build-debug/mms_results/poisson_mag_mms_rates.csv"))


def load_csv(path):
    """Load CSV with numpy, handling header."""
    if path is None or not os.path.exists(path):
        return None
    data = np.genfromtxt(path, delimiter=',', names=True, dtype=None, encoding='utf-8')
    return data


def add_reference_slope(ax, h, y_start, slope, label, color='gray', ls='--'):
    """Add a reference slope line to a log-log plot."""
    h = np.array(h, dtype=float)
    y = y_start * (h / h[0]) ** slope
    ax.plot(h, y, color=color, ls=ls, lw=1.0, alpha=0.6, label=f'$O(h^{{{slope}}})$')


def compute_rate(h, e):
    """Compute convergence rate between consecutive entries."""
    rates = []
    for i in range(1, len(h)):
        if e[i] > 0 and e[i-1] > 0 and h[i] > 0 and h[i-1] > 0:
            rates.append(np.log(e[i-1]/e[i]) / np.log(h[i-1]/h[i]))
        else:
            rates.append(0.0)
    return rates


# ============================================================================
# Style
# ============================================================================
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'lines.markersize': 6,
    'lines.linewidth': 1.5,
})

COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']
MARKERS = ['o', 's', '^', 'D', 'v', 'P']


# ============================================================================
# 1. POISSON
# ============================================================================
def plot_poisson():
    data = load_csv(POISSON_CSV)
    if data is None:
        print("  [SKIP] Poisson CSV not found")
        return
    h = data['h']
    L2 = data['L2_error']
    H1 = data['H1_error']
    Linf = data['Linf_error']

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5))
    ax.loglog(h, L2, '-o', color=COLORS[0], label=f'$L^2$ (rate={compute_rate(h,L2)[-1]:.2f})')
    ax.loglog(h, H1, '-s', color=COLORS[1], label=f'$H^1$ (rate={compute_rate(h,H1)[-1]:.2f})')
    ax.loglog(h, Linf, '-^', color=COLORS[2], label=f'$L^\\infty$ (rate={compute_rate(h,Linf)[-1]:.2f})')

    # Reference slopes
    add_reference_slope(ax, h, L2[0]*1.5, 2, 'O(h²)')
    add_reference_slope(ax, h, H1[0]*1.5, 1, 'O(h¹)')

    ax.set_xlabel('Mesh size $h$')
    ax.set_ylabel('Error')
    ax.set_title('Poisson MMS Convergence (Q1)')
    ax.legend(loc='lower right')
    ax.grid(True, which='both', alpha=0.3)
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'mms_poisson.png'), dpi=200)
    plt.close(fig)
    print(f"  [OK] Poisson: L2={compute_rate(h,L2)[-1]:.2f}, H1={compute_rate(h,H1)[-1]:.2f}, Linf={compute_rate(h,Linf)[-1]:.2f}")


# ============================================================================
# 2. CAHN-HILLIARD
# ============================================================================
def plot_ch():
    data = load_csv(CH_CSV)
    if data is None:
        print("  [SKIP] CH CSV not found")
        return
    h = data['h']
    tL2 = data['theta_L2']
    tH1 = data['theta_H1']
    tLinf = data['theta_Linf']
    pL2 = data['psi_L2']
    pLinf = data['psi_Linf']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # theta
    ax1.loglog(h, tL2, '-o', color=COLORS[0], label=f'$\\theta$ $L^2$ (rate={compute_rate(h,tL2)[-1]:.2f})')
    ax1.loglog(h, tH1, '-s', color=COLORS[1], label=f'$\\theta$ $H^1$ (rate={compute_rate(h,tH1)[-1]:.2f})')
    ax1.loglog(h, tLinf, '-^', color=COLORS[2], label=f'$\\theta$ $L^\\infty$ (rate={compute_rate(h,tLinf)[-1]:.2f})')
    add_reference_slope(ax1, h, tL2[0]*1.5, 3, 'O(h³)')
    add_reference_slope(ax1, h, tH1[0]*1.5, 2, 'O(h²)')
    ax1.set_xlabel('Mesh size $h$')
    ax1.set_ylabel('Error')
    ax1.set_title('CH: Phase field $\\theta$ (Q2)')
    ax1.legend(loc='lower right', fontsize=8)
    ax1.grid(True, which='both', alpha=0.3)
    ax1.invert_xaxis()

    # psi
    ax2.loglog(h, pL2, '-o', color=COLORS[0], label=f'$\\psi$ $L^2$ (rate={compute_rate(h,pL2)[-1]:.2f})')
    ax2.loglog(h, pLinf, '-^', color=COLORS[2], label=f'$\\psi$ $L^\\infty$ (rate={compute_rate(h,pLinf)[-1]:.2f})')
    add_reference_slope(ax2, h, pL2[0]*1.5, 3, 'O(h³)')
    ax2.set_xlabel('Mesh size $h$')
    ax2.set_ylabel('Error')
    ax2.set_title('CH: Chemical potential $\\psi$ (Q2)')
    ax2.legend(loc='lower right', fontsize=8)
    ax2.grid(True, which='both', alpha=0.3)
    ax2.invert_xaxis()

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'mms_cahn_hilliard.png'), dpi=200)
    plt.close(fig)
    print(f"  [OK] CH: theta_L2={compute_rate(h,tL2)[-1]:.2f}, theta_H1={compute_rate(h,tH1)[-1]:.2f}, theta_Linf={compute_rate(h,tLinf)[-1]:.2f}")


# ============================================================================
# 3. MAGNETIZATION
# ============================================================================
def plot_mag():
    data = load_csv(MAG_CSV)
    if data is None:
        print("  [SKIP] Mag CSV not found")
        return
    h = data['h']
    ML2 = data['M_L2']
    MLinf = data['M_Linf']

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5))
    ax.loglog(h, ML2, '-o', color=COLORS[0], label=f'$M$ $L^2$ (rate={compute_rate(h,ML2)[-1]:.2f})')
    ax.loglog(h, MLinf, '-^', color=COLORS[2], label=f'$M$ $L^\\infty$ (rate={compute_rate(h,MLinf)[-1]:.2f})')
    add_reference_slope(ax, h, ML2[0]*1.5, 2, 'O(h²)')
    ax.set_xlabel('Mesh size $h$')
    ax.set_ylabel('Error')
    ax.set_title('Magnetization MMS Convergence (DG-Q1)')
    ax.legend(loc='lower right')
    ax.grid(True, which='both', alpha=0.3)
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'mms_magnetization.png'), dpi=200)
    plt.close(fig)
    print(f"  [OK] Mag: M_L2={compute_rate(h,ML2)[-1]:.2f}, M_Linf={compute_rate(h,MLinf)[-1]:.2f}")


# ============================================================================
# 4. NAVIER-STOKES (all 4 phases in 2x2 figure)
# ============================================================================
def _plot_ns_phase(ax, data, title):
    """Plot a single NS phase on the given axes."""
    h = data['h']
    ux_L2 = data['ux_L2']
    ux_H1 = data['ux_H1']
    ux_Linf = data['ux_Linf']
    p_L2 = data['p_L2']
    p_Linf = data['p_Linf']
    div_L2 = data['div_L2']

    ax.loglog(h, ux_L2, '-o', color=COLORS[0], label=f'$u$ $L^2$ ({compute_rate(h,ux_L2)[-1]:.2f})')
    ax.loglog(h, ux_H1, '-s', color=COLORS[1], label=f'$u$ $H^1$ ({compute_rate(h,ux_H1)[-1]:.2f})')
    ax.loglog(h, ux_Linf, '-^', color=COLORS[2], label=f'$u$ $L^\\infty$ ({compute_rate(h,ux_Linf)[-1]:.2f})')
    ax.loglog(h, p_L2, '-D', color=COLORS[3], label=f'$p$ $L^2$ ({compute_rate(h,p_L2)[-1]:.2f})')
    ax.loglog(h, p_Linf, '-v', color=COLORS[4], label=f'$p$ $L^\\infty$ ({compute_rate(h,p_Linf)[-1]:.2f})')
    ax.loglog(h, div_L2, '-P', color=COLORS[5], label=f'div $L^2$ ({compute_rate(h,div_L2)[-1]:.2f})')

    add_reference_slope(ax, h, p_L2[0]*2, 2, 'O(h²)')
    add_reference_slope(ax, h, ux_L2[0]*0.5, 3, 'O(h³)')

    ax.set_xlabel('Mesh size $h$')
    ax.set_ylabel('Error')
    ax.set_title(title)
    ax.legend(loc='best', fontsize=6.5)
    ax.grid(True, which='both', alpha=0.3)
    ax.invert_xaxis()


def plot_ns():
    data_a = load_csv(NS_CSV_A)
    data_b = load_csv(NS_CSV_B)
    data_c = load_csv(NS_CSV_C)
    data_d = load_csv(NS_CSV_D)
    if data_a is None:
        print("  [SKIP] NS CSV not found")
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    phase_data = [
        (data_a, 'Phase A: Steady Stokes'),
        (data_b, 'Phase B: Unsteady Stokes'),
        (data_c, 'Phase C: Steady NS'),
        (data_d, 'Phase D: Unsteady NS'),
    ]

    for idx, (data, title) in enumerate(phase_data):
        if data is not None:
            ax = axes[idx // 2, idx % 2]
            _plot_ns_phase(ax, data, title)
        else:
            print(f"  [SKIP] NS {title} CSV not found")

    fig.suptitle('Navier-Stokes MMS Convergence (Q2/DG-Q1)', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT_DIR, 'mms_navier_stokes.png'), dpi=200)
    plt.close(fig)

    h = data_a['h']
    for label, data in [('A', data_a), ('B', data_b), ('C', data_c), ('D', data_d)]:
        if data is not None:
            hd = data['h']
            print(f"  [OK] NS Phase {label}: ux_L2={compute_rate(hd,data['ux_L2'])[-1]:.2f}, ux_H1={compute_rate(hd,data['ux_H1'])[-1]:.2f}, p_L2={compute_rate(hd,data['p_L2'])[-1]:.2f}")


# ============================================================================
# 5. POISSON + MAGNETIZATION (coupled)
# ============================================================================
def plot_poisson_mag():
    data = load_csv(POISSON_MAG_CSV)
    if data is None:
        print("  [SKIP] Poisson+Mag CSV not found")
        return
    h = data['h']
    phi_L2 = data['phi_L2']
    phi_H1 = data['phi_H1']
    M_L2 = data['M_L2']

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5))
    ax.loglog(h, phi_L2, '-o', color=COLORS[0], label=f'$\\phi$ $L^2$ (rate={compute_rate(h,phi_L2)[-1]:.2f})')
    ax.loglog(h, phi_H1, '-s', color=COLORS[1], label=f'$\\phi$ $H^1$ (rate={compute_rate(h,phi_H1)[-1]:.2f})')
    ax.loglog(h, M_L2, '-^', color=COLORS[2], label=f'$M$ $L^2$ (rate={compute_rate(h,M_L2)[-1]:.2f})')

    add_reference_slope(ax, h, phi_L2[0]*1.5, 3, 'O(h³)')
    add_reference_slope(ax, h, phi_H1[0]*1.5, 2, 'O(h²)')

    ax.set_xlabel('Mesh size $h$')
    ax.set_ylabel('Error')
    ax.set_title('Poisson+Magnetization Coupled MMS')
    ax.legend(loc='lower right')
    ax.grid(True, which='both', alpha=0.3)
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'mms_poisson_mag.png'), dpi=200)
    plt.close(fig)
    print(f"  [OK] Poisson+Mag: phi_L2={compute_rate(h,phi_L2)[-1]:.2f}, phi_H1={compute_rate(h,phi_H1)[-1]:.2f}, M_L2={compute_rate(h,M_L2)[-1]:.2f}")


# ============================================================================
# 6. COMBINED SUMMARY PLOT (all subsystems in one figure)
# ============================================================================
def plot_combined():
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # (0,0) Poisson
    data = load_csv(POISSON_CSV)
    if data is not None:
        ax = axes[0, 0]
        h = data['h']
        ax.loglog(h, data['L2_error'], '-o', color=COLORS[0], label=f'$L^2$ ({compute_rate(h,data["L2_error"])[-1]:.1f})')
        ax.loglog(h, data['H1_error'], '-s', color=COLORS[1], label=f'$H^1$ ({compute_rate(h,data["H1_error"])[-1]:.1f})')
        ax.loglog(h, data['Linf_error'], '-^', color=COLORS[2], label=f'$L^\\infty$ ({compute_rate(h,data["Linf_error"])[-1]:.1f})')
        add_reference_slope(ax, h, data['L2_error'][0]*1.5, 2, '')
        ax.set_title('Poisson (Q1)')
        ax.legend(loc='lower right', fontsize=7)
        ax.grid(True, which='both', alpha=0.3)
        ax.invert_xaxis()
        ax.set_ylabel('Error')

    # (0,1) Cahn-Hilliard
    data = load_csv(CH_CSV)
    if data is not None:
        ax = axes[0, 1]
        h = data['h']
        ax.loglog(h, data['theta_L2'], '-o', color=COLORS[0], label=f'$\\theta$ $L^2$ ({compute_rate(h,data["theta_L2"])[-1]:.1f})')
        ax.loglog(h, data['theta_H1'], '-s', color=COLORS[1], label=f'$\\theta$ $H^1$ ({compute_rate(h,data["theta_H1"])[-1]:.1f})')
        ax.loglog(h, data['theta_Linf'], '-^', color=COLORS[2], label=f'$\\theta$ $L^\\infty$ ({compute_rate(h,data["theta_Linf"])[-1]:.1f})')
        add_reference_slope(ax, h, data['theta_L2'][0]*1.5, 3, '')
        ax.set_title('Cahn-Hilliard (Q2)')
        ax.legend(loc='lower right', fontsize=7)
        ax.grid(True, which='both', alpha=0.3)
        ax.invert_xaxis()

    # (0,2) Magnetization
    data = load_csv(MAG_CSV)
    if data is not None:
        ax = axes[0, 2]
        h = data['h']
        ax.loglog(h, data['M_L2'], '-o', color=COLORS[0], label=f'$M$ $L^2$ ({compute_rate(h,data["M_L2"])[-1]:.1f})')
        ax.loglog(h, data['M_Linf'], '-^', color=COLORS[2], label=f'$M$ $L^\\infty$ ({compute_rate(h,data["M_Linf"])[-1]:.1f})')
        add_reference_slope(ax, h, data['M_L2'][0]*1.5, 2, '')
        ax.set_title('Magnetization (DG-Q1)')
        ax.legend(loc='lower right', fontsize=7)
        ax.grid(True, which='both', alpha=0.3)
        ax.invert_xaxis()

    # (1,0) NS Phase A
    data = load_csv(NS_CSV_A)
    if data is not None:
        ax = axes[1, 0]
        h = data['h']
        ax.loglog(h, data['ux_L2'], '-o', color=COLORS[0], label=f'$u$ $L^2$ ({compute_rate(h,data["ux_L2"])[-1]:.1f})')
        ax.loglog(h, data['p_L2'], '-D', color=COLORS[3], label=f'$p$ $L^2$ ({compute_rate(h,data["p_L2"])[-1]:.1f})')
        ax.loglog(h, data['div_L2'], '-P', color=COLORS[5], label=f'div $L^2$ ({compute_rate(h,data["div_L2"])[-1]:.1f})')
        add_reference_slope(ax, h, data['p_L2'][0]*2, 2, '')
        ax.set_title('NS: Steady Stokes (Q2/DG-Q1)')
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, which='both', alpha=0.3)
        ax.invert_xaxis()
        ax.set_xlabel('Mesh size $h$')
        ax.set_ylabel('Error')

    # (1,1) NS Phase D
    data = load_csv(NS_CSV_D)
    if data is not None:
        ax = axes[1, 1]
        h = data['h']
        ax.loglog(h, data['ux_L2'], '-o', color=COLORS[0], label=f'$u$ $L^2$ ({compute_rate(h,data["ux_L2"])[-1]:.1f})')
        ax.loglog(h, data['p_L2'], '-D', color=COLORS[3], label=f'$p$ $L^2$ ({compute_rate(h,data["p_L2"])[-1]:.1f})')
        ax.loglog(h, data['div_L2'], '-P', color=COLORS[5], label=f'div $L^2$ ({compute_rate(h,data["div_L2"])[-1]:.1f})')
        add_reference_slope(ax, h, data['div_L2'][0]*2, 2, '')
        ax.set_title('NS: Unsteady NS (Q2/DG-Q1)')
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, which='both', alpha=0.3)
        ax.invert_xaxis()
        ax.set_xlabel('Mesh size $h$')

    # (1,2) Poisson+Mag coupled
    data = load_csv(POISSON_MAG_CSV)
    if data is not None:
        ax = axes[1, 2]
        h = data['h']
        ax.loglog(h, data['phi_L2'], '-o', color=COLORS[0], label=f'$\\phi$ $L^2$ ({compute_rate(h,data["phi_L2"])[-1]:.1f})')
        ax.loglog(h, data['phi_H1'], '-s', color=COLORS[1], label=f'$\\phi$ $H^1$ ({compute_rate(h,data["phi_H1"])[-1]:.1f})')
        ax.loglog(h, data['M_L2'], '-^', color=COLORS[2], label=f'$M$ $L^2$ ({compute_rate(h,data["M_L2"])[-1]:.1f})')
        add_reference_slope(ax, h, data['phi_L2'][0]*1.5, 3, '')
        ax.set_title('Poisson+Mag Coupled')
        ax.legend(loc='lower right', fontsize=7)
        ax.grid(True, which='both', alpha=0.3)
        ax.invert_xaxis()
        ax.set_xlabel('Mesh size $h$')

    fig.suptitle('MMS Convergence Study — All Subsystems', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT_DIR, 'mms_all_subsystems.png'), dpi=200)
    plt.close(fig)
    print(f"  [OK] Combined plot saved")


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  MMS Convergence Plot Generator")
    print("=" * 60)
    print(f"  Output directory: {OUT_DIR}")
    print()

    print("Individual plots:")
    plot_poisson()
    plot_ch()
    plot_mag()
    plot_ns()
    plot_poisson_mag()

    print("\nCombined plot:")
    plot_combined()

    print(f"\nAll plots saved to: {OUT_DIR}/")
    print("=" * 60)
