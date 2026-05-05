#!/usr/bin/env python3
"""
analyze_hedgehog.py — single-pass analysis of a Semi_Coupled hedgehog run.

Reads diagnostics.csv (and optionally energy.csv) from a run directory and
produces:
  - 9-panel time-series figure (mass, energy, interface, velocity, divU,
    magnetic field, force, CFL, mesh cells)
  - Rosensweig instability summary (λ_c theory vs. observed wavelength,
    spike count, spike amplitude)
  - Pre/post-spike-onset comparison panel highlighting the bifurcation event

Usage:
  python3 analyze_hedgehog.py <run_dir>
  python3 analyze_hedgehog.py <run_dir> --out plots/

The script tolerates partial runs (i.e., still in progress) so it can be
used mid-run for quick health checks.
"""
from __future__ import annotations
import argparse
import csv
import math
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_diagnostics(csv_path: Path) -> dict[str, np.ndarray]:
    """Load diagnostics.csv into a dict of column-name -> ndarray of floats.
    Skips comment rows starting with '#' and the header row.
    """
    if not csv_path.exists():
        sys.exit(f"diagnostics.csv not found at {csv_path}")
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = []
        for row in reader:
            if not row or row[0].lstrip().startswith("#"):
                continue
            try:
                int(row[0])  # only numeric step rows
            except ValueError:
                continue
            rows.append(row)
    if not rows:
        sys.exit("no numeric rows found in diagnostics.csv")
    arr = np.array(rows, dtype=float)
    return {name: arr[:, i] for i, name in enumerate(header)}


def parse_run_info(run_dir: Path) -> dict[str, str]:
    """Extract a few key parameters from run_info.txt for use in plots."""
    info_path = run_dir / "run_info.txt"
    out: dict[str, str] = {}
    if not info_path.exists():
        return out
    for line in info_path.read_text().splitlines():
        for key in ("preset", "dt", "t_final", "eps", "lambda", "chi",
                    "pool_depth", "r"):
            if line.strip().startswith(f"{key}"):
                parts = line.split("=")
                if len(parts) >= 2:
                    out[key] = parts[1].strip().split()[0].rstrip(",")
    return out


# ---------------------------------------------------------------------------
# Rosensweig instability theory
# ---------------------------------------------------------------------------
# Reference: Rosensweig, "Ferrohydrodynamics" (1985), Ch. 7. The classical
# linear-stability dispersion relation for a deep ferrofluid pool with a
# vertical applied field is
#
#     ρ̄ ω² = Δρ g k + σ k³ − μ₀ M² k² · F(μ_r)
#
# where Δρ = ρ_ferro − ρ_air is the *density contrast* (NOT the absolute
# density), σ the physical surface tension, M the equilibrium magnetization,
# μ_r = 1+χ, and F(μ_r) = (μ_r − 1)² / (μ_r(μ_r + 1)) is the demag factor.
#
# At neutral stability (ω² = 0), the *critical wavenumber* is
#
#     k_c = √(Δρ g / σ)              ⇒  λ_c = 2π √(σ / (Δρ g))
#
# The associated *critical magnetization* satisfies
#
#     μ₀ M_c² · F(μ_r) = 2 √(Δρ g σ)
#
# Above threshold (B = M²/M_c² > 1), the dominant (most-unstable) wavenumber
# shifts to
#
#     k_m = k_c · √(B − √(B² − 1))      (Rosensweig 1985, Eq. 7.5.18)
#
# In the Cahn-Hilliard formulation used by Nochetto CMAME 2016, the
# physical surface tension σ is related to the (λ, ε) interface
# parameters by the standard equipartition integral:
#
#     σ = (2√2 / 3) · (λ / ε)        for double-well W(θ) = (1−θ²)²/4
#
# This is the constant that calibrates the diffuse-interface energy to a
# sharp-interface free-surface tension.

# Cahn-Hilliard double-well calibration constant. See, e.g., Boyer &
# Lapuerta, M2AN 40 (2006) 653-687, Eq. (1.6).
CH_TENSION_CONST = 2.0 * math.sqrt(2.0) / 3.0  # ≈ 0.9428


def rosensweig_critical_wavelength(sigma: float, delta_rho: float,
                                   g: float) -> float:
    """λ_c = 2π √(σ / (Δρ g)). Δρ is the density *contrast*, not absolute."""
    if delta_rho <= 0 or g <= 0 or sigma <= 0:
        return float("nan")
    return 2.0 * math.pi * math.sqrt(sigma / (delta_rho * g))


def rosensweig_critical_magnetization(sigma: float, delta_rho: float,
                                       g: float, chi: float,
                                       mu0: float = 1.0) -> float:
    """Critical magnetization at instability onset.

    μ₀ M_c² · F(μ_r) = 2 √(Δρ g σ),  where F(μ_r) = (μ_r − 1)² / (μ_r (μ_r + 1)).

    Returns NaN if any input is non-physical.
    """
    if sigma <= 0 or delta_rho <= 0 or g <= 0 or chi <= 0 or mu0 <= 0:
        return float("nan")
    mu_r = 1.0 + chi
    F = (mu_r - 1.0) ** 2 / (mu_r * (mu_r + 1.0))
    if F <= 0:
        return float("nan")
    return math.sqrt(2.0 * math.sqrt(delta_rho * g * sigma) / (mu0 * F))


def rosensweig_dominant_wavelength(sigma: float, delta_rho: float, g: float,
                                    M: float, M_c: float) -> float:
    """Most-unstable wavelength when above threshold (M > M_c).

    k_m = k_c · √(B − √(B² − 1))   with  B = (M/M_c)²

    Returns the critical λ_c at threshold (B≤1) and λ_c · k_c/k_m above it.
    """
    lam_c = rosensweig_critical_wavelength(sigma, delta_rho, g)
    if not math.isfinite(lam_c) or M_c <= 0 or M <= 0:
        return lam_c
    B = (M / M_c) ** 2
    if B <= 1.0:
        return lam_c
    k_ratio = math.sqrt(B - math.sqrt(B * B - 1.0))
    if k_ratio <= 0:
        return lam_c
    return lam_c / k_ratio


def estimate_spike_count(y_max: float, y_mean: float, y_min: float,
                         domain_width: float = 1.0) -> tuple[int, float]:
    """Heuristic spike count from the y_max - y_min spread.
    Returns (n_spikes, amplitude).  If amplitude < 1% of pool, returns 0.
    """
    amp = max(0.0, y_max - y_mean)
    if amp < 0.01:
        return 0, amp
    # rough spacing ≈ 0.2 (for d=2 hedgehog with 5 dipoles)
    return max(1, int(round(domain_width / 0.25))), amp


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_summary(d: dict[str, np.ndarray], info: dict[str, str], out_path: Path):
    t = d["time"]
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    fig.suptitle(f"Hedgehog L5 — diagnostics summary  "
                 f"(rows: {len(t)}; t = {t[0]:.3f}…{t[-1]:.3f}; "
                 f"preset={info.get('preset','?')}, dt={info.get('dt','?')}, "
                 f"ε={info.get('eps','?')})",
                 fontsize=11, y=0.995)

    # ramp end indicator
    t_ramp = 4.2

    def shade_ramp(ax):
        ax.axvline(t_ramp, color="gray", lw=0.8, ls="--", alpha=0.7)
        ax.text(t_ramp, ax.get_ylim()[1], " ramp end",
                fontsize=7, color="gray", va="top")

    # 1. Mass conservation
    ax = axes[0, 0]
    ax.plot(t, d["mass"], lw=0.8)
    ax.set_title("Mass (should be ~constant)")
    ax.set_xlabel("t"); ax.set_ylabel("(θ, 1)")
    ax.grid(alpha=0.3)
    drift = abs(d["mass"][-1] - d["mass"][0])
    ax.text(0.02, 0.97, f"|Δ| = {drift:.2e}",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    # 2. Energy
    ax = axes[0, 1]
    if "E_internal" in d: ax.plot(t, d["E_internal"], lw=0.8, label="E_internal")
    if "E_CH" in d:       ax.plot(t, d["E_CH"], lw=0.8, label="E_CH")
    if "E_kin" in d:      ax.plot(t, d["E_kin"], lw=0.8, label="E_kin")
    if "E_mag" in d:      ax.plot(t, d["E_mag"], lw=0.8, label="E_mag")
    ax.set_title("Energy components")
    ax.set_xlabel("t"); ax.set_ylabel("E")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.legend(fontsize=7, loc="best"); ax.grid(alpha=0.3)
    shade_ramp(ax)

    # 3. Interface position
    ax = axes[0, 2]
    if "interface_y_max"  in d: ax.plot(t, d["interface_y_max"],  lw=0.8, label="y_max")
    if "interface_y_mean" in d: ax.plot(t, d["interface_y_mean"], lw=0.8, label="y_mean")
    if "interface_y_min"  in d: ax.plot(t, d["interface_y_min"],  lw=0.8, label="y_min")
    pool_depth = float(info.get("pool_depth", "0.11"))
    ax.axhline(pool_depth, color="black", lw=0.5, ls=":",
               label=f"pool depth = {pool_depth}")
    ax.set_title("Interface (θ=0 contour)")
    ax.set_xlabel("t"); ax.set_ylabel("y")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    shade_ramp(ax)

    # 4. Velocity
    ax = axes[1, 0]
    if "U_max" in d: ax.plot(t, d["U_max"], lw=0.8, label="‖U‖∞")
    ax.set_title("U_max")
    ax.set_xlabel("t"); ax.set_ylabel("|U|")
    ax.set_yscale("log"); ax.grid(alpha=0.3)
    shade_ramp(ax)

    # 5. Incompressibility
    ax = axes[1, 1]
    if "divU_L2"   in d: ax.plot(t, d["divU_L2"],   lw=0.8, label="‖∇·U‖_L²")
    if "divU_Linf" in d: ax.plot(t, d["divU_Linf"], lw=0.8, label="‖∇·U‖_L∞")
    ax.set_title("Incompressibility residual")
    ax.set_xlabel("t"); ax.set_ylabel("‖∇·U‖")
    ax.set_yscale("symlog", linthresh=1e-8)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    shade_ramp(ax)

    # 6. Magnetic field magnitudes
    ax = axes[1, 2]
    if "H_max" in d: ax.plot(t, d["H_max"], lw=0.8, label="‖H‖∞")
    if "M_max" in d: ax.plot(t, d["M_max"], lw=0.8, label="‖M‖∞")
    ax.set_title("Magnetic field / magnetization")
    ax.set_xlabel("t"); ax.set_ylabel("|·|")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    shade_ramp(ax)

    # 7. Forces
    ax = axes[2, 0]
    if "F_mag_max"  in d: ax.plot(t, d["F_mag_max"],  lw=0.8, label="F_Kelvin")
    if "F_cap_max"  in d: ax.plot(t, d["F_cap_max"],  lw=0.8, label="F_capillary")
    if "F_grav_max" in d: ax.plot(t, d["F_grav_max"], lw=0.8, label="F_gravity")
    ax.set_title("Body forces (max)")
    ax.set_xlabel("t"); ax.set_ylabel("|F|∞")
    ax.set_yscale("symlog", linthresh=1)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    shade_ramp(ax)

    # 8. CFL
    ax = axes[2, 1]
    if "CFL" in d: ax.plot(t, d["CFL"], lw=0.8)
    ax.axhline(1.0, color="red", lw=0.5, ls="--", label="CFL = 1")
    ax.set_title("CFL number")
    ax.set_xlabel("t"); ax.set_ylabel("CFL")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    shade_ramp(ax)

    # 9. Mesh cells
    ax = axes[2, 2]
    if "n_cells" in d: ax.plot(t, d["n_cells"], lw=0.8, label="n_cells")
    if "n_dofs"  in d: ax.plot(t, d["n_dofs"]/100.0, lw=0.8, label="n_dofs / 100")
    ax.set_title("Mesh size")
    ax.set_xlabel("t"); ax.set_ylabel("count")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    shade_ramp(ax)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"  → {out_path}")


def plot_rosensweig_validation(d: dict[str, np.ndarray], info: dict[str, str],
                                out_path: Path):
    t = d["time"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Rosensweig validation: instability detection",
                 fontsize=12, y=0.995)

    # Theory — physical parameters
    eps    = float(info.get("eps", "5e-3"))
    lam    = float(info.get("lambda", "2.5e-2"))
    chi0   = float(info.get("chi", "0.9"))
    # density contrast (Δρ = ρ_ferro − ρ_air = 2r, NOT ρ̄ = 1)
    r      = float(info.get("r", "0.1"))
    delta_rho = 2.0 * r
    g      = 30000.0
    mu0    = 1.0   # nondim units in Nochetto formulation
    # Diffuse-interface → sharp-interface surface-tension calibration
    sigma_eff = CH_TENSION_CONST * lam / eps
    lam_c     = rosensweig_critical_wavelength(sigma_eff, delta_rho, g)
    M_c       = rosensweig_critical_magnetization(sigma_eff, delta_rho, g, chi0, mu0)
    # Bond number from observed M (use last value if available)
    M_now   = float(d.get("M_max", np.array([0.0]))[-1])
    B       = (M_now / M_c) ** 2 if M_c > 0 and math.isfinite(M_c) else 0.0
    lam_m   = rosensweig_dominant_wavelength(sigma_eff, delta_rho, g, M_now, M_c)

    # 1. Interface envelope: y_max - y_min
    ax = axes[0, 0]
    if "interface_y_max" in d and "interface_y_mean" in d:
        amp = d["interface_y_max"] - d["interface_y_mean"]
        ax.plot(t, amp, lw=0.8, label="y_max − y_mean")
    ax.set_title("Spike amplitude")
    ax.set_xlabel("t"); ax.set_ylabel("y_max − y_mean")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.axvline(4.2, color="gray", lw=0.5, ls="--")

    # 2. Bifurcation event detection (large d/dt of y_max)
    ax = axes[0, 1]
    if "interface_y_max" in d:
        ymax = d["interface_y_max"]
        if len(t) > 10:
            dy = np.gradient(ymax, t)
            ax.plot(t, dy, lw=0.8, label="d(y_max)/dt")
            # find peaks > 5σ above median to flag bifurcation events
            thresh = np.median(np.abs(dy)) + 5 * np.std(dy[np.abs(dy) < np.percentile(np.abs(dy), 95)])
            spikes_t = t[np.abs(dy) > thresh]
            for st in spikes_t[:5]:
                ax.axvline(st, color="red", lw=0.4, alpha=0.5)
            if len(spikes_t):
                ax.text(0.02, 0.97, f"bifurcation events flagged: {len(spikes_t)}\n"
                                    f"first at t = {spikes_t[0]:.3f}",
                        transform=ax.transAxes, fontsize=8, va="top",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    ax.set_title("Bifurcation detection (red = events)")
    ax.set_xlabel("t"); ax.set_ylabel("d(y_max)/dt")
    ax.grid(alpha=0.3)

    # 3. Theory comparison
    ax = axes[1, 0]
    ax.axis("off")
    spike_pred_c = 1.0 / lam_c if lam_c > 0 and math.isfinite(lam_c) else float("nan")
    spike_pred_m = 1.0 / lam_m if lam_m > 0 and math.isfinite(lam_m) else float("nan")
    bond_str = f"{B:.3f}" + ("  (above threshold)" if B > 1.0
                              else "  (below threshold)")
    info_text = (
        "Linear-stability theory (Rosensweig 1985)\n"
        "─────────────────────────────────────────\n"
        f"  λ, ε                       = {lam:.3e}, {eps:.3e}\n"
        f"  σ = (2√2/3)·λ/ε            = {sigma_eff:.4f}\n"
        f"  Δρ = 2r (contrast, not ρ̄) = {delta_rho:.3f}\n"
        f"  g, μ₀                      = {g:.0f}, {mu0:.2f}\n"
        f"  χ₀                         = {chi0:.2f}\n"
        f"  ───────────────────────\n"
        f"  λ_c = 2π√(σ/Δρ·g)          = {lam_c:.4f}\n"
        f"  M_c (critical magnetiz.)   = {M_c:.4f}\n"
        f"  M (observed, latest)       = {M_now:.4f}\n"
        f"  Bond B = (M/M_c)²          = {bond_str}\n"
        f"  λ_m (dominant if B>1)      = {lam_m:.4f}\n"
        f"  ───────────────────────\n"
        f"  Domain width               = 1.0\n"
        f"  Predicted spikes (λ_c)     ≈ {spike_pred_c:.2f}\n"
        f"  Predicted spikes (λ_m)     ≈ {spike_pred_m:.2f}\n"
    )
    ax.text(0.0, 0.95, info_text, transform=ax.transAxes,
            fontsize=9, family="monospace", va="top")

    # 4. Latest interface profile (just the 1D summary stats)
    ax = axes[1, 1]
    n_last = min(200, len(t))
    if "interface_y_max" in d:
        ax.plot(t[-n_last:], d["interface_y_max"][-n_last:], lw=1.0, label="y_max")
    if "interface_y_mean" in d:
        ax.plot(t[-n_last:], d["interface_y_mean"][-n_last:], lw=1.0, label="y_mean")
    if "interface_y_min" in d:
        ax.plot(t[-n_last:], d["interface_y_min"][-n_last:], lw=1.0, label="y_min")
    ax.set_title(f"Recent interface (last {n_last} steps)")
    ax.set_xlabel("t"); ax.set_ylabel("y")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"  → {out_path}")


def plot_solver_health(d: dict[str, np.ndarray], info: dict[str, str],
                       out_path: Path):
    """Iteration counts + solve times — useful for tuning Plan A's adaptive
    rebuild threshold and spotting GMRES staleness."""
    t = d["time"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Solver health (Plan A / Plan B telemetry)",
                 fontsize=12, y=0.995)

    def _safe_legend(ax):
        """Show legend only if there is at least one labeled artist."""
        h, _ = ax.get_legend_handles_labels()
        if h:
            ax.legend(fontsize=7)

    def _safe_log(ax, series_list):
        """Switch to log y only if any plotted series has positive values."""
        if any((s > 0).any() for s in series_list if s is not None):
            ax.set_yscale("log")

    # 1. Magnetic GMRES iterations
    ax = axes[0, 0]
    if "mag_iterations" in d:
        ax.plot(t, d["mag_iterations"], lw=0.6, alpha=0.7, label="mag GMRES")
        ax.axhline(50, color="red", lw=0.5, ls="--",
                   label="Plan A rebuild trigger (50)")
        # Recent average
        n_recent = min(500, len(t))
        recent = d["mag_iterations"][-n_recent:]
        ax.text(0.02, 0.97,
                f"recent: min={int(recent.min())}, "
                f"median={int(np.median(recent))}, "
                f"max={int(recent.max())}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    else:
        ax.text(0.5, 0.5, "mag_iterations\nnot in CSV\n(pre-Plan-B run)",
                ha="center", va="center", transform=ax.transAxes, fontsize=10,
                color="gray")
    ax.set_title("Magnetic GMRES iterations / step")
    ax.set_xlabel("t"); ax.set_ylabel("iters")
    ax.grid(alpha=0.3); _safe_legend(ax)

    # 2. CH GMRES iterations
    ax = axes[0, 1]
    if "ch_iterations" in d:
        ax.plot(t, d["ch_iterations"], lw=0.6, alpha=0.7, label="CH GMRES")
    ax.set_title("CH iterations / step")
    ax.set_xlabel("t"); ax.set_ylabel("iters")
    ax.grid(alpha=0.3); _safe_legend(ax)

    # 3. Solve times (CH, mag, NS)
    ax = axes[1, 0]
    times = []
    if "ch_time" in d:
        ax.plot(t, d["ch_time"], lw=0.6, alpha=0.7, label="CH"); times.append(d["ch_time"])
    if "poisson_time" in d:
        ax.plot(t, d["poisson_time"], lw=0.6, alpha=0.7, label="mag"); times.append(d["poisson_time"])
    if "ns_time" in d:
        ax.plot(t, d["ns_time"], lw=0.6, alpha=0.7, label="NS"); times.append(d["ns_time"])
    ax.set_title("Per-subsystem solve time (s)")
    ax.set_xlabel("t"); ax.set_ylabel("s")
    _safe_log(ax, times); ax.grid(alpha=0.3); _safe_legend(ax)

    # 4. Magnetic residual (relative)
    ax = axes[1, 1]
    res = []
    if "mag_residual" in d:
        ax.plot(t, d["mag_residual"], lw=0.6, alpha=0.7, label="mag rel-res")
        res.append(d["mag_residual"])
    if "ch_residual" in d:
        ax.plot(t, d["ch_residual"], lw=0.6, alpha=0.7, label="CH rel-res")
        res.append(d["ch_residual"])
    ax.set_title("Solver relative residuals")
    ax.set_xlabel("t"); ax.set_ylabel("residual")
    _safe_log(ax, res); ax.grid(alpha=0.3); _safe_legend(ax)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"  → {out_path}")


def print_health_summary(d: dict[str, np.ndarray], info: dict[str, str]):
    t = d["time"]
    last = -1
    print("─" * 64)
    print(f"  RUN HEALTH at step {int(d['step'][last])}, t = {t[last]:.4f}")
    print("─" * 64)
    if "mass" in d:
        drift = d["mass"][last] - d["mass"][0]
        print(f"  Mass drift               : {drift:+.3e}  "
              f"({'OK' if abs(drift) < 1e-3 else 'CHECK'})")
    if "theta_min" in d and "theta_max" in d:
        thmin = d["theta_min"][last]; thmax = d["theta_max"][last]
        ok = (thmin > -1.05) and (thmax < 1.05)
        print(f"  θ ∈ [{thmin:.4f}, {thmax:.4f}]    "
              f"({'OK' if ok else 'OVERSHOOT'})")
    if "CFL" in d:
        cfl_max = d["CFL"][-1000:].max() if len(t) > 1000 else d["CFL"].max()
        print(f"  CFL (recent max)         : {cfl_max:.3f}        "
              f"({'OK' if cfl_max < 1 else 'WARN'})")
    if "divU_L2" in d:
        div = d["divU_L2"][last]
        print(f"  divU_L2 (latest)         : {div:.3e}")
    if "U_max" in d:
        print(f"  U_max (latest)           : {d['U_max'][last]:.3f}")
    if "interface_y_max" in d and "interface_y_mean" in d:
        amp = d["interface_y_max"][last] - d["interface_y_mean"][last]
        pool = float(info.get("pool_depth", "0.11"))
        print(f"  spike amplitude (latest) : {amp:.4f}    "
              f"(pool={pool}, amp/pool = {amp/pool:.2f})")
    if "n_cells" in d:
        print(f"  cells (latest)           : {int(d['n_cells'][last])}")
    print("─" * 64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path,
                   help="Path to a run directory (the one containing diagnostics.csv)")
    p.add_argument("--out", type=Path, default=None,
                   help="Output directory for plots (default: <run_dir>/analysis/)")
    args = p.parse_args()

    if not args.run_dir.is_dir():
        sys.exit(f"not a directory: {args.run_dir}")
    out_dir = args.out or args.run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.run_dir / 'diagnostics.csv'} …")
    d = load_diagnostics(args.run_dir / "diagnostics.csv")
    info = parse_run_info(args.run_dir)
    print(f"  {len(d['time'])} rows, t = {d['time'][0]:.3f} … {d['time'][-1]:.3f}")

    print("Generating summary panel …")
    plot_summary(d, info, out_dir / "summary.png")

    print("Generating Rosensweig validation panel …")
    plot_rosensweig_validation(d, info, out_dir / "rosensweig.png")

    print("Generating solver health panel …")
    plot_solver_health(d, info, out_dir / "solver_health.png")

    print_health_summary(d, info)


if __name__ == "__main__":
    main()
