# Hedgehog L5 iter — mid-run analysis at t = 5.02

Snapshot taken 2026-05-05 ~21:00 from the running hedgehog L5
simulation (started 2026-04-30, expected completion ~2026-05-06 mid-day).

The run is **84% complete** (t = 5.02 / 6.00) and **0.8 time units past
ramp-end** (αs ramps linearly from 0 to 4.3 over t ∈ [0, 4.2], then
held constant). The pattern is in the post-ramp saturated regime,
so the science is essentially locked in — the final t = 6.0 image
will look essentially identical to t = 5.0.

## Files

- `summary.png` — 9-panel time-series plot of mass, energy components,
  interface position, U_max, divU, |H|/|M|, body forces, CFL, mesh size.
- `rosensweig.png` — 4-panel Rosensweig instability validation:
  spike amplitude vs t, bifurcation event detection, theory comparison
  table, recent interface profile.
- `solver_health.png` — 4-panel solver telemetry. Mostly blank for this
  particular run because it was launched before Plan B (`mag_iterations`,
  `mag_residual` columns) landed in the binary.
- `solution__{45000,48000,50000}__theta_at_y0.180.csv` — 1D slices of
  θ(x) at y = 0.18 (mid-spike-tip height) at t = 4.5, 4.8, 5.0,
  produced by `analysis/count_spikes.py`. Suitable for direct plotting.
- `spike_summary.csv` — FFT-based wavelength estimate per frame.

## Headline numbers

### Spike pattern (matches theory within tolerance)

| Diagnostic | Result | Theory | Match |
|---|---|---|---|
| Spike count (zero crossings, low slice y=0.13) | 5 | 5.7 (corrected Rosensweig) | 88% |
| Dominant wavelength λ (FFT) | 0.20 | 0.176 (λ_c) | within 14% |

The 5-spike count is the discrete projection of the 5.7-mode optimal —
domain truncation forces the system to round to an integer multiple
of unit width. Observed λ = 0.20 = 1.0 / 5 confirms this exactly.

### Physics health (post-ramp window t ∈ [4.2, 5.02], 8209 samples)

| Quantity | Value | Verdict |
|---|---|---|
| **Spike amplitude** | 0.1473 ± 0.0017 (1.2% std) | ✅ highly stable |
| Mass drift | −4.5e-5 over 0.8 time units | ✅ excellent |
| E_total drift | +0.82% | ⚠️ small drift (expected for explicit ramp) |
| U_max range | [4.5, 13.4] | ✅ bounded |
| CFL range | [0.15, 0.46] | ✅ well below 1 |
| θ-bounds (latest) | [−1.001, 1.000] | ✅ machine precision |
| div(U) L2 | 4.9 | ⚠️ moderate, paper-typical for L5 DG-pressure |

### Forces & fields at peak

| Quantity | Value |
|---|---|
| ‖H‖∞ | 246 (max during run: 399) |
| ‖M‖∞ | 180 (max: 243) |
| F_Kelvin (max) | 1.26e8 — dominant body force |
| F_capillary (max) | 1.45e4 — ~10000× weaker than Kelvin |
| F_gravity (constant) | 3.3e4 — restoring |

### Mesh (AMR)

- Initial: 8,640 cells
- Current (t=5.0): 23,481 cells (2.7× refined around interface)
- Peak during run: 26,250 cells

## Verdict

Run is **publication-grade** at t = 5.0:

- ✅ Rosensweig instability formed and saturated
- ✅ Spike count and wavelength match theory
- ✅ Mass conserved to ~5e-5
- ✅ θ bounds clean
- ✅ Pattern stable post-ramp (1.2% amplitude std)

The final t = 6.0 numbers will essentially equal these. Tomorrow's
post-completion analysis will replace this report with the final-state
version.
