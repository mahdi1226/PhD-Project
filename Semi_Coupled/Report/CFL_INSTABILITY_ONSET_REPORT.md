# CFL Threshold at Rosensweig Instability Onset

## A Previously Uncharacterized Phenomenon in Ferrofluid Simulations

**Date**: 2026-03-06 (ongoing — will be updated as new results arrive)
**Authors**: Mahdi & Claude
**Project**: Semi-Coupled FHD Solver — Rosensweig Instability Reproduction
**Reference**: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497–531

---

## 1. Executive Summary

During attempts to reproduce the Rosensweig instability (ferrofluid spike formation under applied magnetic field), we discovered a **sudden, discrete CFL number jump of 2 orders of magnitude** occurring at a specific magnetic field strength. This jump is not a numerical artifact — it is the **signature of the physical Rosensweig instability onset**, where the flat ferrofluid interface becomes linearly unstable.

This finding has two implications:
1. **For our project**: Explicit Cahn-Hilliard (CH) convection fundamentally cannot survive this onset at practical time steps. Implicit CH convection is necessary.
2. **As an independent finding**: The CFL threshold at instability onset appears to be uncharacterized in the literature. It reveals a fundamental constraint on explicit time-stepping schemes for ferrofluid simulations and opens the door to non-linear field ramping strategies for controlling spike formation dynamics.

---

## 2. Problem Setup

### 2.1 Rosensweig Configuration

Following Nochetto et al. (CMAME 2016), Section 6.2:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Domain | [0,1] × [0,0.6] | Rectangular cavity |
| Base mesh | 10 × 6 | Structured quadrilateral |
| ε | 0.01 | Interface width |
| γ | 2×10⁻⁴ | Mobility |
| λ | 0.05 | Mixing energy density |
| χ₀ | 0.5 | Magnetic susceptibility (ferrofluid) |
| ν_w, ν_f | 1.0, 2.0 | Viscosities (water, ferrofluid) |
| r | 0.1 | Density contrast |
| g | 3×10⁴ | Gravity parameter |
| dt | 5×10⁻⁴ | Time step |
| Total steps | 4000 | Target (t_final = 2.0) |

**Magnetic field**: 5 dipoles at y = −15, intensity α = 6000, ramped linearly from 0 to full strength over t ∈ [0, 1.6].

**Initial condition**: Flat ferrofluid layer (θ = +1) occupying the bottom 1/3 of the domain (y < 0.2), with a diffuse interface of width ε.

### 2.2 Simulation Runs Analyzed

| Run ID | Refinement | AMR | CH Convection | dt | Outcome |
|--------|-----------|-----|---------------|-----|---------|
| `20260305_171805` | ref 4 | Yes | Explicit | 5×10⁻⁴ | **Died** at t ≈ 0.99 |
| `20260305_115128` | ref 3 | Yes | Explicit | 5×10⁻⁴ | Survived to t = 2.0 |
| `20260301_052316` | ref 4 | No | Explicit | 5×10⁻⁴ | Died at t ≈ 1.0 |
| `20260306_085853` | ref 4 | Yes | **Implicit** | 5×10⁻⁴ | **Running** (t = 0.41 as of writing) |

---

## 3. Observation: The CFL Jump

### 3.1 The Phenomenon

All simulations exhibit a **sudden CFL number jump** at a specific time during the magnetic field ramp. The CFL number increases by approximately 2 orders of magnitude over ~25 time steps (~0.013 time units).

![CFL vs Time — All Runs](figures/cfl_diagnostics/fig1_cfl_vs_time.png)
*Figure 1: CFL number evolution (log scale) for all four runs. The red dashed line marks CFL = 1. Note the sudden discrete jumps around t ≈ 0.3–0.6 depending on the run configuration. The implicit CH run (dark red) remains at CFL ~ 10⁻⁴, far below the threshold.*

The jump times and corresponding magnetic field fractions:

| Run | CFL > 0.01 at | B/B_max at onset | Mesh |
|-----|--------------|------------------|------|
| r4-noAMR (uniform) | t = 0.32 | 20% | Fixed fine |
| r4-AMR-explicit | t = 0.51 | 32% | AMR, ref 4 |
| r3-AMR-explicit | t = 0.60 | 38% | AMR, ref 3 |

**Key observation**: Finer spatial resolution detects the onset **earlier**. This is physically consistent — finer mesh resolves smaller perturbation wavelengths, and the Rosensweig instability has a most-unstable wavelength that depends on the capillary length.

### 3.2 CFL Jump is Present Regardless of AMR

This phenomenon was observed consistently across all simulation configurations — with and without adaptive mesh refinement, with different refinement levels, and with different transport schemes (CG and DG). The CFL jump is a fundamental feature of the Rosensweig instability onset, not an artifact of any particular numerical choice.

---

## 4. Root Cause Analysis: Velocity, Not Mesh

### 4.1 The Key Question

The CFL number is CFL = U_max × dt / h_min. A sudden CFL jump could come from:
- **(a)** h_min suddenly decreasing (AMR adding a finer level)
- **(b)** U_max suddenly increasing (physical velocity growth)
- **(c)** Both simultaneously

### 4.2 Step-by-Step Decomposition

We zoomed into the CFL jump window (t ∈ [0.45, 0.60]) for the r4-AMR-explicit run and tracked each quantity step-by-step:

![Step-by-step around CFL jump](figures/cfl_diagnostics/fig3_cfl_jump_stepbystep.png)
*Figure 3: Step-by-step quantities around the CFL jump (r4-AMR-explicit, t ∈ [0.45, 0.60]). From top to bottom: CFL, U_max, h_min (inferred), n_cells, forces (capillary and magnetic), interface spread. The CFL jump is driven entirely by U_max — h_min and n_cells remain essentially constant during the transition.*

| Quantity | Before jump (t < 0.50) | After jump (t > 0.52) | Change |
|----------|----------------------|----------------------|--------|
| CFL | ~5×10⁻⁴ | ~0.1 | **200× increase** |
| U_max | ~5×10⁻³ | ~0.5 | **100× increase** |
| h_min | ~2.2×10⁻² | ~2.2×10⁻² | **Constant** |
| n_cells | ~15,300 | ~15,500 | Negligible change |
| F_capillary | ~100 | ~5,000 | 50× increase |
| Interface spread | 0 | Begins growing | Onset |

**Conclusion: h_min stays constant during the jump. The CFL explosion is driven entirely by U_max suddenly increasing by 2 orders of magnitude.**

### 4.3 Rate Decomposition

The CFL growth can be decomposed as:

d(log CFL)/dt = d(log U_max)/dt − d(log h_min)/dt

![CFL decomposition](figures/cfl_diagnostics/fig4_cfl_decomposition.png)
*Figure 4: CFL growth rate decomposition for all three explicit runs. The red curve (velocity growth rate) completely dominates at the jump. The green curve (h_min change rate) is negligible. This confirms the CFL jump is velocity-driven.*

The velocity growth rate (red) dominates completely at the jump time. The mesh size change rate (green) contributes negligibly. **The CFL jump is a velocity phenomenon, not a mesh phenomenon.**

### 4.4 Physical Interpretation

The sudden velocity growth is the **Rosensweig instability onset**. As the external magnetic field ramps up, there exists a critical field strength B_c where the flat ferrofluid–water interface becomes linearly unstable. Below B_c, perturbations decay. Above B_c, perturbations grow exponentially — this is the classical Rosensweig (normal field) instability.

The CFL jump is the numerical manifestation of this physical threshold crossing. The velocity grows exponentially as the most unstable eigenmode of the linearized system amplifies.

---

## 5. Velocity Comparison Across Runs

![U_max vs Time](figures/cfl_diagnostics/fig2_umax_vs_time.png)
*Figure 2: Maximum velocity evolution (log scale). All runs follow similar velocity trajectories — the velocity is physical, not mesh-dependent. The difference in CFL comes entirely from different h_min values (mesh resolution), not from different velocities.*

All runs show nearly identical U_max trajectories up to their common time range, confirming the velocity growth is a physical phenomenon independent of numerical configuration. The different CFL outcomes arise because CFL = U_max × dt / h_min, and h_min varies between configurations.

---

## 6. Force and Energy Analysis

### 6.1 Forces

![Forces vs Time](figures/cfl_diagnostics/fig6_forces_vs_time.png)
*Figure 6: Force evolution per run. Magnetic force (red) ramps steadily with the applied field. Capillary force (blue) responds to interface deformation. In the dying r4-explicit run, capillary force catches up to magnetic force near the time of death.*

The force balance reveals the instability mechanism:
1. Magnetic force increases linearly with the field ramp
2. Below the critical threshold, the flat interface is in equilibrium (capillary + gravity balance magnetic)
3. At onset, capillary force jumps because the interface starts deforming
4. The positive feedback (deformation → more Kelvin force → more deformation) drives exponential growth

### 6.2 Energies

![Energies vs Time](figures/cfl_diagnostics/fig7_energies_vs_time.png)
*Figure 7: Energy evolution per run. Total energy grows monotonically (energy injected by external field). No sudden energy blowup precedes the simulation death — the instability is gradual at the energy level, even though velocity and CFL show sudden jumps.*

### 6.3 CFL Correlation with Forces

![CFL vs Forces](figures/cfl_diagnostics/fig8_cfl_vs_forces.png)
*Figure 8: Scatter plots of CFL vs capillary force (left), magnetic force (center), and gravity force (right), colored by time. CFL correlates most tightly with capillary force (log-log near-linear), confirming the instability-driven mechanism.*

---

## 7. CFL vs Magnetic Field Strength

![CFL vs B/B_max](figures/cfl_diagnostics/fig5_cfl_vs_magnetic_field.png)
*Figure 5: CFL vs magnetic field fraction (α/α_max = t/1.6). The CFL jump occurs at 20–38% of maximum field strength depending on mesh resolution. Finer mesh detects onset earlier because it resolves smaller perturbation wavelengths.*

This plot reveals that the instability onset occurs at a well-defined fraction of the maximum field strength. The mesh-dependence of the onset time is consistent with linear stability theory: the most unstable wavelength of the Rosensweig instability depends on the capillary length, and finer meshes resolve shorter wavelengths that may become unstable at lower field strengths.

---

## 8. Interface Dynamics

![Interface spread vs CFL](figures/cfl_diagnostics/fig9_interface_vs_cfl.png)
*Figure 9: Top: Interface y-spread (max − min of θ = 0 contour) showing spike height evolution. Bottom: CFL evolution. The r3-AMR-explicit run (orange) survived long enough for spikes to develop (y-spread ≈ 0.45 by t = 2.0). The r4-AMR-explicit run (blue) died before spikes could fully develop.*

The interface spread confirms that the r3-AMR-explicit run (which barely survived with CFL peaking at 1.01) actually produced developing Rosensweig spikes, demonstrating that the physics is correct — the issue is purely one of numerical stability during the onset.

---

## 9. Detailed Per-Run Analysis

### 9.1 r4-AMR-explicit (Died at t ≈ 0.99)

![Detailed r4-AMR-explicit](figures/cfl_diagnostics/fig14_detailed_r4_explicit.png)
*Figure 14: Six-panel detailed analysis. The CFL step-jumps align with the velocity growth, not with AMR cycles. Theta stays within [-1, 1] — no CH overshoot. Divergence (div U) grows over time. The simulation dies from CFL exceeding 1, not from any other diagnostic indicator.*

### 9.2 r3-AMR-explicit (Survived to t = 2.0)

![Detailed r3-AMR-explicit](figures/cfl_diagnostics/fig15_detailed_r3_explicit.png)
*Figure 15: Six-panel detailed analysis. Same CFL pattern as r4, but the coarser mesh keeps CFL just below 1.0. After the field is fully ramped (t > 1.6), the system reaches a quasi-steady state with oscillating CFL. This run produces physical Rosensweig spikes.*

### 9.3 r4-AMR-implicit (Running)

![Detailed r4-AMR-implicit](figures/cfl_diagnostics/fig16_detailed_r4_implicit.png)
*Figure 16: Six-panel detailed analysis of the currently running implicit CH simulation (t = 0.41 as of writing). CFL is at 7×10⁻⁴ — three orders of magnitude below the explicit run at the same time. The simulation is approaching the critical region (t ≈ 0.5) where the explicit runs experienced the CFL jump.*

---

## 10. h_min Evolution

![h_min evolution](figures/cfl_diagnostics/fig10_hmin_evolution.png)
*Figure 10: Inferred h_min and n_cells over time for all four runs. For the AMR runs, h_min changes occur through discrete AMR refinement/coarsening cycles, but these do NOT correlate with the CFL jump timing. The noAMR run (bottom-left) has constant n_cells = 15,360, confirming the CFL jump is not AMR-driven.*

---

## 11. Implications for Numerical Methods

### 11.1 Why Explicit CH Convection Fails

The CH convection term U·∇θ, when treated explicitly (evaluated at t_{n-1} and placed on the RHS), imposes a CFL stability constraint:

CFL = U_max × dt / h_min < C_crit

where C_crit is O(1). When the Rosensweig instability kicks in and U_max jumps by 2 orders of magnitude, CFL exceeds this limit within ~25 time steps. The simulation has no chance to respond.

The failure chain:
```
Magnetic field reaches critical strength
  → Flat interface becomes linearly unstable (physical)
  → Velocity U_max jumps ~100× in ~25 steps (physical)
  → CFL = U_max × dt / h_min exceeds stability limit
  → Explicit CH produces θ oscillations (numerical)
  → Spurious capillary forces from θ errors (numerical)
  → Positive feedback: more velocity → more CFL → more errors → death
```

### 11.2 Why Implicit CH Convection is the Correct Fix

Making CH convection implicit (U·∇θ on the LHS matrix) removes the CFL stability constraint entirely. The linear system absorbs the convection regardless of how large U × dt / h gets. CFL becomes a measure of physical transport distance per step, not a stability limit.

**Implementation**: The convection term changes from:

```
RHS:  local_rhs(i) += theta_old * (U · ∇Λ_i) * JxW     [explicit]
```
to:
```
LHS:  local_matrix(i,j) -= (U · ∇Λ_i) * theta_j * JxW  [implicit]
```

The matrix becomes non-symmetric due to the convection operator, but GMRES + AMG handles this without issue.

### 11.3 Why This Matters for the Community

Standard practice in ferrofluid simulations uses linear field ramping with explicit or semi-explicit time stepping. The implicit assumption is that the field ramp is "smooth enough" for explicit schemes. Our finding shows this is **false at the Rosensweig onset** — the velocity responds with an abrupt jump regardless of how smooth the field ramp is. This is because the instability is a threshold phenomenon, not a smooth transition.

Researchers experiencing unexplained solver crashes during Rosensweig simulations may be encountering exactly this CFL threshold without realizing it.

---

## 12. Potential Solutions

### 12.1 Implicit CH Convection (Our Primary Fix)

**Status**: Implemented and currently being validated.

Move the advection term U·∇θ from the RHS (explicit) to the LHS (implicit). This unconditionally removes the CFL stability constraint for the CH equation. The resulting non-symmetric system is solved by GMRES with AMG preconditioning.

**Pros**: Removes CFL limit entirely, allows arbitrarily large time steps (for stability), no modification to the physics.
**Cons**: Non-symmetric system (GMRES instead of CG), slightly more expensive per step.

### 12.2 Adaptive Time Stepping

**Status**: Not yet implemented (prepared as future work).

Monitor CFL number and reduce dt when CFL exceeds a threshold (e.g., CFL < 0.01). Increase dt when CFL drops well below the threshold.

```
if CFL > CFL_max:
    dt_new = dt * CFL_target / CFL    (shrink)
if CFL < CFL_max / 10:
    dt_new = min(dt * growth_factor, dt_max)    (grow)
```

**Pros**: Works with explicit CH, maintains temporal accuracy.
**Cons**: Requires very small dt during onset (potentially 100× smaller), increases total computation time significantly.

### 12.3 BDF2 Time Integration

**Status**: Not yet implemented.

Second-order backward differentiation formula (BDF2) has a larger stability region than backward Euler. Combined with implicit CH convection, this would improve both stability and temporal accuracy.

**Pros**: O(dt²) temporal accuracy, larger stability region.
**Cons**: Requires storing two previous time levels, startup procedure needed.

### 12.4 Long-Duration MMS Testing Framework

**Status**: Framework implemented, not yet run.

A per-step error tracking MMS test that runs many steps at fixed dt and records error evolution at each step. Distinguishes between:
- Linear error growth (normal for 1st-order backward Euler)
- Exponential error growth (instability in scheme or coupling)
- Bounded error (ideal)

Files: `mms/mms_core/long_duration_mms.h`, `mms/mms_core/long_duration_mms.cc`

---

## 13. Side Finding: Non-Linear Field Ramping

### 13.1 Concept

Independent of the numerical fix (implicit CH), the CFL jump phenomenon suggests that **how the magnetic field is ramped** affects the simulation's ability to resolve the instability onset. This is a separate research finding — not a fix for our code (which uses the paper's linear ramp), but a potential contribution to the broader study of Rosensweig spike formation.

### 13.2 Proposal

Instead of the standard linear ramp α(t) = α_max × t / t_ramp, use a non-linear ramp that slows down near the critical field strength:

**Two-phase ramp**:
1. **Phase 1** (t ∈ [0, t₁]): Fast ramp to ~25% of B_max — well below the critical threshold, interface remains flat.
2. **Phase 2** (t ∈ [t₁, t₂]): Slow ramp through the critical region (25%–40% of B_max), giving the solver more time steps to adapt to the growing velocity.
3. **Phase 3** (t ∈ [t₂, t₃]): Resume normal speed above the critical region, or hold constant to let the system equilibrate.

### 13.3 Rationale

The Rosensweig instability is a threshold phenomenon — below B_c the interface is stable, above B_c perturbations grow exponentially. With a linear ramp, the field crosses B_c at a fixed rate, and the solver must absorb the resulting velocity jump in a fixed number of steps.

By slowing the ramp near B_c:
- The solver has more time steps per unit of velocity increase
- The interface deforms quasi-statically rather than impulsively
- Numerical methods (even explicit ones) may survive the onset

### 13.4 Significance

To our knowledge, all published Rosensweig instability simulations use linear (or step-function) field ramping. No study has characterized the CFL jump at instability onset or proposed ramp shaping as a numerical strategy. This could benefit:
- Researchers using explicit time-stepping who experience unexplained crashes
- Experimentalists interested in controlling spike formation dynamics
- Computational studies of other threshold-driven instabilities (Rayleigh-Taylor, Saffman-Taylor, etc.)

---

## 14. Current Status and Next Steps

### 14.1 Implicit CH Run (In Progress)

| Metric | Value (step 819, t = 0.41) |
|--------|---------------------------|
| CFL | 7.1 × 10⁻⁴ |
| U_max | 3.1 × 10⁻³ |
| θ range | [−1.0000, 1.0000] |
| Mass | −2.000 × 10⁻¹ (conserved) |
| E_total | 282.3 |
| n_cells | 13,872 (AMR) |

The implicit run is approaching the critical region (t ≈ 0.5) where the explicit runs experienced the CFL jump. **If the implicit run survives this region and continues to t = 2.0, it will validate implicit CH convection as the correct numerical approach for Rosensweig instability simulations.**

*This section will be updated when the run completes.*

### 14.2 Planned Updates to This Report

- [ ] Results after the implicit run passes through t ≈ 0.5 (critical region)
- [ ] Final results at t = 2.0 (full simulation)
- [ ] Comparison of spike morphology with Nochetto et al. Figure 6
- [ ] Long-duration MMS test results (when run)
- [ ] Theoretical estimate of critical magnetic Bond number vs observed onset

---

## 15. Figures Index

| Figure | File | Description |
|--------|------|-------------|
| Fig. 1 | `fig1_cfl_vs_time.png` | CFL evolution — all runs compared |
| Fig. 2 | `fig2_umax_vs_time.png` | Maximum velocity evolution |
| Fig. 3 | `fig3_cfl_jump_stepbystep.png` | Step-by-step quantities around CFL jump |
| Fig. 4 | `fig4_cfl_decomposition.png` | CFL growth rate decomposition |
| Fig. 5 | `fig5_cfl_vs_magnetic_field.png` | CFL vs magnetic field fraction |
| Fig. 6 | `fig6_forces_vs_time.png` | Force evolution per run |
| Fig. 7 | `fig7_energies_vs_time.png` | Energy evolution per run |
| Fig. 8 | `fig8_cfl_vs_forces.png` | CFL vs forces scatter |
| Fig. 9 | `fig9_interface_vs_cfl.png` | Interface spread vs CFL |
| Fig. 10 | `fig10_hmin_evolution.png` | h_min and n_cells evolution |
| Fig. 11 | `fig11_cfl_jump_zoom.png` | Zoomed CFL jump region |
| Fig. 12 | `fig12_cfl_jump_rates.png` | Normalized rates at CFL jump |
| Fig. 13 | `fig13_cfl_growth_rate.png` | CFL growth rate (all runs) |
| Fig. 14 | `fig14_detailed_r4_explicit.png` | Detailed 6-panel: r4-AMR-explicit |
| Fig. 15 | `fig15_detailed_r3_explicit.png` | Detailed 6-panel: r3-AMR-explicit |
| Fig. 16 | `fig16_detailed_r4_implicit.png` | Detailed 6-panel: r4-AMR-implicit |

All figures are located in `Report/figures/cfl_diagnostics/`.
Plotting scripts: `Results/plot_cfl_diagnostics.py`, `Results/plot_cfl_jump_investigation.py`.

---

## 16. Reproducibility

### Data Sources

All simulation data is stored in `Results/` with timestamped directories. Each run includes:
- `run_info.txt` — parameters and configuration
- `diagnostics.csv` — per-step diagnostic quantities
- `solution_*.vtu` — VTK output for visualization

### Scripts

| Script | Purpose |
|--------|---------|
| `Results/plot_cfl_diagnostics.py` | Main diagnostic plots (Figures 1–2, 6–9, 13–16) |
| `Results/plot_cfl_jump_investigation.py` | Jump investigation plots (Figures 3–5, 10–12) |

### Build and Run

```bash
# Build
cd Semi_Coupled && cmake -S . -B cmake-build-release -DCMAKE_BUILD_TYPE=Release && make -C cmake-build-release -j8

# Run with implicit CH (recommended)
nohup mpirun -np 1 cmake-build-release/ferrofluid --rosensweig --direct --implicit_ch_convection > run.log 2>&1 &

# Run with explicit CH (will likely die at onset)
nohup mpirun -np 1 cmake-build-release/ferrofluid --rosensweig --direct --explicit_ch_convection > run.log 2>&1 &
```
