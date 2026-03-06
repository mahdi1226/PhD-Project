# Decoupled Ferrofluid Solver -- Session Handoff

## Project Location
`/Users/mahdi/Projects/git/PhD-Project/Decoupled/`
Build: `cd /Users/mahdi/Projects/git/PhD-Project/Decoupled/build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j8`

## Reference Papers
- Zhang, He & Yang, SIAM J. Sci. Comput. 43(1), 2021 -- Parameters, validation, SAV scheme
  PDF: `/Users/mahdi/Desktop/droplet/GZhang_XMHe_XYang.pdf`
- Nochetto, Salgado & Tomas, CMAME 309, 2016 -- PDE formulation (simplified model)
  PDF: `/Users/mahdi/Desktop/RHNochetto_AJSalgado_ITomas.pdf`

---

## Current Status Summary

| Test | Status | Result |
|------|--------|--------|
| **Standalone NS MMS** | **PASS** | All rates 2.00 (projection method, dt∝h²) |
| **Full 4-system MMS** | **PASS** | θ:1.93, U:2.00, p:2.02, φ:2.99, M:2.00 |
| **Rosensweig uniform** (r=4) | PASS | 800+ steps, theta in [-1.01, 1.01], stable |
| **Rosensweig nonuniform** (r=3) | **RUNNING** | Step ~5000/17500, CFL=0.003, stable |

---

## What Was Done Across All Sessions

### Sessions 1-3: SAV CH Assembly Fix + Rosensweig Fix + Equation Audit
- Removed Douglas-Dupont stabilization from CH assembly
- Fixed 10/14 Rosensweig parameters
- Fixed Rosensweig IC (flat at y=0.2, no perturbation)
- Fixed capillary force direction (theta*grad_psi, not psi*grad_theta)
- Fixed Kelvin force grad(h_a) missing from grad(H)
- Added `droplet_nofield` validation preset

### Session 4: Complete Zhang Deviation Audit + 7 Fixes
- Fixed viscous term 2x error (nu/2 -> nu/4)
- Fixed S1 auto-computation: lambda/(4*epsilon)
- Enabled full magnetization PDE (beta=1, tau=1e-4)
- Removed dipole regularization delta
- Sharp step IC for Rosensweig

### Sessions 5-8: MMS Verification + Critical DG Bug Fix
- Built incremental MMS strategy: standalone -> coupled -> full system
- Fixed DG face assembly `FEInterfaceValues` indexing (zero cross-cell coupling)
- Added upwind penalty for optimal O(h^2) DG convergence
- All standalone and coupled MMS tests PASS

### Sessions 9-10: Validation Suite + Nonuniform Rosensweig
- All validation tests rerun post-DG fix
- Nonuniform Rosensweig: two clean spikes form, instability at t~2.07
- Root cause: operator splitting instability under strong coupling (chi_0=0.9)

### Sessions 11-12: Paper Audit + Parameter Cleanup
- Reverted chi/nu to sigmoid interpolation (was wrongly set to linear)
- Added spin-vorticity term to magnetization equation
- Removed S2 and C0 (not in Zhang's paper)
- Paper is sole authority for parameters

### Session 13: AMR Implementation


**Adaptive Mesh Refinement** for all 4 subsystems on shared p4est triangulation.

**New file:** `utilities/amr.h` -- header-only 14-step AMR algorithm:
1. Kelly error estimation on theta (interface field)
2. Mark cells with fixed-fraction (upper/lower)
3. Enforce level limits (max/min)
4. Interface protection (never coarsen where |theta| < threshold)
5. Prepare triangulation
6. Create SolutionTransfer for all subsystems (7 DoFHandlers)
7. Execute mesh refinement
8. Re-setup all subsystems
9. Interpolate solutions to new mesh
10. Apply constraints
11. Clamp theta to [-1, 1]
12. Recompute psi = theta^3 - theta nodally
13. Update all ghost vectors
14. Log diagnostics

**Critical bug found and fixed:** `ghosts_valid_` flag in CH subsystem not reset
by `setup()`. After AMR, `update_ghosts()` skipped the copy, leaving theta_relevant_
at zero. Fixed by adding `ghosts_valid_ = false` in `setup()` and belt-and-suspenders
`invalidate_ghosts()` calls in amr.h.

**Physics-based AMR activation gate:**
- AMR stays dormant until |U|_max exceeds `amr_activation_U` threshold (default 1e-3)
- Once activated, stays active for the rest of the simulation
- When NS is disabled, activates immediately (no velocity to gate on)
- `--amr-activation-U 0` for immediate activation

**AMR parameters (all CLI-configurable):**
```
--amr / --no-amr              Enable/disable (default: OFF)
--amr-interval N              Refine every N steps (default: 5)
--amr-max-level N             Max refinement level (default: 0 = no cap)
--amr-min-level N             Min refinement level (default: 0)
--amr-upper-fraction V        Top fraction to refine (default: 0.3)
--amr-lower-fraction V        Bottom fraction to coarsen (default: 0.10)
--amr-activation-U V          |U| threshold to start AMR (default: 1e-3)
```

**AMR test results:**
- CH-only: 200+ steps, mesh 960->420, energy decreasing
- Full 4-system: 370+ steps at r3, mesh 3840->1140 (70% reduction), all fields preserved
- Uniform Rosensweig with AMR: 100 steps, activation at step 63, mesh 3840->2112->11688
- Physics gate saves 37% wall time vs fixed-interval AMR
- Default path (no AMR): unchanged behavior, no regression

**VTK mesh_level field** added to `write_combined_vtu()` and `write_ch_only_vtu()`
for ParaView AMR visualization.

### Session 14: Material Property Fix (March 4, 2026)
- Isolated sigmoid chi/nu as sole cause of Rosensweig instability (spin-vorticity is safe)
- Reverted to LINEAR chi/nu interpolation (Zhang convention)
- Full Shliomis model: LINEAR chi/nu + spin-vorticity ON = production config

### Sessions 15-16: Magnetization Step 5/6 + Sparsity Analysis (March 5, 2026)

**Zhang Algorithm 3.1 Step 5/6 (magnetization transport splitting):**
- Implemented Step 5 (explicit transport in Picard loop) + Step 6 (implicit DG transport after Picard)
- Goal: unconditional energy stability per Zhang's scheme
- New vectors: `Mx/My_transport_solution_`, `Mx/My_transport_rhs_`, `Mx/My_old_ghosted_`
- Assembly modes: `Step5_Explicit` (mass + relaxation in Picard) + `Step6_Implicit` (full DG transport post-Picard)
- Result: Rosensweig nonuniform still blows up at t=2.184 (was t=2.290 without Step 6)

**Sparsity Analysis & Cuthill-McKee Renumbering:**
- Ported from Semi_Coupled solver: `utilities/sparsity_export.h` (SVG + gnuplot + bandwidth CSV)
- CLI: `--dump-sparsity` (export after step 1), `--renumber-dofs` / `--no-renumber-dofs`
- CM applied to CG systems only (theta, psi, phi, ux, uy); DG systems skipped (magnetization, pressure)
- Fixed parallel crash: Epetra native `ExtractMyRowView` API replaces deal.II iterators (non-contiguous CH map)
- Results (np=1, r=3): Poisson **-77.3%** bandwidth, CH **-11.5%**, Mag unchanged, NS +7.2%

**Rosensweig Nonuniform Blow-Up Analysis (8 runs analyzed):**
- ALL runs blow up at t=2.18-2.68, same cascade: H spikes → M explodes → U → theta → NaN
- No AMR involved (all runs use uniform refinement)
- Halving dt (1e-4) only delays slightly (t=2.362 vs t=2.290)
- Root cause: Poisson-Magnetization feedback instability at interface peaks

---

## AMR Switchability

`use_amr = false` by default. The existing `refine_global(initial_refinement)` path
is unchanged. AMR only activates when `--amr` is passed on the CLI. Presets do NOT
enable AMR. Any existing test/preset works identically without modification.

---

## Key Technical Details

### Phase Field Convention
- Zhang uses Phi in {0,1}, code uses theta in {-1,+1}
- Equivalent via Phi = (theta+1)/2
- Material properties (production config = LINEAR chi/nu):
  - chi(theta) = chi_0 * (theta+1)/2 -- LINEAR
  - nu(theta) = nu_w*(1-theta)/2 + nu_f*(theta+1)/2 -- LINEAR
  - rho(theta) = 1 + r * H(theta/epsilon) -- sigmoid (Zhang Eq 4.2)

### SAV Stabilization (Zhang Eq 3.5-3.6)
- S1 = lambda/(4*epsilon), no S2 or C0 (paper only)
- S does NOT depend on dropped terms (verified from energy proof Eq 3.41)

### Coupling Strategy (Zhang Algorithm 3.1 — No Picard)
```
FOR each timestep:
  1. CH solve (SAV, using U^{n-1})
  2. Mag Step 5 (explicit: mass + relaxation, no transport)
  3. Poisson -> H
  4. Mag Step 6 (implicit: full DG transport)
  5. NS projection method:
     a. Velocity predictor (CG+AMG for ux, uy)
     b. Pressure Poisson (CG+AMG for dp)
     c. Velocity correction (consistent mass CG)
```

### Sessions 17-18: NS Projection Method + Picard Removal (CURRENT, March 6, 2026)

**Stage 1: Remove Picard iteration**
- Single forward pass replaces Picard loop (unconditionally stable per Zhang)
- CLI flags `--picard-*` deprecated with warnings

**Stage 2: NS pressure-correction projection method**
- Monolithic saddle-point system → 3 separate CG+AMG solves (ux, uy, p)
- Pressure FE space: DG-P1 → CG-Q1 (needed for pressure Poisson Laplacian)
- Velocity correction: consistent mass CG solve (not lumped mass — avoids O(dt/h) H1 error)
- CM renumbering on all 3 DoFHandlers

**MMS results:**
- Standalone NS: all rates 2.00 (refs 2-5) — PASS
- Coupled 4-system: θ:1.93, U:2.00, p:2.02, φ:2.99, M:2.00 — PASS
- Note: projection method splitting error O(dt) limits rates to 2.0 with dt∝h² scaling

**Rosensweig validation:**
- Uniform (r=4): 800+ steps stable, θ∈[-1.01,1.01]
- Nonuniform (r=3): running, step ~5000, CFL=0.003, stable past old blow-up initiation zone

---

## Currently Running Tests

**Rosensweig nonuniform** (r=3, dt=2e-4, 42 dipoles): step ~5000/17500, t≈1.0.
Old blow-up was at t≈2.18. ETA to blow-up zone: ~8 hours.

---

## Nonuniform Rosensweig Blow-Up Summary (8 runs, pre-projection-method)

| Run | dt | Blow-up time | Strategy | Notes |
|-----|-----|-------------|----------|-------|
| Step6 (newest) | 2e-4 | t=2.184 | Step5+6 | Earliest blow-up |
| 030226_230101 | 2e-4 | t=2.286 | Step5 | |
| nonuniform.log | 2e-4 | t=2.290 | Step5 | |
| 030326_050402 | 1e-4 | t=2.362 | unknown | Halved dt, still fails |
| 022826_222751 | 2e-4 | t=2.674 | Step5 (old) | Oldest code |
| 030226_050447 | 2e-4 | t=2.678 | Step5 (old) | |

**Cascade (identical in ALL runs):**
1. H (magnetic field) spikes first (single step: H jumps from ~240 to ~30000)
2. M (magnetization) follows (Poisson-Mag Picard coupling)
3. U (velocity) spikes (Kelvin force amplification)
4. theta (phase field) breaks (convection U.grad(theta))
5. Full NaN within 1-3 steps

---

## Pending Tasks (Priority Order)

1. **Nonuniform Rosensweig** -- confirm stability past t=2.18 (old blow-up point) with projection method
2. **Full system MMS with AMR** -- verify convergence rates preserved with mesh adaptation
3. **Extension study** (Phase C.0+) -- parametric sweeps, see `extension.md`

---

*Updated: March 6, 2026 (Sessions 17-18 -- Picard removal, NS projection method, coupled MMS PASS)*
