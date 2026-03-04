# Implementation Plan: Next Steps

## Project: Ferrofluid Phase-Field Model (Nochetto et al.)
## Last Updated: March 3, 2026

---

## Current Status

- All 10 spatial MMS tests pass with optimal convergence rates (np=4)
- 12 bugs found and fixed total (4 MMS, 3 MPI/solver, 1 test harness, 1 H field, 1 viscous scaling,
  1 CH convection explicit->implicit, 1 DG face loop after AMR)
- Block-Gauss-Seidel global iteration implemented and working
- AMR implemented for parallel distributed mesh (start coarse, refine up)
- **CH convection made implicit** in theta (removes CFL constraint)
- **BLOCKING ISSUE:** Rosensweig instability not yet reproduced — needs adequate
  resolution (h < epsilon) with implicit convection. r3 (h=1/80) insufficient for
  epsilon=0.01; need r4 (h=1/160) or AMR

---

## PRIORITY 1: Rosensweig Instability with Adequate Resolution (BLOCKING)

### Problem Statement
The CH convection CFL constraint has been removed (implicit convection, Session 17).
The remaining issue is **mesh resolution**: h must be smaller than epsilon for meaningful
results. r3 (h=1/80) fails for epsilon=0.01.

### Paper Parameters (Section 6.2, p.520-522)
- Domain: [0,1] x [0,0.6], ferrofluid pool depth 0.2
- epsilon=0.01, lambda=0.05, chi_0=0.5, gamma=0.0002, mu_0=1
- v_w=1.0, v_f=2.0, r=0.1 (Boussinesq density ratio)
- g = (0, -30000) (scaled gravity)
- 5 dipoles at y=-15, d=(0,1), ramp from alpha_s=0 to alpha_s=6000 over t in [0,1.6]
- AMR: Kelly error estimator, every 5 steps, 6 refinement levels
- 4000 time steps (dt=5e-4), t_F=2.0

### Options for Next Run
a. **r4 uniform, no AMR** (h=1/160 < epsilon, ~60K cells, 2 ranks)
   - Pro: simple, adequate resolution, directly tests implicit convection
   - Con: slow with direct solver (~60K DoFs per field), may need 4+ ranks
b. **r3 with AMR to L5-L6** (start 3840 cells, refine interface to h=1/320-1/640)
   - Pro: efficient, follows paper approach, adequate interface resolution
   - Con: AMR + implicit convection not yet tested together
c. **r4 with AMR to L6-L7** (best of both)
   - Pro: higher base resolution + AMR at interface
   - Con: most expensive, may not be needed

### Kelvin Force Stability (Secondary)
The earlier odd-even Kelvin force oscillation (Session 13-14) occurred with explicit
convection. It's unclear whether implicit convection also fixes that issue, or whether
the Kelvin force DG skew form has a separate stability problem. Running at r4+ will
clarify this.

---

## PRIORITY 2: Temporal Convergence Verification (DEFERRED)

### Goal
Verify first-order temporal convergence (backward Euler) for all subsystems.

### Status
Current MMS solutions are linear in time (theta ~ t, U ~ t, M ~ t). Backward Euler
is exact for linear-in-time solutions, so temporal tests show flat error (rate 0.0).
This is EXPECTED, not a bug.

### Implementation Needed
- Create t^2 MMS solutions (quadratic in time) so backward Euler produces O(dt) error
- Fix spatial mesh (fine enough to resolve), vary dt
- Expected: O(dt) convergence for all fields

---

## PRIORITY 3: Code Cleanup (DEFERRED)

See detailed cleanup plan at `.claude/plans/mighty-wondering-koala.md`.

Key items:
1. Remove Zhang-He-Yang extension code (beta damping, spin-vorticity)
2. Remove dead/debug code
3. Consolidate duplicated functions (applied field, Heaviside)
4. Fix performance (FEFaceValues inside loop)
5. Extract hardcoded values to parameters

---

## Results Archive

| Directory | Test | Date | Status |
|-----------|------|------|--------|
| `20260228_063736_droplet_r5_direct_amr` | Droplet (no mag) | Feb 28 | Complete |
| `20260228_221805_droplet-uniform-B_r5_direct_Namr` | Droplet + uniform B | Feb 28 | Complete (slow instability) |
| `20260301_052316_rosen_r4_direct_Namr` | Rosensweig r=4 (explicit) | Mar 1 | Failed (odd-even oscillation) |
| `20260303_*_rosen_*_amr` | Rosensweig AMR attempts | Mar 3 | Failed (CFL blowup) |
| `20260303_214052_rosen_r3_direct_Namr` | Rosensweig r3 implicit | Mar 3 | Killed (under-resolved) |

---

## Key Files

### Production Code
| File | Purpose |
|------|---------|
| `assembly/ns_assembler.cc` | NS + Kelvin force assembly (Bug 9 fix here) |
| `assembly/magnetization_assembler.cc` | DG magnetization transport |
| `assembly/ch_assembler.cc` | Cahn-Hilliard assembly |
| `physics/kelvin_force.h` | DG skew form Kelvin force kernels |
| `physics/applied_field.h` | External field computation (dipoles + uniform) |
| `core/phase_field.cc` | Main time loop + BGS iteration |
| `utilities/parameters.cc` | CLI parsing + presets |

### MMS Sources
| File | Purpose |
|------|---------|
| `mms/ns/ns_mms.h` | NS + Kelvin force MMS source |
| `mms/poisson/poisson_mms.h` | Poisson MMS + exact H |
| `mms/magnetization/magnetization_mms.h` | Magnetization MMS source |
| `mms/ch/ch_mms.h` | Cahn-Hilliard MMS source |

### Reference Code
| File | Purpose |
|------|---------|
| `Decoupled/navier_stokes/navier_stokes_assemble.cc` | Working 3-term Kelvin (reference) |
