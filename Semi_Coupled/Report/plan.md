# Implementation Plan: Next Steps

## Project: Ferrofluid Phase-Field Model (Nochetto et al.)
## Last Updated: March 6, 2026

---

## Current Status

- All 10 spatial MMS tests pass with optimal convergence rates (np=4)
- 12 bugs found and fixed total (4 MMS, 3 MPI/solver, 1 test harness, 1 H field, 1 viscous scaling,
  1 CH convection explicit->implicit, 1 DG face loop after AMR)
- Block-Gauss-Seidel global iteration implemented and working
- AMR implemented for parallel distributed mesh (start coarse, refine up)
- **CH convection made implicit** in theta (removes CFL constraint)
- **AMR bulk coarsening fix** prevents oscillation cycle (cells 15K -> 4.6K)
- **CFL instability onset characterized** -- velocity-driven jump at ~30% of max field (see Report)
- **Rosensweig run in progress** (r4, AMR, implicit CH, step 819/4000, t=0.41)
- Long-duration MMS framework prepared (not yet run)

---

## PRIORITY 1: Rosensweig Instability Validation (IN PROGRESS)

### Current Run
Run `20260306_085853_rosen_r4_direct_amr` is in progress with:
- ref=4, AMR (bulk coarsening fix), implicit CH convection, dt=5e-4, direct solver
- At step 819 (t=0.41): CFL=7.1e-4, theta=[-1.000,1.000], mass conserved
- Approaching the critical instability onset region (t~0.5)
- Estimated 10 hours total runtime

### CFL Instability Onset Discovery
The explicit runs failed because the Rosensweig instability onset causes a **100x velocity
jump in ~25 time steps** (~30% of max field). This violates the CFL constraint for explicit
CH convection. The implicit fix removes this constraint. See `Report/CFL_INSTABILITY_ONSET_REPORT.md`.

### What to Expect
- If the implicit run survives t~0.5 and continues to t=2.0: Rosensweig spikes should form
- Compare spike morphology with Nochetto et al. Figure 6
- If it fails: investigate other coupling instabilities (Kelvin force, Picard convergence)

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
| `20260301_052316_rosen_r4_direct_Namr` | Rosensweig r=4 noAMR DG | Mar 1 | Died t~1.0 (CFL) |
| `20260303_*_rosen_*_amr` | Rosensweig AMR attempts | Mar 3 | Failed (CFL blowup) |
| `20260303_214052_rosen_r3_direct_Namr` | Rosensweig r3 implicit | Mar 3 | Killed (under-resolved) |
| `20260305_115128_rosen_r3_direct_amr` | Rosensweig r3 AMR explicit | Mar 5 | **Completed** t=2.0 (CFL~1.0) |
| `20260305_171805_rosen_r4_direct_amr` | Rosensweig r4 AMR explicit | Mar 5 | Died t~0.99 (CFL=1.15) |
| `20260306_085853_rosen_r4_direct_amr` | Rosensweig r4 AMR **implicit** | Mar 6 | **Running** (t=0.41) |

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
