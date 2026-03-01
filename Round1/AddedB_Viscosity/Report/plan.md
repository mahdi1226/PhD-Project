# Implementation Plan: Next Steps

## Project: Ferrofluid Phase-Field Model (Nochetto et al.)
## Last Updated: March 1, 2026

---

## Current Status

- All 10 spatial MMS tests pass with optimal convergence rates (np=4)
- 10 bugs found and fixed total (4 MMS, 3 MPI/solver, 1 test harness, 1 H field, 1 viscous scaling)
- Block-Gauss-Seidel global iteration implemented and working
- **BLOCKING ISSUE:** Kelvin force DG skew form causes numerical instability
  in production Rosensweig runs (odd-even oscillation at ~20% field ramp)

---

## PRIORITY 1: Fix Kelvin Force Numerical Instability (BLOCKING)

### Problem Statement
The DG skew form Kelvin force (Eq. 38, Lemma 3.1) causes violent odd-even numerical
oscillation in the Rosensweig instability test at ~22% of field ramp (step ~630/4000).
F_mag jumps from 32K to 244K in 20 steps while the interface barely moves. The BGS
iteration fails to converge (residual 0.15-0.85).

Additionally, the droplet-uniform-B test (which SHOULD produce zero Kelvin force for a
uniform field on a symmetric droplet) shows F_mag ~ 101 and exponentially growing U_max.

### Key Evidence
- AddedB_Viscosity DG skew form: UNSTABLE
- Decoupled project (3-term Kelvin): STABLE (produces correct Rosensweig spikes)

### Investigation Plan

#### Step 1: Compare Kelvin Force Formulations
Read the Decoupled project's NS assembler carefully:
- File: `Decoupled/navier_stokes/navier_stokes_assemble.cc`
- Lines 527-610: cell Kelvin force (3 terms)
- Lines 925-960: face Kelvin force

The Decoupled project uses three separate terms:
```cpp
// Term 1: mu_0 * (m . grad)H
// Term 2: mu_0/2 * (m x H, curl v)  — antisymmetric stress
// Term 3: mu_0 * (m x curl(H), v)   — magnetic torque
```

Questions to answer:
1. Are the 3 terms together equivalent to the DG skew form (Eq. 38)?
2. Does the DG skew form's 1/2 div(M) term introduce instability?
3. Does the face term -(V.n) [[H]].{M} have the correct magnitude?

#### Step 2: Test with Simplified Kelvin Force
As a diagnostic step, try replacing the DG skew form with just the (M.grad)H cell term
(no div(M) skew correction, no face term). If the droplet-uniform-B test shows F_mag ~ 0,
the problem is in the skew form specifically.

#### Step 3: Consider Alternative Kelvin Formulations
Options:
a. **Standard Kelvin**: `mu_0 * (M . grad)H` (cell only, no DG face repair)
b. **Decoupled 3-term**: Match the working Decoupled project exactly
c. **DG skew form with stabilization**: Add penalty or damping to face term
d. **Nochetto Lemma 3.1 with CG grad(phi)**: Since phi is CG, grad(phi) is continuous
   across cells, so [[H]] = 0 and the face term vanishes automatically

**Option (d) is interesting**: If phi is solved with CG elements, grad(phi) is already
continuous across faces (within the CG space). The jump [[grad phi]] should be zero
at CG nodes and small at quadrature points. The face term may be amplifying numerical
noise from quadrature-level discontinuities of the CG gradient.

### Expected Outcome
One of these approaches should eliminate the spurious forces and stabilize the Rosensweig
simulation.

---

## PRIORITY 2: Rosensweig Instability Validation

### Goal
Reproduce CMAME 2016, Section 6.2: 4-5 spikes forming at the ferrofluid-air interface
under a ramped vertical magnetic field from 5 point dipoles.

### Paper Parameters (Section 6.2, p.520-522)
- Domain: [0,1] x [0,0.6], ferrofluid pool depth 0.2
- epsilon=0.01, lambda=0.05, chi_0=0.5, gamma=0.0002, mu_0=1
- v_w=1.0, v_f=2.0, r=0.1 (Boussinesq density ratio)
- g = (0, -30000) (scaled gravity)
- 5 dipoles at y=-15, d=(0,1), ramp from alpha_s=0 to alpha_s=6000 over t in [0,1.6]
- AMR: Kelly error estimator, every 5 steps, 6 refinement levels
- 4000 time steps (dt=5e-4), t_F=2.0

### Steps
1. Fix Kelvin force instability (Priority 1)
2. Run r=4 to t=2.0 without instability
3. Run r=5 (paper's r=6 is expensive, start with r=5)
4. Enable AMR (currently running with --no_amr)
5. Compare spike count, wavelength, amplitude with paper Figure 1

---

## PRIORITY 3: Temporal Convergence Verification (DEFERRED)

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

## PRIORITY 4: Code Cleanup (DEFERRED)

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
| `20260301_052316_rosen_r4_direct_Namr` | Rosensweig r=4 | Mar 1 | Running (odd-even oscillation) |

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
