# Implementation Plan: Next Steps

## Project: Ferrofluid Phase-Field Model (Nochetto et al.)
## Last Updated: March 6, 2026

---

## Current Status (Updated March 7, 2026)

- All 10 spatial MMS tests pass with optimal convergence rates
- 13 bugs found and fixed total (4 MMS, 3 MPI/solver, 1 test harness, 1 H field, 1 viscous scaling,
  1 CH convection explicit->implicit, 1 DG face loop after AMR, 1 AMR bulk coarsening)
- **BGS set to single pass** (paper-like approach; iterating to convergence diverges)
- **Code optimized**: M_PI centralized, dead code removed, memory management fixed,
  NS assembler consolidated via NSForceData struct, mag face assembly deduplicated
- **CH convection implicit** in theta (removes CFL constraint)
- **AMR bulk coarsening fix** prevents oscillation cycle
- **3 validation benchmarks running** (droplet, square, droplet-nonuniform-B)
- **Decoupled rosensweig-nonuniform running** (4 ranks, comparison reference)
- Long-duration MMS framework prepared (not yet run)

---

## PRIORITY 1: Validation Pyramid (IN PROGRESS)

### Step 1: Benchmark Tests (Running — March 7)
Three tests launched to validate the basic physics before attempting Rosensweig:

| Test | What it validates | Expected result |
|------|-------------------|-----------------|
| Droplet (no mag) | CH + NS coupling, interface stability | Circle stays circular |
| Square (no mag) | CH + NS, energy dissipation, mass conservation | Square relaxes to circle |
| Droplet + nonuniform B | CH + NS + Mag, Kelvin force on interface | Droplet elongates into ellipse |

**Early observations (step ~40):** All three tests running stably.

### Step 2: Dome Test (NEXT)
- Uses `h = h_a` only (reduced magnetic field, no Poisson solve)
- Tests gravity + magnetic force on flat interface
- Simpler than full Rosensweig (no demagnetization field)

### Step 3: Rosensweig Instability (AFTER BENCHMARKS)
- Previous runs showed BGS non-convergence at onset (t~0.5)
- Now using BGS=1 (single pass), implicit CH convection
- Need to verify benchmarks pass first before attempting

### BGS Investigation Conclusion
BGS=20 tested: residual stays 0.2-0.5 at onset — iterating does NOT help.
Paper (Section 6, p.520): "We make no attempt to prove convergence."
Decision: BGS=1 (single pass per time step), matching paper's likely approach.

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

## PRIORITY 3: Code Cleanup (PARTIALLY COMPLETE)

Completed (Session 19):
1. Centralized M_PI in tools.h (removed from 6 files)
2. Removed dead fe_Q1_ member from phase_field
3. Fixed raw new/delete -> unique_ptr in ns_block_preconditioner.cc
4. Deduplicated magnetization face assembly (~40 lines -> lambda)
5. Consolidated 4 NS assembler functions into unified NSForceData struct

Remaining:
1. Remove Zhang-He-Yang extension code (beta damping, spin-vorticity)
2. Consolidate duplicated functions (applied field, Heaviside)
3. Fix performance (FEFaceValues inside loop)
4. Extract hardcoded values to parameters

---

## Results Archive

| Directory | Test | Date | Status |
|-----------|------|------|--------|
| `20260228_063736_droplet_r5_direct_amr` | Droplet (no mag) | Feb 28 | Complete |
| `20260228_221805_droplet-uniform-B_r5_direct_Namr` | Droplet + uniform B | Feb 28 | Complete (slow instability) |
| `20260301_052316_rosen_r4_direct_Namr` | Rosensweig r=4 noAMR DG | Mar 1 | Died t~1.0 (CFL) |
| `20260301_214922_rosen_r4_direct_Namr` | Rosensweig r=4 noAMR | Mar 1 | Died (CFL) |
| `20260302_174643_rosen_r4_direct_Namr` | Rosensweig r=4 noAMR | Mar 2 | Died (CFL) |
| `20260304_065229_square_r5_direct_amr` | Square r=5 AMR | Mar 4 | Partial |
| `20260304_232520_square_r4_direct_Namr` | Square r=4 noAMR | Mar 4 | Complete |
| `20260305_021220_rosen_r3_direct_amr` | Rosensweig r3 AMR | Mar 5 | Partial |
| `20260305_115128_rosen_r3_direct_amr` | Rosensweig r3 AMR explicit | Mar 5 | **Completed** t=2.0 |
| `20260305_171805_rosen_r4_direct_amr` | Rosensweig r4 AMR explicit | Mar 5 | Died t~0.99 (CFL=1.15) |
| `20260306_085853_rosen_r4_direct_amr` | Rosensweig r4 AMR implicit | Mar 6 | Partial (BGS issues) |
| `20260306_171510_rosen_r4_direct_amr` | Rosensweig r4 BGS=20 | Mar 6 | Killed t=0.60 (BGS non-conv) |
| `20260307_*_droplet_r5_direct_amr` | Droplet benchmark | Mar 7 | **Running** |
| `20260307_*_square_r5_direct_amr` | Square benchmark | Mar 7 | **Running** |
| `20260307_*_droplet-nonuniform-B_*` | Droplet + nonuniform B | Mar 7 | **Running** |

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
