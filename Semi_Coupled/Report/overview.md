# Semi_Coupled: Monolithic Electromagnetics Refactor

## Paper Reference
Nochetto, Salgado & Tomas, "A diffuse interface model for two-phase ferrofluid flows",
CMAME 309 (2016) 497-531. arXiv:1511.04381

## What We Are Doing
Replacing the separate Poisson + Magnetization Picard iteration with a monolithic
block system that solves equations (42c) and (42d) from the paper together as one
coupled linear system.

### Previous Approach (archived_Nochetto)
- Poisson equation for demagnetizing potential phi: solved with CG+AMG
- Magnetization M: either DG transport PDE or algebraic projection M = chi(theta)*grad(phi)
- Coupling: Picard iteration (7 iterations, omega=0.35 under-relaxation)
- Problem: this decouples what the paper solves monolithically

### New Approach
- Single FESystem combining DG magnetization (vector) + CG potential (scalar)
- One block matrix assembled and solved together with MUMPS direct solver
- Eliminates Picard iteration entirely
- Faithful to the paper's scheme (42c)-(42d) and its energy stability proof

## Why We Are Doing This

### Evidence from Rosensweig Instability Runs
All runs with the Picard-based code failed to produce the correct Rosensweig peaks:

1. **AMR r7, interval=5** (March 5, completed to t=2.0): Shaky beginning, asymmetric
   spike generation, unstable peaks, only 2 spikes instead of 4.

2. **AMR r6, interval=5** (March 8): Killed at t=0.749 due to horizontal sloshing
   caused by AMR interval too frequent.

3. **AMR r4, interval=50** (March 8): Fixed sloshing by increasing AMR interval to 50.
   But only 1.6 elements across interface width epsilon=0.01 -- too coarse. Interface
   damaged at t~0.95 before peaks could form.

4. **Global r4** (March 8): No AMR artifacts, but same resolution problem. Interface
   damage before peak formation.

5. **Key observation**: At t=0.9, the paper shows visible waves for ALL configurations
   (4-7 refinement levels, 1000-8000 time steps). Our r4 runs showed barely any
   deformation (y_max=0.207, only 7.7mm above pool level).

### The Physics Problem
- The paper's Figure 7 proves that h=h_a (dropping the Poisson solve entirely) causes
  "the Rosensweig instability does not manifest". The demagnetizing field from the
  Poisson equation creates the spatial non-uniformity that drives peak formation.
- Our Picard iteration with omega=0.35 under-relaxation was too weak to properly
  couple phi and M. The paper solves (42c)-(42d) as one monolithic block in the BGS
  inner loop (page 520).

### Energy Stability
The paper's energy estimate (Proposition 4.1, equation 43) includes the dissipation
term ||delta_M^k||^2, which requires the implicit backward Euler time derivative in
equation (42c). The quasi-static shortcut M = chi(theta)*grad(phi) drops this term
and violates the energy stability proof. The monolithic block system preserves it.

## Solver Strategy (All Subsystems)

For 2D development and validation, use MUMPS direct solver for all three subsystems:
- **Magnetics** (M + phi): nonsymmetric block system, MUMPS
- **Cahn-Hilliard** (theta + psi): symmetric, MUMPS (currently block CG+AMG, may switch)
- **Navier-Stokes** (u + p): nonsymmetric saddle-point, MUMPS (currently block preconditioner)

This isolates physics errors from solver convergence issues. If the simulation blows up,
it is the weak forms, BCs, or time step -- not iterative solver failure.

## Audit Against Nochetto CMAME 2016 (March 2026)

Full 6-agent code audit comparing every assembler, solver, and material function
against the paper's equations. Four issues found and fixed:

### Fix 1: CH Convection — Implicit → Explicit (ch_assembler.cc)
**Paper Eq 65a:** `(delta_theta/tau, Lambda) - (U^k theta^{k-1}, grad Lambda) - gamma(grad psi^k, grad Lambda) = 0`

The convection term uses **explicit** `theta^{k-1}` (previous time step) on the RHS.
Code had implicit `theta^k` on the LHS matrix, over-stabilizing the interface.
This is the most likely cause of the flat Rosensweig interface — implicit convection
damps the perturbations that seed spike formation.

**Change:** Removed implicit convection from LHS, added explicit convection to RHS.
Updated MMS source to use `CHSourceThetaWithConvection` (explicit).

### Fix 2: BGS mag_old Overwrite (phase_field.cc)
`mag_old_solution_` was being overwritten at each BGS iteration inside `solve_magnetics()`.
Paper Eq 42c: `delta_M/tau = (M^k - M^{k-1})/tau` — M^{k-1} must refer to the previous
**time step**, not the previous BGS pass. Only affects runs with `bgs_iters > 1`.

**Change:** Save `mag_old_solution_` once before the BGS loop, not inside it.

### Fix 3: Heaviside Cutoff Consistency (material_properties.h)
`susceptibility()` reimplemented its own sigmoid with overflow cutoff at 20, while the
shared `heaviside()` function uses cutoff 30. Both compute `1/(1+exp(-x))` but with
different overflow protection thresholds, causing inconsistency at extreme values.

**Change:** `susceptibility()` now calls `heaviside()` directly: `chi_0 * heaviside(theta/epsilon)`.

### Fix 4: MPI Type Portability (ns_setup.cc)
`MPI_UNSIGNED` was hardcoded for `dealii::types::global_dof_index`, which can be
`uint64_t` on some builds. This could silently corrupt DoF index communication.

**Change:** Added `sizeof`-based MPI type selection: `MPI_UNSIGNED` for 32-bit,
`MPI_UNSIGNED_LONG_LONG` for 64-bit.

### Verified Non-Issues (No Change Needed)
- ILU `ilu_rtol = 1.0`: Correct — rtol=1.0 means "no ILU strengthening" per Trilinos docs.
- `E_internal_prev_`: Not dead code — used in `log_step()` for energy derivative logging.
- Missing `constraints.distribute()` in mag solver: Not missing — caller `solve_magnetics()`
  calls `mag_constraints_.distribute()` after solver returns.

## Elongation Preset Update (March 2026)

Reduced droplet radius from R=0.2 to R=0.1 in `setup_elongation()`:
- **Old:** 40% diameter/domain ratio, 12.6% area fraction — too large, boundary artifacts
- **New:** 20% diameter/domain ratio, 3.1% area fraction — room for 3-4x elongation
- Mesh refined r=6 → r=7 (h=1/128) for proper interface resolution (epsilon/h ~ 2.56)
- Bm reduced from 40.5 to 20.25 (still strong elongation, less extreme)

## AMR Coarsening Fix (March 2026)

Droplet, square, and elongation presets had `amr_min_level = initial_refinement - 2`,
which only coarsened 2 levels below the initial mesh. Bulk cells far from the interface
stayed unnecessarily fine, wasting DoFs and wall time.

**Change:** Set `amr_min_level = 1` for all three presets (matching Rosensweig/Hedgehog).
Elongation also changed from uniform mesh (no AMR) to AMR with aggressive bulk coarsening.

A validation script `run_validation.sh` runs square → droplet → elongation back to back.
Not yet run — validation tests are pending.

## Project Structure
- `Semi_Coupled/` -- this project (monolithic electromagnetics)
- `Archived_Nochetto/` -- previous code with Picard iteration (preserved, not modified)
- `Decoupled/` -- Zhang, He & Yang scheme (separate project, not related)
