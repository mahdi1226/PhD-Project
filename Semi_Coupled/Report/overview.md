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

## Project Structure
- `Semi_Coupled/` -- this project (monolithic electromagnetics)
- `Archived_Nochetto/` -- previous code with Picard iteration (preserved, not modified)
- `Decoupled/` -- Zhang, He & Yang scheme (separate project, not related)
