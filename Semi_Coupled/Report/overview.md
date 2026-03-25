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

## Bug Fixes — 2026-03-20

### Problem: Asymmetric Rosensweig spikes
With monolithic magnetics, the simulation produces 3 spikes instead of 4, asymmetric heights,
and wall-biased nucleation. Full investigation of BCs, equations, ghosting, parameters — all
verified correct. Three bugs found:

### Fix 1: θ-lagging in magnetic assembler
**File:** `core/phase_field.cc` line 752
- χ(θ) was evaluated at θ^k (just-solved) instead of θ^{k-1} (previous step)
- Paper Theorem 4.1 requires material coefficients at OLD theta for energy stability
- Changed `theta_relevant_` → `theta_old_relevant_`

### Fix 2: CH convection — conservative → skew form
**File:** `assembly/ch_assembler.cc`
- Code used conservative form `(θU, ∇Λ)`, missing `+½div(U)θΛ` from paper's skew form (Eq 37)
- div(U) is weakly zero but NOT pointwise zero — the missing term acts as artificial mass source/sink
- Added velocity gradients + `½div(U)θΛ` term to match paper exactly

### Fix 3: Kelvin force face — missing plus-side CG test functions (**likely asymmetry cause**)
**File:** `assembly/ns_assembler.cc`
- Face integral `-μ₀ [[H]]·{M}(V·n)` only assembled minus-cell CG test functions
- Plus-cell test functions (which also have support at the face) were completely skipped
- Which cell is "minus" depends on cell ordering → NOT left-right symmetric → **breaks symmetry**
- Fixed: now assembles both cells' test functions at each face

### Also fixed
- Critical wavelength diagnostic: was using `σ = λ = 0.05`, should be `σ = λ/ε = 5.0`
  - Now correctly predicts λ_c = 0.257 (~3.9 spikes), matching paper's 4 spikes
- Wrong comment in `kelvin_force.h`: claimed `[[H]]=0` for CG φ, but ∇φ IS discontinuous

### Code cleanup (same session)
- Deleted 7 dead/duplicate/stub files (~2,200 lines): `amr_postprocess.h`, `initial_conditions.h`,
  `magnetization_diagnostics.h`, `mms/poisson/poisson_mms.h`, `mms/magnetization/magnetization_mms.h`,
  `questions.h`, `bubble_benchmark.cc`
- Deleted 29 stale `.log` files from project root
- Added `.gitignore`
- Created `Report/refactor_plan.md` for future cleanup tasks

See `Report/bugfixes_20260320.md` for detailed derivations and verification notes.

## Bug Fixes & Changes — 2026-03-22/23

### Fix 4: Capillary coefficient magnitude
**File:** `assembly/ns_assembler.cc`
- Original: `capillary_coeff = -lambda` (= -0.05)
- Fixed: `capillary_coeff = -lambda / epsilon` (= -5.0)
- Paper Eq 14e, 42e: the coefficient is λ/ε, not λ. Was 100× too weak.

### Fix 5: ψ sign convention confirmed
**File:** `assembly/ch_assembler.cc`, `assembly/ns_assembler.cc`
- Paper Eq 1: Ψ_paper = εΔθ − (1/ε)f(θ)
- Code weak form gives: ψ_code = −εΔθ + (1/ε)f(θ) = −Ψ_paper
- **ψ_code = −ψ_paper** (the old comment was CORRECT)
- Capillary force: paper +(λ/ε)(θ∇Ψ, V) → code −(λ/ε)(θ∇ψ_code, V)
- Verified with +λ/ε test: blew up at t=0.1 (driving force, unstable). Confirms −λ/ε is correct.

### Fix 6: Mass matrix coefficient
**File:** `assembly/ns_assembler.cc`
- Reverted from ½ρ(θ)/dt to 1/dt
- Paper Eq 42e has plain (δU^k/τ, V) — no ½, no ρ(θ)
- The ½ appears only in the kinetic energy functional, not the scheme equation

### Fix 7: Sharp step initial condition
**File:** `core/phase_field_setup.cc`
- Changed from smooth tanh equilibrium profile to sharp step: θ = +1 below y=0.2, −1 above
- Paper Eq 41 just says "interpolation of initial data" — the tanh profile was our (incorrect) choice
- Sharp step matches Zhang's approach and is standard for phase-field pool initialization

### Fix 8: Removed θ clamping
**File:** `core/phase_field_amr.cc`
- Removed artificial clamping of θ to [-1, 1] after each step
- Paper uses truncated double-well (Eq 2) which naturally limits growth outside [-1,1]
- Clamping broke mass conservation and distorted interface dynamics
- Result: 3 → 4-5 spikes, better symmetry

### Fix 9: Added `use_h_a_only` flag for dome test
**Files:** `utilities/parameters.h`, `utilities/parameters.cc`, `assembly/ns_assembler.cc`, `assembly/magnetic_assembler.cc`
- Paper Section 5 (Eq 65-66): simplified scheme uses H = h_a (no Poisson equation)
- Added `--h_a_only` CLI flag and `use_h_a_only` parameter
- Dome preset (`--dome`) enables this automatically
- Kelvin force uses h_a directly with (M·∇)h_a via finite differences
- Face loop skipped (h_a smooth → [[H]] = 0)
- Magnetization relaxes toward χ·h_a on RHS instead of implicit ∇φ coupling
- Identity block for φ to keep monolithic system non-singular

---

## Test Results — 2026-03-22/23

### Rosensweig Instability Tests

| Run | Mesh | dt | Clamp | Onset | Spikes | y_max | Notes |
|-----|------|----|-------|-------|--------|-------|-------|
| rosen_coarse_full2 | L4 | 5e-4 | yes | t=1.21 | 3 | 0.460 | Fine dt, late onset |
| rosen_coarse_dt2e3 | L4 | 2e-3 | yes | t=0.576 | 3 | 0.460 | Coarse dt, better onset |
| rosen_noclamp | L4 | 2e-3 | no | t=1.0 | 4 | 0.468 | No clamp: 3→4 spikes |
| rosen_noclamp_L5 | L5 | 2e-3 | no | t=0.866 | 4 | 0.457 | Finer mesh, 4 symmetric spikes |
| rosen_noclamp_L5_dt1e3 | L5 | 1e-3 | no | t=0.953 | ? | 0.461 | Smaller dt delays onset |
| Paper | L6 | 5e-4 | no | ~0.7 | 4-5 | ~0.55 | Reference |

### Key Observations
1. **Coarser dt gives better onset timing** — dt=2e-3 onset ≈ 0.6-0.9 vs dt=5e-4 onset ≈ 1.2
2. **Removing clamping improved spike count** — 3 → 4 spikes, better symmetry
3. **Spike height consistently low** — y_max ≈ 0.46 vs paper's 0.55 across all settings
4. **L5 with dt=2e-3** is the best local result: 4 symmetric spikes

### H = h_a Experiment (Paper Section 5 scheme)
- **Rosensweig with H = h_a only**: flat interface, zero spikes, zero instability
- **Rosensweig with H = h_a + ∇φ**: blows up (double-counts h_a)
- **Rosensweig with H = ∇φ**: spikes ✓ (the Poisson RHS encodes h_a into ∇φ)
- **Conclusion**: ∇φ from the Poisson equation IS the total field H = h_a + h_d. The Poisson RHS `(h_a - M, ∇X)` encodes h_a via natural boundary conditions. h_d (demagnetizing field) creates the interface gradients that drive instability.

### HPC Scaling Tests (L5, dt=1e-3, iterative)
| Ranks | Direct (s/step) | Iterative (s/step) |
|-------|----------------|-------------------|
| 6 | 70.6 | 71.8 |
| 8 | 46.7 | **44.1** |
| 10 | 45.6 | 47.9 |
| 12 | pending | pending |

**Winner: iterative, 8 ranks** (44.1s/step). 8-rank sweet spot due to communication overhead.

### Diagnostics Overhead Test (dome, L7, 8 ranks)
- Diagnostics every step: 6.82s/step
- Diagnostics every 100 steps: 7.33s/step
- **Conclusion**: diagnostics overhead is negligible. Solver dominates.

---

## Confirmed Scaling Bottlenecks

1. **Block Schur preconditioner rebuilt every step** — AMG factorization + Epetra row scan in constructor, destroyed at function return (ns_solver.cc:76, ns_block_preconditioner.cc:51-325)
2. **36 MPI_Allreduce per step** — 34 scalar + 2 vector reductions across 6 diagnostic modules. Negligible cost at current problem sizes.
3. **O(N_dofs) global maps at AMR** — 3 full-size arrays broadcast via MPI_Allreduce (ns_setup.cc:148-153). Only at AMR cycles.

---

## Code Audit Summary (2026-03-23)

### Verified Correct
- CH assembler: all signs, coefficients, implicit/explicit ✓
- NS assembler: mass, convection, viscous, pressure, Kelvin, capillary, gravity ✓
- Magnetic assembler: M transport, Poisson, upwinding ✓
- Parameters, BCs, ICs: all match paper ✓
- Truncated double-well potential (Eq 2) ✓
- Ghosted vectors properly updated before reads ✓
- Constraints applied after every solve ✓
- AMR solution transfer correct ✓

### Known Issues
- **BGS time derivative bug**: when `bgs_max_iterations > 1`, `theta_relevant_` and `mag_solution_old_` shift baseline each iteration. Harmless with default `bgs_max_iterations=1`.
- **Magnetic solver has no convergence flag**: returns void, unconverged solution used silently.
- **Dead code**: `rho_q` computed but unused in NS assembler, `assemble_kelvin_force_gradient` never called (133 lines), `r_density` parameter has no effect.

### Potential Avenues (Not Yet Tried)
1. Higher refinement (L6, L7) — HPC production runs pending
2. Multiple BGS iterations — needs time derivative bug fix first
3. Schur preconditioner caching between AMR cycles

---

## Active Tests

- **Dome test** (Semi_Coupled): `--dome --direct`, 8 ranks, H = h_a, 60000 steps to t=6.0
- **Dome test** (Decoupled): running separately for comparison
- **HPC scaling tests**: L6/L7 rank scaling, results pending
- **HPC production runs**: 9-test matrix (L6/L7, various dt), waiting for scaling results

---

## Project Structure
- `Semi_Coupled/` -- this project (monolithic electromagnetics)
- `Archived_Nochetto/` -- previous code with Picard iteration (preserved, not modified)
- `Decoupled/` -- Zhang, He & Yang scheme (separate project, not related)
