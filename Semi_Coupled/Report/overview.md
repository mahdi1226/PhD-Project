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

## Magnetization Model — Full H = ∇φ + h_a (Paper Eq 42c)

The magnetization assembler implements the paper's full model (Eq 42c):
```
(1/T)(M^k, Z) = (1/T) chi(theta) H^k Z,   where H^k = grad(phi^k) + h_a
```
In the monolithic system, grad(phi^k) is a trial function. The relaxation term
splits into LHS and RHS:
- **LHS (C_M_phi coupling):** `-(1/T) chi(theta) (∇φ_j, Z_i)` — couples M to φ
- **RHS (h_a source):** `+(1/T) chi(theta) (h_a, Z_i)` — known applied field

### History: Incorrect "Simplified Model" (removed March 15, 2026)
A previous version introduced a `--full_mag` / simplified split based on a misreading
of the paper. Section 5 (p.518-519) discusses h = h_a as a **negative control** for the
dome test (Figure 7), not as a simplification used in the actual numerical experiments.
The "simplified" mode dropped the C_M_phi coupling entirely, relaxing M toward
chi(theta)*h_a only — a hybrid not described in the paper. Additionally, the full-model
branch had a bug: the h_a RHS term was incorrectly guarded by `if (!use_full_mag_model)`,
so --full_mag runs relaxed toward chi(theta)*∇φ only, missing h_a entirely.

Both issues are now fixed: the full model (H = ∇φ + h_a) is hardcoded, the simplified
mode and --full_mag flag are removed.

### Observation After Audit Fixes
A coarse test (L4, dt=0.002, 500 steps to t=1.0) with audit fixes produced
**4 spikes** — the correct number from the paper. This was the first time the code
generated any spikes at all. However:
- Onset was delayed compared to the paper (~t=0.9 vs paper's earlier onset)
- Spike shape and wavelength may not match perfectly
- Higher resolution tests needed to confirm

This observation motivated the v2 algorithmic isolation study (see below).

## Algorithmic Variant CLI Flags (March 2026)

To systematically isolate which algorithmic choices affect the Rosensweig pattern,
two boolean CLI flags were added:

| Flag | Parameter | Default | Effect |
|------|-----------|---------|--------|
| `--explicit_ch` | `use_explicit_ch_convection` | `false` | CH convection uses θ^{k-1} (explicit, paper Eq 65a) instead of θ^k (implicit) |
| `--theta_old_chi` | `use_theta_old_for_chi` | `false` | Magnetics uses θ^{n-1} for χ(θ) instead of θ^n from CH |

### Why These Flags Exist
- **CH convection:** Paper Eq 65a uses explicit θ^{k-1}, but implicit θ^k provides
  better U-θ coupling for instability-driven flows. Both are valid first-order schemes.
- **θ time level for χ(θ):** Theorem 4.1's energy stability proof assumes θ^{n-1}
  (frozen from previous time step), but using θ^n (fresh from CH solve) provides
  tighter coupling within the BGS iteration.

### Files Modified
- `utilities/parameters.h` — 2 bool fields added
- `utilities/parameters.cc` — CLI parsing + help text + verbose output
- `assembly/ch_assembler.cc` — Conditional implicit/explicit convection
- `core/phase_field.cc` — Conditional θ_old vs θ pass to magnetic assembler

## Rosensweig v2 Test Matrix — Algorithmic Isolation Study (March 2026)

**Location:** `hpcc_run/rosensweig_v2.dat` (8 tests, SLURM array job)

**Goal:** Run at paper resolution (L6, dt=5e-4) to isolate which algorithmic choices
affect onset time, spike count, and pattern quality. Each test changes ONE variable
from the baseline.

### Group A: Algorithmic Variants (L6, dt=5e-4, ILU, BGS=1)

| ID | Flags | What Changes | Hypothesis |
|----|-------|-------------|------------|
| T1 | baseline | — | Reference: implicit CH + full mag (Eq 42c) + θ^n |
| T2 | `--theta_old_chi` | χ(θ) time level | Does freezing θ^{n-1} for χ affect onset? |
| T3 | `--explicit_ch` | CH convection | Control — expect flat (explicit kills instability) |

### Group B: Coupling Strength (L6, dt=5e-4, ILU)

| ID | Flags | What Changes | Hypothesis |
|----|-------|-------------|------------|
| T4 | `--bgs_iters 2` | BGS iterations | Tighter coupling — improves pattern? |
| T5 | `--bgs_iters 3` | BGS iterations | Even tighter — convergence check |

### Group C: Resolution (ILU, BGS=1)

| ID | Flags | What Changes | Hypothesis |
|----|-------|-------------|------------|
| T6 | `--dt 2.5e-4 --max_steps 8000` | dt (2× finer) | Earlier onset? Better wavelength? |
| T7 | `-r 7` | Mesh (L7) | Finer mesh → different spike count/shape? |
| T8 | `--no_amr -r 6` | AMR off | Isolate AMR artifacts from physics |

### How to Submit
```bash
sbatch --array=1-8 hpcc_run/submit.sub hpcc_run/rosensweig_v2.dat
```

### What to Look For
For each test, extract from `diagnostics.csv`:
1. **Onset time:** when `interface_y_max` first exceeds 0.21
2. **Spike count:** from VTK at t=2.0
3. **Final spike height:** `interface_y_max` at t=2.0
4. **Energy stability:** E_total monotonically decreasing?

### Key Comparisons
- T1 vs T2: Does θ time level for χ matter? → Force balance timing
- T1 vs T3: Confirms explicit CH kills instability (control)
- T1 vs T4/T5: Does BGS coupling fix pattern? → BGS lag issue
- T1 vs T6: Does dt refinement fix onset? → First-order time error
- T1 vs T7: Does mesh refinement fix wavelength? → Mesh-dependent selection
- T1 vs T8: Does AMR affect pattern? → AMR artifact check

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
- `Semi_Coupled/hpcc_run/` -- HPC submission scripts, test matrices, notes
- `Semi_Coupled/Report/` -- implementation plans, derivation, this overview
- `Archived_Nochetto/` -- previous code with Picard iteration (preserved, not modified)
- `Decoupled/` -- Zhang, He & Yang scheme (separate project, not related)

## Status Summary (March 15, 2026)

### Completed
- Monolithic M+φ system (Eq 42c-42d) implemented and MMS-verified
- Full audit against Nochetto CMAME 2016 — 4 fixes applied
- Full mag model (H = ∇φ + h_a) hardcoded per paper Eq 42c
- 2 algorithmic variant CLI flags (`--explicit_ch`, `--theta_old_chi`)
- v2 test matrix (8 tests) created in `hpcc_run/rosensweig_v2.dat`
- Coarse test (L4) produces 4 spikes — first successful spike formation

### Pending
- Submit v2 test matrix on HPC (The Mill) and collect results
- Analyze results to determine which algorithmic choices are critical
- Validation tests (square, droplet, elongation) not yet run
- T1-T3 implementation (Landau-Lifshitz, spin-vorticity, antisymmetric stress) planned but not started
