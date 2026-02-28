# Implementation Plan: Next Steps

## Project: Ferrofluid Phase-Field Model (Nochetto et al.)
## Date: February 2026

---

## Current Status

- All 11 MMS tests pass with optimal spatial convergence rates
- Spatial verification is COMPLETE for the four-subsystem coupled model
- Production code (assemblers, solvers, setup) verified against manufactured solutions

---

## Phase 1: Temporal Convergence Verification (Next)

### Goal
Verify first-order temporal convergence (backward Euler) for all subsystems.

### Approach
1. Create t^2 MMS solutions (quadratic in time) to avoid exact temporal discretization
   - Current M* = t*f(x,y) gives ZERO temporal error with backward Euler
   - Need M* = t^2*f(x,y) so that backward Euler produces O(dt) error
2. Fix spatial mesh (fine enough to resolve), vary dt
3. Expected: O(dt) convergence for all fields (first-order scheme)

### Implementation
- Modify `magnetization_mms.h`: add t^2 variant MMS functions
- Modify `ns_mms.h`, `ch_mms.h`, `poisson_mms.h`: same t^2 variants
- Add `--mode temporal` flag to test_mms.cc
- Run with fixed refs=5, varying steps=10,20,40,80,160

---

## Phase 2: Block-Gauss-Seidel Global Iteration — ✅ COMPLETE

### Status: Implemented and tested (February 27, 2026)

### What Was Done
1. **Added outer BGS loop** in `core/phase_field.cc`:
   - Wraps [CH] → [Mag+Poisson (Picard)] → [NS] in a convergence loop
   - Convergence criterion: max relative change over θ and U < tolerance
   - Defaults: max 5 iterations, tolerance 1e-2
2. **Added parameters** to `utilities/parameters.h/cc`:
   - `enable_bgs` (bool, default true), `bgs_max_iterations`, `bgs_tolerance`
   - CLI flags: `--bgs/--no_bgs`, `--bgs_iters N`, `--bgs_tol TOL`
3. **Added diagnostics**: BGS iterations and residual logged to CSV and console
4. **MMS verification**: All 11 tests still pass with identical convergence rates

### Observed Behavior
- Initial transient (~30 steps): needs all 5 BGS iterations, residual ~0.02-0.14
- Steady state: converges in 2-3 iterations
- AMR steps temporarily increase residual (mesh interpolation error)
- Overhead: ~4-5x per time step vs single-pass

### Files Modified
- `core/phase_field.cc`: BGS outer loop (main change)
- `core/phase_field.h`: `last_bgs_iterations_`, `last_bgs_residual_` members
- `core/phase_field_setup.cc`: initializer list
- `utilities/parameters.h/cc`: BGS parameters and CLI flags
- `utilities/tools.h`: CSV header stamp
- `diagnostics/step_data.h`: `bgs_iterations`, `bgs_residual` fields
- `output/metrics_logger.h`: CSV columns
- `output/console_logger.h`: BGS console note

### Effort
- ~50 lines of code change
- Low risk: single iteration reproduces current behavior exactly

---

## Phase 3: Code Cleanup (See plan file)

A detailed cleanup plan exists at `.claude/plans/mighty-wondering-koala.md`.

### Key Items
1. **Remove Zhang-He-Yang extension code**: beta damping, spin-vorticity (unused)
   - These are remnants from the full Rosensweig/M3AS model (arXiv:1511.04381)
   - Our code implements the simplified CMAME model (no angular momentum)
2. **Remove dead/debug code**: verify_susceptibility_fix(), abandoned face loop, static prints
3. **Consolidate duplicated code**: applied field, Heaviside function
4. **Fix performance**: FEFaceValues inside loop, vector allocations inside loop
5. **Extract hardcoded values**: under-relaxation omega, dipole regularization

### Files Affected
~15 files, mostly assemblers and utilities

---

## Phase 4: Rosensweig Instability Validation

### Goal
Reproduce the Rosensweig (normal field) instability benchmark from CMAME 2016, Section 6.2.

### Paper Reference Parameters (Section 6.2, p.520-522)
- Domain: [0,1] × [0,0.6], ferrofluid pool depth 0.2
- ε=0.01, λ=0.05, χ₀=0.5, γ=0.0002, μ₀=1
- v_w=1.0, v_f=2.0, r=0.1 (Boussinesq density ratio)
- g = (0, -30000) (scaled gravity)
- 5 dipoles at y=-15, d=(0,1), ramp from α_s=0 to α_s=6000 over t∈[0,1.6]
- AMR: Kelly error estimator (Eq. 99), Dörfler marking, refined every 5 steps
- 6 refinement levels, 4000 time steps (dt=5e-4), t_F=2.0
- Solver: Block-Gauss-Seidel + UMFPACK (deal.II)

### Expected Results (from paper)
- 4-5 spikes form inside the domain (paper shows 5 at t=2.0)
- Critical wavelength λ_c = 2π√(σ/(gΔρ)) ≈ 0.25 (4 peaks in unit box)
- Interesting dynamics from t=0.7 to t=1.3
- Interface should remain flat until field reaches critical strength

### Current Run (In Progress)
- Running with np=1, r=4 (4 refinement levels), Release mode
- Parameters match paper exactly via `setup_rosensweig()`
- Single-pass solver (no global iteration) — may need Phase 2 fix

### Steps
1. ✅ Set up initial condition: flat ferrofluid-air interface
2. ✅ Apply dipole field with time ramping
3. ⏳ Time-step to t=2.0 (running)
4. Evaluate: amplitude, spike count, wavelength
5. Compare with paper's Fig. 1 results
6. If instability is absent/wrong: implement Block-Gauss-Seidel (Phase 2)

---

## Phase 5: Dipole Field Validation

### Goal
Validate dipole magnetic field interactions with ferrofluid drop.

### Setup
- Circular ferrofluid droplet in non-magnetic fluid
- Point dipole or line of dipoles creating non-uniform field
- Observe droplet deformation and migration

### Steps
1. Initialize circular droplet using phase field (tanh profile)
2. Position dipole(s) using existing DipoleConfig parameters
3. Run simulation, track droplet shape evolution
4. Compare with analytical predictions where available

---

## Phase 6: Performance Optimization

### Goal
Enable larger 3D simulations and longer time integrations.

### Items
- Profile hotspots (assemblers, solvers)
- Fix MPI np>1 issue (Trilinos/Amesos direct solver MPI_ERR_TRUNCATE)
- Parallel scaling study (weak and strong)
- Consider iterative solvers (GMRES+AMG) for scalability
- Consider matrix-free assembly for large problems

---

## Priority Order

1. **Rosensweig validation** (current — running now)
2. **Block-Gauss-Seidel** (if Rosensweig results require it)
3. **Temporal convergence** (completes verification story)
4. **Code cleanup** (reduces maintenance burden)
5. Dipole validation
6. Performance optimization
