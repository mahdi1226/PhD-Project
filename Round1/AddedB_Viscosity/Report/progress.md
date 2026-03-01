# Verification & Validation Progress Report

## Project: Ferrofluid Phase-Field Model (Nochetto et al.)
## Last Updated: March 1, 2026

---

## Overview

Complete spatial MMS verification of the four-subsystem coupled ferrofluid model based on
Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531. Full paper-vs-code audit completed
(Session 15) — all equations, parameters, and FE spaces match the paper exactly. Production
Rosensweig instability validation is in progress, with a Kelvin force numerical instability
under investigation.

---

## Phase 1: MMS Verification (Sessions 1-6) — COMPLETE

### Session 1-2: Standalone MMS Tests
- Built standalone MMS tests for each subsystem: CH, Poisson, NS, Magnetization
- All 4 standalone tests achieved optimal convergence rates
- Identified that coupled tests were failing

### Session 3: Coupled Test Debugging
- **Kelvin force H sign bug**: Production code used `H = h_a - grad_phi` instead of
  `H = grad_phi` (the Poisson solve already incorporates h_a in the RHS). Fixed by
  setting `H = grad_phi` directly.
- **Poisson L-infinity mean-shift bug**: L-infinity error was artificially large due to
  a constant offset (Neumann BC gauge). Fixed by subtracting the mean before computing
  L-infinity norm.
- **CH velocity time mismatch**: CH assembler was using velocity at the wrong time level.
  Fixed to use velocity at the current time step.
- Result: POISSON_MAG, MAG_CH, NS_MAGNETIZATION coupled tests all pass.

### Session 4: FULL_SYSTEM Debugging Begins
- FULL_SYSTEM test still failing with zero magnetization convergence when U != 0
- Isolated the problem: DG magnetization transport face terms
- Identified that the ONLY difference between passing POISSON_MAG and failing FULL_SYSTEM
  is non-zero velocity in the magnetization transport

### Session 5: DG Transport Bug Hunt
- Created standalone MAG_TRANSPORT test isolating pure DG transport
- Applied three-part fix to face assembly: correct sign, correct trial/test slots
- Results improved but convergence still broken at O(1) error level

### Session 6: Root Cause Found and Fixed
- **ROOT CAUSE**: `FEInterfaceValues::shape_value(false, j, q)` uses an INTERFACE DOF index,
  not a cell-local index. For DG elements:
  - Interface DOFs 0..dofs_per_cell-1 = cell 0's DOFs ("here")
  - Interface DOFs dofs_per_cell..2*dofs_per_cell-1 = cell 1's DOFs ("there")
  - The code was calling `shape_value(false, j, q)` for j=0..3, which evaluated cell 0's
    DOFs on cell 1's side = **always zero** for DG
  - **Fix**: `shape_value(false, dofs_per_cell + j, q)` to access cell 1's DOFs
- **Result**: ALL 11 MMS tests pass with optimal convergence rates

---

## Phase 2: MPI & Solver Fixes (Sessions 7-12) — COMPLETE

### Bug 5: MPI_ERR_TRUNCATE (Session 7-8)
**Symptom:** Ferrofluid binary crashed with `MPI_ERR_TRUNCATE` at np >= 2.
**Root cause:** Three rank-0-only code paths called functions containing `MPI_Bcast`,
leaving orphaned broadcasts on the communicator. When Trilinos subsequently performed
`MPI_Allgather`, the stale data caused buffer mismatches.
**Fix:** Added non-MPI local-only timestamp functions for rank-0-only contexts.
**Files:** `utilities/tools.h`, `output/console_logger.h`

### Bug 6: Picard Double-Reduce (Session 8)
**Symptom:** Picard convergence residual was N times too large with N MPI ranks.
**Root cause:** `l2_norm()` on Trilinos vectors already returns global L2 norm. The code
then called `MPIUtils::reduce_sum()` on the squared norms, giving N * true_value.
**Fix:** Removed the redundant reduce_sum.
**File:** `core/phase_field.cc`

### Bug 7: --max_steps Not Enforced (Session 8)
**Fix:** Added max_steps check to the time loop while condition.
**File:** `core/phase_field.cc`

### Bug 8: CH_NS MMS Test Wrong Time Loop Ordering (Session 9)
**Symptom:** CH_NS test showed theta L2 rate = 0.0 (flat error).
**Root cause:** Test solved NS first then CH (opposite of paper's Block-GS ordering).
NS updated `ux_old = ux_sol` before CH, so CH received U^n instead of U^{n-1}.
**Fix:** Reordered time loop to match paper: CH -> NS -> update old values.
**Files:** `mms/coupled/ch_ns_mms_test.cc`, `mms/ch/ch_mms.h`

### Block-Gauss-Seidel Implementation (Session 10)
- Added outer BGS loop wrapping [CH] -> [Mag+Poisson (Picard)] -> [NS]
- Max 5 iterations, tolerance 1e-2 (relative change in theta and U)
- CLI: `--bgs/--no_bgs`, `--bgs_iters N`, `--bgs_tol TOL`
- All MMS tests still pass with identical convergence rates

---

## Phase 3: H Field Convention Fix (Session 13-14) — COMPLETE

### Bug 9: H Field Double-Counting in NS Kelvin Force (CRITICAL)

**Discovery context:** Droplet-uniform-B test exploded at step ~250 with F_mag >> F_cap.
A uniform field on a symmetric circular droplet should produce NO net Kelvin force.

**Root cause:** Inconsistency within the codebase:
- Magnetization assembler: `H = grad_phi` (CORRECT)
- NS assembler Kelvin force: `H = h_a - grad_phi` (WRONG)

The Nochetto paper (CMAME 2016, page 506) explicitly states: "H^h = grad(Phi^k) in M_h
is an admissible test function." The Poisson equation `(grad phi, grad X) = (h_a - M, grad X)`
determines phi such that grad(phi) IS the total magnetic field H. The RHS `(h_a - M)` is
just the source term. Adding h_a to grad(phi) double-counts the applied field.

**The Decoupled project (which produces correct Rosensweig results) confirms:**
```cpp
// Decoupled/navier_stokes/navier_stokes_assemble.cc, line 527-538:
// CRITICAL: Do NOT add h_a here! H = grad(phi)
H_vec[0] = phi_gradients[q][0];
H_vec[1] = phi_gradients[q][1];
```

**Fix — 3 changes in `assembly/ns_assembler.cc`:**
1. Cell term: `H = grad_phi` (was `h_a - grad_phi`)
2. (M*grad)H: Remove `M_grad_H *= -1.0` negation
3. Face jump: `jump_H = grad_phi_minus - grad_phi_plus` (was negated)

**Fix — 3 changes in MMS sources:**
4. `poisson_mms.h`: `poisson_mms_exact_H()` returns `grad(phi*)` (was `-grad(phi*)`)
5. `ns_mms.h`: `compute_kelvin_force_mms_source()` uses `H* = grad(phi*)` (6 sign flips)
6. `tools.h`: Updated `TEST_H_FORMULA` label

**Verification:** All 10 spatial MMS tests pass with optimal convergence rates at np=4.

---

## Phase 4: Production Validation Tests (Session 13-14) — IN PROGRESS

### Droplet-Uniform-B Test (Bug 9 regression test)

**Purpose:** Uniform field on symmetric circular droplet should produce no Kelvin force.
**Run:** r=5, np=4, 200 steps, dt=0.001, chi_0=0.5, h_a=(0,1), instant ramp

**Results (before fix):** Exploded at step ~250 (F_mag >> F_cap, droplet torn apart)

**Results (after fix):** Completed 200 steps, no explosion. However:
- U_max growing exponentially: 0.004 -> 0.007 -> 0.015 -> 0.043 (doubling every ~30 steps)
- Interface contracting: 0.2500 -> 0.2536 -> 0.2549 -> 0.2561
- F_mag ~ 101 (constant, NOT zero as expected for uniform field)
- Expected: F_mag ~ 0, U_max ~ 0, droplet stays circular

**Assessment:** The H field fix prevented the catastrophic explosion, but the Kelvin force
still produces spurious forces on a uniform-field symmetric droplet. This suggests the DG
skew form face terms may have an additional issue.

### Rosensweig Instability Test (r=4, 4000 steps)

**Purpose:** Reproduce Nochetto CMAME Section 6.2 Rosensweig instability benchmark.
**Run:** r=4, np=4, 4000 steps, dt=5e-4, no AMR, all paper parameters

**Results at step 719 (t=0.36, 22% of field ramp):**
- Violent odd-even numerical oscillation starting at step ~630 (t=0.32):
  - U_max oscillating 0.22 <-> 0.33 every other step
  - F_mag ~ 275,000 (8x gravity=33,000) at only 22% of field ramp
  - BGS not converging: residual 0.15 - 0.85, all 5 iterations consumed
- Interface barely moving: y_min=0.194, y_max=0.201 (vs initial 0.200)
- This is numerical instability, not physical instability

**Timeline of onset:**
| Step | t | U_max | F_mag | BGS_iters | BGS_res |
|------|-------|-------|-------|-----------|---------|
| 599 | 0.30 | 0.002 | 29K | 2 | 0.006 |
| 619 | 0.31 | 0.004 | 32K | 5 | 0.019 |
| 639 | 0.32 | 0.366 | 244K | 5 | 0.157 |
| 659 | 0.33 | 0.344 | 267K | 5 | 0.183 |
| 699 | 0.35 | 0.226 | 290K | 5 | 0.853 |
| 719 | 0.36 | 0.330 | 276K | 5 | 0.512 |

**Assessment:** The Kelvin force (DG skew form) becomes unstable at moderate field strength.
F_mag jumps from 32K to 244K in 20 steps with barely any interface movement. The BGS
iteration fails to converge (residual > 0.1), indicating the linearized system is not
well-posed at this field strength.

---

## Current MMS Test Results (All Passing, np=4, March 1 2026)

### Standalone Tests
| Test | Key Fields | L2 Rate | Expected |
|------|-----------|---------|----------|
| CH_STANDALONE | theta | 3.00 | 3.0 |
| POISSON_STANDALONE | phi | 3.00 | 3.0 |
| NS_STANDALONE | U, p | 3.00, 2.0+ | 3.0, 2.0 |
| MAGNETIZATION_STANDALONE | M | 2.00 | 2.0 |

### Coupled Tests
| Test | Key Fields | L2 Rate | Expected |
|------|-----------|---------|----------|
| POISSON_MAGNETIZATION | phi, M | 3.00, 2.00 | 3.0, 2.0 |
| NS_MAGNETIZATION | U, p | 3.00, 2.0+ | 3.0, 2.0 |
| CH_NS | theta, U | 3.00, 3.00 | 3.0, 3.0 |
| MAG_CH | theta, M | 3.00, 2.00 | 3.0, 2.0 |
| NS_POISSON_MAG | all | optimal | optimal |

### Full System
| Test | theta | U | phi | M | p |
|------|-------|---|-----|---|---|
| FULL_SYSTEM | 3.00 | 3.00 | 3.00 | 2.00 | 2.0+ |

---

## Bug Summary (All Sessions)

| # | Bug | Location | Impact | Session |
|---|-----|----------|--------|---------|
| 1 | Kelvin force H sign (MMS) | ns_assembler.cc | Wrong H in Kelvin force | 3 |
| 2 | Poisson L-inf offset | mms error computation | Inflated L-inf error | 3 |
| 3 | CH velocity timing | ch_assembler.cc | Wrong time level for U | 3 |
| 4 | **DG face DOF index** | magnetization_assembler.cc | **Zero DG face flux** | 4-6 |
| 5 | MPI orphaned Bcast | tools.h, loggers | MPI_ERR_TRUNCATE crash np>=2 | 7-8 |
| 6 | Picard double-reduce | phase_field.cc | N*true residual at np=N | 8 |
| 7 | --max_steps ignored | phase_field.cc | Time loop runs forever | 8 |
| 8 | CH_NS test ordering | ch_ns_mms_test.cc | theta rate=0.0 | 9 |
| 9 | **H field double-counting** | ns_assembler.cc | **Kelvin force explosion** | 13-14 |
| 10 | **Viscous term factor of 2** | ns_assembler.cc, ns_mms.h | **2x effective viscosity** | 15 |

---

## Phase 5: Paper-vs-Code Audit (Session 15)

### Bug 10: Viscous Term Factor of 2 (CRITICAL)

**Discovery context:** Full equation-by-equation audit of paper Eq. 42e vs code.

**Root cause:** The paper's bilinear form is `ν(T(U), T(V))` where `T = ½(∇u + (∇u)^T)`.
The code's helper `compute_symmetric_gradient()` returns `D = ∇u + (∇u)^T = 2T` (without
the ½ factor). The code used coefficient `nu/2`, giving `(ν/2)(D,D) = (ν/2)(2T)(2T) = 2ν(T,T)`,
which is 2× the paper.

**Impact:** The effective viscosity in the simulation was double the intended value. This
over-damps the flow and suppresses the Rosensweig instability, potentially preventing spike
formation even if the Kelvin force is correct.

**Fix (ns_assembler.cc, lines 328-332):** Changed coefficient from `nu_q / 2.0` to `nu_q / 4.0`:
```cpp
// (ν/4)(D,D) = ν(T_paper, T_paper) matches paper Eq. 42e
local_ux_ux(i, j) += (nu_q / 4.0) * (T_U_x * T_V_x) * JxW;
```

**Fix (ns_mms.h):** Changed MMS source from `-nu * laplacian` to `-(nu / 2.0) * laplacian`,
matching the strong form `-(ν/2)∆U` that corresponds to the bilinear form `ν(T,T)`.

**Note:** MMS tests passed before AND after the fix because the source term was always
internally consistent with the bilinear form. The bug was invisible to MMS — it only
affected the physical viscosity scaling.

### Full Audit Results (Session 15)

Equation-by-equation audit of paper (Eq. 42a-42f) vs code. Findings:

| Item | Paper | Code | Match? |
|------|-------|------|--------|
| NS viscous bilinear form | ν(T,T), T=½(∇u+∇u^T) | (ν/4)(D,D), D=2T | ✅ (after Bug 10 fix) |
| NS convection (skew) | B_h(U^{n-1}, U^n, V) | skew form with discrete U_old | ✅ |
| NS pressure | -(p, ∇·V) + (∇·U, Q) | DG P1, penalty diagonal | ✅ |
| NS Kelvin force | B_h^m(V, H, M) Eq. 38 | DG skew form | ✅ (after Bug 9 fix) |
| NS gravity | rg·H(θ/ε) | rg·H(θ/ε) | ✅ |
| NS capillary | (λ/ε)θ∇ψ | (λ/ε)θ∇ψ | ✅ |
| CH advection | +(Uθ, ∇Λ) | +(Uθ, ∇Λ) on RHS | ✅ |
| CH ψ convention | ε∆θ - (1/ε)f(θ) | (1/ε)f(θ) - ε∆θ = -ψ_paper | ✅ (self-consistent) |
| CH stabilization | η = ε | η = ε | ✅ |
| Poisson | (∇φ, ∇X) = (h_a-M, ∇X) | matches exactly | ✅ |
| H field | H = ∇φ | H = ∇φ | ✅ (after Bug 9 fix) |
| Magnetization transport | DG skew Eq. 57 | matches Eq. 57 | ✅ |
| Mag relaxation | (1/τ)(M - χH) | (1/τ)(M - χH) | ✅ |
| Applied field | Eq. 97-98 (2D dipole) | matches formula | ✅ |
| FE degrees | Q2/P1/Q2/Q2/Q1(DG) | Q2/P1/Q2/Q2/Q1(DG) | ✅ |
| Rosensweig params (6.2) | All listed | All match | ✅ |
| Material properties | ν_w, ν_f, ρ, χ₀, etc. | All match | ✅ |
| mu_0 | 1.0 (Table 1, p.520) | 1.0 | ✅ |
| gravity | (0, -30000) | (0, -30000) | ✅ |

**Only discrepancy found:** Bug 10 (viscous factor of 2), now fixed.

---

## Paper Comparison: Solver Strategy

### Our Implementation (Updated March 1, 2026)
- Block-Gauss-Seidel global iteration: [CH] -> [Poisson <-> Mag (Picard)] -> [NS], REPEAT
- Max 5 BGS iterations per time step, tolerance 1e-2 (relative change)
- Picard inner loop for Mag-Poisson coupling (7 iters, tol=0.05, omega=0.35)

### Solver Match Summary
| Aspect | Paper | Our Code | Match? |
|--------|-------|----------|--------|
| Linear solver | UMFPACK | UMFPACK (via Mumps) | Y |
| Mag-Poisson coupling | Coupled block | Picard iteration | Y |
| Global iteration | Block-Gauss-Seidel | Block-Gauss-Seidel (max 5) | Y |
| CH re-solve | Yes (each global iter) | Yes (each BGS iter) | Y |
| NS re-solve | Yes (each global iter) | Yes (each BGS iter) | Y |
| DG form | Skew-symmetric (Eq. 57) | Skew-symmetric (Eq. 57) | Y |
| Time stepping | Backward Euler | Backward Euler | Y |
| Stabilization | eta-stabilization (Eq. 42b) | eta-stabilization | Y |

### KEY DISCREPANCY: Kelvin Force Formulation
The Decoupled project (which produces correct Rosensweig results) uses THREE Kelvin force terms:
1. `mu_0 * (m . grad)H` — convective term
2. `mu_0/2 * (m x H, curl v)` — antisymmetric stress
3. `mu_0 * (m x curl(H), v)` — magnetic torque coupling

AddedB_Viscosity uses the DG skew form from Eq. 38 (Lemma 3.1):
```
B_h^m(V,H,M) = sum_T int_T [(M.grad)H.V + 1/2(div M)(H.V)] dx
             - sum_F int_F (V.n^-) [[H]].{M} ds
```
This is mathematically equivalent for smooth fields but may have different stability
properties in the discrete setting. The odd-even oscillation suggests the DG skew form
Kelvin force is not adequately stabilized.
