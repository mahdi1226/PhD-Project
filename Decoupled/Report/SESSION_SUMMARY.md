# Decoupled Ferrofluid Solver -- Development Summary

## Reference
Nochetto, Salgado & Tomas, *CMAME* **309** (2016) 497-531, Eq. 42a-42f -- PDE formulation
Zhang, He & Yang, *SIAM J. Sci. Comput.* **43**(1) (2021) -- SAV scheme, parameters, validation

---

## 1. Project Overview

A parallel finite element solver for ferrofluid dynamics (Rosensweig instability),
implemented in C++ using **deal.II 9.7.1** with Trilinos, p4est, and MPI.

The solver decomposes the coupled PDE system into **4 independent subsystems**,
each with its own FE space, assembly, solver, and VTK output:

| Subsystem      | Field       | FE Space    | Eq.      | Solver         |
|----------------|-------------|-------------|----------|----------------|
| Poisson        | phi         | CG Q1       | 42d      | CG + AMG       |
| Magnetization  | Mx, My      | DG Q1       | 42c/56-57| Direct / GMRES+ILU |
| Cahn-Hilliard  | theta, psi  | CG Q2       | 42a-b    | Direct (MUMPS) |
| Navier-Stokes  | ux, uy, p   | Q2 / DG-Q1  | 42e-f    | Direct / Block Schur |

---

## 2. Governing Equations

### Eq. 42d -- Poisson (Magnetostatic potential)
```
((1 + chi(theta)) grad(phi), grad(X)) = ((1 - chi(theta)) h_a, grad(X))
```
- Derived from -Laplacian(phi) = div(m) with algebraic m = chi(theta)(grad(phi) + h_a)
- Neumann BC, DoF-0 pinned to zero
- LHS + RHS reassembled each step (variable chi(theta) coefficient)

### Eq. 42c / 56-57 -- Magnetization (DG transport + relaxation)
```
(1/dt + 1/tau_M)(M^k, Z) - B_h^m(U^{n-1}, Z, M^k)
    = (1/tau_M)(chi(theta^{n-1}) H^k, Z) + (1/dt)(M^{n-1}, Z)
```
- DG-Q1 with upwind flux (skew-symmetric transport)
- Mx, My solved sequentially sharing one matrix
- Optional beta-term: beta * M x (M x H) (Landau-Lifshitz damping)

### Eq. 42a-b -- Cahn-Hilliard (Phase field + chemical potential, SAV formulation)
```
(1/dt + S1)(theta^n, chi) + gamma(grad(psi^n), grad(chi)) = (1/dt + S1)(theta^{n-1}, chi) + (U^{n-1} . grad(chi)) theta^{n-1}
(psi^n, xi) + lambda*epsilon(grad(theta^n), grad(xi)) = -(r/sqrt(E1+C0)) * (lambda/epsilon)(f(theta^{n-1}), xi)
```
- SAV (Scalar Auxiliary Variable) scheme from Zhang Eq 3.5-3.6
- S1 = lambda/(4*epsilon) stabilization constant
- r = sqrt(E1(theta) + C0) is the SAV variable
- psi already contains lambda factor (important for NS capillary coupling)
- Double-well potential: F(theta) = (1/4)(theta^2 - 1)^2, f = F' = theta^3 - theta

### Eq. 42e-f -- Navier-Stokes (Velocity + pressure with Kelvin + capillary force)
```
(1/dt + S2)(U^n, V) + (nu(theta) T(U^n), T(V))/2 + B_h(U^{n-1}; U^n, V)
    - (P^n, div(V)) = (1/dt + S2)(U^{n-1}, V)
                     + mu_0 * (M . grad)H . V           [Kelvin force]
                     + theta^{n-1} * grad(psi^n) . V    [Capillary force]
                     + rho(theta) * g . V                [Gravity]
(div(U^n), Q) = 0
```
- Component-split: separate DoFHandler for ux, uy
- S2 = NS stabilization, adaptive: S2 = 1.5*mu0^2*(chi0*|H_max|)^2/(4*nu_min)
- Kelvin force: mu_0 * (M . grad)H, where grad(H) = Hess(phi) + grad(h_a)
- Capillary force: theta * grad(psi), where psi from SAV CH solve (contains lambda)
- Variable viscosity: nu(theta) = nu_w + (nu_f - nu_w) * H(theta/epsilon)
- **Algebraic M mode**: M = chi(theta) * H_total (no magnetization PDE needed)

---

## 3. Coupling Strategy (Semi-coupled / Operator Splitting + Picard)

```
FOR each timestep n:
  1. Cahn-Hilliard:  solve for theta^n using U^{n-1}

  2. Picard loop (k = 0..max_picard):
       a. Poisson:        phi^k from M_relaxed     -> H^k = grad(phi)
       b. Magnetization:  M_raw^k from H^k, U^{n-1}, theta^{n-1}
       c. Under-relax:    M_relaxed = omega*M_raw + (1-omega)*M_old
       d. Check:          if ||M_new - M_old|| / ||M_new|| < tol: break

  3. Navier-Stokes:  solve for U^n using H^k, M^k, theta^{n-1}
```

**Under-relaxation:** omega = 0.35 (stabilizes M -> phi -> H -> M feedback)
**Picard iterations:** max 7, tolerance 0.01

---

## 4. What Was Built (Session Work)

### 4.1 Subsystem Architecture (all 4 subsystems)

Each subsystem follows an identical facade pattern:
```
<subsystem>/
  <subsystem>.h              -- Public facade class
  <subsystem>.cc              -- Constructor, facade methods
  <subsystem>_setup.cc        -- DoF distribution, sparsity, constraints
  <subsystem>_assemble.cc     -- Matrix + RHS assembly
  <subsystem>_solve.cc        -- Linear solver (direct / iterative)
  <subsystem>_output.cc       -- write_vtu() parallel VTK output
  <subsystem>_main.cc         -- Standalone 4-mode driver
  CMakeLists.txt              -- Static library + test + main targets
  tests/
    <subsystem>_mms.h         -- MMS exact solutions + source terms
    <subsystem>_mms_test.cc   -- MMS convergence test executable
```

### 4.2 Shared Infrastructure

- **`utilities/parameters.h/.cc`** -- Centralized runtime configuration with:
  - All physics parameters (epsilon, chi_0, tau_M, beta, mobility, lambda, nu, r, gravity)
  - Rosensweig preset (Section 6.2)
  - CLI parsing: `--mode`, `--ref`, `--steps`, `--refinement`, `--dt`, etc.
  - Run configuration: `params.run.mode`, `params.run.refs`, `params.run.steps`

- **`utilities/solver_info.h`** -- Solver result struct (iterations, residual, timing)

- **`utilities/timestamp.h`** -- Timestamped filename generation

- **`physics/`** -- Shared physics kernels:
  - `material_properties.h` -- Heaviside, chi(theta), nu(theta), rho(theta), F(theta)
  - `kelvin_force.h` -- Kelvin body force DG assembly kernels (cell + face)
  - `skew_forms.h` -- Skew-symmetric DG transport forms (scalar + vector)
  - `applied_field.h` -- 2D line dipole computation + gradient (Jacobian) for Kelvin force

### 4.3 Standalone Drivers (all 4 identical CLI)

All 4 `_main.cc` files support the same interface:
```bash
mpirun -np 4 ./<subsystem>_main --mode mms          # MMS spatial convergence (2D)
mpirun -np 4 ./<subsystem>_main --mode 2d            # Single 2D run + VTK output
mpirun -np 4 ./<subsystem>_main --mode 3d            # 3D run + VTK output
mpirun -np 4 ./<subsystem>_main --mode temporal      # Temporal convergence study
mpirun -np 4 ./<subsystem>_main --ref 2 3 4 5        # Override refinement range
mpirun -np 4 ./<subsystem>_main --refinement 5       # Single refinement (2d/3d)
mpirun -np 4 ./<subsystem>_main --steps 20           # Override time steps
```

### 4.4 MMS Verification Tests

Standalone test executables with pass/fail:
```bash
mpirun -np 4 ./test_poisson_mms --refs 2 3 4 5 6
mpirun -np 4 ./test_cahn_hilliard_mms
mpirun -np 4 ./test_magnetization_mms
mpirun -np 4 ./test_navier_stokes_mms --phase D --ref 2 3 4 5 6
```

### 4.5 Coupled MMS Tests

**Poisson + Magnetization (Picard iteration):**
```bash
mpirun -np 4 build/mms_tests/test_poisson_mag_mms
```
Tests the Picard iteration loop with under-relaxation, verifying coupled convergence.

**Poisson + Magnetization + NS (Kelvin force coupling):**
```bash
mpirun -np 1 build/mms_tests/test_poisson_mag_ns_mms --refs 2 3 4 --steps 1
```
Tests all 3 subsystems together: NS provides velocity for DG magnetization transport,
Poisson provides H for relaxation, and Kelvin force μ₀(M·∇)H couples back into NS.
Supports `--mag-only` (skip NS, use projected U) and `--project-u` (use exact U interpolant)
for incremental debugging.

**Files:**
- `mms_tests/poisson_mag_ns_mms.h` — NS MMS source with Kelvin body force + curl correction
- `mms_tests/poisson_mag_ns_mms_test.cc` — 3-subsystem test harness with extensive diagnostics

---

## 5. Test Results

### 5.1 MMS Spatial Convergence (all PASS)

**Standalone tests** (default refinements: {2, 3, 4, 5, 6}):

| Subsystem      | Field   | Norm | Expected | Achieved | Status |
|----------------|---------|------|----------|----------|--------|
| Poisson        | phi     | L2   | O(h^2)   | 2.00     | PASS   |
| Poisson        | phi     | H1   | O(h^1)   | 1.00     | PASS   |
| Cahn-Hilliard  | theta   | L2   | O(h^3)   | 3.00     | PASS   |
| Cahn-Hilliard  | theta   | H1   | O(h^2)   | 2.00     | PASS   |
| Magnetization  | M       | L2   | O(h^2)   | 2.00     | PASS   |
| Navier-Stokes  | ux      | H1   | O(h^2)   | 2.00     | PASS   |
| Navier-Stokes  | uy      | H1   | O(h^2)   | 2.00     | PASS   |
| Navier-Stokes  | p       | L2   | O(h^2)   | 2.19-2.47| PASS   |

**Coupled tests** (refinements: {2, 3, 4}):

| Test                        | phi L2 | M L2 | ux L2 | p L2 | Status |
|-----------------------------|--------|------|-------|------|--------|
| Poisson-Mag (Picard)        | 2.00   | 2.00 | --    | --   | PASS   |
| Poisson-Mag-NS (μ₀=0.1)    | 3.00   | 1.95 | 3.00  | 2.1  | PASS   |

### 5.2 VTK Output Modes

| Subsystem      | 2D VTK | 3D VTK | Notes |
|----------------|--------|--------|-------|
| Poisson        | Yes    | Yes    | Full working |
| Cahn-Hilliard  | Yes    | Yes    | Full working |
| Magnetization  | Yes    | Stub   | "3D not yet implemented" |
| Navier-Stokes  | Yes    | Stub   | Needs uz component (see Section 7) |

### 5.3 Temporal Convergence

**Status: Known limitation (deferred)**

All 3 time-dependent subsystems (NS, CH, Mag) show FLAT temporal error because
the MMS solutions are **linear in time** (t * f(x,y)). Backward Euler is exact
for linear-in-time data: `(u(t+dt) - u(t))/dt = f(x)` exactly.

**Fix needed:** Change MMS solutions to quadratic time dependence (t^2 * f(x,y))
in all `_mms.h` headers. This is non-trivial but well-defined future work.

---

## 6. File Inventory

### Source Files (by module)

```
utilities/
  parameters.h              -- All physics + run config + CLI
  parameters.cc              -- Rosensweig preset + parse_command_line()
  solver_info.h              -- LinearSolverParams, SolverInfo structs
  timestamp.h                -- timestamped_filename()
  amr.h                      -- Header-only 14-step AMR algorithm (template)

physics/
  material_properties.h      -- Heaviside, chi(theta), nu(theta), F(theta), f(theta)
  kelvin_force.h             -- Kelvin force cell + face assembly kernels
  skew_forms.h               -- DG skew-symmetric transport forms
  applied_field.h            -- 2D line dipole field + gradient computation

poisson/
  poisson.h                  -- PoissonSubsystem<dim> facade
  poisson.cc                 -- Constructor
  poisson_setup.cc           -- DoF, sparsity, constraints
  poisson_assemble.cc        -- Laplacian LHS + source RHS
  poisson_solve.cc           -- CG+AMG solver
  poisson_output.cc          -- write_vtu()
  poisson_main.cc            -- 4-mode standalone driver
  CMakeLists.txt
  tests/poisson_mms.h        -- Exact solution + source for phi
  tests/poisson_mms_test.cc  -- MMS convergence test

magnetization/
  magnetization.h            -- MagnetizationSubsystem<dim> facade
  magnetization.cc           -- Constructor
  magnetization_setup.cc     -- DG DoF, face sparsity
  magnetization_assemble.cc  -- DG cell+face, transport, relaxation, beta-term
  magnetization_solve.cc     -- Direct / GMRES+ILU
  magnetization_output.cc    -- write_vtu()
  magnetization_main.cc      -- 4-mode standalone driver
  CMakeLists.txt
  tests/magnetization_mms.h  -- Exact Mx, My + source + error computation
  tests/magnetization_mms_test.cc

cahn_hilliard/
  cahn_hilliard.h            -- CahnHilliardSubsystem<dim> facade
  cahn_hilliard.cc           -- Constructor
  cahn_hilliard_setup.cc     -- Coupled theta-psi DoF
  cahn_hilliard_assemble.cc  -- Monolithic saddle-point assembly
  cahn_hilliard_solve.cc     -- Direct (MUMPS)
  cahn_hilliard_output.cc    -- write_vtu()
  cahn_hilliard_main.cc      -- 4-mode standalone driver
  CMakeLists.txt
  tests/cahn_hilliard_mms.h  -- Exact theta + source
  tests/cahn_hilliard_mms_test.cc

navier_stokes/
  navier_stokes.h            -- NSSubsystem<dim> facade
  navier_stokes.cc           -- Constructor
  navier_stokes_setup.cc     -- ux, uy, p DoFs + saddle-point sparsity
  navier_stokes_assemble.cc  -- Momentum + continuity + Kelvin force
  navier_stokes_solve.cc     -- Direct / Block Schur FGMRES
  navier_stokes_output.cc    -- write_vtu()
  navier_stokes_main.cc      -- 4-mode standalone driver
  CMakeLists.txt
  tests/navier_stokes_mms.h  -- Exact ux, uy, p + sources + errors
  tests/navier_stokes_mms_test.cc  -- 4-phase MMS test (A/B/C/D)

mms_tests/
  poisson_mag_mms.h          -- Coupled MMS source for phi+M
  poisson_mag_mms_test.cc    -- Picard iteration convergence test
  poisson_mag_ns_mms.h       -- NS MMS source with Kelvin force (body + curl)
  poisson_mag_ns_mms_test.cc -- 3-subsystem Kelvin force convergence test
  CMakeLists.txt
```

### Build Commands

```bash
# Top-level build (all drivers + MMS tests)
cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j8

# Production run example
mpirun -np 4 ./drivers/ferrofluid_decoupled --rosensweig -r 4 --vtk_interval 100

# With AMR enabled
mpirun -np 4 ./drivers/ferrofluid_decoupled --rosensweig -r 3 --amr --amr-interval 10

# MMS test
mpirun -np 1 ./mms_tests/test_coupled_system_mms --refs 2 3 4 --steps 1
```

---

## 6b. Critical Bug Fix: DG Face Assembly (February 2026)

### The Bug

The DG magnetization transport face assembly used `FEInterfaceValues::shape_value`
with incorrect interface DoF indexing, causing **zero cross-cell coupling** on all
interior faces.

In deal.II's `FEInterfaceValues` for DG elements, interface DoFs are numbered:
- `0 .. dpc-1` → "here" cell DoFs
- `dpc .. 2*dpc-1` → "there" cell DoFs

The code used `shape_value(false, i, q)` to get the "there" cell's i-th basis function.
But interface DoF `i` belongs to the "here" cell — evaluating it from the "there" side
returns 0 for DG. The correct call is `shape_value(false, i + dofs_per_cell, q)`.

### Impact

With `Z_i_there = 0` and `M_j_there = 0` at all quadrature points:
- `face_ht`, `face_th`, `face_tt` matrices were identically zero
- Only `face_hh` (here-here block) was non-zero
- The DG method had NO inter-element coupling in the face flux
- Transport errors were O(1) instead of O(h²)

This bug affected ALL validation tests (droplet, rosensweig) where the magnetization
PDE is solved with non-zero velocity. The standalone magnetization MMS test (U=0)
was unaffected because the face flux is zero when U=0.

### Fix

1. **Indexing fix**: `shape_value(false, i + dofs_per_cell, q)` in 4 locations
   (AMR Case 1 + Case 2, both matrix assembly and face_mms_active RHS)
2. **Upwind penalty added**: `+½|U·n|[[Z]][[M]]` for optimal O(h²) convergence
   (central flux alone gives only O(h))

### Verification

| Configuration | M_L2 (ref=4) | Rate |
|---------------|-------------|------|
| Before fix (broken face flux) | 0.121 | O(1) — no convergence |
| After fix, central flux only | 1.58e-4 | 1.0 — sub-optimal |
| After fix + upwind penalty | 7.20e-6 | **1.96** — optimal |

Full Poisson-Mag-NS test with μ₀=0.1: **[PASS]** — all rates within tolerance.

### File

`magnetization/magnetization_assemble.cc` — face loop, both AMR cases

---

## 6c. Validation Test Results (Sessions 9-10, February 2026)

### Completed Tests

All tests rerun after DG face fix (Session 8) and nonuniform preset additions (Session 9).

| Test | Steps | dt | theta range | |U|_max | Picard | Status |
|------|-------|-----|-------------|---------|--------|--------|
| Square (CH-only, r=6) | 5000 | 1e-3 | [-0.992, 1.01] | 0 | N/A | **PASS** |
| Droplet w/field (r=7) | 1500 | 1e-3 | [-1.00, 1.03] | ~0.01 | 12 iters | **PASS** |
| Rosensweig uniform (r=4) | 2000 | 1e-3 | [-1.00, 1.00] | ~0.5 | 10 iters | **PASS** |
| Rosensweig nonuniform | 17500 | 2e-4 | [-1.06, 1.08] | 30+ | 2 iters | **FAIL** |

### Nonuniform Rosensweig Failure Analysis

**Parameters:** 42 dipoles (3 rows at y=-0.5,-0.75,-1.0), chi_0=0.9, h=1/120,
interface at y=0.1, lambda=0.25.

**Timeline:**
- Steps 0-10000 (t=0 to 2.0): Correct behavior. Two spikes form gradually. |U|~0.5.
- Step 10350 (t=2.07): Velocity jumps from 0.52 to 0.73 (40% in one step).
- Steps 10350-10570: |U| explodes to 11.4, theta overshoots to [-1.02, 1.08].
- Steps 10570+: Chaotic dynamics, spike morphology destroyed, |U| reaches 30+.

**Root causes identified:**
1. Picard convergence uses global L2 norm (misses local spike tip dynamics)
2. S2 stabilization lags by one step (computed from phi^{n-1})
3. No outer NS-Mag iteration under strong coupling (operator splitting instability)
4. Nonuniform case has ~7x stronger Kelvin force than uniform (closer dipoles, higher chi_0)

**VTK output:** Frames 0-200 (steps 0-10000) show physically correct two-spike Rosensweig
instability. Frames >207 show numerical artifacts.

---

## 7. Known Gaps and Future Work

### 7.1 Temporal Convergence Tests (Deferred)
- **Problem:** MMS solutions linear in time -> backward Euler has zero temporal error
- **Fix:** Use t^2 * f(x,y) in all `_mms.h` headers, update source terms, ICs, BCs
- **Scope:** ~3 files, non-trivial but well-defined

### 7.2 NS 3D Support (Not Yet Implemented)
- **Problem:** NS uses component-split (separate ux, uy DoFHandlers). 3D needs uz.
- **What's needed:**
  - Add `uz_dof_handler_`, `uz_solution_`, etc. (~10 new member variables)
  - Add uz assembly block (duplicate uy pattern)
  - Extend symmetric gradient helpers from 2D indexing to 3D
  - Grow saddle-point system from 3-block to 4-block
  - Fix divergence constraint: add duz/dz
  - Change vorticity from 2D scalar to 3D vector
- **Estimate:** ~4 hours, mechanical (pattern duplication), not algorithmically hard
- **Note:** NSSubsystem is already `dim`-templated with `dim=3` instantiations in all .cc files

### 7.3 Magnetization 3D Support (Stub)
- Prints "not yet implemented" -- simpler than NS (no component split)
- Just needs 3D mesh generation in the driver

### 7.4 Production Coupled Driver -- IMPLEMENTED (Strategy A)
- `drivers/decoupled_driver.cc` implements Strategy A (Gauss-Seidel splitting)
- Uses SAV (Scalar Auxiliary Variable) energy-stable time integration
- Supports algebraic magnetization M = chi(theta)*H (skips magnetization PDE)
- Validation presets: `--validation square|droplet|droplet_nofield|rosensweig` and `--rosensweig-nonuniform`
- Build: `cd drivers/build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j8`
- Run: `mpirun -np 2 ./ferrofluid_decoupled --validation droplet -r 7 --vtk_interval 100 --sav_S1 5000`

---

## 8. Validation Test Parameters

### 8.1 Rosensweig Instability (Zhang Eq 4.4)

```
Domain:        [0, 1] x [0, 0.6]
IC:            Flat interface at y = 0.2, NO perturbation
Mesh:          refinement level 4
epsilon:       5e-3
chi_0:         0.5
mu_0:          1.0
Relaxation:    tau_M = 1e-6  (algebraic M = chi*H used instead)
Viscosity:     nu_water = 1.0, nu_ferro = 2.0
Density ratio: r = 0.1
Gravity:       |g| = 6e4, direction = (0, -1)
CH mobility:   gamma = 2e-4
lambda:        1
Time step:     dt = 1e-3, max_steps = 2000
Dipoles:       5 line dipoles at y = -0.15, x in {0, 0.25, 0.5, 0.75, 1.0}
               ramp_slope = 5000, intensity_max = 8000
```

### 8.2 Droplet Deformation (Zhang Eq 4.8)

```
Domain:        [0, 1] x [0, 1]
IC:            Circular droplet R = 0.1 at (0.5, 0.5)
Mesh:          refinement level 7
epsilon:       2e-3
chi_0:         2
mu_0:          0.1
Viscosity:     nu_water = nu_ferro = 1
lambda:        1
gravity:       0
ramp_slope:    1000
Time step:     dt = 1e-3, max_steps = 1500
```

---

## 9. Data Flow Between Subsystems

```
                    theta^{n-1}
                   +----------+
                   |          |
                   v          |
     +---------+  chi(theta)  |
     | Poisson |<-------------+
     |  phi^k  |              |
     +----+----+              |
          | H = grad(phi)     |
          v                   |
  +-------+--------+         |
  | Magnetization   |         |
  |  M^k (Mx, My)  |<--------+  chi(theta), U^{n-1}
  +-------+--------+
          |
          | M^k, H^k
          v
  +-------+--------+      +------------------+
  | Navier-Stokes  |<-----| Cahn-Hilliard    |
  |  U^n, p^n      |      |  theta^n, psi^n  |
  +----------------+      +------------------+
     Kelvin force:            Convection:
     mu_0 * B_h^m(V,H,M)     (U^{n-1} . grad) theta
     nu(theta) viscosity
```

---

## 10. AMR (Adaptive Mesh Refinement) — Session 13, March 2026

### 10.1 Overview

Adaptive mesh refinement for all 4 subsystems on the shared p4est triangulation.
Refines near the diffuse interface (|theta| < threshold) and coarsens in bulk regions.
Opt-in via `--amr` CLI flag; default behavior (uniform global refinement) is unchanged.

### 10.2 Algorithm (14 steps)

Located in `utilities/amr.h` (header-only template):

1. Kelly error estimation on theta (interface field)
2. Mark cells with fixed-fraction (upper 30% / lower 10%)
3. Enforce level limits (max/min)
4. Interface protection — never coarsen cells where |theta| < threshold
5. Prepare triangulation (`prepare_coarsening_and_refinement()`)
6. Create `SolutionTransfer` for all subsystems (7 DoFHandlers)
7. Execute mesh refinement (`execute_coarsening_and_refinement()`)
8. Re-setup all subsystems (rebuilds DoFs, constraints, matrices, vectors)
9. Interpolate solutions to new mesh
10. Apply constraints (`distribute()`)
11. Clamp theta to [-1, 1] (interpolation causes overshoot)
12. Recompute psi = theta^3 - theta nodally (restore constitutive relation)
13. Update all ghost vectors (`update_ghosts()` on each subsystem)
14. Log diagnostics (cells before/after, clamped DoFs, theta bounds)

### 10.3 Physics-Based Activation Gate

AMR stays dormant until |U|_max exceeds a threshold (default 1e-3), then uses
fixed-interval refinement. This avoids wasting computation on high-DoF meshes
during the quiescent early phase when the interface hasn't moved.

- Once activated, AMR stays active for the rest of the simulation
- When NS is disabled (CH-only), activates immediately (no velocity to gate on)
- `--amr-activation-U 0` for immediate activation (no gate)

### 10.4 CLI Parameters

```
--amr / --no-amr              Enable/disable (default: OFF)
--amr-interval N              Refine every N steps (default: 5)
--amr-max-level N             Max refinement level (default: 0 = no cap)
--amr-min-level N             Min refinement level (default: 0)
--amr-upper-fraction V        Top fraction to refine (default: 0.3)
--amr-lower-fraction V        Bottom fraction to coarsen (default: 0.10)
--amr-activation-U V          |U| threshold to start AMR (default: 1e-3)
```

### 10.5 Files Modified

| File | Change |
|------|--------|
| `utilities/amr.h` | **NEW** — 14-step AMR algorithm (header-only template) |
| `utilities/parameters.h` | AMR fields in Mesh struct, amr_activation_U |
| `utilities/parameters.cc` | AMR CLI parsing, help text |
| `cahn_hilliard/cahn_hilliard.h` | Mutable DoFHandler accessors |
| `cahn_hilliard/cahn_hilliard.cc` | ghosts_valid_ = false in setup() |
| `navier_stokes/navier_stokes.h` | Mutable solution + DoFHandler accessors |
| `poisson/poisson.h` | Mutable solution + DoFHandler accessor |
| `magnetization/magnetization.h` | Mutable solution + DoFHandler accessors |
| `drivers/decoupled_driver.cc` | AMR call in time loop + activation gate + mesh_level VTK |

### 10.6 Test Results

- CH-only: 200+ steps, mesh 960→420, energy decreasing
- Full 4-system: 370+ steps at r3, mesh 3840→1140 (70% reduction)
- Uniform Rosensweig + AMR: 100 steps, activation at step 63, mesh 3840→2112→11688
- Physics gate saves 37% wall time vs fixed-interval AMR
- Default path (no AMR): unchanged behavior, no regression

### 10.7 Critical Bug Fix: Ghost Validity

After AMR, the CH subsystem's `update_ghosts()` was silently skipping the copy
because `ghosts_valid_` was not reset by `setup()`. Fixed by adding
`ghosts_valid_ = false` in `CahnHilliardSubsystem::setup()`.

---

## 11. Material Property Fix & Full Shliomis Model (Session 14, March 4, 2026)

### 11.1 Root Cause: Sigmoid χ/ν Breaks Rosensweig

Commit `c8666bf` introduced two changes simultaneously: (1) sigmoid interpolation for χ and ν, (2) spin-vorticity coupling ½(∇×U × M, Z). Both were blamed for Rosensweig instability explosion.

**Isolation test** (this session): Built both 800aa2c (linear χ/ν, no spin-vorticity) and HEAD (linear χ/ν, WITH spin-vorticity re-enabled) in parallel worktrees. Both ran 2000 steps on 4 MPI ranks, stable.

| Configuration | theta range | |U| final | Status |
|---------------|-------------|-----------|--------|
| 800aa2c (linear, no spin-vort) | [-1.01, 0.995] | 2.79 | STABLE |
| HEAD (linear + spin-vort) | [-1.00, 0.996] | 0.52 | STABLE |

**Conclusion**: Sigmoid was the sole culprit. Spin-vorticity is physically correct (damps velocity) and safe.

### 11.2 Production Configuration

**Full Shliomis model** = LINEAR χ/ν + spin-vorticity ON:
- χ(θ) = χ₀·(θ+1)/2 — linear (Zhang convention)
- ν(θ) = ν_w·(1-θ)/2 + ν_f·(θ+1)/2 — linear (Zhang convention)
- ρ(θ) = 1 + r·H(θ/ε) — sigmoid (Zhang Eq 4.2, unchanged)
- Spin-vorticity: ½(∇×U × M^{n-1}, Z) in magnetization assembly

### 11.3 Files Modified

| File | Change |
|------|--------|
| `physics/material_properties.h` | Reverted χ and ν from sigmoid to LINEAR |
| `magnetization/magnetization_assemble.cc` | Re-enabled spin-vorticity coupling |

---

*Generated: February 2025*
*Updated: March 4, 2026 (Session 14: Material property fix, full Shliomis model confirmed)*
*Total source code: ~11,000 lines across 4 subsystems + shared libraries*
