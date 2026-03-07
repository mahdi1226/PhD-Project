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

| Subsystem      | Field       | FE Space    | Eq.      | Solver (default)      | Direct option     |
|----------------|-------------|-------------|----------|-----------------------|-------------------|
| Poisson        | phi         | CG Q2       | 42d      | CG + AMG              | `--poisson-direct` |
| Magnetization  | Mx, My      | DG Q1       | 42c/56-57| Direct (MUMPS)        | (already default)  |
| Cahn-Hilliard  | theta, psi  | CG Q2       | 42a-b    | Direct (MUMPS)        | (already default)  |
| Navier-Stokes  | ux, uy, p   | Q2 / CG-Q1  | 42e-f   | CG+AMG (projection)   | `--ns-direct`      |

Use `--all-direct` to force direct solvers for ALL subsystems.

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

## 3. Coupling Strategy (Fully Decoupled, Zhang Algorithm 3.1)

```
FOR each timestep n:
  1. Cahn-Hilliard:  solve for theta^n using U^{n-1}

  2. Poisson-Magnetization (Gauss-Seidel, NO Picard):
       a. Magnetization Step 5: M_explicit from H^{n-1}, M^{n-1} (mass + relaxation only)
       b. Poisson:               phi^n from M_explicit           -> H^n = grad(phi)
       c. Magnetization Step 6:  M^n from H^n, U^{n-1}          (full DG transport, implicit)

  3. Navier-Stokes (Pressure-Correction Projection, Zhang Alg 3.1 Steps 2-4):
       a. Velocity predictor:   solve for u_bar^n (viscous + forces, old pressure)
       b. Pressure Poisson:     -Laplacian(dp) = -(1/dt) div(u_bar^n)
       c. Velocity correction:  M * delta_u = dt * grad(dp),  u^n = u_bar^n + delta_u
```

**No Picard iteration** — single forward pass per timestep (unconditionally stable per Zhang).
**Pressure changed from DG-P1 to CG-Q1** — required for the pressure Poisson Laplacian.
**3 separate CG+AMG solves** replace the monolithic saddle-point system.

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
| Rosensweig nonuniform | 17500 | 2e-4 | blows up | 30+ | 2 iters | **FAIL** (all 8 runs, see Sec 14) |

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

## 12. Zhang Algorithm 3.1 Step 5/6 — Magnetization Transport Splitting (Session 15, March 5, 2026)

### 12.1 Motivation

Zhang's Algorithm 3.1 splits the magnetization transport into two stages for unconditional
energy stability. The original implementation only had Step 5 (explicit transport inside the
Picard loop). Step 6 (implicit DG transport after Picard convergence) was missing.

### 12.2 Implementation

**Step 5 (explicit, inside Picard loop):** Mass + relaxation only (no transport):
```
(1/dt + 1/tau_M)(M^k, Z) = (1/tau_M)(chi(theta) H^k, Z) + (1/dt)(M^{n-1}, Z)
```

**Step 6 (implicit, after Picard converges):** Full DG transport:
```
(1/dt)(M^n, Z) - B_h^m(U^{n-1}, Z, M^n) = (1/dt)(M_picard^k, Z)
```

### 12.3 Files Modified

| File | Change |
|------|--------|
| `magnetization/magnetization.h` | New vectors (transport_solution, transport_rhs, old_ghosted), AssemblyMode enum |
| `magnetization/magnetization_setup.cc` | Allocate new vectors |
| `magnetization/magnetization_assemble.cc` | Split into Step5_Explicit and Step6_Implicit modes |
| `magnetization/magnetization.cc` | New `solve_step6_transport()` method |
| `drivers/decoupled_driver.cc` | Call Step 6 after Picard loop |
| `utilities/parameters.h` | Note on Step 5/6 strategy |

### 12.4 Result

Nonuniform Rosensweig with Step 5/6 blows up at t=2.184 (was t=2.290 without Step 6).
Step 6 did not cure the instability — the blow-up is a Poisson-Magnetization feedback
issue, not a transport stability issue.

---

## 13. Sparsity Analysis & Cuthill-McKee Renumbering (Session 16, March 5, 2026)

### 13.1 Overview

Ported sparsity analysis infrastructure from Semi_Coupled solver. Enables:
- Bandwidth measurement for all system matrices
- SVG + gnuplot sparsity pattern visualization (small matrices only)
- Per-row NNZ distribution CSV
- Cuthill-McKee DoF renumbering to reduce bandwidth

### 13.2 Implementation

**New file:** `utilities/sparsity_export.h` — self-contained header for sparsity export:
- `analyze_sparsity()` — compute bandwidth, NNZ stats, density
- `export_sparsity_pattern()` — write SVG, gnuplot, bandwidth CSV
- `write_sparsity_summary()` — combined summary CSV for all matrices

**CLI flags:**
```
--dump-sparsity         Export sparsity patterns after step 1
--renumber-dofs         Apply Cuthill-McKee DoF renumbering (reduces bandwidth)
--no-renumber-dofs      Disable DoF renumbering (default)
```

**CM applied to CG systems only:**
- Poisson (phi): CG Q1 — large benefit
- Cahn-Hilliard (theta, psi): CG Q2 — moderate benefit
- Navier-Stokes (ux, uy): CG Q2 — slight increase (expected)
- Magnetization: DG Q1 — **skipped** (no benefit for DG)
- Pressure: DG Q1 — **skipped**

### 13.3 Parallel Fix (Non-Contiguous Maps)

The CH coupled matrix has non-contiguous row ownership (theta block + psi block with a gap).
deal.II's `local_range()` and iterators assume contiguous ownership, causing `std::length_error`
crashes on np>1.

**Fix:** Replaced all deal.II Trilinos iterator usage with Epetra's native API:
- `ExtractMyRowView(local_row, ...)` for row data access
- `RowMap().GID(local_i)` for global row indices
- `ColMap().GID(local_col)` for global column indices

### 13.4 Bandwidth Reduction Results (np=1, r=3)

| Matrix | Without CM | With CM | Reduction |
|--------|-----------|---------|-----------|
| Poisson (15617x15617) | 2,727 | 618 | **-77.3%** |
| CH (31234x31234) | 18,344 | 16,235 | **-11.5%** |
| Magnetization (15360x15360) | 2,395 | 2,395 | 0% (DG, skipped) |
| NS (42754x42754) | 33,063 | 35,460 | +7.2% |

### 13.5 Files Modified

| File | Change |
|------|--------|
| `utilities/sparsity_export.h` | **NEW** — sparsity analysis + export (Epetra native API) |
| `utilities/parameters.h` | `renumber_dofs`, `dump_sparsity` flags |
| `utilities/parameters.cc` | CLI parsing + help text |
| `poisson/poisson_setup.cc` | CM after distribute_dofs() |
| `poisson/poisson.h` | `get_system_matrix()` getter |
| `cahn_hilliard/cahn_hilliard_setup.cc` | CM for theta + psi |
| `cahn_hilliard/cahn_hilliard.h` | `get_system_matrix()` getter |
| `navier_stokes/navier_stokes_setup.cc` | CM for ux + uy |
| `drivers/decoupled_driver.cc` | Sparsity dump after step 1 + banner |

---

## 14. Nonuniform Rosensweig Blow-Up Analysis (March 5, 2026)

### 14.1 Summary of All Runs

8 distinct runs analyzed. ALL blow up in the range t=2.18 to t=2.68.

| Run | dt | Blow-up time | Mag Strategy |
|-----|-----|-------------|-------------|
| Step6 (newest) | 2e-4 | t=2.184 | Step5+6 |
| 030226_230101 | 2e-4 | t=2.286 | Step5 |
| nonuniform.log | 2e-4 | t=2.290 | Step5 |
| 030326_050402 | **1e-4** | t=2.362 | unknown |
| 022826_222751 | 2e-4 | t=2.674 | Step5 (old) |
| 030226_050447 | 2e-4 | t=2.678 | Step5 (old) |

### 14.2 Blow-Up Cascade (Identical in ALL Runs)

1. **H spikes first**: |H| jumps from ~240 to ~30,000 in a single step
2. **M follows**: Picard coupling amplifies (|M| from 144 to infinity in 2 steps)
3. **U blows up**: Kelvin force mu_0*(M.grad)H drives velocity to ~10^5
4. **theta breaks**: Convection U.grad(theta) destroys phase field bounds
5. **Full NaN**: 1-3 steps after theta breaks

### 14.3 Key Observations

- **NOT AMR-related**: All runs use uniform refinement (no AMR)
- **NOT CFL-limited**: Halving dt to 1e-4 only delays by ~0.07s (t=2.362 vs t=2.290)
- **NOT Step 5/6 related**: Step 6 made it slightly worse (t=2.184 vs t=2.290)
- **Root cause**: Poisson-Magnetization feedback instability at interface peaks
  - chi(theta) coefficient creates sharp gradients in phi at interface
  - H concentrates at spike tips, Picard under-relaxation (omega=0.35) insufficient
  - Feedback gain > 1 at critical interface curvature → divergence

---

## 15. Zhang Algorithm 3.1: NS Pressure-Correction Projection (Sessions 17-18, March 6, 2026)

### 15.1 Motivation

The nonuniform Rosensweig blow-up (Section 14) was traced to the Picard iteration loop
creating a Poisson-Magnetization feedback instability. Zhang's Algorithm 3.1 eliminates
Picard iteration entirely: a single forward pass (CH → Mag Step 5 → Poisson → Mag Step 6 → NS)
is unconditionally energy-stable.

The NS solver was the main bottleneck: the monolithic saddle-point system (Q2/DG-P1)
required a direct solver or block-Schur preconditioner. Zhang's scheme replaces this
with a **pressure-correction projection method** using 3 separate scalar CG+AMG solves.

### 15.2 Stage 1: Remove Picard Iteration

- Removed the Picard iteration loop from `decoupled_driver.cc`
- Single forward pass: CH → Mag Step 5 → Poisson → Mag Step 6 → NS
- Picard CLI flags (`--picard-*`) deprecated with runtime warnings

### 15.3 Stage 2: NS Pressure-Correction Projection Method

**Architecture change:** Replaced monolithic NS saddle-point system with 3 separate matrices:
- `ux_matrix_` (CG Q2): velocity-x predictor
- `uy_matrix_` (CG Q2): velocity-y predictor
- `p_matrix_` (CG Q1): pressure Poisson (Laplacian)

**Pressure FE space changed:** DG-P1 → CG-Q1 (required for Laplacian in pressure Poisson step)

**Projection method steps (Zhang Alg 3.1, Steps 2-4):**
1. **Velocity predictor** (Step 2): Solve separate ux, uy systems with viscous + Kelvin + capillary + gravity terms, using old pressure gradient on RHS
2. **Pressure Poisson** (Step 3): Solve −Δ(δp) = −(1/dt)∇·ū for pressure correction
3. **Velocity correction** (Step 4): Solve M·δu = dt·∇(δp) via consistent mass CG, then u = ū + δu

**Key implementation detail:** Velocity correction uses **consistent mass CG solve** (not lumped mass).
Lumped mass M_L(i) = O(h²) causes O(dt/h) boundary layer error in H1 norm.
Consistent mass CG solve preserves O(h²) accuracy.

**Symmetric gradient helpers:** `compute_T_test_ux/uy` functions compute the test function
contributions to the viscous bilinear form (ν/4)(T(U), T(V)) for the component-split assembly.

### 15.4 Files Modified

| File | Change |
|------|--------|
| `navier_stokes/navier_stokes.h` | 3 separate matrices (ux, uy, p), vel_mass_matrix_, projection method API |
| `navier_stokes/navier_stokes_setup.cc` | 3 sparsity patterns, CG-Q1 pressure, CM renumbering on all 3 |
| `navier_stokes/navier_stokes_assemble.cc` | Component-split viscous assembly, pressure Poisson, velocity correction |
| `navier_stokes/navier_stokes_solve.cc` | 3 separate CG+AMG solves (solve_velocity, solve_pressure) |
| `navier_stokes/navier_stokes.cc` | Updated constructor and facade methods |
| `navier_stokes/navier_stokes_output.cc` | Updated for CG pressure output |
| `navier_stokes/navier_stokes_main.cc` | dt ∝ h² scaling for MMS (splitting error) |
| `navier_stokes/tests/navier_stokes_mms_test.cc` | Added projection steps, default phases B,D only |
| `drivers/decoupled_driver.cc` | Removed Picard loop, added projection method calls |
| `mms_tests/coupled_system_mms.h` | Updated expected rates to 2.0 (projection method limit) |
| `mms_tests/coupled_system_mms_test.cc` | Added projection steps + dt ∝ h² scaling |
| `utilities/parameters.h` | Removed block-Schur config, updated NS solver params |
| `utilities/parameters.cc` | Deprecated Picard CLI flags |

### 15.5 MMS Convergence Results

**Standalone NS MMS** (refs 2-5, dt ∝ h²):
All rates exactly **2.00** for ux_L2, ux_H1, uy_L2, uy_H1, p_L2, div_L2. **[PASS]**

**Full coupled 4-system MMS** (refs 1-3, dt ∝ h²):

| Variable | Rate (ref1→2) | Rate (ref2→3) | Status |
|----------|:--:|:--:|:--:|
| θ_L2     | 1.70 | 1.93 | PASS |
| θ_H1     | 1.91 | 1.97 | PASS |
| ux_L2    | 1.97 | 2.00 | PASS |
| p_L2     | 2.04 | 2.02 | PASS |
| φ_L2     | 3.00 | 2.99 | PASS |
| M_L2     | 2.00 | 2.00 | PASS |

**Note:** Projection method has O(dt) splitting error. With dt ∝ h² scaling, all rates
are capped at 2.0. Higher-order variables (θ, φ) show rates approaching 2.0 from below
as the temporal splitting error dominates at finer meshes.

### 15.6 Rosensweig Validation

**Uniform** (r=4, 800+ steps): Completely stable. θ∈[-1.01, 1.01], |U| settling to ~0.03.

**Nonuniform** (r=3, dt=2e-4, 42 dipoles): Running. At step ~5000 (t≈1.0), CFL=0.003,
all fields bounded. Previously blew up at t≈2.18 with Picard iteration. Test in progress.

---

*Generated: February 2025*
*Updated: March 6, 2026 (Sessions 17-18: Picard removal, NS projection method, coupled MMS PASS)*
*Total source code: ~11,500 lines across 4 subsystems + shared libraries*
