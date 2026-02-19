# Decoupled Ferrofluid Solver -- Development Summary

## Reference
Nochetto, Salgado & Tomas, *CMAME* **309** (2016) 497-531, Eq. 42a-42f
Zhang, He & Yang, *CMAME* **371** (2020) -- Landau-Lifshitz extension

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
(grad(phi), grad(X)) = (h_a - M, grad(X))
```
- Neumann BC, DoF-0 pinned to zero
- LHS assembled ONCE (constant coefficient Laplacian)
- RHS reassembled each Picard iteration

### Eq. 42c / 56-57 -- Magnetization (DG transport + relaxation)
```
(1/dt + 1/tau_M)(M^k, Z) - B_h^m(U^{n-1}, Z, M^k)
    = (1/tau_M)(chi(theta^{n-1}) H^k, Z) + (1/dt)(M^{n-1}, Z)
```
- DG-Q1 with upwind flux (skew-symmetric transport)
- Mx, My solved sequentially sharing one matrix
- Optional beta-term: beta * M x (M x H) (Landau-Lifshitz damping)

### Eq. 42a-b -- Cahn-Hilliard (Phase field + chemical potential)
```
(1/dt)(theta^n, chi) + (U^{n-1} . grad(theta^n), chi) + gamma(grad(psi), grad(chi)) = (1/dt)(theta^{n-1}, chi)
lambda*epsilon(grad(theta^n), grad(xi)) + (lambda/epsilon)(f(theta), xi) - (psi^n, xi) = 0
```
- Coupled theta-psi saddle-point system
- Double-well potential: F(theta) = (1/4)(theta^2 - 1)^2

### Eq. 42e-f -- Navier-Stokes (Velocity + pressure with Kelvin force)
```
(1/dt)(U^n, V) + (nu(theta) T(U^n), T(V))/2 + B_h(U^{n-1}; U^n, V)
    - (P^n, div(V)) = (1/dt)(U^{n-1}, V) + mu_0 * B_h^m(V, H^k, M^k) + (f, V)
(div(U^n), Q) = 0
```
- Component-split: separate DoFHandler for ux, uy
- Kelvin force: mu_0 * B_h^m(V, H, M) = mu_0 * [(M.grad)H.V + 0.5*div(M)(H.V)]
- Variable viscosity: nu(theta) = nu_w + (nu_f - nu_w) * H(theta/epsilon)

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
  - `applied_field.h` -- 2D line dipole computation with ramp

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

### 4.5 Coupled MMS Test (Poisson + Magnetization)

```bash
mpirun -np 4 ./test_poisson_mag_mms
```
Tests the Picard iteration loop with under-relaxation, verifying coupled convergence.

---

## 5. Test Results

### 5.1 MMS Spatial Convergence (all PASS)

Default refinements: {2, 3, 4, 5, 6}

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

physics/
  material_properties.h      -- Heaviside, chi(theta), nu(theta), F(theta), f(theta)
  kelvin_force.h             -- Kelvin force cell + face assembly kernels
  skew_forms.h               -- DG skew-symmetric transport forms
  applied_field.h            -- 2D line dipole field computation

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
  CMakeLists.txt
```

### Build Commands

```bash
# Navier-Stokes (ninja)
cd navier_stokes/cmake-build-debug && cmake .. && ninja

# Cahn-Hilliard (make)
cd cahn_hilliard/build && cmake .. && make -j$(nproc)

# Poisson (make)
cd poisson/build && cmake .. && make -j$(nproc)

# Magnetization (ninja)
cd magnetization/cmake-build-debug && cmake .. && ninja

# Coupled MMS test (ninja)
cd mms_tests/cmake-build-debug && cmake .. && ninja
```

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

### 7.4 Production Coupled Driver
- No unified top-level driver orchestrating all 4 subsystems together
- The Picard coupling pattern exists in `mms_tests/poisson_mag_mms_test.cc`
- Production driver needs: time loop, all 4 subsystems, VTK output, diagnostics
- **Three coupling strategies planned:** fully decoupled, semi-coupled, fully coupled

---

## 8. Rosensweig Instability Parameters (Section 6.2)

```
Domain:        [0, 1] x [0, 0.6]
Mesh:          10x6 initial cells, refinement level 5
Interface:     epsilon = 0.01
Susceptibility: chi_0 = 0.5
Relaxation:    tau_M = 1e-6
Viscosity:     nu_water = 1.0, nu_ferro = 2.0
Density ratio: r = 0.1
Gravity:       |g| = 30000, direction = (0, -1)
CH mobility:   gamma = 0.0002
Surface tension: lambda = 0.05
Time step:     dt = 5e-4, t_final = 2.0
Dipoles:       5 line dipoles at y = -15, x in {-0.5, 0, 0.5, 1, 1.5}
               intensity = 6000, ramp time = 1.6s
Picard:        max 7 iterations, tolerance 0.01
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

*Generated: February 2025*
*Total source code: ~10,000 lines across 4 subsystems + shared libraries*
