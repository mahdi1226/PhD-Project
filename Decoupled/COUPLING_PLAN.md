# Ferrofluid Coupled Solver -- Implementation Plan

## Three Coupling Strategies for the Full 4-Subsystem Problem

Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) Eq. 42a-42f

---

## 0. The Problem

We have 4 verified subsystems that need to be orchestrated into a production
time-stepping loop:

```
Poisson        (phi)          -- Eq. 42d, magnetostatic potential
Magnetization  (Mx, My)       -- Eq. 42c, DG transport + relaxation
Cahn-Hilliard  (theta, psi)   -- Eq. 42a-b, phase field + chemical potential
Navier-Stokes  (ux, uy, p)    -- Eq. 42e-f, momentum + continuity + Kelvin force
```

**Data dependencies between subsystems:**

| Subsystem     | Reads from          | What it uses                              |
|---------------|---------------------|-------------------------------------------|
| Poisson       | Magnetization       | M in RHS: (h_a - M, grad(X))             |
| Magnetization | Poisson             | H = grad(phi) in relaxation term          |
| Magnetization | Cahn-Hilliard       | chi(theta) susceptibility                 |
| Magnetization | Navier-Stokes       | U in transport: (U . grad)M              |
| Cahn-Hilliard | Navier-Stokes       | U in convection: (U . grad)theta          |
| Navier-Stokes | Cahn-Hilliard       | theta for nu(theta) viscosity             |
| Navier-Stokes | Poisson             | H = grad(phi) for Kelvin force            |
| Navier-Stokes | Magnetization       | M for Kelvin force: mu_0 * B_h^m(V, H, M)|

The three strategies below differ in HOW TIGHTLY these dependencies are resolved
within each time step.

---

## Strategy A: Fully Decoupled (Gauss-Seidel Splitting)

### Algorithm
```
FOR each time step n (t_{n-1} -> t_n):

  1. Cahn-Hilliard:
     Solve theta^n, psi^n using U^{n-1}
     (Convection uses lagged velocity)

  2. Poisson:
     Solve phi^n using M^{n-1}
     (Demagnetizing field uses lagged magnetization)
     Compute H^n = grad(phi^n)

  3. Magnetization:
     Solve M^n using H^n, U^{n-1}, theta^{n-1}
     (Transport uses lagged velocity, susceptibility uses lagged theta)

  4. Navier-Stokes:
     Solve U^n, p^n using H^n, M^n, theta^{n-1}
     (Kelvin force uses current H and M, viscosity uses lagged theta)

  Output VTK if needed
  Advance time
```

### Characteristics
- **Solves per step:** 4 (one per subsystem), NO iteration
- **Cost per step:** Cheapest
- **Stability:** First-order in time. Energy-stable IF lagged coefficients used
  (Theorem 4.1 in paper). May require smaller dt for strong coupling (large chi_0).
- **Accuracy:** O(dt) splitting error on top of O(dt) backward Euler.
  For small dt this is acceptable; for large dt the splitting error dominates.
- **When to use:** Production runs where dt is already small for accuracy reasons,
  or when chi_0 is small (weak magnetic coupling).

### Pros
- Simplest to implement and debug
- Each subsystem solved exactly once -- predictable runtime
- No convergence issues (no iteration to fail)
- Easily parallelizable (subsystems are independent linear solves)

### Cons
- Splitting error can be O(dt) -- not reduced by mesh refinement
- For strong Poisson-Magnetization coupling (large chi_0), may need very small dt
- No feedback within a time step -- lag can cause oscillations

### Files to Create
```
drivers/
  decoupled_driver.cc       -- Main time loop, Gauss-Seidel order
  CMakeLists.txt            -- Links all 4 subsystem libraries
```

### Implementation Complexity: LOW (~200-300 lines)

---

## Strategy B: Semi-Coupled (Operator Splitting + Picard Refinement)

### Algorithm
```
FOR each time step n (t_{n-1} -> t_n):

  1. Cahn-Hilliard:
     Solve theta^n, psi^n using U^{n-1}

  2. Picard iteration (k = 0, 1, ..., max_picard):

     a. Poisson:
        Solve phi^k using M_relaxed^{k-1}
        Compute H^k = grad(phi^k)

     b. Magnetization:
        Solve M_raw^k using H^k, U^{n-1}, theta^{n-1}

     c. Under-relax:
        M_relaxed^k = omega * M_raw^k + (1 - omega) * M_relaxed^{k-1}

     d. Convergence check:
        IF ||M_relaxed^k - M_relaxed^{k-1}|| / ||M_relaxed^k|| < tol: BREAK

  3. Navier-Stokes:
     Solve U^n, p^n using H^k_final, M^k_final, theta^{n-1}

  Output VTK if needed
  Advance time
```

### Characteristics
- **Solves per step:** 2 + 2*K + 1 where K = Picard iterations (typically 3-7)
- **Cost per step:** ~3-5x more than fully decoupled
- **Stability:** Tighter than decoupled. Under-relaxation (omega = 0.35) damps the
  M -> phi -> H -> M feedback spiral. Proven stable for chi_0 up to ~1.0.
- **Accuracy:** Splitting error between {Poisson, Mag} and {CH, NS} is O(dt).
  Within {Poisson, Mag}, the Picard iteration resolves coupling to tolerance.
- **When to use:** Production Rosensweig simulations. This is the paper's scheme.

### Pros
- Resolves the strongest coupling (Poisson <-> Magnetization) to convergence
- Under-relaxation handles large chi_0 robustly
- CH and NS still use lagged coupling (energy-stable, Theorem 4.1)
- Already validated: `mms_tests/poisson_mag_mms_test.cc` demonstrates this pattern

### Cons
- Picard iteration cost is variable (hard to predict runtime)
- NS still sees lagged theta -- splitting error remains O(dt) for CH-NS coupling
- Under-relaxation parameter omega needs tuning for different chi_0 values
- More complex than fully decoupled

### Picard Parameters (from existing code)
```
max_picard = 50        (safety limit)
picard_tol = 1e-10     (relative residual)
omega      = 0.35      (under-relaxation)
```
Production values (from Parameters):
```
picard_iterations = 7   (max)
picard_tolerance  = 0.01
```

### Files to Create
```
drivers/
  semi_coupled_driver.cc    -- Time loop with Picard inner loop
  CMakeLists.txt            -- Links all 4 subsystem libraries
```

### Implementation Complexity: MEDIUM (~400-500 lines)
Most of the Picard pattern can be lifted from `mms_tests/poisson_mag_mms_test.cc`.

---

## Strategy C: Fully Coupled (Global Picard / Block Iteration)

### Algorithm
```
FOR each time step n (t_{n-1} -> t_n):

  Outer Picard iteration (k = 0, 1, ..., max_outer):

    1. Cahn-Hilliard:
       Solve theta^k using U^{k-1}
       (Uses CURRENT velocity iterate, not lagged from previous step)

    2. Inner Picard iteration (j = 0, 1, ..., max_inner):

       a. Poisson:
          Solve phi^j using M_relaxed^{j-1}
          Compute H^j = grad(phi^j)

       b. Magnetization:
          Solve M_raw^j using H^j, U^{k-1}, theta^k
          (Uses CURRENT theta iterate)

       c. Under-relax M:
          M_relaxed^j = omega * M_raw^j + (1 - omega) * M_relaxed^{j-1}

       d. Inner convergence check on M

    3. Navier-Stokes:
       Solve U^k, p^k using H^j_final, M^j_final, theta^k
       (Uses CURRENT theta and magnetic fields)

    4. Outer convergence check:
       IF ||U^k - U^{k-1}|| + ||theta^k - theta^{k-1}|| < tol: BREAK

  Output VTK if needed
  Advance time
```

### Characteristics
- **Solves per step:** (2 + 2*J + 1) * K_outer where J = inner Picard, K_outer = outer
  Typically: (2 + 2*5 + 1) * 3 = 39 solves per step
- **Cost per step:** ~10-20x more than fully decoupled
- **Stability:** Most robust. All couplings resolved to tolerance within each step.
  No splitting error at all -- only time discretization error O(dt).
- **Accuracy:** Pure O(dt) backward Euler error. No splitting artifacts.
  Allows larger dt than decoupled strategies for the same accuracy.
- **When to use:** Benchmark computations, validation against literature,
  cases where splitting error is unacceptable, or very large dt.

### Pros
- No splitting error -- cleanest results
- Can use larger dt (fewer total steps) while maintaining accuracy
- All fields are self-consistent within each time step
- Best for convergence studies and paper-quality results

### Cons
- Most expensive per step by far
- Outer iteration may not converge for very strong coupling
- Complex implementation with nested iteration loops
- Diminishing returns if dt is already small

### Key Difference from Semi-Coupled
In semi-coupled, CH and NS use **lagged** fields (from previous time step).
In fully coupled, CH and NS participate in the **outer Picard loop**, using
current iterates. This eliminates the CH-NS splitting error.

### Convergence Parameters
```
Inner Picard (Poisson <-> Mag):
  max_inner  = 50
  inner_tol  = 1e-10
  omega      = 0.35

Outer Picard (CH <-> NS <-> Magnetic):
  max_outer  = 10
  outer_tol  = 1e-6
```

### Files to Create
```
drivers/
  fully_coupled_driver.cc   -- Nested Picard loops
  CMakeLists.txt            -- Links all 4 subsystem libraries
```

### Implementation Complexity: HIGH (~600-800 lines)
Nested iteration with multiple convergence checks. Need careful vector management
to avoid excessive memory allocation.

---

## Comparison Summary

| Aspect                 | Fully Decoupled    | Semi-Coupled       | Fully Coupled       |
|------------------------|--------------------|--------------------|---------------------|
| Solves per step        | 4                  | ~10-15             | ~30-40              |
| Cost per step          | 1x (baseline)      | 3-5x               | 10-20x              |
| Splitting error        | O(dt)              | O(dt) for CH-NS    | None                |
| Poisson-Mag coupling   | Lagged (O(dt))     | Resolved (Picard)  | Resolved (Picard)   |
| CH-NS coupling         | Lagged (O(dt))     | Lagged (O(dt))     | Resolved (outer)    |
| Energy stability       | Theorem 4.1        | Theorem 4.1        | Stronger            |
| Implementation         | ~250 lines         | ~450 lines         | ~700 lines          |
| Robustness (large chi) | May need small dt   | Robust (omega)     | Most robust          |
| Best for               | Fast production    | Paper's scheme     | Benchmarks           |
| Lines of code          | LOW                | MEDIUM             | HIGH                |

---

## Implementation Plan (Recommended Order)

### Phase 1: Shared Driver Infrastructure

Create `drivers/` directory with shared utilities:

```
drivers/
  driver_common.h           -- Shared time loop utilities:
                               - Diagnostic output (energy, mass, max|U|, max|M|)
                               - VTK output scheduler (every N steps)
                               - CSV logging (time, energy, Picard iters, solver stats)
                               - Initial condition setup (flat interface, random, custom)
                               - Command line: --strategy decoupled|semi|coupled
  CMakeLists.txt            -- Top-level build linking all 4 subsystem libs
```

**Estimated effort:** 1-2 hours

### Phase 2: Fully Decoupled Driver

Implement Strategy A first (simplest, proves the wiring works):
1. Create `drivers/decoupled_driver.cc`
2. Wire all 4 subsystems on shared triangulation
3. Implement Gauss-Seidel time loop
4. Add VTK output at intervals
5. Test with Rosensweig preset

**Estimated effort:** 2-3 hours

### Phase 3: Semi-Coupled Driver

Implement Strategy B (the paper's actual scheme):
1. Create `drivers/semi_coupled_driver.cc`
2. Copy Picard loop from `mms_tests/poisson_mag_mms_test.cc`
3. Add CH and NS around the Picard loop
4. Add Picard convergence diagnostics
5. Test with Rosensweig preset, compare to Strategy A

**Estimated effort:** 3-4 hours

### Phase 4: Fully Coupled Driver

Implement Strategy C:
1. Create `drivers/fully_coupled_driver.cc`
2. Add outer Picard loop around CH + inner Picard + NS
3. Add outer convergence monitoring
4. Test convergence with decreasing dt, compare to B

**Estimated effort:** 4-5 hours

### Phase 5: Comparison Study

Run all 3 strategies on the same problem and compare:
- Total wall time for same dt
- Solution differences between strategies
- Picard iteration counts
- Energy conservation
- Interface dynamics (VTK comparison)

**Estimated effort:** 2-3 hours

---

## Directory Structure (Final)

```
PhD-Project/Decoupled/
  utilities/                 -- Shared parameters, solver info, timestamps
  physics/                   -- Material properties, Kelvin force, applied field
  poisson/                   -- Poisson subsystem (lib + main + test)
  magnetization/             -- Magnetization subsystem (lib + main + test)
  cahn_hilliard/             -- Cahn-Hilliard subsystem (lib + main + test)
  navier_stokes/             -- Navier-Stokes subsystem (lib + main + test)
  mms_tests/                 -- Coupled MMS verification (Poisson+Mag)
  drivers/                   -- NEW: Production coupled drivers
    driver_common.h          --   Shared diagnostics, IC setup, output
    decoupled_driver.cc      --   Strategy A: Gauss-Seidel splitting
    semi_coupled_driver.cc   --   Strategy B: Picard for Poisson-Mag
    fully_coupled_driver.cc  --   Strategy C: Global Picard
    CMakeLists.txt           --   Build all 3 drivers
```

---

## CMakeLists.txt for drivers/

```cmake
cmake_minimum_required(VERSION 3.13)
find_package(deal.II 9.4 REQUIRED HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR})
deal_ii_initialize_cached_variables()

project(ferrofluid_drivers)

set(SRC_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
include_directories(${SRC_ROOT})

# Import all 4 subsystem libraries
add_subdirectory(${SRC_ROOT}/poisson       ${CMAKE_BINARY_DIR}/poisson)
add_subdirectory(${SRC_ROOT}/magnetization ${CMAKE_BINARY_DIR}/magnetization)
add_subdirectory(${SRC_ROOT}/cahn_hilliard ${CMAKE_BINARY_DIR}/cahn_hilliard)
add_subdirectory(${SRC_ROOT}/navier_stokes ${CMAKE_BINARY_DIR}/navier_stokes)

# Strategy A: Fully Decoupled
add_executable(ferrofluid_decoupled decoupled_driver.cc)
target_link_libraries(ferrofluid_decoupled
    poisson_lib magnetization_lib cahn_hilliard_lib navier_stokes_lib)
deal_ii_setup_target(ferrofluid_decoupled)

# Strategy B: Semi-Coupled (Paper's scheme)
add_executable(ferrofluid_semi_coupled semi_coupled_driver.cc)
target_link_libraries(ferrofluid_semi_coupled
    poisson_lib magnetization_lib cahn_hilliard_lib navier_stokes_lib)
deal_ii_setup_target(ferrofluid_semi_coupled)

# Strategy C: Fully Coupled
add_executable(ferrofluid_coupled fully_coupled_driver.cc)
target_link_libraries(ferrofluid_coupled
    poisson_lib magnetization_lib cahn_hilliard_lib navier_stokes_lib)
deal_ii_setup_target(ferrofluid_coupled)
```

---

## Usage (Target CLI)

```bash
# Build all drivers
cd drivers && mkdir build && cd build
cmake .. -DDEAL_II_DIR=/path/to/dealii && make -j$(nproc)

# Strategy A: Fully Decoupled
mpirun -np 4 ./ferrofluid_decoupled --rosensweig --refinement 5

# Strategy B: Semi-Coupled (paper's scheme)
mpirun -np 4 ./ferrofluid_semi_coupled --rosensweig --refinement 5

# Strategy C: Fully Coupled
mpirun -np 4 ./ferrofluid_coupled --rosensweig --refinement 5

# Comparison run (smaller problem for testing)
mpirun -np 4 ./ferrofluid_decoupled     --rosensweig --refinement 3 --t_final 0.1
mpirun -np 4 ./ferrofluid_semi_coupled   --rosensweig --refinement 3 --t_final 0.1
mpirun -np 4 ./ferrofluid_coupled        --rosensweig --refinement 3 --t_final 0.1
```

---

## Energy Law (for Validation)

The discrete energy (Theorem 4.1, Eq. 40):

```
E^n = (1/2)||U^n||^2 + (lambda*epsilon/2)||grad(theta^n)||^2
    + (lambda/epsilon)(F(theta^n), 1) + (mu_0/(2*tau_M))||M^n||^2
    - mu_0(M^n, H^n)
```

Should be non-increasing (or bounded) for all 3 strategies.
The fully coupled strategy should show the tightest energy decay.
All drivers should log this energy at each time step for comparison.

---

*Plan created: February 2025*
