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
| Navier-Stokes | Cahn-Hilliard       | theta for nu(theta), psi for capillary: theta*grad(psi) |
| Navier-Stokes | Poisson             | H = grad(phi) + h_a, Hess(phi) + grad(h_a) for Kelvin |
| Navier-Stokes | Magnetization       | M for Kelvin force (algebraic: M = chi*H, no PDE) |

The three strategies below differ in HOW TIGHTLY these dependencies are resolved
within each time step.

---

## Strategy A: Fully Decoupled (Gauss-Seidel Splitting) -- IMPLEMENTED

**Status: Implemented in `drivers/decoupled_driver.cc`**

### Algorithm (with SAV energy-stable scheme, Zhang Eq 3.5-3.11)
```
FOR each time step n (t_{n-1} -> t_n):

  1. Cahn-Hilliard (SAV):
     Solve theta^n, psi^n using U^{n-1}
     SAV variable: r^n = sqrt(E1(theta^{n-1}) + C0)
     Stabilization: S1 = lambda/(4*epsilon)
     psi = lambda*(-epsilon*Laplacian(theta) + (1/epsilon)*f(theta))

  2. Poisson (with algebraic M = chi*H):
     Solve ((1+chi)*grad(phi), grad(X)) = ((1-chi)*h_a, grad(X))
     H_total = grad(phi) + h_a

  3. Magnetization: SKIPPED (algebraic M = chi(theta) * H_total)
     No PDE solve needed -- M computed inline in NS assembly

  4. Navier-Stokes:
     Solve U^n, p^n using H^n, theta^{n-1}, psi^n
     Capillary force: theta^{n-1} * grad(psi^n)     [Phi * grad(W)]
     Kelvin force:    mu_0 * (M . grad)H              [grad(H) = Hess(phi) + grad(h_a)]
     Gravity:         rho(theta) * g
     Stabilization:   S2 = 1.5*mu0^2*(chi0*|H_max|)^2/(4*nu_min)

  Output VTK if needed
  Advance time
```

### Characteristics
- **Solves per step:** 3 (CH, Poisson, NS) with algebraic M, NO iteration
- **Cost per step:** Cheapest
- **Stability:** Energy-stable via SAV with S1, S2 stabilizers (Zhang Theorem 3.1)
- **Accuracy:** O(dt) splitting error on top of O(dt) backward Euler
- **When to use:** Validation tests, production runs with small dt

### Implementation
- File: `drivers/decoupled_driver.cc` (~1500 lines)
- Build: `cd drivers/build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j8`
- Run: `mpirun -np 2 ./ferrofluid_decoupled --validation droplet -r 7 --vtk_interval 100`
- Validation presets: `square`, `droplet`, `droplet_nofield`, `rosensweig`

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

### SAV Energy (Zhang Theorem 3.1, Eq 3.38)

With SAV stabilization, the modified energy:
```
E_SAV^n = (1/2)||U^n||^2 + (r^n)^2 + (S1 - lambda*L/2)||theta^n||^2
```
where r^n = sqrt(E1(theta^n) + C0) is the SAV variable.

The discrete energy inequality (Eq 3.41) guarantees:
```
E_SAV^n + dissipation_terms <= E_SAV^{n-1}
```

### Key stabilization constants
- S1 = lambda/(4*epsilon) -- CH stabilization (depends only on CH parameters)
- S2 = mu0^2 * (chi0 * |H_max|)^2 / (4 * nu_min) -- NS stabilization
- S1 does NOT depend on dropped terms (beta, spin vorticity, Maxwell stress)
- All dropped terms are dissipative (positive norms on LHS of energy inequality)
- With algebraic M = chi*H, all 3 dropped terms are exactly zero

### Note on algebraic magnetization
When using algebraic M = chi(theta)*H (tau_M -> 0 limit), the Poisson-Magnetization
Picard iteration is unnecessary. M is computed inline from the current H field.
This eliminates the magnetization PDE solve entirely.

---

## Code-vs-Zhang Deviation Audit (Session 4)

**Goal: Match Zhang, He & Yang (SIAM J. Sci. Comput. 43(1), 2021) EXACTLY.**

All subsystem PDEs originate from Nochetto et al. (CMAME 2016), but the coupling
strategy, time-stepping scheme, stabilization, and validation parameters all come
from Zhang. The following deviations were found by equation-by-equation comparison.

### Deviation 1: VISCOUS TERM — FACTOR OF 2 ERROR (CRITICAL)

**Zhang Eq 2.6:** `-div(ν(Φ) D(u))` where `D(u) = ½(∇u + ∇u^T)`
Weak form: `(ν D(u), D(v))`

**Nochetto Eq 14e/42e:** `(ν_Θ T(U), T(V))` where `T(u) = ½(∇u + ∇u^T)` (defined below Eq 14f)
So Nochetto's T = D = ½(∇u + ∇u^T), and `(ν T, T) = (ν D, D)`. Same as Zhang.

**Code (`navier_stokes_assemble.cc` line 29):**
Comment says `T(U) = ∇U + (∇U)^T` (WITHOUT ½).
`compute_T_test_ux()` builds T = ∇U + ∇U^T = 2D.
Line 214: `(nu/2.0) * (T_U * T_V)` = `(ν/2)(2D:2D)` = `2ν(D:D)`.

**Result: Effective viscosity is 2× the paper's value.**

**Fix:** Change `(nu / 2.0) * (T_U * T_V)` to `(nu / 4.0) * (T_U * T_V)`.
This gives `(ν/4)(2D:2D) = ν(D:D)` — matching both Zhang and Nochetto.
Applied in 3 functions: `assemble_stokes()`, `assemble_coupled()`, `assemble_coupled_algebraic_M()`.

**Files:** `navier_stokes/navier_stokes_assemble.cc` (lines 214-217, ~540-543, ~1087-1090)

---

### Deviation 2: S1 STABILIZATION CONSTANT (HIGH)

**Zhang p.B182:** "In all numerical tests, we choose L = 1/(2ε),
thus the stabilizing constant is S = λ/(4ε)."

For Rosensweig (λ=1, ε=5e-3): S1 = 1/(4×0.005) = **50**

**Code (`decoupled_driver.cc` line 1295):** `sav_S1 = 1.0 / epsilon` = **200**

4× over-stabilization. The S1(θ^{n+1} − θ^n) term adds artificial diffusion
to the phase field, damping interface dynamics.

**Fix:** Change auto-computation to `sav_S1 = lambda / (4.0 * epsilon)`.

**Files:** `drivers/decoupled_driver.cc` (line ~1295)

---

### Deviation 3: NS SOLVER — DIRECT SADDLE-POINT vs PRESSURE PROJECTION (MEDIUM)

**Zhang Steps 2-4 (Eq 3.11-3.13):**
- Step 2: Velocity predictor (NO pressure in LHS — old p^n on RHS only)
- Step 3: Pressure Poisson: `(∇p^{n+1}, ∇q) = (∇p^n, ∇q) − (1/dt)(div ũ, q)`
- Step 4: Velocity correction: `u^{n+1} = ũ − dt·∇(p^{n+1} − p^n)`

This is a Chorin-type pressure-correction scheme — velocity and pressure are
DECOUPLED into separate scalar/vector solves.

**Code:** Solves monolithic saddle-point `[A B^T; B 0][u; p] = [f; 0]` directly.

**Impact:** Direct solve has NO velocity-pressure splitting error (arguably better).
But it's structurally different from Zhang, meaning our solver is solving a slightly
different linear system each step. For exact replication, implement projection.

**Fix:** Implement Zhang's 3-step projection scheme:
1. Solve velocity-only system for ũ (use old pressure on RHS)
2. Solve pressure Poisson for p^{n+1}
3. Correct velocity: u^{n+1} = ũ − dt·∇(p^{n+1} − p^n)

**Files:** New projection solver needed in `navier_stokes/`, driver changes in
`drivers/decoupled_driver.cc`

---

### Deviation 4: PHASE FIELD CONVENTION — {-1,+1} vs {0,1} (HIGH)

**Zhang:** Φ ∈ {0, 1}. Ferrofluid = 1, non-magnetic = 0.
- Double-well: F(Φ) = Φ²(1−Φ)²/4, f(Φ) = Φ(1−Φ)(1−2Φ) (modified)
- Susceptibility: χ(Φ) = χ₀ · Φ (LINEAR interpolation)
- Viscosity: ν(Φ) = ν_f·Φ + ν_w·(1−Φ) (LINEAR interpolation)
- Density: ρ(Φ) = 1 + r/(1+exp((1−2Φ)/ε)) (Eq 4.2)
- IC: Φ = 1 if ferrofluid region, Φ = 0 otherwise (sharp step)

**Code:** θ ∈ {−1, +1}. Ferrofluid = +1, non-magnetic = −1.
- Double-well: F(θ) = (θ²−1)²/4, f(θ) = θ³−θ (standard Ginzburg-Landau)
- Susceptibility: χ(θ) = χ₀ · H(θ/ε) (SIGMOID interpolation)
- Viscosity: ν(θ) = ν_w + (ν_f−ν_w)·H(θ/ε) (SIGMOID interpolation)
- Density: ρ(θ) = 1 + r·H(θ/ε) (SIGMOID interpolation)

The transformation Φ = (θ+1)/2 maps the equilibria correctly, but:
1. Zhang's χ(Φ) = χ₀·Φ is LINEAR. Code's χ(θ) = χ₀·H(θ/ε) is a SIGMOID.
   In the diffuse interface region, these give DIFFERENT susceptibility profiles.
2. Zhang's density formula (Eq 4.2) is NOT a simple sigmoid — it uses a specific
   logistic form 1/(1+exp((1−2Φ)/ε)). This doesn't exactly match H(θ/ε).
3. The double-well polynomials differ but are equivalent under the mapping.

**Fix:** Either:
- (a) Switch code to {0,1} convention (most invasive but cleanest), OR
- (b) Keep {−1,+1} but use equivalent formulas:
  - χ(θ) = χ₀·(θ+1)/2  (linear, not sigmoid)
  - ν(θ) = ν_w·(1−θ)/2 + ν_f·(θ+1)/2  (linear, not sigmoid)
  - ρ(θ) = 1 + r/(1+exp(−θ/ε))  (matches Zhang's Eq 4.2 under Φ=(θ+1)/2)

Option (b) preserves all existing infrastructure while matching Zhang's physics.

**Files:** `physics/material_properties.h` (chi, nu, rho functions)

---

### Deviation 5: MAGNETIZATION — ALGEBRAIC vs FULL PDE (HIGH)

**Zhang Eq 4.4 (Rosensweig):** β = 1, τ = 1e-4. Solves the FULL magnetization PDE:
```
(1/τ)(∂m/∂t) + (u·∇)m = (1/τ)(χ(Φ)h) − β·m×(m×h)
```
This is Zhang's Step 5 (Eq 3.15-3.16), solved as DG transport + relaxation.

**Code:** `use_algebraic_magnetization = true`, `beta = 0`.
M = χ(θ)·H computed algebraically. No magnetization PDE at all.

**Impact:** With τ=1e-4 and dt=1e-3, τ/dt = 0.1 — small but not negligible.
The PDE gives magnetization that LAGS behind equilibrium during rapid field
changes (the ramp). Also β=1 Landau-Lifshitz damping rotates m toward h.

**Fix:** Enable magnetization PDE:
- Set `use_algebraic_magnetization = false`
- Set `beta = 1.0, tau_M = 1e-4`
- The magnetization subsystem already exists and is MMS-verified
- Need to wire it into the driver's Gauss-Seidel loop properly

**Files:** `utilities/parameters.cc` (Rosensweig preset), `drivers/decoupled_driver.cc`

---

### Deviation 6: DIPOLE FIELD — REGULARIZATION PARAMETER δ (LOW-MEDIUM)

**Zhang's dipole formula:** `φ_s(x) = d·(x_s − x) / |x_s − x|²` (NO regularization)

**Code (`applied_field.h` line 114):**
```cpp
const double delta = 0.01 * min_dipole_dist;  // δ = 0.01 * |y_dipole|
```
Uses regularized `R² = |r|² + δ²` instead of `|r|²`.

With dipoles at y = −15: δ = 0.01 × 15 = 0.15.
This smooths the field slightly near (but never at) the dipoles.

**Impact:** At the domain (distance ~15 from dipoles), δ²/r² ≈ 0.15²/15² ≈ 1e-4.
Negligible effect on h_a values in the domain. But for exact replication, remove δ.

**Fix:** Set `delta = 0.0` or remove regularization entirely.
The dipoles are far enough from the domain that no singularity protection is needed.

**Files:** `physics/applied_field.h` (lines 114-115, and corresponding lines in gradient)

---

### Deviation 7: ROSENSWEIG IC — tanh vs SHARP STEP (LOW)

**Zhang Eq 4.3:** `Φ₀ = 1 if y ≤ 0.2, Φ₀ = 0 otherwise` (sharp discontinuity)

**Code (`decoupled_driver.cc`):** Uses `θ = tanh((0.2 − y)/(√2 ε))` (smooth tanh profile)

**Impact:** The tanh profile approximates the sharp interface and converges to it
as ε → 0. For ε = 5e-3, the interface thickness is ~O(ε) = 0.005, so the profile
is very close to a step. Minor effect on early dynamics.

**Fix (optional):** Use sharp step: θ = +1 if y ≤ 0.2, θ = −1 otherwise.
Or keep tanh (physically more meaningful as initial diffuse profile).

---

### Deviation 8: S2 ADAPTIVE FORMULA (LOW-MEDIUM)

**Zhang p.B182 (below Eq 3.38):** The NS stabilization bound is part of the energy
estimate. The exact formula for the required S2 involves operator norms.

**Code:** `S2 = 1.5 * μ₀² * (χ₀ * |H_max|)² / (4 * ν_min)` with safety factor 1.5.

**Impact:** This is an heuristic bound. May be too large (over-stabilizing velocity)
or too small (missing stability). Zhang's paper doesn't give an explicit S2 formula
for the fully decoupled case — it's embedded in the energy estimate.

**Fix:** Keep for now, tune if needed after other fixes. The 1.5× safety factor
provides margin.

---

### Priority Order for Fixes

| Priority | Deviation | Effort | Impact |
|----------|-----------|--------|--------|
| **1** | #1 Viscous factor of 2 | 10 min | CRITICAL — all flow dynamics wrong |
| **2** | #2 S1 = 200 → 50 | 5 min | HIGH — interface over-damped |
| **3** | #4 Phase field convention | 1 hr | HIGH — all material properties differ |
| **4** | #5 Algebraic M → PDE M | 30 min | HIGH — magnetization dynamics differ |
| **5** | #3 Saddle-point → projection | 4-6 hr | MEDIUM — different pressure algorithm |
| **6** | #6 Dipole regularization δ | 5 min | LOW — negligible at domain distance |
| **7** | #7 IC tanh vs step | 5 min | LOW — converges to same profile |
| **8** | #8 S2 formula | defer | LOW — tune empirically |

**Recommended approach:**
1. Fix deviations #1, #2, #6, #7 (30 min) — quick wins
2. Fix deviation #4 (1 hr) — material properties
3. Fix deviation #5 (30 min) — enable magnetization PDE
4. Rerun droplet + Rosensweig tests
5. If results still differ from Zhang, implement deviation #3 (projection scheme)

---

---

## Critical Fix: DG Face Assembly (Session 8, February 27, 2026)

**Deviation #5 (Magnetization PDE)** is now fully functional and MMS-verified.
A critical bug in `magnetization_assemble.cc` caused zero cross-cell coupling
in the DG face flux. See `SESSION_SUMMARY.md` Section 6b for details.

With the fix + upwind penalty, the magnetization transport achieves O(h²)
convergence in the full Poisson-Mag-NS coupled test (μ₀=0.1, real NS velocity).

This means **all validation tests (droplet, rosensweig) now have correct DG
magnetization transport** and should be rerun.

---

*Plan created: February 2025*
*Updated: February 27, 2026 (Session 8 — DG face fix, Poisson-Mag-NS MMS verified)*
