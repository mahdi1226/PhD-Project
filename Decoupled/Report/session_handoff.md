# Decoupled Ferrofluid Solver -- Session Handoff

## Project Location
`/Users/mahdi/Projects/git/PhD-Project/Decoupled/`
Build: `cd /Users/mahdi/Projects/git/PhD-Project/Decoupled/drivers/build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j8`

## Reference Papers
- Zhang, He & Yang, SIAM J. Sci. Comput. 43(1), 2021 -- Parameters, validation, SAV scheme
  PDF: `/Users/mahdi/Desktop/droplet/GZhang_XMHe_XYang.pdf`
- Nochetto, Salgado & Tomas, CMAME 309, 2016 -- PDE formulation (simplified model)
  PDF: `/Users/mahdi/Desktop/RHNochetto_AJSalgado_ITomas.pdf`

---

## Current Status Summary

| Test | Status | Result | Directory |
|------|--------|--------|-----------|
| **Square** (CH-only) | Done | 5000 steps, theta in [-0.992, 1.01] | `Results/022126_051823_square_r6/` |
| **Droplet WITH field** (post-fix) | Done | 1500 steps, theta in [-1.00, 1.03] | `Results/022126_065225_droplet_wfield_r7/` |
| **Droplet WITHOUT field** | Done | 1500 steps, theta in [-1.00, 1.01], |U|=0 | `Results/022126_055622_droplet_wofield_r7/` |
| **Rosensweig** | Pending rerun | Previously unstable (theta overshoot to 1.63); bugs now fixed, needs rerun | -- |

---

## What Was Done Across All Sessions

### Session 1: SAV CH Assembly Fix
- **Removed Douglas-Dupont stabilization** from CH assembly (was over-stabilizing)
- **Added lambda factors** to diffuse-interface energy terms in SAV formulation
- File: `cahn_hilliard/cahn_hilliard_assemble.cc`

### Session 2: Rosensweig Fix + Validation Runs
1. **Fixed Rosensweig parameters** (10 of 14 were wrong) -- `utilities/parameters.cc`
2. **Fixed Rosensweig IC** -- flat at y=0.2, no perturbation -- `drivers/decoupled_driver.cc`
3. **Added field ramp cap** -- `physics/applied_field.h`: alpha(t) = min(slope*t, intensity_max)
4. **Ran square test** -> PASSED (5000 steps)
5. **Ran Rosensweig** -> UNSTABLE (theta overshoot to 1.63 at t~0.88), killed
6. **Ran droplet WITH field** (2 CPU cores) -> PASSED (1500 steps, theta in [-1.00, 1.03])

### Session 3: Equation Audit + Bug Fixes + droplet_nofield
User identified 6 potential bugs. Thorough audit against Zhang and Nochetto papers.

**Bug audit results:**

| # | Issue | Verdict |
|---|-------|---------|
| 1 | chi_0 = 0.5 for Rosensweig | Correct |
| 2 | mu_0 = 1.0 for Rosensweig | Correct |
| **3** | **Capillary force direction + extra lambda** | **BUG -- Fixed** |
| 4 | lambda in W equation | Correct |
| 5 | Curl terms (m x h) omitted | Correct (zero for algebraic M) |
| **6** | **grad(H) missing grad(h_a) in Kelvin force** | **BUG -- Fixed** |

**Bug #3 details (capillary force):**
- Code had `lambda * psi * grad_theta` (W*grad(Phi))
- Both Zhang Eq 2.6 and Nochetto Eq 42e/65d say it should be **theta * grad_psi** (Phi*grad(W))
- Also: SAV psi already contains lambda, so extra lambda multiplication gave lambda^2
- Fixed in both `assemble_coupled()` and `assemble_coupled_algebraic_M()`

**Bug #6 details (Kelvin force grad(h_a)):**
- Code's Poisson gives phi as demagnetizing potential, so H = grad(phi) + h_a
- grad(H) = Hess(phi) + grad(h_a), but code only used Hess(phi)
- For dipole fields (spatially varying), grad(h_a) != 0
- Added `compute_applied_field_gradient<dim>()` to `applied_field.h`
- Fixed in both assembly functions

**Additional Session 3 work:**
- Added `droplet_nofield` validation preset (droplet w/o magnetic field)
- Fixed segfault: NS assembly branch for no-magnetic case (use `assemble_stokes()`)
- Re-ran square test (passed)
- Ran droplet without field test (passed, |U|=0 as expected)
- Confirmed stabilizer S1 = lambda/(4*epsilon) does NOT depend on dropped terms (beta, spin vorticity, Maxwell stress) -- these are all dissipative and exactly zero for algebraic M = chi*H

**Git commits:**
- `4f53b44` -- SAV assembly fix, NS/Poisson coupling, Rosensweig correction (Sessions 1-2)
- `af2ca79` -- Fix capillary force direction (theta*grad_psi) and add grad(h_a) to Kelvin force (Session 3)

### Session 4: Complete Zhang Deviation Audit + 7 Fixes

Equation-by-equation comparison of code vs Zhang, He & Yang (SIAM J. Sci. Comput. 43(1), 2021).
Found 8 deviations, fixed 7. All changes compile cleanly.

| # | Deviation | Fix Applied |
|---|-----------|-------------|
| **1** | **Viscous term 2× too large** | `nu/2.0` → `nu/4.0` in 3 functions (12 lines) |
| **2** | **S1 = 200 instead of 50** | Auto-compute: `lambda/(4*epsilon)` |
| **3** | Saddle-point vs pressure projection | DEFERRED — structurally different |
| **4** | **Sigmoid vs linear material properties** | chi, nu now linear in Φ=(θ+1)/2 |
| **5** | **Algebraic M instead of mag PDE** | `use_algebraic_magnetization=false`, β=1, τ=1e-4 |
| **6** | **Dipole regularization δ** | Removed (δ=0, matching Zhang) |
| **7** | **tanh IC instead of sharp step** | Sharp step for Rosensweig + droplet |

**Files modified:**
- `navier_stokes/navier_stokes_assemble.cc` — viscous factor fix (#1), comments
- `drivers/decoupled_driver.cc` — S1 formula (#2), sharp step ICs (#7)
- `physics/material_properties.h` — linear chi/nu, sigmoid density (#4)
- `utilities/parameters.cc` — β=1, τ=1e-4, algebraic_M=false for both presets (#5)
- `physics/applied_field.h` — remove δ regularization (#6)

### Sessions 5-8: MMS Verification Framework + Critical DG Bug Fix

**Incremental MMS validation strategy** — test each coupling layer before the full system:

| Test | Status | u_L2 rate | p_L2 rate | φ_L2 rate | M_L2 rate |
|------|--------|-----------|-----------|-----------|-----------|
| Poisson standalone | PASS | — | — | 3.0 | — |
| Magnetization standalone (U=0) | PASS | — | — | — | 2.0 |
| Poisson-Mag coupled (Picard) | PASS | — | — | 3.0 | 2.0 |
| **Poisson-Mag-NS (Kelvin, μ₀=0.1)** | **PASS** | **3.0** | **2.1** | **3.0** | **1.95** |
| Full system (CH+NS+Poisson+Mag) | Pending | | | | |

**Critical bug found and fixed: DG face assembly `FEInterfaceValues` indexing**

The DG magnetization transport had an `FEInterfaceValues::shape_value` indexing bug that
caused **zero cross-cell coupling** in the face flux. For DG elements, interface DoFs are numbered:
- 0..dpc-1: "here" cell DoFs
- dpc..2*dpc-1: "there" cell DoFs

The code used `shape_value(false, i, q)` to get "there" cell's i-th basis function, but this
actually looks up interface DoF `i` (a "here" DoF) from the "there" side → always 0 for DG.
**Fix**: `shape_value(false, i + dofs_per_cell, q)`.

Effect: Face flux had NO cross-cell coupling → DG transport gave O(1) errors instead of O(h²).
After fix + upwind penalty: M_L2 convergence rate = 1.95 (expected 2.0).

**Files modified:**
- `magnetization/magnetization_assemble.cc` — Fixed `shape_value` indexing in 4 places
  (AMR Case 1 + Case 2, both matrix and face_mms_active sections) + added upwind penalty
- `mms_tests/poisson_mag_ns_mms.h` — NEW: NS MMS source with Kelvin force (body + curl)
- `mms_tests/poisson_mag_ns_mms_test.cc` — NEW: 3-subsystem test harness, diagnostics
- `mms_tests/CMakeLists.txt` — Added test targets

**Run command:**
```bash
cmake --build build -j8 --target test_poisson_mag_ns_mms
mpirun -np 1 build/mms_tests/test_poisson_mag_ns_mms --refs 2 3 4 --steps 1
```

---

## Pending Tasks (In Priority Order)

### 1. Rebuild Driver and Run Full Validation Tests
The DG face assembly bug fix (Session 8) affects ALL validation tests that use
the magnetization PDE with non-zero velocity. Rebuild and rerun:
```bash
cd /Users/mahdi/Projects/git/PhD-Project/Decoupled/drivers/build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j8
mpirun -np 2 ./ferrofluid_decoupled --validation droplet -r 7 --vtk_interval 20
mpirun -np 2 ./ferrofluid_decoupled --rosensweig -r 4 --vtk_interval 20
```
- Quick sanity checks (20 steps) passed for both droplet and rosensweig after fix
- Output now goes to `/Users/mahdi/Projects/git/PhD-Project/Decoupled/Results/`
- Rosensweig previously unstable (θ overshoot to 1.63) — may now be stable with correct DG transport

### 2. Poisson-Mag-NS-CH MMS Test (Full System)
- Next step in incremental MMS strategy: add CH to the coupled test
- All sub-tests pass: standalone, Poisson-Mag, Poisson-Mag-NS
- Needs MMS source terms for the full 4-subsystem coupling

### 3. Implement Pressure Projection (Deviation #3, DEFERRED)
- Zhang uses Chorin-type 3-step projection
- Current code uses direct saddle-point (arguably more accurate)
- Implement if results don't match Zhang after other fixes

---

## Key Technical Details

### Phase Field Convention
- Zhang uses Phi in {0,1}, code uses theta in {-1,+1}
- Equivalent via Phi = (theta+1)/2
- Material properties now use LINEAR interpolation matching Zhang:
  - chi(θ) = χ₀·(θ+1)/2 (not sigmoid)
  - nu(θ) = ν_w·(1-θ)/2 + ν_f·(θ+1)/2 (not sigmoid)
  - rho(θ) = 1 + r·H(θ/ε) (sigmoid — matches Zhang Eq 4.2)

### Magnetization
- Code uses **algebraic M = chi(theta)*H** (tau->0 limit), skips magnetization PDE
- With algebraic M, beta term and curl terms are identically zero

### SAV Stabilization (Zhang Eq 3.5-3.6)
- S1 = CH stabilization = lambda/(4*epsilon). Override with `--sav_S1`
- S2 = NS stabilization, adaptive: S2 = 1.5*mu0^2*(chi0*|H_max|)^2/(4*nu_min)
- S1 does NOT depend on the 3 dropped terms (verified from energy proof Eq 3.41)

### Capillary Force (Fixed in Session 3)
- Correct form: theta * grad(psi) = Phi * grad(W) (Zhang Eq 2.6)
- SAV psi = lambda*(-epsilon*Laplacian(theta) + (1/epsilon)*f(theta)) already contains lambda

### Kelvin Force (Fixed in Session 3)
- (M . grad)H where H = grad(phi) + h_a
- grad(H) = Hess(phi) + grad(h_a) -- both terms now included
- `compute_applied_field_gradient<dim>()` computes analytical Jacobian of dipole field

---

## Test Parameters

### Droplet (Zhang Eq 4.8)
```
Domain:     [0,1] x [0,1]
IC:         Circular droplet R=0.1 at (0.5, 0.5)
epsilon:    2e-3
Mobility:   2e-4
chi_0:      2
mu_0:       0.1
nu_f=nu_w:  1
lambda:     1
gravity:    0
ramp_slope: 1000
dt:         1e-3
max_steps:  1500
```

### Rosensweig (Zhang Eq 4.4)
```
Domain:     [0,1] x [0,0.6]
IC:         Flat interface at y=0.2, NO perturbation
epsilon:    5e-3
Mobility:   2e-4
chi_0:      0.5
mu_0:       1.0
nu_f:       2.0,  nu_w: 1.0
lambda:     1
r:          0.1
gravity:    (0, -6e4)
ramp_slope: 5000,  intensity_max: 8000
dt:         1e-3
max_steps:  2000
```

---

## Files Modified Across All Sessions

| File | What Changed |
|------|-------------|
| `cahn_hilliard/cahn_hilliard_assemble.cc` | S1: Removed Douglas-Dupont, added lambda to SAV |
| `utilities/parameters.cc` | S2: Fixed Rosensweig params. S3: Added `droplet_nofield` preset |
| `utilities/parameters.h` | S8: Output folder → absolute path `Decoupled/Results/` |
| `drivers/decoupled_driver.cc` | S2: Fixed Rosensweig IC. S3: Added `droplet_nofield` IC/output, NS assembly branch for no-magnetic |
| `physics/applied_field.h` | S2: Added intensity_max cap. S3: Added `compute_applied_field_gradient<dim>()` |
| `navier_stokes/navier_stokes_assemble.cc` | S3: Fixed capillary (theta*grad_psi), added grad(h_a) to Kelvin in both assembly functions |
| `magnetization/magnetization_assemble.cc` | **S8: Fixed FEInterfaceValues indexing bug** (shape_value offset for "there" DoFs) + added upwind penalty `½\|U·n\|[[Z]][[M]]` |
| `mms_tests/poisson_mag_ns_mms.h` | S7: **NEW** — NS MMS source with Kelvin body force + curl correction |
| `mms_tests/poisson_mag_ns_mms_test.cc` | S7-8: **NEW** — 3-subsystem test harness with diagnostics |
| `mms_tests/CMakeLists.txt` | S7: Added `test_poisson_mag_ns_mms` target |

## Key Files (Read-Only Reference)

| File | Purpose |
|------|---------|
| `utilities/parameters.cc` | All presets: square, droplet, droplet_nofield, rosensweig |
| `drivers/decoupled_driver.cc` | IC setup + time loop (Strategy A: Gauss-Seidel) |
| `physics/applied_field.h` | h_a computation + gradient (Jacobian) for Kelvin force |
| `physics/material_properties.h` | Double-well, chi, nu, rho |
| `physics/kelvin_force.h` | DG skew form: cell kernel + face kernel |
| `cahn_hilliard/cahn_hilliard_assemble.cc` | SAV CH assembly (coupled theta-psi) |
| `navier_stokes/navier_stokes_assemble.cc` | NS assembly: capillary + Kelvin + viscous + convection |
| `poisson/poisson_assemble.cc` | ((1+chi)*grad(phi), grad(X)) = ((1-chi)*h_a, grad(X)) |

---

## User Preferences
- **2 CPU cores** for validation tests (6 cores available if needed)
- Results auto-saved to `Results/` with timestamp prefix
- User opens VTK files in **ParaView** to verify visually
- Following Zhang for parameters/validation, Nochetto for PDE formulation (simplified model)
- beta, spin vorticity, Maxwell stress tensor **dropped for now**

---

*Updated: February 27, 2026 (Session 8 — DG face fix + Poisson-Mag-NS MMS PASS)*
