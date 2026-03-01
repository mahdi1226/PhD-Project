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
| **Square** (CH-only, r=6) | PASS | 5000 steps, theta in [-0.992, 1.01] | `Results/022826_062447_square_r6/` |
| **Droplet WITH field** (r=7) | PASS | 1500 steps, theta in [-1.00, 1.03] | `Results/022826_000008_droplet_wfield_r7/` |
| **Droplet WITHOUT field** | PASS | 1500 steps, theta in [-1.00, 1.01], |U|=0 | `Results/022126_055622_droplet_wofield_r7/` |
| **Rosensweig uniform** (Section 4.3, r=4) | PASS | 2000 steps, theta in [-1.00, 1.00], 5 dipoles at y=-15 | `Results/022826_132131_rosensweig_r4/` |
| **Rosensweig nonuniform** (Section 4.4) | **UNSTABLE** | Two spikes form correctly by step 10000, then numerical instability onset at step ~10350 (t=2.07). |U| explodes from 0.52 to 30+, theta overshoots [-1.04, 1.08]. Does NOT match Zhang. | `Results/022826_222751_rosensweig_nonuniform_r3/` |

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
| **1** | **Viscous term 2x too large** | `nu/2.0` -> `nu/4.0` in 3 functions (12 lines) |
| **2** | **S1 = 200 instead of 50** | Auto-compute: `lambda/(4*epsilon)` |
| **3** | Saddle-point vs pressure projection | DEFERRED -- structurally different |
| **4** | **Sigmoid vs linear material properties** | chi, nu changed to linear -- **WRONG, reverted to sigmoid in S11** |
| **5** | **Algebraic M instead of mag PDE** | `use_algebraic_magnetization=false`, beta=1, tau=1e-4 |
| **6** | **Dipole regularization delta** | Removed (delta=0, matching Zhang) |
| **7** | **tanh IC instead of sharp step** | Sharp step for Rosensweig + droplet |

**Files modified:**
- `navier_stokes/navier_stokes_assemble.cc` -- viscous factor fix (#1), comments
- `drivers/decoupled_driver.cc` -- S1 formula (#2), sharp step ICs (#7)
- `physics/material_properties.h` -- chi/nu changed to linear (#4) -- **reverted to sigmoid in S11**
- `utilities/parameters.cc` -- beta=1, tau=1e-4, algebraic_M=false for both presets (#5)
- `physics/applied_field.h` -- remove delta regularization (#6)

### Sessions 5-8: MMS Verification Framework + Critical DG Bug Fix

**Incremental MMS validation strategy** -- test each coupling layer before the full system:

| Test | Status | u_L2 rate | p_L2 rate | phi_L2 rate | M_L2 rate |
|------|--------|-----------|-----------|-------------|-----------|
| Poisson standalone | PASS | -- | -- | 3.0 | -- |
| Magnetization standalone (U=0) | PASS | -- | -- | -- | 2.0 |
| Poisson-Mag coupled (Picard) | PASS | -- | -- | 3.0 | 2.0 |
| **Poisson-Mag-NS (Kelvin, mu_0=0.1)** | **PASS** | **3.0** | **2.1** | **3.0** | **1.95** |
| Full system (CH+NS+Poisson+Mag) | Pending | | | | |

**Critical bug found and fixed: DG face assembly `FEInterfaceValues` indexing**

The DG magnetization transport had an `FEInterfaceValues::shape_value` indexing bug that
caused **zero cross-cell coupling** in the face flux. For DG elements, interface DoFs are numbered:
- 0..dpc-1: "here" cell DoFs
- dpc..2*dpc-1: "there" cell DoFs

The code used `shape_value(false, i, q)` to get "there" cell's i-th basis function, but this
actually looks up interface DoF `i` (a "here" DoF) from the "there" side -> always 0 for DG.
**Fix**: `shape_value(false, i + dofs_per_cell, q)`.

Effect: Face flux had NO cross-cell coupling -> DG transport gave O(1) errors instead of O(h^2).
After fix + upwind penalty: M_L2 convergence rate = 1.95 (expected 2.0).

**Files modified:**
- `magnetization/magnetization_assemble.cc` -- Fixed `shape_value` indexing in 4 places
  (AMR Case 1 + Case 2, both matrix and face_mms_active sections) + added upwind penalty
- `mms_tests/poisson_mag_ns_mms.h` -- NEW: NS MMS source with Kelvin force (body + curl)
- `mms_tests/poisson_mag_ns_mms_test.cc` -- NEW: 3-subsystem test harness, diagnostics
- `mms_tests/CMakeLists.txt` -- Added test targets

**Run command:**
```bash
cmake --build build -j8 --target test_poisson_mag_ns_mms
mpirun -np 1 build/mms_tests/test_poisson_mag_ns_mms --refs 2 3 4 --steps 1
```

### Sessions 9-10: Full Validation Suite + Nonuniform Rosensweig

**Completed validation tests (all rerun post-DG fix):**

1. **Square test** (CH-only, r=6): 5000 steps, theta in [-0.992, 1.01]. PASS.
   - Directory: `Results/022826_062447_square_r6/`

2. **Droplet with field** (r=7): 1500 steps, theta in [-1.00, 1.03]. PASS.
   - Directory: `Results/022826_000008_droplet_wfield_r7/`

3. **Uniform Rosensweig** (Section 4.3, r=4): 2000 steps, theta in [-1.00, 1.00]. PASS.
   - 5 dipoles at y=-15, uniform spacing, chi_0=0.5, ramp 5000, max 8000
   - Picard: 10 iters per step, all converging
   - Spike formation visible, matches Zhang qualitatively
   - Directory: `Results/022826_132131_rosensweig_r4/`

4. **Nonuniform Rosensweig** (Section 4.4): 17500 steps planned, **UNSTABLE after step ~10350**.
   - 42 dipoles at y in {-0.5, -0.75, -1.0}, chi_0=0.9, dt=2e-4, h=1/120 (r=3 custom)
   - Configurable y_interface=0.1 (added in Session 9)
   - Two clean Rosensweig spikes form by step 10000 (t=2.0)
   - **Instability onset at step ~10350 (t=2.07):**
     - |U| jumps from 0.52 -> 0.73 -> 1.19 -> 2.13 -> ... -> 30+ over ~200 steps
     - theta overshoots to [-1.04, 1.08] by step 10570
     - Spike morphology distorts from clean peaks to chaotic mixing
     - Does NOT match Zhang's smooth, gradual spike evolution
   - Directory: `Results/022826_222751_rosensweig_nonuniform_r3/`
   - VTK frames 0-200 (steps 0-10000) are valid; frames >207 show numerical artifacts

**New code (Session 9-10):**

- Added `--rosensweig-nonuniform` preset to `utilities/parameters.cc`:
  ```
  chi_0=0.9, mu_0=1.0, nu_f=2.0, nu_w=1.0, lambda=0.25, epsilon=5e-3
  42 dipoles (3 rows: y=-0.5, y=-0.75, y=-1.0, 14 per row, x in [0, 0.975])
  dt=2e-4, gravity=(0,-6e4), r=0.1, ramp=5000, alpha_max=8000
  h=1/120 (subdivided 120x72), interface at y=0.1
  ```
- Added `y_interface` parameter to `utilities/parameters.h`
- Updated `drivers/decoupled_driver.cc`: IC uses configurable `y_interface` instead of hardcoded 0.2

**Analytical predictions for parametric study (C.0 extension):**

Derived Rosensweig linear stability theory from the governing equations:
- Dispersion: omega^2(k) = -Delta_rho*g*k - sigma*k^3 + mu_0*f(chi_0)*H_0^2*k^2
- Surface tension: sigma = lambda*sqrt(2)/3 (from CH energy with F=(theta^2-1)^2/16)
- Critical wavenumber: k_c = sqrt(Delta_rho*g/sigma)
- Predictions: chi_0 controls onset/height (NOT spike count), lambda controls spike count/spacing
- Updated `Report/extension.md` with full parametric study design (162 runs, geometric extensions)

---

## CRITICAL: Nonuniform Rosensweig Numerical Instability (Debugging Required)

### Symptoms
- Two clean Rosensweig spikes visible at frame 200 (step 10000, t=2.0)
- At step ~10350 (t~2.07): sudden velocity explosion, |U| goes from 0.52 to 30+ in ~200 steps
- theta overshoots [-1.04, 1.08] — maximum principle violated
- Spike morphology distorts into chaotic mixing — NOT physical
- Zhang's paper shows smooth, gradual spike growth without such discontinuities

### Instability Timeline (from log)
```
step 10000  t=2.000  theta=[-1.00, 0.999]  |U|=0.498   |M|=94.6   S2=4.5e3   <- Clean
step 10300  t=2.060  theta=[-1.00, 1.000]  |U|=0.518   |M|=97.2   S2=4.8e3   <- Still OK
step 10350  t=2.070  theta=[-1.00, 1.000]  |U|=0.731   |M|=99.6   S2=4.8e3   <- ONSET
step 10370  t=2.074  theta=[-1.00, 1.000]  |U|=2.13    |M|=104    S2=5.1e3   <- Growing
step 10400  t=2.080  theta=[-1.00, 1.000]  |U|=5.19    |M|=108    S2=7.3e3   <- Rapid
step 10470  t=2.094  theta=[-1.00, 1.01]   |U|=11.4    |M|=111    S2=7.5e3   <- Exploding
step 10570  t=2.114  theta=[-1.02, 1.08]   |U|=7.48    |M|=107    S2=7.8e3   <- theta overshoot
step 11000  t=2.200  theta=[-1.06, 0.993]  |U|=27.0    |M|=124    S2=1.5e4   <- Invalid
step 11420  t=2.284  theta=[-1.03, 1.000]  |U|=18.3    |M|=120    S2=1.1e4   <- Continuing
```

### Root Cause Analysis

1. **Picard convergence is too weak (global L2 norm):**
   - `rel_change = |M_new_norm - M_k_norm| / M_new_norm` uses GLOBAL L2 norms
   - Reports `rel_change=0.0` and 2 iterations even during violent dynamics
   - Misses LOCAL changes at spike tips (small volume, large gradients)
   - Fix: Use pointwise or element-local convergence criteria

2. **S2 stabilization lags by one step:**
   - S2 computed from phi^{n-1} using H_max from previous step (lines 1387-1400 in driver)
   - During rapid H changes at spike tips, S2 cannot keep up
   - Creates positive feedback: larger H -> need larger S2, but S2 based on old H

3. **No outer NS<->Mag iteration (operator splitting instability):**
   - NS uses stale M^{n-1}, phi^{n-1} for Kelvin force
   - Picard only iterates Poisson<->Mag, NOT NS<->Mag
   - For strong coupling (nonuniform: chi_0=0.9, close dipoles, |H|~200), the
     splitting error becomes dynamically unstable
   - Uniform case works because coupling is weaker (chi_0=0.5, dipoles at y=-15)

4. **Nonuniform case is ~7x harder than uniform:**
   - Dipoles at y={-0.5,-0.75,-1.0} vs y=-15 -> much stronger field gradients
   - chi_0=0.9 vs 0.5 -> stronger magnetic response
   - Interface at y=0.1 vs y=0.2 -> closer to dipoles

### Proposed Fixes (Next Session Priority)

1. **Reduce dt further** (quick test): try dt=1e-4 instead of 2e-4
2. **Tighten Picard:** increase max_picard=50, decrease tol=1e-6 (from 1e-2)
3. **Add outer NS-Mag iteration** (Strategy B from COUPLING_PLAN.md):
   - After NS solve, re-do Poisson->Mag Picard with updated U
   - Check outer convergence of U
4. **Finer mesh**: increase from h=1/120 to h=1/160 to resolve spike tip gradients
5. **S2 predictor**: use extrapolated H_max instead of lagged

---

## Pending Tasks (In Priority Order)

### 1. Debug Nonuniform Rosensweig Instability (CRITICAL)
The uniform case passes (Section 4.3), but the nonuniform case (Section 4.4) develops
numerical instability after ~10000 steps. This is the main blocking issue. See root
cause analysis above.

### 2. Poisson-Mag-NS-CH MMS Test (Full System)
- Next step in incremental MMS strategy: add CH to the coupled test
- All sub-tests pass: standalone, Poisson-Mag, Poisson-Mag-NS
- Needs MMS source terms for the full 4-subsystem coupling

### 3. Implement Pressure Projection (Deviation #3, DEFERRED)
- Zhang uses Chorin-type 3-step projection
- Current code uses direct saddle-point (arguably more accurate)
- Implement if results don't match Zhang after other fixes

### 4. Extension Study (Phase C.0+)
- Parametric study design complete in `Report/extension.md`
- CLI parameter overrides needed: --chi0, --alpha_max, --Lx, --Ly, --dipole_curve
- DO NOT CODE YET -- wait for validation to complete

---

## Key Technical Details

### Phase Field Convention
- Zhang uses Phi in {0,1}, code uses theta in {-1,+1}
- Equivalent via Phi = (theta+1)/2
- Material properties now use LINEAR interpolation matching Zhang:
  - chi(theta) = chi_0*(theta+1)/2 (not sigmoid)
  - nu(theta) = nu_w*(1-theta)/2 + nu_f*(theta+1)/2 (not sigmoid)
  - rho(theta) = 1 + r*H(theta/epsilon) (sigmoid -- matches Zhang Eq 4.2)

### Magnetization
- Code now uses **full magnetization PDE** with beta=1, tau=1e-4
- Picard iteration resolves Poisson<->Mag coupling within each step
- DG Q1 with upwind flux (MMS-verified O(h^2))

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

### Reduced Magnetic Field Infrastructure
- `use_reduced_magnetic_field` flag already implemented in magnetization assemblers
- CLI: `--reduced_field` to switch from total field to reduced field formulation
- Not yet tested in validation runs

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

### Rosensweig Uniform (Zhang Eq 4.3)
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
Dipoles:    5 at y=-15, uniform spacing x in {0, 0.25, 0.5, 0.75, 1.0}
```

### Rosensweig Nonuniform (Zhang Eq 4.4)
```
Domain:     [0,1] x [0,0.6]
IC:         Flat interface at y=0.1
epsilon:    5e-3
Mobility:   2e-4
chi_0:      0.9
mu_0:       1.0
nu_f:       2.0,  nu_w: 1.0
lambda:     0.25
r:          0.1
gravity:    (0, -6e4)
ramp_slope: 5000,  intensity_max: 8000
dt:         2e-4
max_steps:  17500
Mesh:       120x72 subdivided rectangle (h=1/120)
Dipoles:    42 total, 3 rows:
            - 14 at y=-0.5, x in [0, 0.975]
            - 14 at y=-0.75
            - 14 at y=-1.0
```

---

## Files Modified Across All Sessions

| File | What Changed |
|------|-------------|
| `cahn_hilliard/cahn_hilliard_assemble.cc` | S1: Removed Douglas-Dupont, added lambda to SAV |
| `utilities/parameters.cc` | S2: Fixed Rosensweig params. S3: Added `droplet_nofield`. **S9: Added `rosensweig-nonuniform` preset (42 dipoles, chi_0=0.9, dt=2e-4, h=1/120)** |
| `utilities/parameters.h` | S8: Output folder -> absolute path. **S9: Added `y_interface` parameter** |
| `drivers/decoupled_driver.cc` | S2: Fixed Rosensweig IC. S3: Added `droplet_nofield` IC/output, NS assembly branch for no-magnetic. **S9: IC uses configurable `y_interface`** |
| `physics/applied_field.h` | S2: Added intensity_max cap. S3: Added `compute_applied_field_gradient<dim>()` |
| `navier_stokes/navier_stokes_assemble.cc` | S3: Fixed capillary (theta*grad_psi), added grad(h_a) to Kelvin in both assembly functions |
| `magnetization/magnetization_assemble.cc` | **S8: Fixed FEInterfaceValues indexing bug** (shape_value offset for "there" DoFs) + added upwind penalty |
| `mms_tests/poisson_mag_ns_mms.h` | S7: **NEW** -- NS MMS source with Kelvin body force + curl correction |
| `mms_tests/poisson_mag_ns_mms_test.cc` | S7-8: **NEW** -- 3-subsystem test harness with diagnostics |
| `mms_tests/CMakeLists.txt` | S7: Added `test_poisson_mag_ns_mms` target |
| `Report/extension.md` | **S10: NEW** -- Phase C extension research roadmap (parametric study, geometric extensions, dipole geometry) |
| `physics/material_properties.h` | **S11: D1 FIX** -- Reverted chi(theta) and nu(theta) from LINEAR to SIGMOID H(theta/eps). Session 4 wrongly changed these to linear. |
| `magnetization/magnetization.h` | **S11: D3 FIX** -- Added spin_vort_rhs_x/y_ cache vectors for spin-vorticity term |
| `magnetization/magnetization_setup.cc` | **S11: D3 FIX** -- Allocate spin-vorticity cache vectors |
| `magnetization/magnetization_assemble.cc` | **S11: D3 FIX** -- Added +1/2(curl_u x m^{n-1}, Z) spin-vorticity on RHS, cached for Picard reuse |

## Key Files (Read-Only Reference)

| File | Purpose |
|------|---------|
| `utilities/parameters.cc` | All presets: square, droplet, droplet_nofield, rosensweig, rosensweig-nonuniform |
| `drivers/decoupled_driver.cc` | IC setup + time loop (Strategy A: Gauss-Seidel) |
| `physics/applied_field.h` | h_a computation + gradient (Jacobian) for Kelvin force |
| `physics/material_properties.h` | Double-well, chi, nu, rho |
| `physics/kelvin_force.h` | DG skew form: cell kernel + face kernel |
| `cahn_hilliard/cahn_hilliard_assemble.cc` | SAV CH assembly (coupled theta-psi) |
| `navier_stokes/navier_stokes_assemble.cc` | NS assembly: capillary + Kelvin + viscous + convection |
| `poisson/poisson_assemble.cc` | ((1+chi)*grad(phi), grad(X)) = ((1-chi)*h_a, grad(X)) |
| `Report/extension.md` | Phase C extension research roadmap with analytical predictions |

---

## Session 11: Systematic Code-vs-Paper Audit

Full line-by-line audit of code against Zhang, He & Yang (SIAM J. Sci. Comput. 43(1), 2021).
Read all 27 pages of the paper and compared every equation, parameter, and algorithmic step.

### Discrepancies Found and Fixed (Session 11)

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| **D1** | chi(theta) and nu(theta) were LINEAR, should be SIGMOID H(theta/eps) per Zhang p. B170 | MAJOR | **FIXED** |
| **D3** | Missing spin-vorticity term -1/2(curl_u x m, Z) in magnetization (Zhang Eq 3.14) | MED-HIGH | **FIXED** |

### Remaining Discrepancy: D2 (MAJOR) -- Missing Eq 3.17 Final Magnetization Step

**This is the HIGHEST PRIORITY remaining item.**

Zhang Algorithm 3.1 has **6 Steps**. Our code implements Steps 1-5 but **SKIPS Step 6**.

**Step 6 (Eq 3.17):** After solving for intermediate m_tilde^n (Step 5), solve for the FINAL m^n:
```
(d_t m^n, n) + b(u_tilde^n, m^n, n) - 1/2(curl_u^n x m^n, n)
= -(1/tau)(m^n, n) + (1/tau)(chi(Phi^n) h_tilde^n, n) + beta(m^{n-1} x h_tilde^n, m^n x n)
```

**Key difference from Step 5 (Eq 3.14):**
- Step 5 uses SKEW form: ((u.grad)m + (div u)m, n) -- for energy stability
- Step 6 uses STANDARD trilinear: b(u, m, n) = ((u.grad)m, n) -- for final accuracy

**Why it matters:** The paper's energy stability proof (Theorem 3.1) requires BOTH steps. Without Step 6, m_tilde from Step 5 is used as the final m, which is only an intermediate variable.

**Implementation plan for D2:**
1. Add a second magnetization solve per timestep
2. Use standard trilinear form b(u,m,n) = ((u.grad)m, n) instead of skew form
3. Include spin-vorticity with CURRENT m^n (not m^{n-1})
4. This is implicit in m^n, so it needs a separate matrix assembly
5. Approximate cost: doubles the magnetization solve time per step

### Other Discrepancies (Acceptable, No Fix Needed)

| # | Issue | Why Acceptable |
|---|-------|----------------|
| D4 | FE spaces: CG Q2/DG Q1 vs paper's CG P1/P2 Taylor-Hood | DG choices defensible for transport. MMS tests pass. |
| D5 | Monolithic saddle-point vs paper's pressure projection | Our approach is MORE accurate (no splitting error) |
| D6 | No h_tilde L2-projection (Eq 3.16) | Direct eval at quad points is as accurate |

### Items Verified Correct

Viscous ν/4(T:T), NS convection (skew), capillary (theta_old*grad_psi), all 3 Kelvin terms,
all 3 b_stab terms, S1=lambda/(4*epsilon), S2 adaptive, double-well F(theta)=(theta^2-1)^2/16,
lambda/gamma conversion, CH convection/SUPG, magnetization relaxation/beta, DG transport
with upwind, Poisson weak form and natural Neumann BC, density sigmoid, gravity Boussinesq,
all parameter values for uniform and nonuniform Rosensweig, solve order CH->NS->Poisson->Mag.

---

## Git Status

**Latest commit:** `800aa2c` (Add coupled ferrofluid solver with full MMS verification)

**Uncommitted changes (to be committed in Session 11):**
- `physics/material_properties.h` -- **D1 FIX**: chi and nu reverted from LINEAR to SIGMOID H(theta/eps)
- `magnetization/magnetization.h` -- **D3 FIX**: Added spin_vort_rhs_ cache vectors
- `magnetization/magnetization_setup.cc` -- **D3 FIX**: Allocate spin_vort_rhs_ vectors
- `magnetization/magnetization_assemble.cc` -- **D3 FIX**: Compute +1/2(curl_u x m^{n-1}, Z) term, cache for Picard
- `drivers/decoupled_driver.cc` -- configurable y_interface in IC
- `utilities/parameters.cc` -- rosensweig-nonuniform preset (42 dipoles, Section 4.4 params)
- `utilities/parameters.h` -- y_interface parameter
- `Report/session_handoff.md` -- Updated with audit findings

**Untracked:**
- `Report/extension.md` -- Phase C extension research roadmap

---

## User Preferences
- **4 CPU cores** for nonuniform Rosensweig (6 available), 2 cores for standard tests
- Results auto-saved to `Results/` with timestamp prefix
- User opens VTK files in **ParaView** to verify visually
- Following Zhang for parameters/validation, Nochetto for PDE formulation (simplified model)
- Extension study: DO NOT CODE YET -- wait for validation to complete

---

## Next Steps (Priority Order)
1. **Re-run Rosensweig tests** with sigmoid chi/nu and spin-vorticity fixes
2. **Implement D2 (Eq 3.17)** if instability persists -- second magnetization solve per timestep
3. **Re-run MMS tests** to confirm convergence rates preserved after changes
4. Visual validation against paper figures in ParaView

---

*Updated: March 1, 2026 (Session 11 -- Systematic code-vs-paper audit, D1+D3 fixes)*
