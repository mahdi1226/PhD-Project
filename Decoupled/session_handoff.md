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
| **Droplet WITH field** | Running (post-fix) | Step ~330/1500, theta in [-1.00, 1.01] | (in progress) |
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

---

## Pending Tasks (In Priority Order)

### 1. Droplet WITH field (post-fix) -- Running Now
Currently at step ~330/1500 with bug fixes applied. ETA ~1 hour.

### 2. Rosensweig Test -- Rerun After Bug Fixes
```bash
cd /Users/mahdi/Projects/git/PhD-Project/Decoupled/drivers/build
mpirun -np 2 ./ferrofluid_decoupled --validation rosensweig -r 4 --vtk_interval 100
```
- Bug fixes may resolve the theta overshoot instability
- If still unstable, try higher S1 or smaller dt

### 3. Add Back Dropped Physics (Future)
- beta, spin vorticity, Maxwell stress tensor dropped for now
- With algebraic M = chi*H, all 3 terms are exactly zero anyway
- Will add back when switching to full magnetization PDE

---

## Key Technical Details

### Phase Field Convention
- Zhang uses Phi in {0,1}, code uses theta in {-1,+1}
- Equivalent via Phi = (theta+1)/2

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
| `drivers/decoupled_driver.cc` | S2: Fixed Rosensweig IC. S3: Added `droplet_nofield` IC/output, NS assembly branch for no-magnetic |
| `physics/applied_field.h` | S2: Added intensity_max cap. S3: Added `compute_applied_field_gradient<dim>()` |
| `navier_stokes/navier_stokes_assemble.cc` | S3: Fixed capillary (theta*grad_psi), added grad(h_a) to Kelvin in both assembly functions |

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

*Updated: February 21, 2025 (Session 3)*
