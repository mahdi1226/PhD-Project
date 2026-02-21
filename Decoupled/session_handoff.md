# Decoupled Ferrofluid Solver — Session Handoff

## Project Location
`/Users/mahdi/Projects/git/PhD-Project/Decoupled/`
Build: `cd /Users/mahdi/Projects/git/PhD-Project/Decoupled/drivers/build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j8`

## Reference Paper
Zhang, He & Yang, SIAM J. Sci. Comput. 43(1), 2021
PDF: `/Users/mahdi/Desktop/droplet/GZhang_XMHe_XYang.pdf`

---

## Current Status Summary

| Test | Status | Result | Directory |
|------|--------|--------|-----------|
| **Square** (CH-only, λ=100) | ✅ Done | 5000 steps, θ∈[-0.992, 1.01], E_CH→117.05 | `Results/022026_234513_square_r6/` |
| **Droplet WITH field** | ✅ Done | 1500 steps, θ∈[-1.00, 1.03], stable | `Results/022126_010723_droplet_wfield_r7/` |
| **Droplet WITHOUT field** | ⏳ Pending | Not yet run | — |
| **Rosensweig** | ❌ Postponed | Killed at step ~1160, θ overshoot to 1.63 | `Results/022026_234445_rosensweig_r4/` |

---

## What Was Done Across All Sessions

### Session 1: SAV CH Assembly Fix
- **Removed Douglas-Dupont stabilization** from CH assembly (was over-stabilizing)
- **Added λ factors** to diffuse-interface energy terms in SAV formulation
- File: `cahn_hilliard/cahn_hilliard_assemble.cc`

### Session 2: Rosensweig Fix + Validation Runs
1. **Fixed Rosensweig parameters** (10 of 14 were wrong) — `utilities/parameters.cc`, function `setup_rosensweig()`
2. **Fixed Rosensweig IC** — flat at y=0.2, no perturbation — `drivers/decoupled_driver.cc`
3. **Added field ramp cap** — `physics/applied_field.h`: `α(t) = min(slope·t, intensity_max)`
4. **Ran square test** → PASSED (5000 steps)
5. **Ran Rosensweig** → UNSTABLE (θ overshoot to 1.63 at t≈0.88), killed by user
6. **Ran droplet WITH field** (2 CPU cores) → PASSED (1500 steps, θ∈[-1.00, 1.03])

---

## Pending Tasks (In Priority Order)

### 1. Run Droplet WITHOUT Magnetic Field (2 CPU cores)
```bash
cd /Users/mahdi/Projects/git/PhD-Project/Decoupled/drivers/build
mpirun -np 2 ./ferrofluid_decoupled --validation droplet_nofield -r 7 --vtk_interval 100 --sav_S1 5000
```
- Should show circular droplet remaining circular (no magnetic force)
- 1500 steps, dt=1e-3, ~85 min with 2 cores

### 2. Rosensweig Test (Postponed)
- User said: "Forget rosensweig for now"
- Issue: θ overshoot to 1.63 at t≈0.88, needs investigation
- May need: higher S₁, smaller dt, or different stabilization
- Parameters are now correct (Zhang Eq 4.4), just stability needs work

### 3. Add Back Dropped Physics (Future)
- User said: "beta term and spin vorticity plus maxwell tensor is dropped, we will add them back later"
- β term: implemented in `magnetization/magnetization_assemble.cc`, only active when `enable_beta_term=true`
- Spin vorticity & Maxwell stress tensor: not yet implemented

---

## Key Technical Details

### Phase Field Convention
- Zhang uses Φ ∈ {0,1}, code uses θ ∈ {-1,+1}
- Equivalent via Φ = (θ+1)/2
- Code is internally consistent with {-1,+1}

### Magnetization
- Code uses **algebraic M = χ(θ)H** (τ→0 limit), skips magnetization PDE
- Zhang solves full PDE with τ=1e-4, β=1 — but τ so small that algebraic is good approx
- With algebraic M, β term is irrelevant

### SAV Stabilization
- S₁ = CH stabilization constant. Auto-computed as 1/ε but can be overridden with `--sav_S1`
- For droplet (ε=2e-3): S₁=5000 = 10/ε works well
- For Rosensweig (ε=5e-3): S₁=200 auto, may need higher
- S₂ = NS stabilization, computed adaptively each step: S₂ = 1.5·μ₀²·(χ₀·|H_max|)²/(4·ν_min)

### Key Material Properties (`physics/material_properties.h`)
- Double-well: F(θ) = (1/4)(θ²-1)², f(θ) = θ³-θ
- Susceptibility: χ(θ) = χ₀·H(θ/ε) where H is smoothed Heaviside
- Viscosity: ν(θ) = ν_w + (ν_f-ν_w)·H(θ/ε)
- Density: ρ(θ) = 1 + r·H(θ/ε) — only in gravity term

---

## Droplet Test Parameters (Zhang Eq 4.8) — Verified Correct
```
Domain:    [0,1]×[0,1]
IC:        Circular droplet R=0.1 at (0.5, 0.5)
ε:         2e-3
Mobility:  2e-4
χ₀:        2
μ₀:        0.1
ν_f=ν_w:   1
λ:         1
gravity:   0
ramp_slope: 1000
dt:        1e-3
max_steps: 1500
```

## Rosensweig Parameters (Zhang Eq 4.4) — Corrected in This Session
```
Domain:    [0,1]×[0,0.6]
IC:        Flat interface at y=0.2, NO perturbation
ε:         5e-3
Mobility:  2e-4
χ₀:        0.5
μ₀:        1.0
ν_f:       2.0,  ν_w: 1.0
λ:         1
r:         0.1
gravity:   (0, -6e4)
ramp_slope: 5000,  intensity_max: 8000
dt:        1e-3
max_steps: 2000
```

---

## Files Modified in These Sessions

| File | What Changed |
|------|-------------|
| `cahn_hilliard/cahn_hilliard_assemble.cc` | Removed Douglas-Dupont, added λ factors to SAV |
| `utilities/parameters.cc` | Fixed `setup_rosensweig()` — 10 parameters corrected to Zhang Eq 4.4 |
| `drivers/decoupled_driver.cc` | Fixed Rosensweig IC: y=0.2, no perturbation |
| `physics/applied_field.h` | Added `intensity_max` cap to both uniform and dipole ramp modes |

## Key Files (Read-Only Reference)

| File | Purpose |
|------|---------|
| `utilities/parameters.cc` | All presets: droplet (line ~308), square, rosensweig |
| `drivers/decoupled_driver.cc` | IC setup + time loop |
| `physics/applied_field.h` | Applied field h_a computation |
| `physics/material_properties.h` | Double-well, χ, ν, ρ |
| `cahn_hilliard/cahn_hilliard_assemble.cc` | SAV CH assembly |
| `navier_stokes/navier_stokes_assemble.cc` | NS assembly (ρ only in gravity term) |
| `magnetization/magnetization_assemble.cc` | Mag PDE (unused with algebraic M) |

---

## User Preferences
- **2 CPU cores** for all tests going forward
- Results auto-saved to `Results/` with timestamp prefix
- User opens VTK files in **ParaView** to verify visually
- Parameters must match **Zhang's paper exactly** — no combining papers
- β, spin vorticity, Maxwell stress tensor **dropped for now** (add later)

## Important User Quotes
- "NO, you are incorrect. All the papers I have seen have a flat surface and the spikes start at t=0.7"
- "beta term and spin vorticity plus maxwell tensor is dropped, so we will add them back later"
- "use only 2 cpu nodes"
- "if you want first do droplet w magnetic field. Forget rosensweig for now"
- "Earlier run was for incorrect physics, although it passed, we have to repeat it with the correct values from paper"

---

## Next Immediate Action
Run the droplet test WITHOUT magnetic field:
```bash
cd /Users/mahdi/Projects/git/PhD-Project/Decoupled/drivers/build
mpirun -np 2 ./ferrofluid_decoupled --validation droplet_nofield -r 7 --vtk_interval 100 --sav_S1 5000
```
Then verify in ParaView that the droplet stays circular.
