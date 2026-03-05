# Phase C: Novel Extensions to Two-Phase Ferrohydrodynamics

## Overview

Zhang, He & Yang (2021) — the base scheme we reproduce in Phase B — is the state
of the art for **isothermal, constant-concentration** two-phase FHD. Three natural
extensions can each yield a standalone publication while building toward a
comprehensive ferrofluid model.

| Extension | New Physics | New PDEs | Novelty | Impact |
|-----------|------------|----------|---------|--------|
| **C.0 Parametric** | Systematic parameter sweeps | None (existing code) | First chi_0 > 1 onset map | Fills analytical gap |
| **C.1 Thermal** | Temperature-dependent magnetization | Heat equation | Energy-stable thermo-FHD | Cooling, hyperthermia |
| **C.2 Concentration** | Nanoparticle migration (magnetophoresis) | Concentration transport | Feedback instability | Separation, stability |
| **C.3 Unified** | All of the above + cross-coupling | Heat + Concentration | Grand unified model | Thesis capstone |

### Current system (Phase B)

| # | Equation | Unknown | Type |
|---|----------|---------|------|
| 1 | Cahn-Hilliard (phase field) | theta, psi | CG Q2 |
| 2 | Navier-Stokes (momentum + pressure) | u, p | CG Q2 + DG P1 |
| 3 | Magnetostatic Poisson | phi | CG Q2 |
| 4 | Magnetization transport | M | DG Q1 |

---

# C.1 — Thermomagnetic Two-Phase Ferrofluid Flows

## 1.1 Motivation

Real ferrofluids are almost never isothermal. The magnetization of ferrofluid
nanoparticles drops sharply with temperature (pyromagnetic effect), creating a
unique body force: ferrofluid flows spontaneously from hot to cold regions in the
presence of a magnetic field. This **thermomagnetic convection** is exploited in
engineering applications but has no rigorous numerical framework for two-phase flows.

**Gap:** No provably energy-stable, fully decoupled scheme exists for two-phase
thermo-ferrohydrodynamic flows.

## 1.2 New Equation: Heat / Energy

    rho * c_p * (dT/dt + u . grad(T)) = div(k(Phi) * grad(T)) + Q_visc + Q_mag

where:
- T is temperature
- k(Phi) = k_f * Phi + k_w * (1 - Phi) is phase-dependent thermal conductivity
- Q_visc = 2 * nu(Phi) * |D(u)|^2 is viscous dissipation (often negligible)
- Q_mag = (mu_0 / tau) * |M - chi(Phi,T) * H|^2 is magnetic relaxation heating

Boundary conditions: dT/dn = 0 (insulated) or T = T_wall (isothermal walls).

## 1.3 Temperature-Dependent Constitutive Laws

The key physical coupling is through temperature-dependent magnetization:

    chi(Phi, T) = chi_0 * Phi * g(T)

where g(T) models the pyromagnetic effect. Common choices:

1. **Linear model:** g(T) = 1 - beta_T * (T - T_ref)
   - Simple, valid for small temperature variations
   - beta_T = pyromagnetic coefficient (~1e-3 to 1e-2 per K for typical ferrofluids)

2. **Langevin model:** g(T) = coth(alpha/T) - T/alpha
   - More physical, captures saturation
   - alpha = mu_0 * m_p * H / k_B where m_p is particle magnetic moment

Additional temperature dependencies:
- Viscosity: nu(Phi, T) = nu(Phi) * exp(A * (1/T - 1/T_ref))  [Arrhenius]
- Surface tension: lambda(T) = lambda_0 * (1 - gamma_T * (T - T_ref))  [Marangoni]

## 1.4 Extended System (6 equations)

| # | Equation | Unknown | Coupling |
|---|----------|---------|----------|
| 1 | Cahn-Hilliard | theta, psi | u (advection), lambda(T) |
| 2 | Navier-Stokes | u, p | theta (viscosity, capillary), M,H (Kelvin), T (buoyancy, Marangoni) |
| 3 | Poisson | phi | M (source), theta (chi) |
| 4 | Magnetization | M | u (transport), H (relaxation), theta,T (chi) |
| 5 | **Heat** | **T** | **u (advection), Phi (conductivity), M,H (heating source)** |

## 1.5 Numerical Scheme

Extend Zhang's decoupled Gauss-Seidel splitting to 6 substeps:

    Given (theta^n, u^n, M^n, phi^n, T^n):

    Step 1: Solve Cahn-Hilliard     -> theta^{n+1}, psi^{n+1}
    Step 2: Solve Heat equation      -> T^{n+1}
    Step 3: Solve Navier-Stokes      -> u^{n+1}, p^{n+1}
    Step 4: Solve Magnetization      -> M^{n+1}
    Step 5: Solve Poisson            -> phi^{n+1}
    Step 6: Update SAV variable      -> r^{n+1}

Heat equation placed after CH (needs theta^{n+1} for k(Phi)) and before NS
(provides T^{n+1} for buoyancy and thermomagnetic force).

**Discretization:** CG Q2 elements (same as theta, phi). Semi-implicit:
- Advection: explicit (use u^n)
- Diffusion: implicit (k(Phi^{n+1}) * Laplacian)
- Source terms: explicit (use M^n, H^n)

**Energy stability:** Total energy extends to E = E_mix + E_kin + E_mag + E_thermal.
Strategy:
- Lag chi(T^n) explicitly in magnetization equation
- Add stabilization S_T * |T^n - T^{n-1}|^2
- Thermal dissipation k * ||grad(T)||^2 provides natural damping
- Expected: unconditional stability under CFL-independent condition on S_T

## 1.6 Competing Interface Mechanisms

### Thermocapillary (Marangoni) effect
- Surface tension depends on temperature: sigma = sigma(T)
- Temperature gradient along interface -> tangential stress -> flow
- Drives flow from hot to cold along the interface
- Enters through: lambda(T) in Cahn-Hilliard, or as Marangoni stress in NS

### Thermomagnetic effect
- Susceptibility depends on temperature: chi = chi(T)
- Hot ferrofluid has lower chi -> weaker magnetic force
- Cold ferrofluid is pulled toward high-field regions
- Enters through: Kelvin force mu * (M . grad)H with M = chi(Phi,T) * H

**Novel physics:** When both effects are present, they can **cooperate or compete**
depending on geometry:
- Heated droplet in field: Marangoni spreads, thermomagnetic compresses/elongates
- Rosensweig instability + bottom heating: thermal gradient modifies spike pattern

## 1.7 Proposed Experiments

1. **Validation: Thermomagnetic Rayleigh-Benard convection**
   - Single-phase limit (Phi = 1), bottom-heated cavity + vertical field
   - Compare with analytical critical Rayleigh number (Finlayson 1970)

2. **Benchmark: Heated ferrofluid droplet in uniform field**
   - Circular droplet + uniform field + temperature gradient
   - Study equilibrium shape vs Bo_m, Ma, beta_T
   - Quantify thermal correction to isothermal droplet deformation

3. **Application: Thermomagnetic droplet manipulation**
   - Droplet in microchannel + non-uniform field + localized heating
   - Demonstrate thermal-magnetic steering for lab-on-chip

4. **Application: Rosensweig instability with thermal gradient**
   - Bottom heating modifies critical field, spike height, wavelength
   - Compare isothermal vs thermomagnetic results

5. **Application: Ferrofluid thermosiphon cooling**
   - Closed cavity, ferrofluid on hot surface, vertical field
   - Quantify Nusselt number enhancement vs field strength

## 1.8 Implementation Plan

**New files:**
- `heat/heat.h` — HeatSubsystem class (facade)
- `heat/heat_setup.cc` — DoF distribution, sparsity, vectors
- `heat/heat_assemble.cc` — Weak form assembly (advection-diffusion)
- `heat/heat_solve.cc` — CG solver with AMG preconditioner
- `heat/heat_output.cc` — VTK output
- `heat/tests/heat_mms.h` — MMS exact solutions and sources
- `heat/tests/heat_mms_test.cc` — Standalone convergence test

**Modified files:**
- `physics/material_properties.h` — Add chi(Phi, T), nu(Phi, T), k(Phi)
- `drivers/decoupled_driver.cc` — Add heat substep to time loop
- `utilities/parameters.h` — Add thermal parameters (k_f, k_w, c_p, beta_T, ...)
- `navier_stokes/navier_stokes_assemble.cc` — Add buoyancy + thermomagnetic force
- `magnetization/magnetization_assemble.cc` — Use chi(Phi, T) instead of chi(Phi)

**Verification sequence:**
1. Standalone heat MMS (advection-diffusion, no coupling)
2. Heat + NS coupled MMS (buoyancy-driven flow)
3. Heat + Poisson + Mag coupled (thermomagnetic force)
4. Full 6-system coupled MMS
5. Rayleigh-Benard validation (single-phase limit)
6. Heated droplet benchmark

## 1.9 Expected Contribution

**Title (working):** "Energy-stable fully decoupled scheme for two-phase
thermomagnetic ferrofluid flows"

**Novel contributions:**
1. First provably energy-stable, fully decoupled scheme for two-phase
   thermo-ferrohydrodynamic flows (6 equations, 6 substeps)
2. Rigorous energy stability proof with temperature-dependent chi(Phi, T)
3. Systematic study of competing thermocapillary and thermomagnetic effects
4. Quantitative benchmarks for heated ferrofluid droplet deformation

**Target journals:** J. Comput. Physics, CMAME, SIAM J. Sci. Comput.

**Timeline:** 14-22 weeks

| Phase | Task | Duration |
|-------|------|----------|
| C.1a | Heat subsystem + standalone MMS | 2-3 weeks |
| C.1b | Couple to NS (buoyancy) + coupled MMS | 1-2 weeks |
| C.1c | Couple to Mag (chi(T)) + full MMS | 2-3 weeks |
| C.1d | Energy stability proof | 2-4 weeks |
| C.1e | Validation experiments | 3-4 weeks |
| C.1f | Paper writing | 4-6 weeks |

---

# C.2 — Concentration-Dependent Ferrofluid with Magnetophoresis

## 2.1 Motivation

Standard ferrofluid models assume **uniform nanoparticle concentration**. In reality,
particles migrate due to:
- **Brownian diffusion** (D) — random thermal motion
- **Magnetophoresis** (D_m) — particles drift toward stronger magnetic field
- **Thermophoresis / Soret effect** (D_T S_T) — particles drift in temperature gradients

This migration changes the local susceptibility chi(c), creating a **feedback loop**:
field gradient -> particles concentrate -> chi increases locally -> field distorted ->
more migration. This is a known instability mechanism observed experimentally but
never modeled with a phase-field approach and energy-stable numerics.

**Gap:** No phase-field FHD model includes nanoparticle concentration dynamics with
provable energy stability. Existing models (Rosensweig 1985, Odenbach 2002) are
either analytical/1D or use ad-hoc numerics.

## 2.2 New Equation: Nanoparticle Concentration Transport

    dc/dt + u . grad(c) = div( D * grad(c)
                              + D_m * c * grad(|H|)
                              + D_T * S_T * c * (1-c) * grad(T) )

where:
- c is volume fraction of magnetic nanoparticles (0 <= c <= 1)
- D is Brownian diffusion coefficient
- D_m is magnetophoretic mobility: D_m = (mu_0 * m_p * V_p) / (6 * pi * eta * r_p)
  - m_p = particle magnetic moment, V_p = particle volume, r_p = radius, eta = viscosity
- D_T * S_T is Soret coefficient (thermophoresis, only if heat equation is present)

Boundary conditions: zero-flux (no particles leave domain).

**Key feature:** The magnetophoretic term D_m * c * grad(|H|) is **nonlinear** and
creates the feedback instability. Particles accumulate where |H| is large, which
increases local chi, which further concentrates the field.

## 2.3 Concentration-Dependent Constitutive Laws

    chi(Phi, c) = chi_0 * Phi * (c / c_ref)

where c_ref is the reference (initial uniform) concentration. Additionally:
- Viscosity: nu(Phi, c) = nu_w + (nu_f(c) - nu_w) * Phi
  - nu_f(c) follows Einstein or Krieger-Dougherty model for suspensions
- Density: rho(Phi, c) = rho_w + (rho_f(c) - rho_w) * Phi

## 2.4 Extended System (6 equations)

| # | Equation | Unknown | Coupling |
|---|----------|---------|----------|
| 1 | Cahn-Hilliard | theta, psi | u (advection) |
| 2 | Navier-Stokes | u, p | theta (viscosity, capillary), M,H (Kelvin), c (density) |
| 3 | Poisson | phi | M (source), theta,c (chi) |
| 4 | Magnetization | M | u (transport), H (relaxation), theta,c (chi) |
| 5 | **Concentration** | **c** | **u (advection), H (magnetophoresis), Phi (confinement)** |

## 2.5 Numerical Scheme

    Given (theta^n, u^n, M^n, phi^n, c^n):

    Step 1: Solve Cahn-Hilliard       -> theta^{n+1}, psi^{n+1}
    Step 2: Solve Concentration        -> c^{n+1}
    Step 3: Solve Navier-Stokes        -> u^{n+1}, p^{n+1}
    Step 4: Solve Magnetization        -> M^{n+1}
    Step 5: Solve Poisson              -> phi^{n+1}
    Step 6: Update SAV variable        -> r^{n+1}

**Discretization options for concentration:**
- **CG Q2** with SUPG: simple, works if magnetophoretic Peclet < O(10)
- **DG Q1** (like magnetization): better for sharp concentration fronts, naturally
  conservative, handles the nonlinear flux D_m * c * grad(|H|)

DG is recommended because c can develop sharp gradients near high-field regions.

**Energy stability challenge:** The magnetophoretic term D_m * c * grad(|H|) is
nonlinear. Key issues:
- grad(|H|) is not smooth (|H| has corners where H = 0)
- The feedback chi(c) -> H -> grad(|H|) -> c creates a potential blow-up
- Strategy: linearize around c^n, add stabilization S_c * |c^n - c^{n-1}|^2
- Need to prove the dissipation D * ||grad(c)||^2 controls the magnetophoretic source

## 2.6 Novel Physics: Magnetophoretic Instability

The feedback loop chi(c) -> H -> c is a genuine **instability mechanism**:
1. Small perturbation in c creates local chi variation
2. Field concentrates where chi is higher
3. Stronger field gradient drives more particles there (magnetophoresis)
4. c increases further -> positive feedback

This is stabilized only by Brownian diffusion. The balance gives a characteristic
**critical magnetic field** above which concentration becomes non-uniform:

    H_crit ~ sqrt(D * k_B * T / (mu_0 * m_p * V_p * c_0))

Below H_crit: uniform concentration (stable). Above: concentration spikes form.

**Two-phase aspect:** In a ferrofluid droplet, particles accumulate at the poles
(where H is strongest), creating an anisotropic susceptibility that modifies the
droplet shape beyond what isothermal, uniform-concentration models predict.

## 2.7 Proposed Experiments

1. **Validation: 1D magnetophoresis in a tube**
   - Ferrofluid in a tube, non-uniform field applied externally
   - Compare concentration profile with analytical steady-state solution
   - Validates the magnetophoretic flux implementation

2. **Benchmark: Concentration instability onset**
   - Uniform ferrofluid layer, increasing vertical field
   - Measure when concentration becomes non-uniform vs H_crit
   - Compare with linear stability analysis

3. **Application: Ferrofluid droplet with concentration feedback**
   - Circular droplet in uniform field (same as Zhang Section 4.5)
   - With concentration dynamics: particles accumulate at poles
   - Compare droplet aspect ratio with/without concentration effects
   - Shows correction to the isothermal Bo_m curve (Fig 4.16 of Zhang)

4. **Application: Magnetic nanoparticle separation**
   - Two-phase system: ferrofluid injected into non-magnetic carrier
   - Non-uniform field collects particles in target region
   - Study separation efficiency vs field parameters
   - Relevant to environmental remediation and biomedical applications

5. **Application: Long-term ferrofluid stability**
   - Ferrofluid at rest in a non-uniform field (e.g., near a permanent magnet)
   - Simulate hours/days of magnetophoretic drift
   - Predict when/where particle sedimentation occurs
   - Practical for ferrofluid seal and bearing design

## 2.8 Implementation Plan

**New files:**
- `concentration/concentration.h` — ConcentrationSubsystem class
- `concentration/concentration_setup.cc` — DG Q1 setup
- `concentration/concentration_assemble.cc` — DG transport + magnetophoretic flux
- `concentration/concentration_solve.cc` — GMRES solver
- `concentration/concentration_output.cc` — VTK output
- `concentration/tests/concentration_mms.h` — MMS with known magnetophoretic field
- `concentration/tests/concentration_mms_test.cc` — Convergence test

**Modified files:**
- `physics/material_properties.h` — Add chi(Phi, c), nu(Phi, c)
- `drivers/decoupled_driver.cc` — Add concentration substep
- `utilities/parameters.h` — Add D, D_m, c_ref parameters
- `magnetization/magnetization_assemble.cc` — Use chi(Phi, c)

**Verification sequence:**
1. Standalone concentration MMS (advection-diffusion, prescribed H)
2. Concentration + Poisson coupled (magnetophoretic feedback)
3. Full 6-system coupled MMS
4. 1D magnetophoresis validation
5. Concentration instability benchmark

## 2.9 Expected Contribution

**Title (working):** "Phase-field model for two-phase ferrofluid flows with
nanoparticle concentration dynamics and magnetophoresis"

**Novel contributions:**
1. First phase-field FHD model incorporating nanoparticle transport with
   magnetophoretic feedback
2. Energy stability analysis for the concentration-coupled system
3. Numerical characterization of the magnetophoretic instability in two-phase flows
4. Quantitative prediction of concentration-dependent droplet deformation

**Target journals:** J. Comput. Physics, Physics of Fluids, J. Magn. Magn. Mater.

**Timeline:** 12-18 weeks (builds on C.1 infrastructure)

| Phase | Task | Duration |
|-------|------|----------|
| C.2a | Concentration subsystem (DG) + standalone MMS | 2-3 weeks |
| C.2b | Couple to Poisson/Mag (chi(c) feedback) | 2-3 weeks |
| C.2c | Magnetophoretic instability analysis | 2-3 weeks |
| C.2d | Validation + application experiments | 3-4 weeks |
| C.2e | Paper writing | 3-5 weeks |

---

# C.3 — Grand Unified Model: Thermo-Magneto-Diffusive Two-Phase Flows

## 3.1 Motivation

The ultimate goal: a comprehensive two-phase ferrofluid model that captures ALL
the dominant physics simultaneously. This is the **thesis capstone** — combining
C.1 (thermal) and C.2 (concentration) into a single framework with full
cross-coupling.

No such model exists in the literature. The closest works are:
- Rosensweig (1985): analytical treatment, single-phase, simplified geometry
- Odenbach (2002): experimental focus, no rigorous numerics
- Nochetto et al. (2016): two-phase FHD, but isothermal, no concentration

## 3.2 Full System (7 equations, 7 unknowns)

| # | Equation | Unknown | Type |
|---|----------|---------|------|
| 1 | Cahn-Hilliard | theta, psi | CG Q2 |
| 2 | Heat | T | CG Q2 |
| 3 | Concentration | c | DG Q1 |
| 4 | Navier-Stokes | u, p | CG Q2 + DG P1 |
| 5 | Magnetization | M | DG Q1 |
| 6 | Poisson | phi | CG Q2 |
| 7 | SAV update | r | scalar |

**Master constitutive law:**

    chi = chi(Phi, T, c) = chi_0 * Phi * g(T) * (c / c_ref)

All material properties depend on (Phi, T, c):
- nu(Phi, T, c) — viscosity (phase + Arrhenius + Einstein)
- k(Phi, T) — thermal conductivity
- rho(Phi, c) — density
- lambda(T) — surface tension (Marangoni)

## 3.3 Cross-Coupling Map

                    theta
                   / | \ \
                  /  |  \ \
               nu  lambda chi  k
                |    |    |    |
                v    v    v    v
    c -----> chi -----> H -----> Kelvin force
    ^         |                      |
    |     grad(|H|)                  v
    |         |                     u -----> advects theta, c, T, M
    |         v                     ^
    +--- magnetophoresis            |
              |                  buoyancy (T)
              |                  Marangoni (T)
              v
         concentration
              |
              +-----> Soret (needs T)

**Three feedback loops:**
1. chi(T) -> H -> Kelvin force -> u -> advects T (thermomagnetic)
2. chi(c) -> H -> grad(|H|) -> magnetophoresis -> c (concentration instability)
3. T -> Soret -> c -> chi -> H -> heating -> T (thermo-diffusive)

## 3.4 Time-Stepping: 7 Decoupled Substeps

    Given all fields at t^n:

    Step 1: Cahn-Hilliard      -> theta^{n+1}, psi^{n+1}
    Step 2: Heat               -> T^{n+1}
    Step 3: Concentration      -> c^{n+1}
    Step 4: Navier-Stokes      -> u^{n+1}, p^{n+1}
    Step 5: Magnetization      -> M^{n+1}
    Step 6: Poisson            -> phi^{n+1}
    Step 7: SAV update         -> r^{n+1}

**Energy stability:** The full energy functional:

    E = E_mix(theta) + E_kin(u) + E_mag(H, M)
      + E_thermal(T) + E_concentration(c)

where E_concentration = D * ||grad(c)||^2 (regularized).

The proof requires controlling ALL cross-coupling terms simultaneously.
Stabilization constants: S1 (CH), S_T (heat), S_c (concentration), S2 (NS/mag).

## 3.5 Proposed Experiments

1. **Heated ferrofluid droplet with particle migration**
   - Droplet in field + thermal gradient
   - Particles migrate due to BOTH magnetophoresis and thermophoresis (Soret)
   - Triple coupling: shape (CH) + temperature (heat) + concentration (transport)
   - No existing model captures all three simultaneously

2. **Rosensweig instability: full physics**
   - Uniform Rosensweig (Section 4.3) with heating AND concentration dynamics
   - How do concentration gradients modify spike formation?
   - Does particle sedimentation eventually destroy the spikes?

3. **Ferrofluid-based heat exchanger optimization**
   - Two-phase ferrofluid in a channel with hot walls and magnetic field
   - Optimize field configuration for maximum heat transfer
   - Account for particle migration (which degrades performance over time)
   - Direct engineering application

4. **Magnetic hyperthermia in two-phase medium**
   - AC field: add oscillating h_a(t) = H_0 * sin(omega * t)
   - Heating source: Q = mu_0 * omega * chi'' * H_0^2
   - Two-phase: ferrofluid (injected) vs tissue (carrier)
   - Study temperature distribution, particle migration under AC field
   - Biomedical application: cancer treatment planning

## 3.6 Expected Contribution

**Title (working):** "A comprehensive phase-field framework for thermo-magneto-
diffusive two-phase ferrofluid flows: theory, numerics, and applications"

**Novel contributions:**
1. Most complete two-phase ferrofluid model in the literature (7 coupled PDEs)
2. Provably energy-stable, fully decoupled scheme (7 substeps)
3. First numerical study of combined thermomagnetic + magnetophoretic effects
   in two-phase ferrofluid systems
4. Engineering-relevant applications: heat exchangers, hyperthermia, droplet control

**Target:** Journal paper (JCP/CMAME) or thesis chapter

**Timeline:** 8-12 weeks (incremental over C.1 + C.2)

| Phase | Task | Duration |
|-------|------|----------|
| C.3a | Integrate C.1 + C.2 into unified driver | 2-3 weeks |
| C.3b | Full 7-system coupled MMS | 2-3 weeks |
| C.3c | Cross-coupling experiments | 2-3 weeks |
| C.3d | Paper/thesis chapter | 2-3 weeks |

---

# C.0 — Parametric Study of Rosensweig Instability

## 0.1 Motivation

Nochetto, Salgado & Tomas (2016) explicitly note that a parametric/sensitivity
study of the Rosensweig instability "would be highly desirable, but that would
involve an ambitious separate analysis." They also point out that **no analytical
results exist for highly paramagnetic ferrofluids (chi_0 > 1)**, and effects
related to the demagnetizing field are poorly understood.

Our Phase B code can fill this gap immediately — no new equations needed, just
systematic parameter sweeps. This is the **lowest-hanging fruit** and can be run
on HPC with many simulations in parallel.

**Gap:** No systematic numerical parametric study of two-phase Rosensweig instability
exists across the full (chi_0, alpha, lambda, depth) parameter space, especially
for chi_0 > 1.

## 0.2 Analytical Predictions (Linear Stability Theory)

Before running simulations, we derive testable predictions from the governing
equations. This allows **theory-first validation**: run the simulation, compare
against the analytical scaling, and check whether deviations reveal nonlinear or
finite-geometry effects.

### Dispersion relation

Perturbing a flat ferrofluid interface eta(x,t) = eta_0 exp(ikx + omega*t) under
a normal magnetic field H_0, the three body forces from our NS equation give:

    omega^2(k) = -Delta_rho * g * k  -  sigma * k^3  +  mu_0 * f(chi_0) * H_0^2 * k^2

where f(chi_0) = chi_0^2 / (2 + chi_0) and sigma = lambda * sqrt(2)/3 is the
surface tension from the CH energy E = int (lambda*eps/2)|grad(theta)|^2 + (lambda/eps)*F(theta).

Setting omega = 0 and minimizing over k:

    Critical wavenumber:  k_c = sqrt(Delta_rho * g / sigma)
    Capillary length:     l_c = 1/k_c = sqrt(sigma / (Delta_rho * g))
    Critical wavelength:  lambda_c = 2*pi / k_c
    Critical field:       H_c^2 = 2*(2+chi_0) * sqrt(Delta_rho * g * sigma) / (mu_0 * chi_0^2)

### Prediction 1: chi_0 controls onset and height, NOT spike count

    H_c  proportional to  sqrt(2 + chi_0) / chi_0

k_c depends only on (g, sigma) — not on chi_0. So:

| chi_0 | sqrt(2+chi_0)/chi_0 | H_c ratio vs baseline |
|-------|---------------------|-----------------------|
| 0.5   | 3.162               | 1.00                  |
| 0.9   | 1.892               | 0.60                  |
| 2.0   | 1.000               | 0.32                  |
| 5.0   | 0.529               | 0.17                  |

Testable predictions:
- chi_0 = 0.5 -> 0.9: critical field drops ~40%, spikes appear earlier in ramp
- chi_0 = 0.5 -> 2.0: critical field drops ~68%, onset at ~1/3 the field
- Spike count is INDEPENDENT of chi_0
- At fixed H > H_c, higher chi_0 -> taller spikes (larger supercriticality)

### Prediction 2: lambda controls spike count and spacing

Since sigma = lambda * sqrt(2)/3:

    Spike spacing:  lambda_c  proportional to  sqrt(lambda)
    Spike count:    N = L_x / lambda_c  proportional to  1/sqrt(lambda)
    Critical field: H_c  proportional to  lambda^{1/4}

| lambda_theta | N / N_baseline | H_c / H_c,baseline | Effect                             |
|-------------|----------------|---------------------|------------------------------------|
| 0.125 (half)| 1.41           | 0.84                | 41% more spikes, 16% easier onset  |
| 0.25 (base) | 1.00           | 1.00                | Baseline                           |
| 0.50 (2x)   | 0.71           | 1.19                | 29% fewer spikes, 19% harder onset |
| 1.00 (4x)   | 0.50           | 1.41                | Half as many spikes, 41% harder    |

### Prediction 3: Domain width L_x selects spike count (quantized modes)

In a finite domain, allowed wavenumbers are quantized: k_n = 2*pi*n / L_x.
The system picks whichever k_n is closest to k_c, so:

    N = L_x / lambda_c  (linear scaling)

| L_x | N (predicted) | Note                                    |
|-----|---------------|-----------------------------------------|
| 0.5 | ~2-3          | May suppress instability if lambda_c > L_x |
| 0.75| ~3-4          | Mode competition possible               |
| 1.0 | ~5 (baseline) | Baseline                                |
| 1.5 | ~7-8          | Expect proportional increase            |
| 2.0 | ~10           | Verify N proportional to L_x            |

Below a critical width L_x < lambda_c, the most unstable mode does not fit and
spikes are completely suppressed. This is testable.

### Summary: parameter-to-observable map

| Parameter   | Spike count | Spike height      | Critical onset |
|-------------|------------|-------------------|----------------|
| chi_0 up    | unchanged  | taller            | earlier        |
| lambda up   | fewer      | taller (per spike)| later          |
| g up        | more       | shorter           | later          |
| Delta_rho up| more       | shorter           | later          |
| L_x up      | more (linear) | unchanged      | unchanged      |

Key insight: chi_0 and lambda have cleanly separated effects — chi_0 controls
the magnetic response (onset, height) while lambda controls the geometric pattern
(count, spacing). This makes them independently verifiable.

---

## 0.3 Baseline Parameters (Zhang Section 4.3)

    chi_0 = 0.5, alpha_max = 8000, lambda = 1, y_interface = 0.2
    epsilon = 5e-3, nu_f = 2, nu_w = 1, r = 0.1, g = 6e4
    Domain [0,1] x [0,0.6], 5 dipoles at y=-15, flat arrangement
    Refinement r=4, dt = 1e-3, max_steps = 2000

---

## 0.4 Tier 1 — Single-Parameter Screening (33 runs)

**Goal:** Vary one parameter at a time, all others at baseline. Identify which
parameters most strongly affect onset, spike count, and spike height. Results
from Tier 1 inform which parameter pairs to cross in Tier 2.

### 1A. Physical parameters

| ID   | Parameter       | Symbol    | Baseline | Values                                          | Runs |
|------|-----------------|-----------|----------|-------------------------------------------------|------|
| 1A-1 | Susceptibility  | chi_0     | 0.5      | 0.1, 0.25, **0.5**, 0.75, 1.0, 1.5, 2.0       | 7    |
| 1A-2 | Field strength  | alpha_max | 8000     | 5000, 7000, **8000**, 9000, 10000               | 5    |
| 1A-3 | Surface tension | lambda    | 1.0      | 0.5, 0.75, 0.9, **1.0**, 1.1, 1.25, 1.5, 2.0  | 8    |
| 1A-4 | Pool depth      | y_interface| 0.2     | 0.05, 0.1, **0.2**, 0.4                        | 4    |
| 1A-5 | Interface width | epsilon   | 5e-3     | 3e-3, **5e-3**, 7e-3, 1e-2                     | 4    |

### 1B. Geometric parameters

| ID   | Parameter       | Symbol    | Baseline | Values                       | Runs |
|------|-----------------|-----------|----------|------------------------------|------|
| 1B-1 | Domain width    | L_x       | 1.0      | 0.5, 0.75, **1.0**, 1.5, 2.0| 5    |
| 1B-2 | Domain height   | L_y       | 0.6      | 0.4, **0.6**, 1.0            | 3    |

### 1C. Demagnetizing field comparison

| ID   | Parameter       | Symbol             | Baseline | Values       | Runs |
|------|-----------------|--------------------|----------|--------------|------|
| 1C-1 | Field model     | use_reduced_field  | false    | false, true  | 2    |

h = h_a + h_d (full, default) vs h = h_a (reduced, Nochetto approach). Tests
the demagnetizing field correction on spike formation. Expected: reduced field
over-predicts spike height (no self-demagnetization to oppose growth).

### Tier 1 summary

**31 runs** (baseline counted once; subtract duplicates) x 4 core-hours = **124 core-hours**
On HPC (31 runs x 4 cores): ~4 hours wall time.

**CLI needed:** `--chi0`, `--alpha_max`, `--lambda`, `--y_interface`, `--epsilon`,
`--Lx`, `--Ly`, `--reduced_field`

**Exit criterion:** Tier 1 analysis done. Identify top-3 most influential
parameters, confirm analytical predictions (Section 0.2), proceed to Tier 2.

---

## 0.5 Tier 2 — 2D Interaction Matrices (91 runs)

**Goal:** Cross the most influential parameters pairwise. Maps onset boundaries,
reveals parameter interactions not visible from one-at-a-time sweeps.

**Prerequisite:** Tier 1 complete and analyzed.

### Matrix A: chi_0 x alpha (susceptibility vs field strength) — 30 runs

The most important sweep. Maps the **onset boundary** for each chi_0.

                  alpha = 2000   4000   8000   16000   32000
    chi_0 = 0.1     .       .       .       .       .
    chi_0 = 0.25    .       .       .       .       .
    chi_0 = 0.5     .       .      (base)   .       .
    chi_0 = 1.0     .       .       .       .       .
    chi_0 = 2.0     .       .       .       .       .
    chi_0 = 5.0     .       .       .       .       .

**Maps uncharted territory for chi_0 > 1.**

### Matrix B: lambda x alpha (surface tension vs field) — 20 runs

Does higher lambda suppress spikes or just change their wavelength?

                  alpha = 2000   4000   8000   16000
    lambda = 0.1     .       .       .       .
    lambda = 0.5     .       .       .       .
    lambda = 1.0     .       .      (base)   .
    lambda = 5.0     .       .       .       .
    lambda = 10.0    .       .       .       .

### Matrix C: chi_0 x pool_depth (susceptibility vs finite-depth) — 20 runs

Does finite pool depth change the instability threshold at high chi_0?

                  y_if = 0.05   0.1    0.2    0.4
    chi_0 = 0.1     .       .       .       .
    chi_0 = 0.5     .       .      (base)   .
    chi_0 = 1.0     .       .       .       .
    chi_0 = 2.0     .       .       .       .
    chi_0 = 5.0     .       .       .       .

### Matrix D: curvature x chi_0 (dipole geometry vs susceptibility) — 12 runs

Does higher susceptibility amplify nonuniformity from curved magnets?

                  curvature = concave(5)  flat(inf)  convex(5)
    chi_0 = 0.5        .           (base)       .
    chi_0 = 1.0        .             .          .
    chi_0 = 2.0        .             .          .
    chi_0 = 5.0        .             .          .

### Matrix E: curvature x lambda (dipole geometry vs surface tension) — 9 runs

Does surface tension smooth out asymmetric spike patterns from curved magnets?

                  curvature = concave(5)  flat(inf)  convex(5)
    lambda = 0.1       .            .          .
    lambda = 1.0       .          (base)       .
    lambda = 5.0       .            .          .

### Tier 2 summary

**91 runs** (subtract overlaps with Tier 1 and baseline) x 4 core-hours = **364 core-hours**
On HPC (91 runs x 4 cores): ~4 hours wall time.

**CLI needed (additional):** `--dipole_curve concave R` / `--dipole_curve convex R`

**Exit criterion:** Onset boundary mapped for chi_0 x alpha. Interaction effects
quantified. Scaling laws extracted. Proceed to Tier 3 if three-way interactions
are suspected from Tier 2 results.

---

## 0.6 Tier 3 — Three-Way Tensor + Dipole Geometry (38 runs)

**Goal:** Resolve three-way parameter interactions and fully characterize the
novel dipole geometry effect. Only run if Tier 2 suggests interactions exist.

**Prerequisite:** Tier 2 complete and analyzed.

### 3A. Three-way tensor: chi_0 x alpha x lambda — 27 runs

Does the critical field depend on surface tension differently at high chi_0?

    chi_0  = {0.5, 2.0, 5.0}
    alpha  = {4000, 8000, 16000}
    lambda = {0.5, 1.0, 5.0}

### 3B. Dipole arrangement geometry sweeps — 11 runs

Curved magnet arrangements break translational symmetry, creating spatially
varying field across the interface. Relevant to real applications:
- MRI contrast agent manipulation (curved magnets)
- Ferrofluid seals (concentric magnets)
- Lab-on-chip (curved microchannels + magnets)

**Dipole configurations (5 dipoles, varying y-position):**

| Config          | Dipole y-positions                           | Expected result              |
|-----------------|----------------------------------------------|------------------------------|
| Flat (baseline) | y_i = -15 for all i                          | 5 equal-height spikes        |
| Concave (U)     | y_i = -15 + R - sqrt(R^2 - (x_i - 0.5)^2)  | Central spikes taller        |
| Convex (cap)    | y_i = -15 - R + sqrt(R^2 - (x_i - 0.5)^2)  | Edge spikes taller/uniform   |
| Deep-V          | y_i = -15 + slope * |x_i - 0.5|              | Single dominant central spike|

Curvature radius R is continuous (R -> inf recovers flat):

    Concave: R = {5, 10, 15, 50, inf}     -> 4 new + baseline
    Convex:  R = {5, 10, 15, 50, inf}     -> 4 new (baseline shared)
    Deep-V:  slope = {5, 10, 20}          -> 3 new

**Physics:** |H(x)| ~ alpha / |r_dipole(x)|^2. Concave focuses field at center
(taller central spike). Convex focuses at edges. Deep-V isolates one dominant spike.

**CLI needed (additional):** `--dipole_curve deepv slope`

### Tier 3 summary

**38 runs** x 4 core-hours = **152 core-hours**
On HPC: ~4 hours wall time.

**Exit criterion:** All 160 runs complete. Full parameter space characterized.
Ready for post-processing and paper write-up.

---

## 0.7 All-Tiers Run Summary

| Tier | Description                          | Runs | Core-hours | Prerequisite |
|------|--------------------------------------|------|------------|--------------|
| 1    | Single-parameter screening           | 31   | 124        | CLI flags    |
| 2    | 2D interaction matrices (A-E)        | 91   | 364        | Tier 1 done  |
| 3    | 3-way tensor + dipole geometry       | 38   | 152        | Tier 2 done  |
| **Total** |                               | **160** | **640** |              |

On HPC (all tiers run independently, 4 cores each): **~4 hours wall time per tier**

---

## 0.8 Measured Quantities

For each simulation, extract from diagnostics.csv and VTK output:

| Metric | How measured | What it tells us |
|--------|-------------|-----------------|
| **Spike count** | Count peaks in theta along y=y_interface at final time | Pattern wavelength |
| **Max spike height** | Maximum y where theta > 0 at final time | Instability strength |
| **Onset time** | First time E_CH exceeds 1.05 x E_CH(0) | Critical condition |
| **Wavelength** | FFT of theta along x at interface height | Pattern selection |
| **Max velocity** | Peak |U| over entire simulation | Flow intensity |
| **Steady state** | Is d(E_CH)/dt < tolerance at t_final? | Equilibrium reached? |
| **Spike morphology** | Sharp tips vs rounded (curvature at peaks) | Capillary effects |

## 0.9 Post-Processing and Visualization

Python script to generate:
1. **Phase diagram** (chi_0 vs alpha): color = spike/no-spike, contour = onset boundary
2. **Heatmaps**: spike height as function of (chi_0, alpha), (lambda, alpha), etc.
3. **Scaling plots**: spike height vs magnetic Bond number Bo_m = mu * H^2 / (lambda * kappa)
4. **Morphology gallery**: representative theta snapshots across parameter space
5. **Energy evolution curves**: E_CH(t) for different parameter sets overlaid
6. **Domain width scaling**: N vs L_x plot — verify linear proportionality, identify
   minimum width for instability onset
7. **Dipole geometry gallery**: side-by-side theta snapshots for flat / concave / convex /
   Deep-V at matched field strength — shows spike-height envelope controlled by magnet shape
8. **Spike height envelope plots**: for each curvature R, plot spike height vs x-position
   along the interface — reveals how magnet geometry imprints on the surface pattern
9. **Theory vs simulation overlay**: plot measured spike count and H_c against analytical
   predictions (Section 0.2) — quantify regime of validity for linear stability theory

## 0.10 Expected Results and Novelty

1. **Onset boundary for chi_0 > 1**: First numerical mapping of the critical field
   strength in the highly paramagnetic regime. Analytical theory (Cowley & Rosensweig
   1967) only covers chi_0 << 1.

2. **Finite-depth correction**: Quantify how pool depth modifies the critical
   wavelength — relevant for thin-film ferrofluid applications.

3. **Scaling laws**: Do spike height and wavelength follow power-law scaling with
   Bo_m? Does the exponent change at high chi_0?

4. **Ferrofluid hedgehog**: At very high chi_0 and alpha, reproduce the "hedgehog"
   instability pattern described by Nochetto et al.

5. **Domain width quantization**: Verify N proportional to L_x and identify the
   critical minimum domain width below which instability is suppressed. This tests
   the quantized-mode prediction from linear stability theory.

6. **Demagnetizing field effect**: Side-by-side comparison of h = h_a + h_d (full)
   vs h = h_a (reduced, Nochetto approach). Expected: reduced model over-predicts
   spike height and underestimates the critical field threshold.

7. **Magnet geometry control of spike morphology** (novel): First systematic study
   of how curved dipole arrangements modify Rosensweig spike patterns:
   - Concave magnets produce graded spike-height envelopes (tall center, short edges)
   - Convex magnets produce edge-dominated or uniform patterns
   - Deep-V arrangements can isolate a single dominant spike
   - The curvature radius R provides continuous control over pattern nonuniformity
   This has direct engineering relevance: magnet shape as a design parameter for
   controlled ferrofluid surface patterning.

## 0.11 Implementation

**Minimal code changes needed:**
- Add CLI overrides for geometric parameters: `--Lx`, `--Ly`, `--dipole_curve`
  (concave/convex/V-shape with radius/slope parameter)
- Add `--reduced_field` flag already exists in the code
- Existing `--chi0`, `--lambda`, `--gravity` overrides handle physics parameters

**Batch infrastructure:**
- `scripts/parametric_sweep.py` — generates parameter combinations and SLURM scripts
- `scripts/post_process_sweep.py` — reads all diagnostics.csv, extracts metrics
- `scripts/plot_parametric.py` — generates phase diagrams, heatmaps, scaling plots,
  dipole geometry galleries, theory-vs-simulation overlays

**HPC submission**: Each run is independent — embarrassingly parallel. Submit as
an array job (SLURM/PBS). 4 cores per run, ~4 hrs each.

## 0.12 Timeline

| Phase | Task | Wall time (HPC) | Human time |
|-------|------|-----------------|------------|
| 0-pre | CLI overrides + dipole geometry code | — | 1-2 days |
| 0a | Screening: physics + geometry (33 runs) | 4 hrs | 1-2 days (analyze) |
| 0b | Interaction matrices A-E (91 runs) | 4 hrs | 3-4 days (analyze) |
| 0c | 3-way tensor (27 runs) | 4 hrs | 1-2 days (analyze) |
| 0d | Dipole geometry sweeps (11 runs) | 4 hrs | 1-2 days (analyze) |
| 0e | Post-processing + plots | — | 3-5 days |
| 0f | Write-up (theory + results) | — | 1-2 weeks |
| **Total** | **160 runs** | **~16 hrs compute** | **~4-5 weeks** |

---

# Summary: Recommended Roadmap

    Phase A (DONE):  Single-phase FHD (Nochetto et al.)
    Phase B (NOW):   Two-phase FHD (Zhang et al.) — isothermal, uniform concentration
         |
         v
    Phase C.0:  Parametric study of Rosensweig instability       [Paper 0 / chapter]
         |      (no new PDEs, just systematic sweeps on HPC)
         v
    Phase C.1:  + Heat equation         -> thermomagnetic convection     [Paper 1]
         |
         v
    Phase C.2:  + Concentration equation -> magnetophoresis, instability [Paper 2]
         |
         v
    Phase C.3:  Unified model           -> full cross-coupling           [Thesis capstone]

**Phase C.0 can start immediately** — the code needs only minor CLI additions
(dipole geometry flags) and batch scripts. It can run **in parallel** with C.1
development (parametric runs on HPC while implementing the heat equation locally).
The 160 runs are embarrassingly parallel and complete in ~4 hours wall time on HPC.

**Total estimated timeline for all of Phase C: 40-65 weeks (~10-15 months)**

Each phase produces a standalone publication while building toward the final
comprehensive model. The infrastructure (architecture pattern, MMS framework,
decoupled splitting) carries over directly from Phase B.

---

# References

**Base scheme:**
- Zhang, He & Yang, SIAM J. Sci. Comput. 43(1), B167-B193 (2021)
- Nochetto, Salgado & Tomas, CMAME 309, 497-531 (2016)

**Thermomagnetic convection:**
- Finlayson, J. Fluid Mech. 40(4), 753-767 (1970)
- Odenbach, "Magnetoviscous Effects in Ferrofluids", Springer (2002)
- Rosensweig, "Ferrohydrodynamics", Dover (2014)

**Concentration / magnetophoresis:**
- Rosensweig, Ch. 13: "Ferrohydrostatics" — particle sedimentation theory
- Odenbach & Thurm, "Magnetoviscous Effects", Ch. 10 (2002)
- Pshenichnikov, Mekhonoshin & Lebedev, J. Magn. Magn. Mater. 145, 319-326 (1995)

**Droplet deformation:**
- Afkhami et al., J. Fluid Mech. 663, 358-384 (2010)
- Nochetto, Salgado & Tomas, Math. Models Methods Appl. Sci. 26, 2393-2449 (2016)

**Hyperthermia / AC heating:**
- Rosensweig, J. Magn. Magn. Mater. 252, 370-374 (2002) — heating model
- Hedayatnasab et al., Materials & Design 123, 174-196 (2017) — review
