# FHD_PUMP: Two-Phase Ferrofluid Droplet Transport

## Research Goal

Simulate ferrofluid droplet manipulation in a non-magnetic carrier fluid (water/blood)
using external magnetic fields. The pumping channel setup from Nochetto et al. (Section 7.2)
provides the magnetic actuation; Cahn-Hilliard provides the two-phase interface tracking.

**Physical scenario**: Ferrofluid droplets in a channel filled with a non-magnetic carrier.
A traveling-wave magnetic field (array of dipoles) selectively transports the ferrofluid
droplets via the Kelvin force, which acts only on the magnetic phase (chi != 0).

**Why this is novel**: Nochetto et al. solve single-phase FHD. Extending to two-phase
with rigorous Cahn-Hilliard coupling and phase-dependent magnetic properties is a genuine
research contribution with clear biomedical applications (drug delivery, lab-on-chip).

---

## Mathematical Formulation

### Cahn-Hilliard Equation

Phase field phi in [-1, 1]: phi = +1 (ferrofluid), phi = -1 (carrier fluid).

```
phi_t + u . grad(phi) = gamma * Delta(mu)              (CH transport)
mu = Psi'(phi) - epsilon^2 * Delta(phi)                (chemical potential)
```

Where:
- Psi(phi) = (1/4)(phi^2 - 1)^2  (double-well potential, or Flory-Huggins logarithmic)
- epsilon: interface thickness parameter (~ h for diffuse interface)
- gamma: mobility coefficient
- mu: chemical potential (auxiliary variable for split formulation)

### Split Formulation (two second-order equations)

Avoids 4th-order equation. Two CG fields: phi (phase) and mu (chemical potential).

```
(phi^k - phi^{k-1})/tau + u^k . grad(phi^k) = gamma * Delta(mu^k)
mu^k = Psi'(phi^{k-1}) + (1/epsilon^2)(phi^k - phi^{k-1}) - epsilon^2 * Delta(phi^k)
```

Note: The `(1/epsilon^2)(phi^k - phi^{k-1})` term is the convex-concave splitting
(Eyre's method) for unconditional energy stability.

### Phase-Dependent Properties

Material properties interpolated between phases:

```
chi(phi)  = chi_ferro * H(phi)                    (susceptibility: ferro only)
nu(phi)   = nu_carrier + (nu_ferro - nu_carrier) * H(phi)   (viscosity)
rho(phi)  = rho_carrier + (rho_ferro - rho_carrier) * H(phi) (density, if needed)
```

Where H(phi) = (1 + phi)/2 is a smooth interpolation (phi=+1 -> ferro, phi=-1 -> carrier).

### Capillary Force in NS

Surface tension adds a force to the momentum equation:

```
f_capillary = sigma * kappa * n * delta_interface
            = sigma * mu * grad(phi)    (diffuse interface form)
```

Where sigma is the surface tension coefficient. In the diffuse-interface formulation,
this becomes a body force proportional to mu * grad(phi).

### Modified NS Equation

```
rho(phi) * [u_t + (u . grad)u] - div(2 * nu(phi) * D(u)) + grad(p)
    = mu_0 * [(M . grad)H + 1/2 div(M) H]     (Kelvin force, acts where chi != 0)
    + sigma * mu_CH * grad(phi)                  (capillary force)
    + 2 * nu_r(phi) * curl(w)                    (micropolar, if retained)
```

### Modified Magnetization Equation

Susceptibility becomes phase-dependent:

```
M_eq = chi(phi) * H    (equilibrium magnetization depends on phase)
```

In pure carrier fluid (phi = -1): chi = 0, so M = 0 and no magnetic response.
In ferrofluid (phi = +1): chi = kappa_0, full magnetic response.

---

## Implementation Plan

### Project Structure

```
Droplet/                          (new folder, parallel to FHD/)
├── CMakeLists.txt
├── utilities/                    (copy from FHD, extend parameters)
├── mesh/                         (copy from FHD)
├── physics/                      (copy from FHD, add capillary force)
├── cahn_hilliard/                (NEW — CH subsystem)
│   ├── cahn_hilliard.h
│   ├── cahn_hilliard_setup.cc
│   ├── cahn_hilliard_assemble.cc
│   ├── cahn_hilliard_solve.cc
│   ├── cahn_hilliard_output.cc
│   └── tests/
│       ├── cahn_hilliard_mms.h
│       └── cahn_hilliard_mms_test.cc
├── poisson/                      (copy from FHD)
├── magnetization/                (copy from FHD, add chi(phi) dependency)
├── navier_stokes/                (copy from FHD, add capillary force + rho(phi)/nu(phi))
├── angular_momentum/             (copy from FHD, optional for two-phase)
├── passive_scalar/               (remove — replaced by CH)
├── mms_tests/                    (new coupled MMS tests)
├── drivers/
│   ├── ch_benchmark_driver.cc    (standalone CH: droplet + square tests)
│   ├── passive_ch_driver.cc      (one-way: FHD velocity advects CH, no feedback)
│   └── fhd_droplet_driver.cc    (full two-way coupled driver)
└── Report/
```

### Phase 1: Standalone Cahn-Hilliard (no magnetics)

**Goal**: Verified CH subsystem with MMS convergence rates.

1. **CH subsystem**: Split formulation, CG Q2 for both phi and mu
   - Backward Euler time stepping
   - Convex-concave (Eyre) splitting for energy stability
   - Convection term: u . grad(phi) (velocity from external source)
   - BCs: natural (no-flux) for both phi and mu
2. **MMS test**: Manufactured smooth solution, verify L2/H1 convergence
3. **Solver**: CG + AMG (system is symmetric if no convection)

### Phase 2: CH Benchmarks (no magnetics)

**Goal**: Verify physics without magnetics.

1. **Circular droplet equilibrium**
   - IC: circular droplet (phi=+1) of radius R in carrier (phi=-1)
   - Domain: [0,1]^2, droplet centered at (0.5, 0.5)
   - Expected: droplet stays circular, pressure jump = sigma/R (Young-Laplace)
   - Verify: mass conservation, energy dissipation, pressure jump
2. **Square relaxation**
   - IC: square region (phi=+1) in carrier (phi=-1)
   - Expected: square relaxes to circle, conserving area
   - Verify: shape evolution, mass conservation, energy monotone decrease
3. **Advected droplet** (with prescribed velocity)
   - IC: circular droplet in a shear flow or uniform flow
   - Verify: droplet translates/deforms correctly, mass conserved

### Phase 3: Passive Phase Field in Pumping Flow (one-way coupling)

**Goal**: Quick proof of concept — ferrofluid droplet transported by magnetic pumping.

1. Copy the FHD pumping driver (Section 7.2 setup)
2. Replace passive scalar with CH equation
3. One-way coupling: FHD velocity advects CH, but phase doesn't affect flow
4. IC: ferrofluid droplet (phi=+1) placed in the channel
5. Run pumping: observe droplet transport, deformation, breakup

This gives visual results quickly without full two-way coupling.

### Phase 4: Two-Way Coupling

**Goal**: Full coupled system — phase affects magnetic response and flow.

1. **Phase-dependent chi(phi)**: Magnetization equation uses chi(phi) instead of constant chi
   - In carrier fluid: chi = 0, M = 0 (no magnetic response)
   - In ferrofluid: chi = kappa_0 (full response)
   - Interface: smooth interpolation
2. **Capillary force**: Add sigma * mu * grad(phi) to NS RHS
3. **Phase-dependent viscosity**: nu(phi) interpolation
4. **Modified Picard iteration**: Add CH solve to the iteration loop
5. **MMS verification**: Full coupled system with manufactured solution
6. **Production runs**: Droplet transport in pumping channel with full physics

---

## Key Parameters

### Cahn-Hilliard
- epsilon: interface thickness (typically ~ 2-4 * h_min for well-resolved interface)
- gamma: mobility (affects interface dynamics speed)
- sigma: surface tension coefficient
- Psi: double-well or Flory-Huggins potential

### Two-Phase Material Properties
- chi_ferro = kappa_0 = 5.0 (from Nochetto paper)
- chi_carrier = 0 (non-magnetic)
- nu_ferro = 0.5 (from Nochetto paper)
- nu_carrier = 0.5 (start with matched viscosity, then explore mismatch)
- sigma: to be determined (controls capillary number Ca = nu * U / sigma)

### Pumping Channel (from Section 7.2)
- Domain: [0, L] x [0, 1], L = 6
- 64 dipoles (32 below, 32 above)
- Traveling wave: f = 10 Hz, lambda = 1, q = 5
- alpha_0 = 5.0

---

## Research Questions

1. Can a traveling-wave magnetic field selectively transport ferrofluid droplets
   through a non-magnetic carrier fluid?
2. How do droplet size, field frequency, and field intensity affect transport
   efficiency and droplet integrity (breakup vs. cohesion)?
3. What is the role of surface tension in maintaining droplet shape under
   magnetic actuation?
4. How does viscosity contrast between phases affect transport dynamics?
5. Can multiple droplets be independently manipulated (sorting/merging)?

---

## Literature to Review

- Nochetto, Salgado & Tomas (2015): arXiv:1511.04381 (base FHD model)
- Nochetto, Salgado & Tomas (2016): Rosensweig instability paper (full vs simplified model)
- Cahn-Hilliard FEM: deal.II tutorial step-36/step-63, or Kim & Lowengrub (2005)
- Two-phase ferrohydrodynamics: Afkhami et al. (2008, 2010)
- Diffuse interface magnetic fluids: Nochetto & Walker (2013)
- Biomedical ferrofluid applications: Pankhurst et al. (2003)

---

## Success Criteria

1. CH subsystem passes MMS (L2 rate >= 2 for CG Q2)
2. Circular droplet: pressure jump within 5% of Young-Laplace
3. Square relaxation: mass conservation < 0.1%, energy monotone
4. Passive CH in pumping: visible droplet transport in VTK
5. Full coupling: ferrofluid droplet transported while carrier stays still
6. At least one novel physical insight about droplet transport under magnetic actuation
