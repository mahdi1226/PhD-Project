# Comparison Study: Numerical Approaches for Two-Phase FHD

Decision document for choosing the numerical path for Phase B
(ferrofluid droplet transport in pumping channel).

Date: 2026-03-03

---

## 1. Problem Statement

We need a two-phase ferrohydrodynamics solver coupling:
- Cahn-Hilliard (interface tracking)
- Navier-Stokes (flow)
- Angular Momentum (micropolar spin, optional)
- Magnetization (Shliomis relaxation)
- Magnetostatics (scalar potential, h = grad(phi))

Our code currently implements the **Nochetto et al. (2015/2016) model** for single-phase FHD
(Phase A, fully verified), extended with Cahn-Hilliard and phase-dependent properties
(chi(phi), nu(phi)). All subsystems pass MMS tests.

**Issue**: Rosensweig instability benchmark does not produce spikes with our current scheme,
but Zhang et al. (2021) successfully produce spikes with the same PDE model. We need to
understand why and decide which path to take.

---

## 2. Critical Finding: All Papers Use the Same PDE

After reviewing all 13 reference papers (see cheatsheet.md), we confirm:

- **NO paper includes a magnetic energy term in the CH chemical potential**
- The chemical potential is always: W = -eps*Delta(phi) + (1/eps)*f(phi)
- Magnetic coupling enters ONLY through:
  1. chi(phi) in the magnetization equation
  2. Kelvin force mu_0*(m.grad)h in NS
  3. Capillary stress phi*grad(W) in NS

This means the difference between Nochetto and Zhang is purely numerical, not in the PDE.

---

## 3. Scheme Comparison: Nochetto vs Zhang

### 3.1 Nochetto, Salgado & Tomas (2016)

| Aspect | Details |
|--------|---------|
| Time integration | BDF1 (backward Euler) |
| Coupling | **Fully coupled** (nonlinear system per step) |
| Nonlinearity | Implicit, requires Newton-like iteration |
| FE for H | **Nedelec H(curl)** (edge elements) |
| CH treatment | Standard semi-implicit, Eyre splitting |
| Magnetization | Simplified Shliomis (no spin coupling, no beta) |
| Energy stability | Unconditionally stable (proven) |
| Solver | UMFPACK (serial direct) |
| Angular momentum | Included (full micropolar) |
| Gravity | Boussinesq approximation |

**Strengths**: Mathematically rigorous, energy-stable, includes full micropolar physics.
**Weaknesses**: Fully coupled = expensive, implicit coupling may over-damp instabilities.

### 3.2 Zhang, He & Yang (2021)

| Aspect | Details |
|--------|---------|
| Time integration | BDF1 (backward Euler) |
| Coupling | **Fully decoupled** (all linear systems) |
| Nonlinearity | IEQ linearization (no Newton) |
| FE for H | **Scalar potential, standard Lagrange** |
| CH treatment | IEQ (Invariant Energy Quadratization) |
| Magnetization | Full Shliomis (spin coupling + beta term) |
| Energy stability | Unconditionally stable (proven) |
| Solver | Standard iterative (constant-coefficient matrices) |
| Angular momentum | Not separate (absorbed into magnetization) |
| Gravity | Not included |

**Strengths**: Efficient (only linear solves), constant-coefficient matrices, produces physical results.
**Weaknesses**: First-order only, IEQ adds auxiliary variables, no angular momentum equation.

### 3.3 Key Differences That Likely Affect Rosensweig

1. **Coupled vs decoupled**: Nochetto's implicit coupling may numerically suppress
   growing instability modes. Zhang's explicit treatment (old-time extrapolation)
   allows instabilities to grow more naturally.

2. **Nedelec vs scalar potential**: Nochetto's H(curl) elements enforce tangential
   continuity of H, which may over-constrain the field near interface tips where
   field concentration drives the instability. Scalar potential is more flexible.

3. **Extra physics**: Zhang includes spin torque (mu/2)*curl(m x h) and beta*m x (m x h).
   These contribute additional angular momentum transfer that helps drive instability.

4. **IEQ treatment of double-well**: Zhang's IEQ linearizes the double-well potential
   differently from Eyre's splitting, potentially affecting interface dynamics.

---

## 4. State-of-the-Art Evolution (2021-2026)

| Paper | Year | Order | Key Innovation |
|-------|------|-------|---------------|
| Nochetto et al. | 2016 | BDF1 | First two-phase FHD model, coupled |
| Zhang/He/Yang | 2021 | BDF1 | First decoupled scheme (IEQ+ZEC) |
| Zhang/He/Yang | 2022 | BDF2 | Nonlocal Q technique for MHD |
| Wu/Yang et al. | 2024 | BDF1+2 | SAV+ZEC, both orders |
| Yang et al. | 2025 | BDF2 | SAV+ZEC+projection, state-of-the-art |
| Chen/Li/Li/He | 2025 | BDF1 | Variable density/viscosity |
| Zhang/Zhou et al. | 2026 | BDF2 | Porous media (Darcy) |

**Trend**: Field moved from coupled -> decoupled, BDF1 -> BDF2, IEQ -> SAV.
All recent papers use fully decoupled schemes with ZEC stabilization.

---

## 5. Available Paths

### Path A: Stay with Current (Nochetto-like Coupled)
- **Effort**: Low (parameter tuning only)
- **Risk**: High (may never produce Rosensweig spikes)
- **Novelty**: None
- **Verdict**: Not recommended as primary path

### Path B: Full Switch to Zhang's IEQ+ZEC Decoupled
- **Effort**: Very High (major rewrite of time integration, add IEQ/SAV variables)
- **Risk**: Low (proven to work for all benchmarks)
- **Novelty**: Low (already published)
- **Verdict**: Too much effort for a replication

### Path C: Hybrid -- Decouple CH from Magnetics (RECOMMENDED)
- **Effort**: Medium (modify driver time loop, keep subsystem assemblers)
- **Risk**: Medium (energy stability not formally proven)
- **Novelty**: Medium (Nochetto's full micropolar model + decoupled CH is new)
- **What changes**:
  1. CH solved first with old velocity (explicit convection)
  2. Mag+Poisson Picard loop with old phi (explicit chi(phi))
  3. NS+AngMom with old M,H and new phi (explicit Kelvin, semi-implicit capillary)
  4. No IEQ/SAV needed -- keep Eyre splitting for CH
- **Why this might work**: The key insight from Zhang is that explicit treatment of
  coupling terms allows instabilities to grow. Our Picard loop already does this
  for Mag+Poisson. Extending to CH coupling is natural.
- **Verdict**: Best effort-to-benefit ratio

### Path D: BDF2 + SAV + ZEC (State-of-the-Art)
- **Effort**: Very High (BDF2 time stepping, SAV variables, ZEC terms, pressure projection)
- **Risk**: Low (proven, most accurate)
- **Novelty**: High if combined with angular momentum (no paper does BDF2+micropolar+CH)
- **Verdict**: Future work / potential methods paper

### Path E: Chen et al. (2025) Variable Density
- **Effort**: High (artificial compressibility, sigma=sqrt(rho) reformulation)
- **Risk**: Low (handles rho_f != rho_w)
- **Novelty**: Low (already published)
- **Verdict**: Only needed if density mismatch matters for pumping

---

## 6. Novelty Analysis

### What Has Been Done
- BDF1 coupled two-phase FHD (Nochetto 2016)
- BDF1 decoupled IEQ+ZEC (Zhang 2021)
- BDF2 decoupled SAV+ZEC without angular momentum (Wu/Yang 2024, Yang 2025)
- Variable density BDF1 (Chen 2025)
- Porous media BDF2 (Zhang 2026)

### What Has NOT Been Done (Novelty Opportunities)

1. **BDF2 decoupled + full micropolar (angular momentum + spin torque) + CH**
   - Every decoupled paper drops angular momentum or simplifies spin coupling
   - We have all 6 subsystems implemented and MMS-verified
   - Combining BDF2 accuracy with full micropolar physics = methods paper

2. **Two-phase ferrofluid pumping with traveling-wave field**
   - Nobody has simulated diffuse-interface droplet transport under traveling-wave fields
   - Closest: Mao et al. (2011) is experimental only, single-phase
   - This is the PhD's primary novelty (application paper)

3. **Adaptive time stepping with provable energy stability**
   - All current schemes use fixed dt
   - Energy-stable adaptive stepping for coupled FHD-CH is open

4. **Pressure-robust two-phase FHD**
   - Scott-Vogelius or divergence-free elements for NS
   - Addresses the pressure robustness issue observed in Section 7.3

---

## 7. Recommendation

**Short-term (now)**: Path C -- Hybrid decoupling
- Minimal code changes to existing subsystems
- Focus on getting Rosensweig benchmark working
- If parameter tuning (eps, viscosity, gravity) solves it, even better

**Medium-term (pumping)**: Keep Path C, focus on application
- Droplet deformation benchmark (validate coupling)
- Pumping channel with droplet transport (PhD novelty)
- This is the primary research contribution

**Long-term (methods paper)**: Path D -- BDF2 + full micropolar
- Upgrade time integration to BDF2
- Add SAV/ZEC for energy stability
- Publish as methods paper: first BDF2 energy-stable decoupled scheme with
  full micropolar physics (angular momentum + spin torque + magnetization + CH)

---

## 8. Parameter Check: Zhang vs Our Rosensweig Setup

Before code changes, verify parameters match Zhang Section 4.3:

| Parameter | Our Current | Zhang (2021) | Notes |
|-----------|------------|--------------|-------|
| epsilon | 0.02 | ~h (mesh-dependent) | Ours may be too large |
| viscosity | nu=2 | Check paper | Ours may be too high |
| chi_0 | 0.5 | Check paper | Susceptibility |
| H field | 10-30 | Check paper | Applied field strength |
| Gravity | None | Check paper | Rosensweig needs gravity? |
| Domain | [0,1]x[0,0.6] | Check paper | |
| dt | 1e-3 | Check paper | |

**Action**: Read Zhang Section 4.3 carefully for exact parameters before
concluding the scheme is the problem.
