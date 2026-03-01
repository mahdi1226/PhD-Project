# FHD Project: Ferrohydrodynamics Finite Element Solver

## Objective

Reproduce the numerical method of **Nochetto, Salgado & Tomas** (arXiv:1511.04381, 2015) for the ferrohydrodynamics (FHD) equations using deal.II + Trilinos.

**Phase A** (current): Single-phase ferrofluid (Algorithm 42).
**Phase B** (future): Two-phase extension with Cahn-Hilliard diffuse interface.

## Governing Equations (Phase A)

The Rosensweig model couples four subsystems on a bounded domain with time stepping:

### Navier-Stokes (Eq. 42e) — Velocity u, Pressure p

    (u^k/tau, v) + (nu+nu_r)(D(u^k), D(v)) + b_h(u^{k-1}; u^k, v)
      - (p^k, div v) + (div u^k, q)
      = (u^{k-1}/tau, v) + mu_0 B_h^m(v, h^k, m^k) + 2 nu_r(w^{k-1}, curl v) + (f, v)

- FE: CG Q2 velocity / DG P1 pressure (Taylor-Hood saddle-point)
- Kelvin force: mu_0 [(M . grad)H + 1/2(div M) H]
- Micropolar coupling: 2 nu_r (w, curl v)
- NS convection b_h(u; u, v) enabled in production (skew-symmetric form)

### Angular Momentum (Eq. 42f) — Angular velocity w

    j(w^k/tau, z) + c_1(grad w^k, grad z) + 4 nu_r(w^k, z)
      = j(w^{k-1}/tau, z) + 2 nu_r(curl u^k, z) + mu_0(m^k x h^k, z)

- FE: CG Q2
- Curl coupling from velocity: 2 nu_r (curl u, z)
- Magnetic torque: mu_0 (m x h, z)
- Convection: disabled per paper Eq. 42f

### Magnetization Transport (Eq. 42c) — Magnetization m

    (m^k/tau, z) + sigma a_h^m(m^k, z) + B_h^m(u^{k-1}; m^k, z) + (m^k/T, z)
      = (m^{k-1}/tau, z) + (kappa_0 h^k / T, z)

- FE: DG Q2 (vector, component-wise)
- DG transport B_h^m: skew-symmetric + upwind flux (Eq. 62)
- SIP diffusion a_h^m (if sigma > 0)
- Debye relaxation toward equilibrium magnetization kappa_0 h

### Poisson (Eq. 42d) — Magnetic potential phi

    (grad phi^k, grad psi) = (h_a - m^k, grad psi)

- FE: CG Q2
- **h = grad(phi) is the TOTAL magnetic field** (paper p.8: "use that h = ∇φ")
- The applied field h_a is encoded into phi via the Poisson RHS
- Assemblers use H = ∇φ only — DO NOT add h_a separately (double-counting)
- Neumann BCs with mean-zero constraint

### Passive Scalar (Eq. 104) — Concentration c

    (c^k/tau, z) + (u^k . grad c^k, z) + alpha(grad c^k, grad z)
      = (c^{k-1}/tau, z)

- FE: CG Q2
- Pure convection-diffusion, one-way coupled (velocity advects scalar, no back-coupling)
- alpha = 0.001 (small diffusion)
- Step function IC: c = 1 for y < 0.5, c = 0 for y >= 0.5
- Neumann BCs (no-flux boundaries)

## Applied Magnetic Field (Eq. 101-103)

Point dipole potential (2D, Eq. 102):

    phi_s(x) = d . (x_s - x) / |x_s - x|^2

Applied field:

    h_a = sum_s alpha_s grad(phi_s)

Supports:
- Multiple dipoles with per-dipole positions, directions, and intensities
- Time-dependent intensities and directions (ramp functions, orbiting, traveling waves)
- Uniform field mode: h_a = direction * intensity

## Algorithm (per time step)

1. **Update applied field** (time-dependent dipole intensities/directions)
2. **Picard iteration** (under-relaxed): Poisson(M_relaxed) <-> Mag(M_old, H, u_old)
   - Iterate until M converges: M_relaxed = omega M_raw + (1-omega) M_prev
3. **NS solve**: Using u_old, w_old, converged M and phi (Kelvin force + micropolar + convection)
4. **AngMom solve**: Using w_old, u_new, converged M and phi (curl + torque)
5. **Passive scalar** (if enabled): Advect c by u_new
6. **Diagnostics output**: U_max, divU, pressure, energy, CFL, scalar bounds, etc.
7. **VTK output** (at configured interval)

## Physical Parameters

| Symbol | Name | Default (paper) | Description |
|--------|------|-----------------|-------------|
| nu | Kinematic viscosity | 1.0 | Fluid viscosity |
| nu_r | Vortex viscosity | 1.0 | Micropolar coupling strength |
| mu_0 | Permeability | 1.0 | Magnetic permeability of free space |
| j | Microinertia | 1.0 | Angular momentum inertia |
| c_1 | Angular viscosity | 1.0 | c_a + c_d |
| sigma | Magnetic diffusion | 0.0 | DG IP diffusion (0 = pure transport) |
| T | Relaxation time | 1.0 | Debye relaxation (T_relax) |
| kappa_0 | Susceptibility | 1.0 | Magnetic susceptibility chi_0 |
| alpha | Scalar diffusivity | 0.001 | Passive scalar diffusion |

## Reference

Nochetto, R.H., Salgado, A.J. & Tomas, I. (2015). A diffuse interface model for two-phase
ferrofluid flows. *Computer Methods in Applied Mechanics and Engineering*, 309, 497-531.
arXiv:1511.04381.
