# Pumping Project: Two-Phase Ferrofluid Droplet Transport

## Objective

Simulate ferrofluid droplet manipulation in a non-magnetic carrier fluid (water/blood)
using external magnetic fields. Extends the single-phase FHD solver (Phase A) with
Cahn-Hilliard two-phase interface tracking and phase-dependent material properties.

**Base solver**: FHD (Phase A) -- Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015)
**Extension**: Cahn-Hilliard + FHD coupling for ferrofluid droplets

## Physical Scenario

Ferrofluid droplets (chi != 0) in a channel filled with non-magnetic carrier fluid.
An external magnetic field deforms and transports the droplet via the Kelvin force,
which acts only on the magnetic phase.

## Governing Equations

### Existing (from FHD Phase A)
- Navier-Stokes (Eq. 42e): CG Q2/DG P1, Kelvin force + micropolar
- Angular Momentum (Eq. 42f): CG Q2, curl coupling + magnetic torque
- Magnetization Transport (Eq. 42c): DG Q2, SIP + transport + Debye relaxation
- Poisson (Eq. 42d): CG Q2, h = grad(phi) total field convention

### New (Phase B additions)
- Cahn-Hilliard: Split formulation (phi + mu), CG Q2, Eyre's convex-concave splitting
- Phase-dependent susceptibility: chi(phi) = chi_ferro * (phi+1)/2 (linear)
- Phase-dependent viscosity: nu(phi) = nu_w*(1-phi)/2 + nu_f*(phi+1)/2 (linear)
- Capillary force: sigma * mu_CH * grad(phi) in NS

### Key Coupling (confirmed by literature review)
- Chemical potential W = -eps*Delta(phi) + (1/eps)*f(phi) -- NO magnetic term
- Magnetic coupling through chi(phi) in magnetization + Kelvin force in NS
- All 13 reviewed papers confirm this formulation (see References/cheatsheet.md)

## Final Status (2026-03-07) -- PROJECT STOPPED

- Steps 3a-3e COMPLETE: phase-dependent properties, full coupled driver, MMS verified
- Step 3f ABANDONED: Rosensweig instability (spikes never form with Nochetto scheme)
- Step 3g COMPLETE: Droplet deformation benchmark (D vs Bo_m, 6 field strengths)
- Literature review COMPLETE: 18 papers in FHD/Reference/ (including Afkhami 2008/2010)
- Sub-cell interface tracking implemented (phi=0 edge-crossing interpolation)

## Deformation Benchmark Results (final)

Bo_m sweep at chi=1.19, R=0.2, ref 6, eps=0.02, t_final=3.0:
- D increases monotonically with Bo_m (qualitatively correct)
- D falls between 2D theory (D=0.039*Bo_m) and 3D theory (D=0.249*Bo_m)
- Linear regime: D/Bo_m is NOT constant -- superlinear D ~ Bo_m^1.3
- Numerical D is 1.2-1.8x above 2D sharp-interface theory
- Likely cause: diffuse interface (eps/R=0.1) and/or mesh resolution (ref 6)
- Proper validation needs: eps-convergence study, mesh convergence study
- Mass conservation: excellent (< 0.01%)

## Dependencies

- deal.II >= 9.4 (with MPI, Trilinos, p4est)
- FHD Phase A code (copied into this folder as starting point)

## Build

    cd Pumping && mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DDEAL_II_DIR=/path/to/dealii
    make -j$(nproc)

## Key References

See References/cheatsheet.md for full analysis. Primary papers:
1. Nochetto, Salgado & Tomas (2016) -- foundational two-phase FHD model
2. Zhang, He & Yang (2021) -- decoupled IEQ+ZEC scheme
3. Afkhami et al. (2010) JFM 663 -- droplet deformation benchmark (our validation)
4. Afkhami et al. (2008) JFM 610 -- droplet motion in viscous media
5. Mao et al. (2011) -- ferrofluid pumping experiment (target application)
