# Pumping Project: Two-Phase Ferrofluid Droplet Transport

## Objective

Simulate ferrofluid droplet manipulation in a non-magnetic carrier fluid (water/blood)
using external magnetic fields. Extends the single-phase FHD solver (Phase A) with
Cahn-Hilliard two-phase interface tracking and phase-dependent material properties.

**Base solver**: FHD (Phase A) -- Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015)
**Extension**: Cahn-Hilliard + FHD coupling for ferrofluid droplets in pumping channel
**Numerical path**: Hybrid decoupling (Path C) -- see References/comparison_study.md

## Physical Scenario

Ferrofluid droplets (chi != 0) in a channel filled with non-magnetic carrier fluid.
A traveling-wave magnetic field (array of dipoles) selectively transports the ferrofluid
droplets via the Kelvin force, which acts only on the magnetic phase.

## Governing Equations

### Existing (from FHD Phase A)
- Navier-Stokes (Eq. 42e): CG Q2/DG P1, Kelvin force + micropolar
- Angular Momentum (Eq. 42f): CG Q2, curl coupling + magnetic torque
- Magnetization Transport (Eq. 42c): DG Q2, SIP + transport + Debye relaxation
- Poisson (Eq. 42d): CG Q2, h = grad(phi) total field convention

### New (Phase B additions)
- Cahn-Hilliard: Split formulation (phi + mu), CG Q2, Eyre's convex-concave splitting
- Phase-dependent susceptibility: chi(phi) via sigmoid interpolation
- Phase-dependent viscosity: nu(phi) via sigmoid interpolation
- Capillary force: sigma * mu_CH * grad(phi) in NS

### Key Coupling (confirmed by literature review)
- Chemical potential W = -eps*Delta(phi) + (1/eps)*f(phi) -- NO magnetic term
- Magnetic coupling through chi(phi) in magnetization + Kelvin force in NS
- All 13 reviewed papers confirm this formulation (see References/cheatsheet.md)

## Current Status (2026-03-03)

- Steps 3a-3e COMPLETE: phase-dependent properties, full coupled driver, MMS verified
- Step 3f IN PROGRESS: Rosensweig instability benchmark (spikes not forming)
- Literature review COMPLETE: 13 papers analyzed, comparison study written
- Decision: Hybrid decoupling (Path C) recommended for next steps

## Dependencies

- deal.II >= 9.4 (with MPI, Trilinos, p4est)
- FHD Phase A code (copied into this folder as starting point)

## Build

    cd Pumping && mkdir cmake-build-release && cd cmake-build-release
    cmake .. -DCMAKE_BUILD_TYPE=Release -DDEAL_II_DIR=/path/to/dealii
    make -j$(nproc)

## Key References

See References/cheatsheet.md for full analysis. Primary papers:
1. Nochetto, Salgado & Tomas (2016) -- foundational two-phase FHD model
2. Zhang, He & Yang (2021) -- decoupled IEQ+ZEC scheme (produces Rosensweig spikes)
3. Wu, Yang et al. (2024) -- BDF2 SAV+ZEC (state-of-the-art accuracy)
4. Chen, Li, Li & He (2025) -- variable density extension
5. Mao et al. (2011) -- ferrofluid pumping experiment (our target application)
