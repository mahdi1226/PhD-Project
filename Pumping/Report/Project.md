# Pumping Project: Two-Phase Ferrofluid Droplet Transport

## Objective

Simulate ferrofluid droplet manipulation in a non-magnetic carrier fluid (water/blood)
using external magnetic fields. Extends the single-phase FHD solver (Phase A) with
Cahn-Hilliard two-phase interface tracking and phase-dependent material properties.

**Base solver**: FHD (Phase A) — Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015)
**Extension**: Cahn-Hilliard + FHD coupling for ferrofluid droplets in pumping channel

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
- Phase-dependent susceptibility: chi(phi) = chi_ferro * (1+phi)/2
- Phase-dependent viscosity: nu(phi) interpolation between phases
- Capillary force: sigma * mu_CH * grad(phi) in NS

## Dependencies

- deal.II >= 9.4 (with MPI, Trilinos, p4est)
- FHD Phase A code (copied into this folder as starting point)

## Build

    cd Pumping && mkdir build && cd build
    cmake .. -DDEAL_II_DIR=/path/to/dealii
    make -j$(nproc)
