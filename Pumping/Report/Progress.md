# Progress Tracker

## Completed

### Project Setup
- [x] Copied FHD Phase A source code as starting point
- [x] Created project structure: utilities, mesh, physics, subsystems, drivers
- [x] Created CMakeLists.txt for standalone build
- [x] Created Report files

### Inherited from FHD Phase A (all verified)
- [x] Poisson subsystem: CG Q2, MMS L2=3.0, H1=2.0
- [x] Magnetization subsystem: DG Q2, MMS L2=3.0
- [x] Navier-Stokes subsystem: CG Q2/DG P1, MMS U_L2=3.0, U_H1=2.0, p_L2=2.1
- [x] Angular Momentum subsystem: CG Q2, MMS L2=3.0, H1=2.0
- [x] Full 4-system coupling: all MMS rates verified
- [x] Block-Schur preconditioner: 26x speedup at ref 7
- [x] Kelvin force face integrals (Eq. 38, 2nd line)
- [x] Production driver with presets and diagnostics

## In Progress

### Phase 1: Standalone Cahn-Hilliard
- [ ] CH subsystem implementation (split formulation, CG Q2)
- [ ] MMS test with manufactured solution
- [ ] Verify L2/H1 convergence rates

## Remaining Work

### Phase 2: CH Benchmarks
- [ ] Circular droplet equilibrium (Young-Laplace pressure jump)
- [ ] Square relaxation (area conservation, energy dissipation)
- [ ] Advected droplet (prescribed velocity, mass conservation)

### Phase 3: Passive CH in Pumping
- [ ] Pumping driver with CH replacing passive scalar
- [ ] One-way coupling: velocity advects phase field
- [ ] Visual validation of droplet transport

### Phase 4: Full Two-Way Coupling
- [ ] Phase-dependent chi(phi) in magnetization
- [ ] Capillary force in NS
- [ ] Phase-dependent viscosity nu(phi)
- [ ] Full coupled MMS verification
- [ ] Production droplet transport runs
