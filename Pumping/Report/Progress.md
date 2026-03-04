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

### Phase 1: Standalone Cahn-Hilliard
- [x] CH subsystem implementation (split phi+mu, CG Q2, Eyre splitting)
- [x] MMS test with manufactured solution
- [x] L2/H1 convergence rates verified (L2>=3, H1>=2)

### Phase 2: CH Benchmarks (partial)
- [x] CH+NS coupled MMS verified
- [x] `ch_benchmark` driver for droplet and square tests

### Phase 3: Phase-Dependent Properties (Step 3a-3e)
- [x] Step 3a: chi(phi) in magnetization assembler (cross-mesh FEValues)
- [x] Step 3b: nu(phi) in NS assembler (reuse CH FEValues)
- [x] Step 3c: Parameters update (chi_ferro, nu_carrier, nu_ferro, enable_phase_dependent_properties)
- [x] Step 3d: Full coupled driver `fhd_ch_driver.cc` (all 6 subsystems)
- [x] Step 3e: Full coupled MMS test `full_ch_fhd_mms_test.cc` (all convergence rates pass)
- [x] Unified VTK output (all fields: ux, uy, p, phi_mag, w, Mx, My, phi, mu in single file)

### Literature Review
- [x] Reviewed all 13 reference papers in References/
- [x] Created References/cheatsheet.md (per-paper analysis + 5 comparison tables)
- [x] Created References/comparison_study.md (decision document for numerical path)
- [x] Critical finding: NO paper includes magnetic energy term in CH chemical potential
- [x] Capillary force forms are equivalent (sigma*mu*grad(phi) vs (lambda/eps)*phi*grad(W))

## In Progress

### Step 3f: Rosensweig Instability Benchmark
- [x] Rosensweig preset in fhd_ch_driver (domain, ICs, parameters)
- [x] FerrofluidPoolIC with optional perturbation
- [x] Interface tracking (y_min, y_max of phi=0 contour)
- [x] CLI flags: --perturbation-amp, --perturbation-modes, --field-strength, --ramp-time, --gamma
- [ ] **Spikes not forming** — multiple runs attempted:
  - r=3, H=10, 500 steps: U_max=0.016, no visible deformation
  - r=4, H=10, 280 steps: U_max=0.005, interface unchanged
  - r=4, H=30, 545 steps: U_max=0.165, still no spikes
- [ ] Root cause analysis: likely numerical (see comparison_study.md)
- [ ] Need to check Zhang's exact parameters before concluding scheme is at fault

## Remaining Work

### Step 3g: Droplet Deformation Benchmark
- [ ] Circular droplet under uniform field
- [ ] Compare aspect ratio b/a vs magnetic Bond number Bo_m
- [ ] Analytical formula: Bo_m = [(1/chi_0)+k]^2 * (b/a)^(1/3) * [2*b/a - (b/a)^(-2) - 1]

### Numerical Path Decision (see comparison_study.md)
- [ ] Verify Zhang's exact Rosensweig parameters (eps, nu, H, gravity)
- [ ] If parameter issue: tune and re-run
- [ ] If scheme issue: implement hybrid decoupling (Path C)

### Pumping Application (PhD Primary Contribution)
- [ ] Droplet transport in pumping channel with traveling-wave field
- [ ] Parametric study: droplet size, field frequency, intensity
- [ ] Role of surface tension in droplet integrity
- [ ] Viscosity contrast effects
- [ ] Multiple droplet manipulation

### Future: BDF2 + Full Micropolar (Methods Paper Potential)
- [ ] Upgrade time integration from BDF1 to BDF2
- [ ] Add SAV/ZEC for energy stability after decoupling
- [ ] Prove energy stability for full system (NS+AngMom+Mag+Poisson+CH)
- [ ] Convergence study (second-order in time)
