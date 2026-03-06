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
- [x] Read Afkhami 2008 (JFM 610): droplet motion, VOF method, chi=1.19
- [x] Read Afkhami 2010 (JFM 663): droplet deformation, D vs Bo_m, experimental validation
- [x] Read Afkhami 2017 (J. Eng. Math. 107): review paper (drug targeting, droplet, thin films)

### Rosensweig Instability (Step 3f) -- ABANDONED
- [x] Rosensweig preset in fhd_ch_driver (domain, ICs, parameters)
- [x] FerrofluidPoolIC with optional perturbation
- [x] Interface tracking (y_min, y_max of phi=0 contour)
- [x] Multiple runs attempted: r=3-4, H=10-30, 500+ steps
- [x] Spikes never formed -- consistent with Nochetto's scheme across multiple projects
- [x] Decision: Not a convergence study in the paper, just proof of concept. Move on.

### Kelvin Force Two-Phase Fix
- [x] Fixed Kelvin force for two-phase: phase-dependent chi(phi) in magnetic stress
- [x] Verified with vertical and horizontal elongation tests
- [x] Linear material properties chi(phi), nu(phi) for stability (sigmoid was unstable)

### AMR Infrastructure
- [x] utilities/amr.h: phase-field gradient refinement
- [x] DG face loop fix (is_active check)
- [x] Working with p4est + Trilinos parallel

### Droplet Deformation Benchmark (Step 3g) -- IN PROGRESS
- [x] `--droplet-deformation` preset: [0,1]^2, R=0.2, chi=1.19, sigma=1.0
- [x] CLI flags: --chi, --field-strength, --ramp-time, --sigma-ch, --epsilon, --gamma
- [x] Sub-cell interface tracking: phi=0 contour via edge-crossing interpolation
  (old cell-midpoint method gave quantized AR values; new method gives continuous values)
- [x] Deformation sweep scripts: `scripts/deformation_sweep.sh` + `scripts/analyze_deformation_sweep.py`
- [x] First sweep (t_final=1.0, old tracking): D increases monotonically with Bo_m
- [ ] Second sweep (t_final=3.0, sub-cell tracking): RUNNING (6 jobs, ~3000 steps each)
- [ ] Final D vs Bo_m comparison plot with 2D/3D theory curves

## Remaining (NOT planned -- project stopping after deformation benchmark)

### Pumping Application (PhD Primary Contribution -- future work)
- [ ] Droplet transport in pumping channel with traveling-wave field
- [ ] Parametric study: droplet size, field frequency, intensity
- [ ] Role of surface tension in droplet integrity
- [ ] Viscosity contrast effects
- [ ] Multiple droplet manipulation
