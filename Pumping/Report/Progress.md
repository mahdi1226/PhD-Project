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

### Droplet Deformation Benchmark (Step 3g) -- COMPLETE
- [x] `--droplet-deformation` preset: [0,1]^2, R=0.2, chi=1.19, sigma=1.0
- [x] CLI flags: --chi, --field-strength, --ramp-time, --sigma-ch, --epsilon, --gamma
- [x] Sub-cell interface tracking: phi=0 contour via edge-crossing interpolation
- [x] Deformation sweep scripts: `deformation_sweep.sh` + `analyze_deformation_sweep.py`
- [x] First sweep (t_final=1.0, old tracking): grid-quantized AR (all low-Bo_m identical)
- [x] Second sweep (t_final=3.0, sub-cell tracking): 6 H0 values, all completed
- [x] D vs Bo_m comparison plots with 2D/3D theory curves
- [x] Corrected 2D theory: D = Bo_m * chi / (3*(2+chi)^2) = 0.039 * Bo_m

#### Final Results (ref 6, eps=0.02, t_final=3.0)

| H0 | Bo_m | AR   | D_num | D_2D  | Ratio |
|----|------|------|-------|-------|-------|
| 1  | 0.24 | 1.02 | 0.011 | 0.009 | 1.23  |
| 1.5| 0.54 | 1.07 | 0.032 | 0.021 | 1.53  |
| 2  | 0.95 | 1.14 | 0.066 | 0.037 | 1.78  |
| 3  | 2.14 | 1.35 | 0.147 | 0.084 | 1.76  |
| 4  | 3.81 | 1.60 | 0.231 | 0.148 | 1.55  |
| 5  | 5.95 | 1.90 | 0.310 | 0.232 | 1.34  |

#### Conclusions
- Qualitatively correct: monotonic D(Bo_m), prolate deformation along field
- D falls between 2D and 3D small-deformation theories
- Linear regime issue: D/Bo_m is NOT constant -- superlinear D ~ Bo_m^1.3
- Numerical D is 1.2-1.8x above 2D theory even at lowest Bo_m
- Likely cause: diffuse interface too thick (eps/R=0.1) and/or insufficient mesh
- No eps-convergence or mesh-convergence study performed
- Sub-cell tracking essential (old cell-midpoint was grid-quantized)

### Convergence Study Infrastructure (added 2026-03-07)
- [x] `scripts/eps_convergence_sweep.sh`: runs eps = 0.04, 0.02, 0.01, 0.005 at fixed H0 and ref
- [x] `scripts/mesh_convergence_sweep.sh`: runs ref = 5, 6, 7 at fixed H0 and eps
- [x] `scripts/analyze_convergence.py`: auto-detects runs, computes D(eps) and D(h), plots convergence
- [x] `params.txt` output in each run directory (epsilon, H0, chi, etc. for auto-detection)

### Code Efficiency Cleanup (2026-03-07)
- [x] Removed dead code: `compute_applied_field_gradient` from applied_field.h (never called)
- [x] Consolidated duplicate: `skew_angular_convection_scalar` delegates to `skew_magnetic_cell_value_scalar`
- [x] Removed duplicate M_PI definition in fhd_ch_driver.cc (already in benchmark_initial_conditions.h)
- [x] Removed unused includes: `fe_interface_values.h` and `fe_q.h` from magnetization_assemble.cc
- [x] Updated material_properties.h comments: documented that sigmoid (not linear) is used for chi/nu

## PROJECT STOPPED (2026-03-07)

### Not Pursued (future work)
- [ ] eps-convergence study for deformation benchmark (infrastructure ready, not run)
- [ ] Mesh convergence study ref 5/6/7 at fixed eps (infrastructure ready, not run)
- [ ] Droplet transport in pumping channel with traveling-wave field
- [ ] Parametric study: droplet size, field frequency, intensity
