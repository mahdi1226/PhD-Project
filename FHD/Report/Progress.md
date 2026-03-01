# Progress Tracker

## Completed

### Shared Infrastructure
- [x] `utilities/parameters.h/.cc` — full runtime configuration with MMS presets + experiment presets
- [x] `utilities/solver_info.h` — solver iteration/residual tracking
- [x] `mesh/mesh.h` — mesh creation (unit square, configurable refinement)
- [x] `physics/skew_forms.h` — skew-symmetric forms for NS convection and DG transport
- [x] `physics/applied_field.h` — point-dipole and uniform applied field (per-dipole directions/intensities)
- [x] `CMakeLists.txt` — modular build with per-subsystem libraries and test targets

### Standalone Subsystems (all MMS PASS)
- [x] **Poisson** (Eq. 42d): CG Q2, Neumann BCs, mean-zero pressure pin
  - Rates: L2=3.0, H1=2.0
- [x] **Magnetization** (Eq. 42c): DG Q2, SIP diffusion, DG transport (skew + upwind)
  - Rates: L2=3.0
- [x] **Navier-Stokes** (Eq. 42e): CG Q2/DG P1, monolithic saddle-point, direct solver
  - Rates: U_L2=3.0, U_H1=2.0, p_L2=2.1
- [x] **Angular Momentum** (Eq. 42f): CG Q2, reaction-diffusion
  - Rates: L2=3.0, H1=2.0

### Pairwise Coupling (all MMS PASS)
- [x] **Poisson <-> Magnetization** (Picard): Under-relaxed iteration (omega=0.35)
  - Rates: phi_L2=3.0, phi_H1=2.0, M_L2=3.0
- [x] **NS + Angular Momentum** (micropolar): Sequential solve per time step
  - Rates: U_L2=3.0, U_H1=2.0, p_L2=2.4, w_L2=3.2, w_H1=2.1

### Full 4-System Coupling (MMS PASS)
- [x] Kelvin force in NS assembly: mu_0 [(M.grad)H + 1/2(div M)H]
- [x] Magnetic torque in AngMom assembly: mu_0 (m x h, z)
- [x] Velocity transport in Magnetization: B_h^m(u; M, z) with face flux
- [x] Full coupled MMS test driver with Picard + sequential solve
- [x] NS convection MMS: skew form source with div_U_old_disc
- [x] All convergence rates verified:

| Field | Rate | Expected |
|-------|------|----------|
| U_L2  | 2.99 | 3.0 |
| U_H1  | 2.00 | 2.0 |
| p_L2  | 2.35 | 2.0 |
| w_L2  | 3.09 | 3.0 |
| w_H1  | 2.06 | 2.0 |
| phi_L2| 3.00 | 3.0 |
| phi_H1| 2.00 | 2.0 |
| M_L2  | 2.92 | 3.0 |

### Production Driver
- [x] `drivers/fhd_driver.cc` — full Algorithm 42 time loop
- [x] Presets: `--spinning-magnet`, `--pumping`, `--stirring-1`, `--stirring-2`, `--stirring-2-enhanced`
- [x] CLI: `--steps N`, `--dt`, `--vtk_interval`, `-r`, `-v`
- [x] NS convection enabled (`include_convection=true`) in production
- [x] Diagnostics CSV output per time step
- [x] VTK output at configurable intervals

### Passive Scalar Subsystem
- [x] `passive_scalar/` — CG Q2 convection-diffusion (Eq. 104)
- [x] Assembly: SUPG stabilization, backward Euler time integration
- [x] Step function IC: c=1 for y<0.5, c=0 for y>=0.5
- [x] Neumann BCs (no-flux)
- [x] Integrated into production driver

### Paper Validation — Section 7.1: Spinning Magnet (COMPLETE)
- [x] Single orbiting dipole at R=0.9, omega=pi rad/s
- [x] Phase 1 (t in [0,1]): linear ramp 0 -> 10, 2-3 Picard iters
- [x] Phase 2 (t in [1,2]): constant field, 1 Picard iter (steady state reached)
- [x] Phase 3 (t in [2,4]): orbiting dipole, 3-5 Picard iters
- [x] 400 steps, dt=0.01, ref 5 (32x32), wall time 889s
- [x] Results qualitatively match paper Section 7.1

### Paper Validation — Section 7.2: Pumping (COMPLETE)
- [x] 64 dipoles (32 below y=-0.1, 32 above y=1.1) in x in [2,4]
- [x] Traveling wave: alpha_s = |sin(omega*t - kappa*x_s)|^{2q}, f=10Hz, lambda=1, q=5
- [x] Domain [0,6] x [0,1], ref 5 (192x32 cells)
- [x] 200 steps, dt=0.01, 10 Picard iters/step, wall time 1723s
- [x] Results qualitatively match paper Section 7.2

### Physics Completeness Fixes (Steps 7a-7b)
- [x] **Spin-magnetization coupling (M × W)** (Eq. 52d, Issue #13): Treated explicitly,
  Mx RHS: -w·My_old, My RHS: +w·Mx_old. Extended mag assembler + 6 MMS source files.
- [x] **Angular momentum convection** (Eq. 52c, Issue #14): Enabled include_convection=true
  in driver and full MMS test. Extended 4 AngMom MMS source files.
- [x] All MMS convergence rates preserved after both fixes:

| Field | Rate | Expected |
|-------|------|----------|
| U_L2  | 2.99 | 3.0 |
| U_H1  | 2.00 | 2.0 |
| p_L2  | 2.37 | 2.0 |
| w_L2  | 3.09 | 3.0 |
| w_H1  | 2.06 | 2.0 |
| phi_L2| 3.00 | 3.0 |
| phi_H1| 2.00 | 2.0 |
| M_L2  | 2.92 | 3.0 |

### Paper Validation — Section 7.3: Stirring (IN PROGRESS)
- [x] **Approach 1** (`--stirring-1`): 2 dipoles at y=-0.4, opposite polarity, f=20Hz
  - 400 steps completed, ref 5, wall time 1275s
  - U_max = 0.016, divU_L2 = 0.006, p = [-9, +100]
  - c in [0, 1.001], mass conserved to 0.0007% — excellent quality
  - Matches paper expectation: minimal mixing (Figure 15)
- [x] **Approach 2** (`--stirring-2`): 8 dipoles at y=-0.1, traveling wave, f=20Hz
  - 100 steps completed, ref 6 (64x64), wall time ~3500s
  - U_max = 1.75 (paper: 4.33), divU_L2 = 4.0 — quality issue (see Diagnostics)
  - Mesh-dependent: ref 5 gives U=5.9, ref 6 gives U=1.7, paper (~100x100) gives U=4.33
- [ ] **Approach 2 Enhanced** (`--stirring-2-enhanced`): f=40Hz, nu=nu_r=0.1, t=4.0 (Figure 19)
  - Preset added, not yet run
- [ ] Run approach 2 at ref 7 (128x128) to match paper's ~100x100 mesh
- [ ] Visual validation of VTK output against paper Figures 15-19

## Remaining Work

### Near-term (Section 7.3 Completion)
- [ ] Run approach 2 at ref 7 (128x128) — paper uses "100 elements in each space direction"
- [ ] Run approach 2 enhanced at ref 7 (Figure 19 validation)
- [ ] Verify divU quality improves at ref 7
- [ ] Visual comparison of passive scalar mixing patterns against Figures 15-19

### Medium-term (Phase A Completion)
- [ ] AMR support: Adaptive mesh refinement in coupled solver (subface handling in DG transport)
- [ ] Performance optimization for ref 7 runs (~4x slower than ref 6)

### Long-term (Phase B)
- [ ] Cahn-Hilliard subsystem with Flory-Huggins logarithmic free energy
- [ ] Phase-dependent material properties (viscosity, susceptibility)
- [ ] Capillary force coupling to Navier-Stokes
- [ ] Two-phase MMS verification
- [ ] Two-phase benchmark validations

## File Structure

    FHD/
    ├── CMakeLists.txt
    ├── utilities/          — parameters, solver_info
    ├── mesh/               — mesh creation
    ├── physics/            — skew_forms, applied_field, material_properties
    ├── poisson/            — Poisson subsystem + MMS test
    ├── magnetization/      — DG magnetization subsystem + MMS test
    ├── navier_stokes/      — Micropolar NS subsystem + MMS test
    ├── angular_momentum/   — Angular momentum subsystem + MMS test
    ├── passive_scalar/     — Passive scalar (Eq. 104) CG Q2 convection-diffusion
    ├── mms_tests/          — Coupled MMS tests (pairwise + full)
    ├── drivers/            — Production driver (fhd_driver.cc)
    ├── experiments/        — Validation experiment configs
    ├── Reference/          — Paper PDF (arXiv:1511.04381)
    └── Report/             — This documentation
