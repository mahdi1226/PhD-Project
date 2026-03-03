# Implementation Plan

## Architecture

Each subsystem follows a consistent pattern:
- `subsystem.h` — public facade (class definition)
- `subsystem_setup.cc` — DoF distribution, constraints, sparsity
- `subsystem_assemble.cc` — weak form assembly (matrix + RHS)
- `subsystem_solve.cc` — linear system solve
- `subsystem_output.cc` — VTK output
- `tests/subsystem_mms.h` — MMS exact solutions, sources, error computation
- `tests/subsystem_mms_test.cc` — convergence study driver

Shared infrastructure:
- `utilities/parameters.h/.cc` — all runtime configuration + experiment presets
- `utilities/solver_info.h` — solver iteration/residual tracking
- `mesh/mesh.h` — mesh creation helpers
- `physics/skew_forms.h` — skew-symmetric transport + convection forms
- `physics/applied_field.h` — external magnetic field h_a (point dipoles + uniform)
- `physics/material_properties.h` — phase-dependent properties (Phase B)

Production driver:
- `drivers/fhd_driver.cc` — full Algorithm 42 time loop with diagnostics and VTK output

## Phase A Implementation Steps

### Step 1: Standalone Subsystems [DONE]
Build and verify each subsystem independently with MMS convergence tests.

1. Poisson (CG Q2, Neumann BCs, mean-zero pin) — L2=3, H1=2
2. Magnetization (DG Q2, SIP + transport, Debye relaxation) — L2=3
3. Navier-Stokes (CG Q2/DG P1, saddle-point, direct solver) — U_L2=3, U_H1=2, p_L2=2
4. Angular Momentum (CG Q2, reaction-diffusion) — L2=3, H1=2

### Step 2: Pairwise Coupling [DONE]
Verify coupling terms between subsystem pairs.

1. Poisson <-> Magnetization (Picard iteration with under-relaxation)
2. NS + Angular Momentum (micropolar: curl coupling)

### Step 3: Full 4-System Coupling [DONE]
Wire all cross-coupling terms and verify with full MMS test.

- Kelvin force in NS: mu_0 [(M . grad)H + 1/2 (div M) H]
- Micropolar in NS: 2 nu_r (w, curl v)
- Curl coupling in AngMom: 2 nu_r (curl u, z)
- Magnetic torque in AngMom: mu_0 (m x h, z)
- Velocity transport in Mag: B_h^m(u; M, z) with skew + upwind
- Poisson-Mag Picard for H-M coupling
- NS convection: b_h(u_old; u, v) skew-symmetric form

### Step 4: Production Driver [DONE]
- Configurable physics parameters via presets and CLI
- VTK output at regular intervals
- Diagnostics CSV logging (velocity, pressure, divergence, energy, CFL, scalar bounds)
- Passive scalar subsystem integrated

### Step 5: Paper Validation — Section 7.1 Spinning Magnet [DONE]
- Single orbiting dipole, 3 phases (ramp, steady, orbiting)
- 400 steps, dt=0.01, ref 5

### Step 6: Paper Validation — Section 7.2 Pumping [DONE]
- 64 dipoles, traveling wave activation
- 200 steps, dt=0.01, ref 5

### Step 7: Paper Validation — Section 7.3 Stirring [COMPLETED — with known limitation]

#### Completed:
- [x] Step 7a: Spin-magnetization coupling (M × W) — all MMS pass
- [x] Step 7b: Angular momentum convection enabled — all MMS pass
- [x] Approach 1: 2 dipoles, f=20Hz — U=0.016, matches paper ("less than 10⁻²") ✅
- [x] Approach 2: 8 dipoles traveling wave — ref 5 U=4.31 matches paper U=4.33 ✅
- [x] Approach 2 Enhanced: f=40Hz, nu=0.1 — U range [1.85, 3.67]
- [x] Kelvin face integrals (Eq. 38, 2nd line) — required for energy identity
- [x] Block-Schur preconditioner — 26× speedup at ref 7
- [x] SUPG stabilization for passive scalar
- [x] Kelvin force diagnostics — cell/face L2 norms and resultant force

#### Known limitation: velocity mesh-dependence
Velocity decreases with mesh refinement (ref 5: U=4.31, ref 7: U=0.51) despite
converged Kelvin force (cell_L2 ≈ 3.73e4, Fy ≈ -7676 at all refinements).
Diagnosis: pressure robustness issue with Q2/DG-P1 for near-singular body force.
Paper likely uses ~32×32 mesh for this proof-of-concept section.
See Diagnostics Issue #10 for full analysis.

#### Decision: Move to Phase B
Section 7.3 is a proof of concept in the paper. The mesh-dependence is a known
limitation of standard mixed FE for near-singular forcing, not a code bug.
Sections 7.1 and 7.2 are fully validated. Moving to two-phase research (Phase B)
is more productive. See FHD_PUMP.md for the Phase B plan.

## Phase B: Two-Phase Ferrofluid Droplet Transport [NEXT]

**Research goal**: Ferrofluid droplets in non-magnetic carrier (water/blood),
manipulated by traveling-wave magnetic field in pumping channel.

**Separate project folder**: `Droplet/` (parallel to `FHD/`)
**Full plan**: See `/PhD-Project/FHD_PUMP.md`

### Implementation phases:
1. **Standalone CH**: Split formulation (phi + mu), CG Q2, MMS verified
2. **CH Benchmarks**: Circular droplet equilibrium + square relaxation
3. **Passive CH in pumping**: One-way coupling (FHD velocity advects CH)
4. **Full two-way coupling**: Phase-dependent chi(phi), nu(phi), capillary force

### Key additions to FHD framework:
- Cahn-Hilliard subsystem (split formulation, Eyre's convex-concave splitting)
- Phase-dependent material properties: chi(phi), nu(phi)
- Capillary force: sigma * mu_CH * grad(phi) in NS
- Modified magnetization with phase-dependent susceptibility

## Build System

    cd FHD && mkdir build && cd build
    cmake .. -DDEAL_II_DIR=/path/to/dealii
    make -j$(nproc)

Dependencies: deal.II >= 9.4 (with MPI, Trilinos, p4est)

## Test Execution

    # Standalone MMS tests
    mpirun -np 2 ./poisson/test_poisson_mms --refs 3 4 5
    mpirun -np 2 ./magnetization/test_magnetization_mms --refs 3 4 5
    mpirun -np 2 ./navier_stokes/test_navier_stokes_mms --refs 3 4 5
    mpirun -np 2 ./angular_momentum/test_angular_momentum_mms --refs 3 4 5

    # Coupled MMS tests
    mpirun -np 2 ./mms_tests/test_poisson_mag_mms --refs 3 4 5
    mpirun -np 2 ./mms_tests/test_ns_angmom_mms --refs 3 4 5
    mpirun -np 2 ./mms_tests/test_full_coupled_mms --refs 3 4 5

    # Production experiments
    mpirun -np 1 ./drivers/fhd_driver --spinning-magnet -r 5 --steps 400
    mpirun -np 1 ./drivers/fhd_driver --pumping -r 5 --steps 200
    mpirun -np 1 ./drivers/fhd_driver --stirring-1 -r 7 --block-schur --steps 400
    mpirun -np 1 ./drivers/fhd_driver --stirring-2 -r 5 --block-schur --steps 200
    mpirun -np 1 ./drivers/fhd_driver --stirring-2-enhanced -r 7 --block-schur --steps 400

    # Note: 1 MPI rank is fastest for direct solver at current problem sizes
    # Use --block-schur for ref 7 runs (26x speedup)
    # Use --cells N for exact NxN mesh (overrides -r)
