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

### Step 7: Paper Validation — Section 7.3 Stirring [IN PROGRESS]

#### Completed:
- Approach 1: 2 dipoles at y=-0.4, 400 steps ref 5 — good quality, matches paper
- Approach 2: 8 dipoles at y=-0.1, 100 steps ref 6 — mesh resolution insufficient
- Passive scalar integrated and working
- Configurable dipole frequency parameter
- Enhanced preset (--stirring-2-enhanced) added

#### Next steps (in order of priority):

**Step 7a: Add missing (M × W) spin-magnetization coupling [DONE]**
Treated explicitly (M^{k-1}), moves to RHS as -(M_old × W, Z). Extended magnetization
assembler with optional w_relevant + w_dof_handler. Updated all MMS sources (6 files).
Full coupled MMS rates preserved. See Diagnostics #13.

**Step 7b: Enable angular momentum convection [DONE]**
Changed include_convection to true in driver and full coupled MMS test. Extended AngMom
MMS sources (4 files) with U_old_disc, div_U_old_disc, include_convection. Full coupled
MMS rates preserved. See Diagnostics #14.

**Step 7c: Re-run Sections 7.1-7.3 with corrected physics**
After implementing Steps 7a-7b, re-run all experiments to see if results change:
- Spinning magnet: likely minimal change (w is small)
- Approach 1: likely minimal change
- Approach 2: could see differences

**Step 7d: Run Approach 2 at ref 7 (128x128)**
- Paper uses "100 elements in each space direction" — ref 7 is the closest match
- This should resolve the divU_L2 = 4.0 incompressibility issue
- Run: `mpirun -np 2 ./drivers/fhd_driver --stirring-2 -r 7 --steps 200`
- Estimated wall time: ~8 hours

**Step 7e: Run Approach 2 Enhanced at ref 7 (Figure 19)**
- f=40Hz, nu=nu_r=0.1, t=4.0
- Run: `mpirun -np 2 ./drivers/fhd_driver --stirring-2-enhanced -r 7 --steps 400`
- Compare scalar mixing pattern against Figure 19

**Step 7f: Visual Validation**
- Generate VTK snapshots at key times
- Compare velocity/magnetization/scalar fields against paper Figures 15-19
- Document quantitative comparison (U_max, mixing efficiency)

## Phase B: Cahn-Hilliard Extension [FUTURE]
Add two-phase diffuse interface model:
- Cahn-Hilliard equation with Flory-Huggins logarithmic free energy
- Phase-dependent material properties (viscosity, susceptibility)
- Capillary force coupling to NS
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
    mpirun -np 2 ./drivers/fhd_driver --spinning-magnet -r 5 --steps 400
    mpirun -np 2 ./drivers/fhd_driver --pumping -r 5 --steps 200
    mpirun -np 2 ./drivers/fhd_driver --stirring-1 -r 5 --steps 400
    mpirun -np 2 ./drivers/fhd_driver --stirring-2 -r 7 --steps 200
    mpirun -np 2 ./drivers/fhd_driver --stirring-2-enhanced -r 7 --steps 400
