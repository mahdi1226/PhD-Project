# Implementation Plan

## Overview

Four-phase implementation: standalone CH, CH benchmarks, passive CH in pumping,
full two-way coupling. See `FHD_PUMP.md` in project root for detailed research plan.

## Phase 1: Standalone Cahn-Hilliard

Build and verify CH subsystem independently with MMS convergence tests.

1. CH subsystem: Split formulation, CG Q2 for both phi and mu
   - Backward Euler time stepping
   - Convex-concave (Eyre) splitting for energy stability
   - Convection term: u . grad(phi) (velocity from external source)
   - BCs: natural (no-flux) for both phi and mu
2. MMS test: Manufactured smooth solution, verify L2/H1 convergence
3. Solver: CG + AMG (system is symmetric if no convection)

**Expected rates**: L2 >= 3, H1 >= 2 for CG Q2

## Phase 2: CH Benchmarks

Verify physics without magnetics.

1. Circular droplet equilibrium — pressure jump = sigma/R (Young-Laplace)
2. Square relaxation — square relaxes to circle, conserving area
3. Advected droplet — prescribed velocity, mass conservation

## Phase 3: Passive CH in Pumping Flow

One-way coupling proof of concept.

1. Copy FHD pumping setup (Section 7.2: 64 dipoles, traveling wave)
2. Replace passive scalar with CH equation
3. One-way: FHD velocity advects CH, phase doesn't affect flow
4. IC: ferrofluid droplet (phi=+1) in carrier (phi=-1) in channel
5. Observe droplet transport, deformation under magnetic pumping

## Phase 4: Two-Way Coupling

Full coupled system — phase affects magnetic response and flow.

1. Phase-dependent chi(phi): chi = 0 in carrier, chi = kappa_0 in ferrofluid
2. Capillary force: sigma * mu_CH * grad(phi) in NS RHS
3. Phase-dependent viscosity: nu(phi) interpolation
4. Modified Picard iteration with CH solve in loop
5. MMS verification of full coupled system
6. Production runs: droplet transport in pumping channel

## Architecture

Each subsystem follows the established pattern:
- `subsystem.h` — public facade (class definition)
- `subsystem_setup.cc` — DoF distribution, constraints, sparsity
- `subsystem_assemble.cc` — weak form assembly (matrix + RHS)
- `subsystem_solve.cc` — linear system solve
- `subsystem_output.cc` — VTK output
- `tests/subsystem_mms.h` — MMS exact solutions, sources, error computation
- `tests/subsystem_mms_test.cc` — convergence study driver

## Build System

    cd Pumping && mkdir build && cd build
    cmake .. -DDEAL_II_DIR=/path/to/dealii
    make -j$(nproc)
