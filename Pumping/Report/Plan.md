# Implementation Plan

## Overview

Four-phase implementation: standalone CH, CH benchmarks, full coupling, pumping application.
See `FHD_PUMP.md` in project root for detailed research plan.
See `References/comparison_study.md` for numerical scheme decision analysis.

## Completed Phases

### Phase 1: Standalone Cahn-Hilliard (DONE)
- CH subsystem: Split formulation, CG Q2 for both phi and mu
- Backward Euler, convex-concave (Eyre) splitting, energy stability
- MMS verified: L2>=3, H1>=2

### Phase 2: CH Benchmarks (PARTIAL)
- CH+NS coupled MMS verified
- `ch_benchmark` driver working
- Remaining: Young-Laplace pressure jump, square relaxation quantitative

### Phase 3: Phase-Dependent Properties (DONE - Steps 3a-3e)
- chi(phi) in magnetization assembler
- nu(phi) in NS assembler
- Full coupled driver `fhd_ch_driver.cc` (6 subsystems)
- Full coupled MMS test (all rates pass)
- Unified VTK output

### Rosensweig Instability (Step 3f -- ABANDONED)
- Multiple runs with H=10-30 at ref 3-4: no spike formation
- Consistent with Nochetto scheme limitation (confirmed across 3+ projects)
- Not a convergence study in the paper; moved on to droplet deformation

### Droplet Deformation (Step 3g -- COMPLETE)
- Afkhami et al. (2010) JFM 663: D vs Bo_m under uniform field
- Chi=1.19, R=0.2, sigma=1.0, H0=1.0-5.0 (Bo_m=0.24-5.95)
- Sub-cell phi=0 contour tracking (edge-crossing interpolation)
- Sweep completed: t_final=3.0, ref 6, 6 H0 values
- D between 2D and 3D theories; linear regime not quantitatively matched
- Needs eps/mesh convergence study for proper validation

## Project Status: STOPPED (2026-03-07)

Solver infrastructure established. Deformation benchmark shows qualitatively
correct behavior. Quantitative validation requires convergence studies not
performed here. See Progress.md for final results table and conclusions.

## Architecture

Each subsystem follows the established pattern:
- `subsystem.h` -- public facade (class definition)
- `subsystem_setup.cc` -- DoF distribution, constraints, sparsity
- `subsystem_assemble.cc` -- weak form assembly (matrix + RHS)
- `subsystem_solve.cc` -- linear system solve
- `subsystem_output.cc` -- VTK output
- `tests/subsystem_mms.h` -- MMS exact solutions, sources, error computation
- `tests/subsystem_mms_test.cc` -- convergence study driver

## Build System

    cd Pumping && mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DDEAL_II_DIR=/path/to/dealii
    make -j$(nproc)
