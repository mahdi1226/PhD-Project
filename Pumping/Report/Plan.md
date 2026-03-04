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

## Current Phase: Benchmarks + Scheme Decision

### Rosensweig Instability (Step 3f, IN PROGRESS)
Spikes not forming with current scheme. Two possible causes:
1. **Parameter mismatch**: our eps/nu/H may differ from Zhang's
2. **Numerical scheme**: fully coupled treatment may over-damp instability

**Action plan**:
1. Read Zhang Section 4.3 for exact parameters
2. If parameter issue: adjust and re-run
3. If scheme issue: implement hybrid decoupling (Path C)

### Hybrid Decoupling (Path C, if needed)
Minimal changes to existing code:
1. CH solved first with old velocity (explicit convection)
2. Mag+Poisson Picard loop with old phi (explicit chi(phi))
3. NS+AngMom with old M,H and new phi
4. No IEQ/SAV -- keep Eyre splitting for CH
This is essentially what our driver already does; may need to tune explicit vs implicit treatment.

### Droplet Deformation (Step 3g)
- Circular droplet under uniform applied field
- Compare aspect ratio vs magnetic Bond number
- Validates coupling without requiring spike formation

## Future Phases

### Pumping Application (PhD Primary Contribution)
1. Traveling-wave magnetic field + ferrofluid droplet in channel
2. Phase-dependent Kelvin force transports droplet selectively
3. Parametric studies: size, frequency, intensity, surface tension, viscosity
4. Novel: no prior simulation of diffuse-interface droplet transport under traveling wave

### BDF2 + Full Micropolar (Methods Paper Potential)
1. Upgrade BDF1 -> BDF2 time integration
2. Add SAV variable for double-well energy
3. Add ZEC terms for coupling decoupling
4. Pressure-correction projection for NS
5. Prove energy stability for 6-field system
6. First BDF2 energy-stable decoupled scheme with angular momentum
7. Novelty: no existing paper combines BDF2 + decoupled + micropolar + CH

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

    cd Pumping && mkdir cmake-build-release && cd cmake-build-release
    cmake .. -DCMAKE_BUILD_TYPE=Release -DDEAL_II_DIR=/path/to/dealii
    make -j$(nproc)
