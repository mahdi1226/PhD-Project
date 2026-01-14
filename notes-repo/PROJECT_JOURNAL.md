# Ferrofluid Phase Field Solver - Project Journal

**Last Updated:** December 12, 2025
**Reference:** Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531

---

## Project Overview

Implementing a ferrofluid solver based on Nochetto's paper with:
- Cahn-Hilliard (phase separation)
- Poisson (magnetostatic potential)
- Navier-Stokes (fluid flow with Kelvin force)

**Goal:** Reproduce Rosensweig instability (ferrofluid spikes under magnetic field)

---

## Current File Structure

```
PhaseField/
├── CMakeLists.txt
├── main.cc
├── core/
│   ├── phase_field.h          # Main problem class
│   ├── phase_field.cc         # Constructor, run(), do_time_step(), solve_ns()
│   └── phase_field_setup.cc   # setup_mesh(), setup_dof_handlers(), setup_*_system()
├── assembly/
│   ├── ch_assembler.h/.cc     # Cahn-Hilliard assembly
│   ├── poisson_assembler.h/.cc # Poisson with μ(θ), dipole BCs
│   └── ns_assembler.h/.cc     # Navier-Stokes with all forces
├── solvers/
│   ├── ch_solver.h/.cc        # UMFPACK for CH
│   ├── poisson_solver.h/.cc   # CG+SSOR or UMFPACK for Poisson
│   └── ns_solver.h/.cc        # UMFPACK for NS (saddle point)
├── setup/
│   ├── ch_setup.h/.cc         # CH coupled system setup
│   ├── poisson_setup.h/.cc    # Poisson sparsity
│   └── ns_setup.h/.cc         # NS 3-field coupled system
├── physics/
│   ├── material_properties.h  # viscosity(θ), susceptibility(θ), permeability(θ)
│   ├── kelvin_force.h         # compute_magnetic_field(), compute_kelvin_force()
│   └── initial_conditions.h   # InitialTheta, InitialPsi, etc.
├── diagnostics/
│   └── ch_mms.h/.cc           # CH MMS verification (O(h²) verified)
├── utilities/
│   ├── parameters.h/.cc       # All parameters + command-line parsing
│   └── tools.h                # timestamped_folder()
└── output/
    └── (vtk_writer, logger - not integrated yet)
```

---

## What's Working ✅

| Component | Status | Verification |
|-----------|--------|--------------|
| Cahn-Hilliard | ✅ Complete | MMS verified O(h²) |
| Poisson | ✅ Complete | μ(θ) ∈ [1, 1.5], dipole BCs ramping |
| Navier-Stokes | ✅ Complete | Forces computed, flow developing |
| Full integration | ✅ Complete | Staggered CH→Poisson→NS |
| VTK output | ✅ Complete | θ, ψ, φ, ux, uy |
| Rosensweig spikes | ✅ Forming! | Visible in ParaView |

---

## Parameters from Paper (Section 6.2, p.520-522)

```cpp
// Domain
x_min = 0.0, x_max = 1.0
y_min = 0.0, y_max = 0.6

// Cahn-Hilliard
epsilon = 0.01      // Interface thickness
gamma = 0.0002      // Mobility
lambda = 0.05       // Capillary coefficient

// Navier-Stokes
nu_water = 1.0      // Water viscosity
nu_ferro = 2.0      // Ferrofluid viscosity
mu_0 = 1.0          // Permeability of free space
r = 0.1             // Density ratio

// Gravity
g = 30000           // Magnitude
direction = (0, -1) // Downward

// Magnetization
chi_0 = 0.5         // Susceptibility (κ₀ in paper)

// Dipoles (5 sources) - CORRECTED
positions = [(-0.5, -15), (0, -15), (0.5, -15), (1, -15), (1.5, -15)]
direction = (0, 1)  // Upward
intensity: 0 → 6000 linearly over t ∈ [0, 1.6], constant after

// Time stepping
dt = 5e-4
t_final = 2.0

// Initial condition
Pool depth = 0.2 (20% of domain height)
θ = +1 (ferrofluid) below, θ = -1 (air/water) above
```

---

## Key Equations (from paper)

### Cahn-Hilliard (Eq. 14a-14b)
```
θ_t + u·∇θ - γΔψ = 0
ψ = -εΔθ + (1/ε)f'(θ)   where f'(θ) = θ³ - θ
```

### Poisson/Magnetostatics (Eq. 14d)
```
-∇·(μ(θ)∇φ) = 0   in Ω
φ = φ_dipole       on ∂Ω

μ(θ) = 1 + χ₀·H(θ/ε)   where H(x) = 1/(1+e^(-x))
```

### Dipole potential (Eq. 97)
```
φ_s(x) = α(t) × (d·r) / |r|²
where r = x - x_dipole, d = direction
```

### Navier-Stokes (Eq. 14e-14f)
```
u_t + (u·∇)u - ∇·(ν(θ)∇u) + ∇p = F_cap + F_mag + F_grav
∇·u = 0

F_cap = (λ/ε)θ∇ψ           (capillary/surface tension)
F_mag = μ₀χ(θ)(H·∇)H       (Kelvin force, H = -∇φ)
F_grav = (1 + r·H(θ/ε))g   (Boussinesq gravity)
```

---

## Command-Line Usage

```bash
# CH only (baseline test)
./ferrofluid --dt 5e-4 --t_final 0.1

# CH + Poisson
./ferrofluid --magnetic --dt 5e-4 --t_final 0.1

# Full solver (CH + Poisson + NS)
./ferrofluid --ns --magnetic --dt 5e-4 --t_final 0.5

# MMS verification
./ferrofluid --mms --dt 1e-4 --t_final 0.2

# Help
./ferrofluid --help
```

---

## TODO List (Priority Order)

### HIGH Priority
1. **CSV diagnostics output** - Monitor mass, energy, forces, CFL, divergence
2. **Investigate instability** - Time step may need adjustment
3. **Pressure output** - Currently not in VTK (different FE)

### MEDIUM Priority
4. **Poisson MMS verification**
5. **NS MMS verification**
6. **Iterative NS solver** (GMRES + Schur preconditioner)
7. **Picard iteration** for nonlinear coupling
8. **Adaptive time stepping**

### LOW Priority
9. **AMR integration**
10. **Full coupled NSCH MMS**
11. **File cleanup** (remove unused files)

---

## Known Issues

1. **Potential instability** at current dt - needs diagnostics to investigate
2. **Pressure not in VTK** - uses Q1 elements, different from Q2 fields
3. **No CSV output yet** - hard to track convergence/stability
4. **Boundary conditions** - using no-slip everywhere for NS

---

## Files to Remove (unused)

- `output/vtk_writer` - using DataOut directly
- `physics/applied_field` - integrated into poisson_assembler
- `physics/boundary_conditions` - integrated into assemblers
- `utilities/block_structure` - not using block vectors
- `utilities/trilinear_forms` - not needed
- `utilities/tensor_operations` - using deal.II directly
- `utilities/linear_algebra` - not using block types

---

## Recent Bug Fixes

1. **Dipole y-position:** Was -1.5, corrected to **-15** (paper value)
2. **kelvin_force.h:** Created but wasn't being used - now integrated into ns_assembler
3. **fe_phase_ → fe_Q2_:** Renamed for consistency with Q2/Q1 elements

---

## Test Results

### CH Only (MMS Verified)
```
Refinement 5: θ L2 error = 9.64e-06, H1 error = 5.09e-05
Refinement 6: θ L2 error = 2.41e-06, H1 error = 1.27e-05
Convergence rate: O(h²) ✓
```

### Full Solver (t=0.1)
```
DoFs: θ=4225, ψ=4225, φ=4225, ux=4225, uy=4225, p=1089
Forces at step 200:
  |F_cap| = 14,504
  |F_mag| = 505,597  (dominant!)
  |F_grav| = 33,000
  |u|_max = 25.6
```

---

## Architecture Decisions

1. **Separate DoFHandlers** for each field (AMR-compatible)
2. **Index maps** to build coupled systems from scalar fields
3. **Header-only physics** (material_properties.h, kelvin_force.h) for performance
4. **Free functions** for setup (avoid circular dependencies)
5. **Staggered time stepping:** CH → Poisson → NS (not monolithic)

---

## Next Conversation Starting Point

1. Upload this journal
2. Upload current source files (or describe location)
3. Priority: Implement CSV diagnostics to debug stability
4. Then: Work through TODO list

---

*End of Journal*
