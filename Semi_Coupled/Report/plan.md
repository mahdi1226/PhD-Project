# Implementation Plan: Monolithic Electromagnetics Block

## Goal
Combine the Poisson (phi) and Magnetization (M) subsystems into a single monolithic
block system, solving equations (42c) and (42d) from Nochetto et al. together.

---

## Phase 1: New Magnetics Module (standalone, MMS-testable)

### Step 1.1: magnetic_setup.h / magnetic_setup.cc
Create the FESystem and block infrastructure.

**FESystem definition:**
```cpp
FESystem<dim>(FE_DGQ<dim>(1), dim,   // M_x, M_y (DG, vector)
              FE_Q<dim>(2),   1)      // phi      (CG, scalar)
```

**Block structure:**
- Block 0: M DoFs (DG) -- M_x and M_y components
- Block 1: phi DoFs (CG) -- demagnetizing potential

**Tasks:**
- Single DoFHandler for the combined system
- DoFTools::count_dofs_per_block for block sizing
- BlockSparsityPattern with flux sparsity (DG face couplings in M-M block only)
- Constraints: hanging nodes for phi (CG), pin phi DoF 0 for Neumann nullspace,
  unconstrained for M (DG)
- Initialize BlockSparseMatrix and block vectors

**Coupling table for sparsity:**
- Cell couplings: all blocks couple (M-M, M-phi, phi-M, phi-phi)
- Face couplings: only M-M block (DG transport upwind flux)

### Step 1.2: magnetic_assembler.h / magnetic_assembler.cc
Monolithic assembly of the 2x2 block system.

**FEValuesExtractors:**
```cpp
const FEValuesExtractors::Vector M(0);       // components 0,1
const FEValuesExtractors::Scalar phi(dim);   // component 2
```

**Cell loop -- the 2x2 block system:**

| Block     | Term | Equation |
|-----------|------|----------|
| A_M (M-M) | (1/tau)(M^k, Z) + (1/T)(M^k, Z) + B_h^m(U^k, Z, M^k) | (42c) mass + relaxation + transport |
| C_M_phi (M-phi) | -(1/T) chi(theta) (grad phi^k, Z) | (42c) equilibrium source couples phi into M |
| C_phi_M (phi-M) | -(M^k, grad X) | (42d) magnetization as Poisson source |
| A_phi (phi-phi) | (grad phi^k, grad X) | (42d) Laplacian |

**RHS:**
- M block: (1/tau)(M^{k-1}, Z) + (1/T) chi(theta) (h_a, Z)
- phi block: (h_a, grad X)

**Face loop (DG transport):**
- Only contributes to M-M block
- Keep existing upwind flux from magnetization_assembler.cc
- Use FEValuesExtractors::Vector M to extract DG shape functions
- phi components have zero jumps (CG) -- no face contribution

**Additional inputs needed at quadrature points:**
- theta (phase field) for chi(theta) evaluation
- U (velocity) for transport term B_h^m
- h_a (applied field)

### Step 1.3: magnetic_solver.h / magnetic_solver.cc
MUMPS direct solver for the monolithic block system.

**Implementation:**
- Take the assembled BlockSparseMatrix
- Pass directly to SolverDirect (Amesos_Mumps via Trilinos)
- No preconditioner needed -- direct solve
- Return solver info (iterations=1 for direct, residual)

**Why MUMPS:**
- Block matrix is nonsymmetric (DG advection + skew cross-coupling)
- 2D problem, manageable size even at AMR level 7
- Eliminates iterative solver convergence as a debugging variable
- Already have MUMPS working in existing magnetization_solver.cc

### Step 1.4: MMS Test for Magnetics Block
Create standalone MMS test for the monolithic magnetics system.

**Location:** `mms/magnetic/magnetic_mms_test.cc`, `mms/magnetic/magnetic_mms.h`

**Strategy:**
- Manufacture exact solutions for both phi and M simultaneously
- phi_exact: smooth function satisfying Neumann BCs (e.g., cos(pi*x)*cos(pi*y/L_y))
- M_exact: consistent with phi_exact through the coupling
- Compute MMS source terms for both equations (42c) and (42d)
- Verify convergence rates:
  - phi (CG Q2): L2 rate = 3, H1 rate = 2
  - M (DG Q1): L2 rate = 2

**Test levels:**
1. Magnetics standalone (no flow, constant theta): verify block assembly + solve
2. Magnetics with variable theta: verify chi(theta) coefficient handling

---

## Phase 2: Integration into Full System

### Step 2.1: Modify core/phase_field.h
- Remove: separate Poisson and Magnetization DoFHandlers, matrices, assemblers, solvers
- Remove: Picard iteration members (picard_iterations, picard_omega, picard_tolerance)
- Add: single magnetic DoFHandler, BlockSparseMatrix, block vectors
- Add: MagneticSolver (MUMPS)
- Keep: existing CH and NS subsystems unchanged

### Step 2.2: Modify core/phase_field.cc
- Replace solve_poisson_magnetization_picard() with solve_magnetics()
- solve_magnetics() does: assemble block system -> MUMPS solve -> extract phi and M
  from block solution vector
- Remove: solve_poisson(), solve_magnetization(), project_equilibrium_magnetization()
- phi and M components extracted from block solution using block indices

### Step 2.3: Modify core/phase_field_setup.cc
- Replace setup_poisson_system() + setup_magnetization_system() with setup_magnetics_system()
- setup_magnetics_system(): FESystem, block sparsity, constraints, vector allocation

### Step 2.4: Modify core/phase_field_amr.cc
- Update AMR transfer to handle the combined DoFHandler
- SolutionTransfer for the block vector (handles DG+CG mixed transfer)

### Step 2.5: Update utilities/parameters.h/.cc
- Remove: picard_iterations, picard_omega, picard_tolerance, use_dg_transport
- Remove: separate magnetization solver params
- Add/keep: magnetics solver params (type=direct, solver=mumps)

### Step 2.6: Update CMakeLists.txt
- Remove old source files: poisson_assembler.cc, magnetization_assembler.cc,
  poisson_setup.cc, magnetization_setup.cc, poisson_solver.cc, magnetization_solver.cc
- Add new source files: magnetic_assembler.cc, magnetic_setup.cc, magnetic_solver.cc

---

## Phase 3: Delete Old Files

After Phase 2 compiles and MMS tests pass, delete:

**Assembly:**
- assembly/poisson_assembler.h, .cc
- assembly/magnetization_assembler.h, .cc

**Setup:**
- setup/poisson_setup.h, .cc
- setup/magnetization_setup.h, .cc

**Solvers:**
- solvers/poisson_solver.h, .cc
- solvers/magnetization_solver.h, .cc

**Diagnostics:**
- diagnostics/poisson_diagnostics.h
- diagnostics/magnetization_diagnostics.h

**Old MMS tests:**
- mms/poisson/ (entire directory)
- mms/magnetization/ (entire directory)
- mms/coupled/poisson_mag_mms_test.cc (Picard coupling test)

---

## Phase 4: Coupled MMS Tests

Update remaining coupled MMS tests to use new magnetics block:
- mms/coupled/ns_magnetization_mms_test.cc -> ns_magnetics
- mms/coupled/ns_poisson_mag_mms_test.cc -> ns_magnetics
- mms/coupled/full_system_mms_test.cc -> uses new magnetics

---

## Phase 5: Physical Validation

1. Square relaxation (CH only, h_a=0): verify CH still works
2. Droplet no-field (CH+NS): verify NS coupling unchanged
3. Rosensweig uniform r4: first magnetics physics test
4. Rosensweig uniform r6: production resolution
5. Compare with paper's Figure 1 (interface heights at t=0.9, 1.0, 1.5, 2.0)

---

## Key Equations Reference

**Equation (42c) -- Magnetization:**
(delta_M^k / tau, Z) - B_h^m(U^k, Z, M^k) + (1/T)(M^k, Z) = (1/T)(chi_theta H^k, Z)

where H^k = grad(phi^k) + h_a, and B_h^m is the DG skew-symmetric transport form.

**Equation (42d) -- Poisson:**
(grad phi^k, grad X) = (h_a^k - M^k, grad X)

**Block system [A]{x} = {b}:**

| A_M        C_M_phi | | M^k   |   | f_M   |
|                     | |       | = |       |
| C_phi_M    A_phi   | | phi^k |   | f_phi |

---

## Implementation Order
1. Start with Step 1.4 (MMS test) and Step 1.2 (assembler) in parallel
2. Then Step 1.1 (setup) and Step 1.3 (solver)
3. Verify with MMS standalone test
4. Then Phase 2 (integration)
5. Then Phase 3 (cleanup) and Phase 4 (coupled MMS)
6. Finally Phase 5 (physical validation)

---

## Status
- [ ] Phase 1: New magnetics module
- [ ] Phase 2: Integration
- [ ] Phase 3: Delete old files
- [ ] Phase 4: Coupled MMS tests
- [ ] Phase 5: Physical validation
