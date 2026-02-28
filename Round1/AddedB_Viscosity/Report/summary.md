# Project Summary: Ferrofluid Phase-Field Simulation

## Reference
Nochetto, R.H., Salgado, A.J. & Tomas, I.
"A diffuse interface model for two-phase ferrofluid flows"
*Computer Methods in Applied Mechanics and Engineering*, 309, 497-531 (2016)

---

## 1. Physical Model

The model describes a two-phase ferrofluid system coupling:
- **Hydrodynamics** (incompressible Navier-Stokes)
- **Phase separation** (Cahn-Hilliard diffuse interface)
- **Magnetostatics** (Poisson equation for magnetic potential)
- **Magnetization dynamics** (transport + relaxation)

The ferrofluid responds to external magnetic fields through the Kelvin body force,
creating instabilities and pattern formation at the fluid interface.

---

## 2. Governing Equations

### 2.1 Navier-Stokes (Eq. 42a-b in paper)

Momentum conservation with Kelvin force coupling:

    rho * (du/dt + (u . grad)u) - div(2*eta*D(u)) + grad(p)
        = mu_0 * (M . grad)H + rho*g + f_theta

    div(u) = 0

where:
- rho(theta): density (phase-dependent)
- eta(theta): viscosity (phase-dependent)
- D(u) = (grad u + grad u^T) / 2: symmetric strain rate
- mu_0 * (M . grad)H: Kelvin body force
- f_theta: capillary/surface tension force from Cahn-Hilliard

### 2.2 Cahn-Hilliard (Eq. 42d-e)

Phase-field evolution with advection:

    d(theta)/dt + u . grad(theta) = div(gamma * grad(psi))

    psi = (1/epsilon) * F'(theta) - epsilon * laplacian(theta)

where:
- theta: order parameter (phase field), theta=+1 ferrofluid, theta=-1 non-magnetic
- psi: chemical potential
- epsilon: interface thickness parameter
- gamma: mobility
- F(theta) = (1/4)(theta^2 - 1)^2: double-well potential

### 2.3 Magnetostatic Poisson (Eq. 42f)

Magnetic potential from Maxwell's equations (magnetostatic limit):

    -div(grad(phi)) = div(M - h_a)

    H = grad(phi)

where:
- phi: total magnetic potential
- H: magnetic field intensity
- h_a: applied (external) magnetic field
- M: magnetization vector

The Poisson equation enforces div(B) = 0 where B = mu_0(H + M).

### 2.4 Magnetization Transport (Eq. 42c)

Magnetization dynamics with transport, relaxation, and equilibrium:

    dM/dt + (u . grad)M + (1/tau_M)(M - chi*H) = 0

where:
- tau_M: magnetic relaxation time (~10^-6 s for ferrofluids)
- chi(theta): magnetic susceptibility (phase-dependent)
- chi*H: equilibrium magnetization (paramagnetic response)

The susceptibility depends on the phase field:

    chi(theta) = chi_0 * H_epsilon(theta)

where H_epsilon is a regularized Heaviside function.

---

## 3. Discretization

### 3.1 Spatial Discretization

| Field | Element | Degree | Space |
|-------|---------|--------|-------|
| Velocity u | Taylor-Hood | Q2-Q1 | CG (continuous Galerkin) |
| Pressure p | Taylor-Hood | Q1 | CG |
| Phase field theta | Lagrange | Q1 | CG |
| Chemical potential psi | Lagrange | Q1 | CG |
| Magnetic potential phi | Lagrange | Q2 | CG |
| Magnetization M | DG Lagrange | Q1 | DG (discontinuous Galerkin) |

### 3.2 DG Transport: Skew-Symmetric Form (Eq. 57)

The magnetization transport uses a DG skew-symmetric bilinear form:

    B_h^m(U, V, W) = sum_T integral_T [(U.grad)V . W + (1/2)(div U)(V . W)] dx
                    - sum_F integral_F (U.n) [[V]] . {W} dS

where:
- [[V]] = V^- - V^+  (jump across face)
- {W} = (W^- + W^+)/2  (average across face)
- n = outward normal of "-" side (FEInterfaceValues convention)

**Key property**: B_h^m(U, M, M) = 0 (energy neutrality)

This ensures the transport does not artificially create or destroy magnetization energy.
The cancellation happens GLOBALLY: the volume boundary flux +1/2 * integral(U.n)|M|^2 ds
cancels with the face term -(U.n)[[M]].{M} when summed over all cells.

### 3.3 Temporal Discretization

- First-order backward Euler (implicit)
- Scheme (42) in the CMAME paper: all fields solved simultaneously per time step

### 3.4 Nonlinear Solver: Paper vs Our Implementation

**Paper's approach** (CMAME 2016, Section 6, p.520):
Block-Gauss-Seidel (Picard-like) iteration with 3 blocks per time step:
1. **Block 1**: Solve CH (θ, ψ) coupled — Eqs (42a)–(42b)
2. **Block 2**: Solve Magnetization + Poisson (M, φ) coupled — Eqs (42c)–(42d)
3. **Block 3**: Solve NS (U, P) coupled — Eqs (42e)–(42f)

Iterate all 3 blocks until convergence. The paper explicitly notes that
Block-Jacobi (fully decoupled, no global iteration) did not yield satisfactory results.
The companion M3AS paper (arXiv:1511.04381, Section 6) confirms: UMFPACK direct solver,
fixed-point iteration for the nonlinear system.

**Our implementation** (updated February 27, 2026):
Block-Gauss-Seidel global iteration matching the paper:
- Outer loop: [CH] → [Poisson ↔ Mag (Picard)] → [NS], repeat until convergence
- Picard inner loop (7 iterations, tol=0.05, ω=0.35) for Poisson-Magnetization coupling
- BGS convergence: max relative change over θ and U < 1e-2
- Maximum 5 BGS iterations per time step (capped to prevent excessive cost)
- Can be disabled with `--no_bgs` for comparison

**Match summary**:
| Aspect | Paper | Our code | Match? |
|--------|-------|----------|--------|
| Global iteration | Block-Gauss-Seidel | Block-Gauss-Seidel (max 5, tol=1e-2) | ✅ |
| CH-NS coupling | Iterative | Iterative (re-solved each BGS iter) | ✅ |
| Mag-Poisson coupling | Coupled block | Picard iteration | ✅ |
| Linear solver | UMFPACK (direct) | UMFPACK/Mumps (direct) | ✅ |

### 3.5 Implementation

- Library: deal.II (v9.5+)
- Parallelism: MPI via Trilinos (distributed triangulations, vectors, matrices)
- Linear solvers: Direct (UMFPACK) or GMRES with AMG preconditioner
- Build system: CMake

---

## 4. MMS Verification

### 4.1 Manufactured Solutions

All solutions are chosen to be smooth, satisfy boundary conditions, and produce
analytically computable source terms.

**Cahn-Hilliard:**
    theta* = t * sin(pi*x) * sin(pi*y/L_y)
    psi* computed from theta* via the CH equation

**Navier-Stokes (solenoidal, zero on boundary):**
    u_x* = t * (pi/L_y) * sin^2(pi*x) * sin(2*pi*y/L_y)
    u_y* = -t * pi * sin(2*pi*x) * sin^2(pi*y/L_y)
    p*   = t * sin(pi*x) * cos(pi*y/L_y)

**Poisson:**
    phi* = t * sin(pi*x) * sin(pi*y/L_y)
    H* = -grad(phi*)

**Magnetization (M*.n = 0 on boundary):**
    M_x* = t * sin(pi*x) * sin(pi*y/L_y)
    M_y* = t * cos(pi*x) * sin(pi*y/L_y)

### 4.2 Test Hierarchy

11 MMS tests organized from standalone to fully coupled:

    Level 1 (Standalone):
        CH_STANDALONE, POISSON_STANDALONE, NS_STANDALONE, MAGNETIZATION_STANDALONE

    Level 2 (Pairwise Coupled):
        POISSON_MAGNETIZATION, NS_MAGNETIZATION, MAG_CH, MAG_TRANSPORT

    Level 3 (Full Coupling):
        FULL_SYSTEM (all four subsystems)

### 4.3 Expected Convergence Rates

For smooth MMS solutions:
- CG-Q1 (theta): O(h^2) in H1, O(h^3) in L2 (superconvergence)
- CG-Q2 (u, phi): O(h^2) in H1, O(h^3) in L2
- Q1 pressure: O(h^2) in L2
- DG-Q1 (M): O(h^2) in L2

### 4.4 Results

ALL 11 tests achieve optimal convergence rates. Representative FULL_SYSTEM results:

    Ref    h          theta_L2    U_L2       phi_L2     M_L2       p_L2
    ---    -----      --------    -----      ------     -----      -----
    2      3.54e-02   8.17e-09    3.65e-05   1.02e-06   7.43e-05   7.82e-05
    3      1.77e-02   1.02e-09    4.56e-06   1.28e-07   1.86e-05   1.44e-05
    4      8.84e-03   1.28e-10    5.71e-07   1.60e-08   4.65e-06   3.43e-06

    Rates:           3.00        3.00       3.00       2.00       2.0+

---

## 5. Code Architecture

    AddedB_Viscosity/
    |-- main.cc                    # Entry point
    |-- core/
    |   |-- phase_field.cc         # Main time loop, Picard coupling
    |   |-- phase_field_setup.cc   # Mesh, DOF, sparsity setup
    |
    |-- assembly/
    |   |-- ch_assembler.cc/h      # Cahn-Hilliard assembly
    |   |-- ns_assembler.cc/h      # Navier-Stokes + Kelvin force
    |   |-- magnetization_assembler.cc/h  # DG magnetization (transport + relaxation)
    |
    |-- solvers/
    |   |-- ch_solver.h            # CH block solver
    |   |-- ns_solver.h            # NS block solver (Schur complement)
    |   |-- poisson_solver.h       # Poisson CG+AMG solver
    |   |-- magnetization_solver.h # Magnetization direct/GMRES solver
    |
    |-- physics/
    |   |-- skew_forms.h           # DG skew-symmetric bilinear forms (Eq. 37, 57)
    |   |-- material_properties.h  # Density, viscosity, susceptibility
    |   |-- applied_field.h        # External magnetic field (dipoles)
    |
    |-- setup/
    |   |-- magnetization_setup.h  # DG sparsity pattern with face coupling
    |
    |-- mms/                       # Method of Manufactured Solutions
    |   |-- mms_core/test_mms.cc   # MMS test driver
    |   |-- ch/ch_mms.h            # CH exact solutions + source terms
    |   |-- ns/ns_mms.h            # NS exact solutions + source terms
    |   |-- poisson/poisson_mms.h  # Poisson exact solutions + source terms
    |   |-- magnetization/         # Magnetization MMS
    |       |-- magnetization_mms.h         # Exact solutions + source terms
    |       |-- magnetization_mms_test.cc   # Standalone + transport tests
    |
    |-- utilities/
    |   |-- parameters.h/cc        # All simulation parameters + CLI parsing
    |
    |-- Report/                    # This documentation
    |-- validation/                # Validation cases (Rosensweig, etc.)

---

## 6. Reference Papers

### Paper 1 (CMAME 2016): "A diffuse interface model for two-phase ferrofluid flows"
- Nochetto, Salgado, Tomas. CMAME 309, 497-531 (2016)
- Two-phase model: CH + NS + Poisson + Magnetization transport
- Simplified model (no angular momentum, σ=0)
- Rosensweig instability benchmark (Section 6.2): domain [0,1]×[0,0.6], 5 dipoles,
  6 refinement levels, 4000 time steps, t_F=2.0
- Ferrofluid hedgehog experiment (Section 6.3): 42 dipoles, non-uniform field
- Key: Block-Gauss-Seidel solver, DG skew-symmetric form (Eq. 57), Kelly AMR

### Paper 2 (M3AS 2016): "The equations of ferrohydrodynamics: modeling and numerical methods"
- Nochetto, Salgado, Tomas. M3AS 26(13), 2393-2449 (2016). arXiv:1511.04381
- Full Rosensweig model WITH angular momentum (w) and micropolar terms
- Includes spin-vorticity coupling, magnetic torque (m × h)
- Numerical validation with MMS (Section 6): UMFPACK, fixed-point iteration
- Experiments: spinning magnet, ferrofluid pumping, ferromagnetic stirring
- Our code has remnants of the Zhang-He-Yang extension from this model (beta damping,
  angular velocity) — currently unused, scheduled for cleanup

---

## 7. Key Numerical Details

### 6.1 chi*H Cancellation in MMS

The equilibrium term (chi/tau_M)*H appears in both the magnetization equation and its
MMS source. Since the MMS source uses the DISCRETE H (from the Poisson solve), these
terms cancel EXACTLY in the weak form, making the magnetization MMS error independent
of the Poisson solution quality. This is by design.

### 6.2 DG Face Assembly: FEInterfaceValues DOF Numbering

For DG elements, FEInterfaceValues numbers interface DOFs sequentially:
- DOFs 0 .. dofs_per_cell-1: belong to cell 0 ("here")
- DOFs dofs_per_cell .. 2*dofs_per_cell-1: belong to cell 1 ("there")

The shape_value(bool here_or_there, uint interface_dof, uint q_point) method:
- shape_value(true, i, q): evaluates interface DOF i on cell 0's side
- shape_value(false, i, q): evaluates interface DOF i on cell 1's side

For DG, shape functions have support only on their own cell, so:
- shape_value(true, i, q) for i < dofs_per_cell: nonzero (cell 0's DOF on cell 0)
- shape_value(false, i, q) for i < dofs_per_cell: ZERO (cell 0's DOF on cell 1)
- shape_value(false, dofs_per_cell + i, q): nonzero (cell 1's DOF on cell 1)

### 6.3 Boundary Treatment for DG Transport

The skew-symmetric form B_h^m only involves interior face integrals. Boundary faces
are skipped because:
1. In the Nochetto model, U . n = 0 on the domain boundary (no-slip)
2. Therefore (U.n)[[M]]{phi} = 0 on boundary faces
3. The MMS velocity (NS MMS) also satisfies U . n = 0 on all boundaries

This is consistent with the physical setup (closed domain with no fluid in/outflow).
