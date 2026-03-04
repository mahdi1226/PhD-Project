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

### 2.1 Navier-Stokes (Eq. 42e in paper)

Momentum conservation with Kelvin force coupling:

    rho * (du/dt + (u . grad)u) - div(eta*T(u)) + grad(p)
        = mu_0 * B_h^m(V, H, M) + rho*g + f_theta

    div(u) = 0

where:
- rho(theta): density (phase-dependent)
- eta(theta): viscosity (phase-dependent)
- T(u) = (grad u + grad u^T) / 2: symmetric strain rate (paper convention)
  - Code helper returns D = grad u + grad u^T = 2T; bilinear form uses (ν/4)(D,D) = ν(T,T)
- B_h^m(V, H, M): DG skew form Kelvin body force (Eq. 38, Lemma 3.1)
- f_theta: capillary/surface tension force from Cahn-Hilliard

### 2.2 Cahn-Hilliard (Eq. 42a-b)

Phase-field evolution with advection:

    d(theta)/dt + u . grad(theta) = div(gamma * grad(psi))

    psi = (1/epsilon) * F'(theta) - epsilon * laplacian(theta)

where:
- theta: order parameter (phase field), theta=+1 ferrofluid, theta=-1 non-magnetic
- psi: chemical potential
- epsilon: interface thickness parameter
- gamma: mobility
- F(theta) = (1/4)(theta^2 - 1)^2: double-well potential

**MODIFICATION (Session 17):** CH convection made implicit in theta to remove CFL constraint.
Paper scheme: `(delta_theta/tau, Lambda) - (U theta_old, grad Lambda) + gamma(grad psi, grad Lambda) = 0`
Modified:     `(delta_theta/tau, Lambda) - (U theta_new, grad Lambda) + gamma(grad psi, grad Lambda) = 0`
The convection `-(U theta, grad Lambda)` moves to LHS, making the matrix non-symmetric.
This removes the hard CFL = U*dt/h < 1 constraint that caused blowup during instability onset.

### 2.3 Magnetostatic Poisson (Eq. 42d)

Magnetic potential from Maxwell's equations (magnetostatic limit):

    (grad(phi), grad(X)) = (h_a - M, grad(X))    for all test X

    H = grad(phi)

where:
- phi: total magnetic potential
- H = grad(phi): magnetic field intensity (TOTAL field, includes h_a effect)
- h_a: applied (external) magnetic field (appears only in Poisson RHS)
- M: magnetization vector

**CRITICAL CONVENTION (page 506 of paper):**
The Poisson equation determines phi such that grad(phi) IS the total magnetic field H.
Do NOT add h_a to grad(phi) — that double-counts the applied field. The applied field
h_a appears ONLY in the Poisson right-hand side as a source term.

### 2.4 Magnetization Transport (Eq. 42c)

Magnetization dynamics with transport, relaxation, and equilibrium:

    dM/dt + (u . grad)M + (1/tau_M)(M - chi*H) = 0

where:
- tau_M: magnetic relaxation time (~10^-6 s for ferrofluids)
- chi(theta): magnetic susceptibility (phase-dependent)
- chi*H: equilibrium magnetization (paramagnetic response)

### 2.5 DG Skew Form Kelvin Force (Eq. 38, Lemma 3.1)

    B_h^m(V, H, M) = sum_T int_T [(M.grad)H . V + 1/2(div M)(H.V)] dx
                    - sum_F int_F (V.n^-) [[H]] . {M} ds

where:
- [[H]] = H^- - H^+ (jump using minus-side normal convention)
- {M} = 1/2(M^- + M^+) (average)
- n^- = outward normal of minus cell

**Energy identity (Lemma 3.1):** B_h^m(H, H, M) = 0
This cancellation requires all three invariants to hold:
1. V.n on faces (full dot product, not component-wise)
2. Single minus-side normal (never flip per cell)
3. Elementwise div(M) from DG gradients + face repair

---

## 3. Discretization

### 3.1 Spatial Discretization

| Field | Element | Degree | Space |
|-------|---------|--------|-------|
| Velocity u | Taylor-Hood | Q2 | CG |
| Pressure p | Taylor-Hood | DG P1 | DG |
| Phase field theta | Lagrange | Q1 | CG |
| Chemical potential psi | Lagrange | Q1 | CG |
| Magnetic potential phi | Lagrange | Q2 | CG |
| Magnetization M | DG Lagrange | Q1 | DG |

### 3.2 DG Transport: Skew-Symmetric Form (Eq. 57)

The magnetization transport uses a DG skew-symmetric bilinear form:

    B_h^t(U, V, W) = sum_T integral_T [(U.grad)V . W + (1/2)(div U)(V . W)] dx
                    - sum_F integral_F (U.n) [[V]] . {W} dS

Key property: B_h^t(U, M, M) = 0 (energy neutrality for transport)

### 3.3 Temporal Discretization

- First-order backward Euler (implicit)
- dt = 5e-4 for Rosensweig benchmark (4000 steps, t_F = 2.0)

### 3.4 Nonlinear Solver

Block-Gauss-Seidel iteration (paper Section 6, p.520):
1. **Block 1**: Solve CH (theta, psi) — Eqs (42a)-(42b)
2. **Block 2**: Solve Magnetization + Poisson (M, phi) via Picard — Eqs (42c)-(42d)
3. **Block 3**: Solve NS (U, P) — Eq (42e)

Iterate until convergence (max 5 BGS iterations, tolerance 1e-2).

### 3.5 Implementation

- Library: deal.II (v9.5+)
- Parallelism: MPI via Trilinos (distributed triangulations, vectors, matrices)
- Linear solvers: Direct (UMFPACK/Mumps)
- Build system: CMake

---

## 4. MMS Verification (COMPLETE)

### 4.1 Results (All 10 Spatial Tests Pass, np=4)

| Test | Key Fields | L2 Rate | Expected |
|------|-----------|---------|----------|
| CH_STANDALONE | theta | 3.00 | 3.0 |
| POISSON_STANDALONE | phi | 3.00 | 3.0 |
| NS_STANDALONE | U, p | 3.00, 2.0+ | 3.0, 2.0 |
| MAGNETIZATION_STANDALONE | M | 2.00 | 2.0 |
| POISSON_MAGNETIZATION | phi, M | 3.00, 2.00 | 3.0, 2.0 |
| NS_MAGNETIZATION | U, p | 3.00, 2.0+ | 3.0, 2.0 |
| CH_NS | theta, U | 3.00, 3.00 | 3.0, 3.0 |
| MAG_CH | theta, M | 3.00, 2.00 | 3.0, 2.0 |
| NS_POISSON_MAG | all | optimal | optimal |
| FULL_SYSTEM | all | optimal | optimal |

### 4.2 Bug History

10 bugs found and fixed across 15 sessions:
- 4 MMS-related bugs (Sessions 1-6)
- 3 MPI/solver bugs (Sessions 7-9)
- 1 test harness bug (Session 9)
- 1 H field convention bug (Sessions 13-14)
- 1 viscous term scaling bug (Session 15)

---

## 5. Production Validation Status (IN PROGRESS)

### 5.1 Droplet Tests

**Droplet (no magnetic):** Stable, Laplace pressure approximately correct.

**Droplet + Uniform B:** After H field fix (Bug 9), no longer explodes. However, still
shows spurious Kelvin force (F_mag ~ 101) and exponentially growing velocity on a
configuration that should produce ZERO Kelvin force. This indicates a remaining issue
in the DG skew form Kelvin force discretization.

### 5.2 Rosensweig Instability

Multiple attempts to reproduce Section 6.2 Rosensweig instability:

**With explicit CH convection (Sessions 13-14):** Odd-even oscillation at t=0.32, F_mag
jumps to 275K. BGS fails to converge.

**With AMR (Session 17):** Start-coarse-refine-up approach. All runs blew up due to CFL
from explicit CH convection (L6 at t=0.335, L7 at t=0.072).

**With implicit CH convection (Session 17):** CFL crash eliminated. But r3 (h=1/80)
under-resolves interface (epsilon=0.01). Spurious spikes at t=0.066.

**Current status:** Need r4 (h=1/160 < epsilon) or AMR with implicit convection.

### 5.3 AMR Implementation

AMR implemented for parallel distributed mesh (Session 16):
- Kelly error estimator, SolutionTransfer for all fields
- Start coarse (r3), refine up to target level
- Interface protection, level enforcement, post-AMR clamping
- Working but untested with implicit CH convection at adequate resolution

### 5.4 Known Working Reference

The Decoupled project uses a different Kelvin force formulation (3 separate terms instead
of the DG skew form) and produces correct Rosensweig instability patterns. Parallel
Rosensweig runs with the Decoupled project are in progress for comparison.

---

## 6. Code Architecture

    AddedB_Viscosity/
    |-- main.cc                    # Entry point
    |-- core/
    |   |-- phase_field.cc         # Main time loop, BGS + Picard coupling
    |   |-- phase_field_setup.cc   # Mesh, DOF, sparsity setup
    |
    |-- assembly/
    |   |-- ch_assembler.cc/h      # Cahn-Hilliard assembly
    |   |-- ns_assembler.cc/h      # Navier-Stokes + Kelvin force
    |   |-- magnetization_assembler.cc/h  # DG magnetization (transport + relaxation)
    |
    |-- solvers/
    |   |-- ch_solver.h            # CH block solver
    |   |-- ns_solver.h            # NS block solver
    |   |-- poisson_solver.h       # Poisson CG+AMG solver
    |   |-- magnetization_solver.h # Magnetization direct solver
    |
    |-- physics/
    |   |-- kelvin_force.h         # DG skew form Kelvin force kernels
    |   |-- skew_forms.h           # DG skew-symmetric bilinear forms (Eq. 37, 57)
    |   |-- material_properties.h  # Density, viscosity, susceptibility
    |   |-- applied_field.h        # External magnetic field (dipoles + uniform)
    |
    |-- mms/                       # Method of Manufactured Solutions
    |   |-- mms_core/test_mms.cc   # MMS test driver
    |   |-- ch/ch_mms.h            # CH exact solutions + source terms
    |   |-- ns/ns_mms.h            # NS exact solutions + source terms
    |   |-- poisson/poisson_mms.h  # Poisson exact solutions + source terms
    |   |-- magnetization/         # Magnetization MMS
    |
    |-- utilities/
    |   |-- parameters.h/cc        # All simulation parameters + CLI parsing
    |   |-- tools.h                # Timestamps, CSV headers, utilities
    |
    |-- Report/                    # This documentation
    |-- Results/                   # Simulation output directories

---

## 7. Key Numerical Details

### 7.1 H = grad(phi) Convention

The Poisson equation `(grad phi, grad X) = (h_a - M, grad X)` determines phi such that
grad(phi) IS the total magnetic field H. The applied field h_a enters ONLY as a source
term in the Poisson RHS. This is confirmed by:
- Nochetto CMAME 2016, page 506: "H^h = grad(Phi^k) in M_h"
- Decoupled project (working): `H = grad(phi)` with explicit "Do NOT add h_a" comment
- Magnetization assembler (correct): `H = grad(phi)`

### 7.2 chi*H Cancellation in MMS

The equilibrium term (chi/tau_M)*H appears in both the magnetization equation and its
MMS source. Since the MMS source uses the DISCRETE H (from the Poisson solve), these
terms cancel EXACTLY in the weak form, making the magnetization MMS error independent
of the H convention used.

### 7.3 DG Face Assembly: FEInterfaceValues DOF Numbering

For DG elements, FEInterfaceValues numbers interface DOFs sequentially:
- DOFs 0 .. dofs_per_cell-1: belong to cell 0 ("here")
- DOFs dofs_per_cell .. 2*dofs_per_cell-1: belong to cell 1 ("there")

Must use `shape_value(false, dofs_per_cell + j, q)` to access cell 1's DOFs.

### 7.4 Viscous Term: ν(T,T) Convention

Paper Eq. 42e uses bilinear form `ν(T(U), T(V))` where `T = ½(∇u + (∇u)^T)`.
The code's helper `compute_symmetric_gradient()` returns `D = ∇u + (∇u)^T = 2T`.

Bilinear form: `(ν/4)(D,D) = (ν/4)(2T)(2T) = ν(T,T)` ← matches paper.
Strong form: `-(ν/2)∆U` (since `div(T) = ½∆u` for incompressible flow).

**Bug 10 (Session 15):** Code previously used coefficient `ν/2`, giving `2ν(T,T)` —
double the paper's viscosity. Fixed to `ν/4`. MMS source also corrected from `-ν∆U`
to `-(ν/2)∆U`. MMS tests were unaffected (self-consistent error).

### 7.5 DG Skew Form Kelvin Force Instability (OPEN ISSUE)

The DG skew form B_h^m (Eq. 38) produces spurious forces and odd-even oscillations
in production runs. The face term `-(V.n) [[H]].{M}` may be amplifying numerical noise
from CG gradient discontinuities at quadrature points. Since phi is CG, grad(phi)
should be nearly continuous across faces, making [[H]] ~ 0 in exact arithmetic but
potentially O(h) at off-node quadrature points.

The Decoupled project avoids this issue by using a different 3-term formulation that
does not involve jumps in H across faces.
