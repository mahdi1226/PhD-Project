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

Single pass per time step (BGS=1, paper-like approach). Testing showed iterating to
convergence (BGS=5 or 20) diverges during strong coupling onset.

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

### 5.1 Validation Pyramid (March 7, 2026)

Testing proceeds from simple to complex to isolate physics-layer issues:
1. **Square** (CH+NS only): relaxation to circle — mass & energy conservation
2. **Droplet** (CH+NS only): circular droplet stability
3. **Droplet + nonuniform B** (CH+NS+Mag): droplet elongation in magnetic field
4. **Dome** (CH+NS+Mag+Grav, h=h_a): flat interface with gravity + magnetic force
5. **Rosensweig** (full system, h=h_a+h_d): instability benchmark

Tests 1-3 currently running (March 7). Early results: all stable.

### 5.2 BGS Coupling Investigation

**BGS=5:** Non-convergent at Rosensweig onset (t~0.5), residual 0.2-0.5.
**BGS=20:** Same result — iterating more does NOT help.
**Paper (Section 6, p.520):** "We make no attempt to prove convergence."
**Decision:** BGS=1 (single pass per time step). With single pass, the coupling is:
[CH] -> [Mag+Poisson (Picard)] -> [NS], with no circular dependency.

### 5.3 Rosensweig Instability

Multiple attempts across Sessions 13-19:
- Explicit CH convection: CFL blowup at onset (100x velocity jump in ~25 steps)
- Implicit CH convection: CFL crash eliminated, but BGS non-convergence persists
- BGS=20: no improvement over BGS=1 during onset

**Next approach:** Run with BGS=1 after validation pyramid benchmarks confirm basic
physics layers work correctly.

### 5.4 AMR Implementation

AMR implemented for parallel distributed mesh (Session 16):
- Kelly error estimator, SolutionTransfer for all fields
- Start coarse (r3), refine up to target level
- Interface protection, level enforcement, post-AMR clamping
- **Bulk coarsening fix (Session 18):** Force-coarsen cells where |theta|>0.95 to prevent
  oscillation cycle. Reduces cell count by ~70%.

### 5.5 Code Optimization (Session 19)

Comprehensive cleanup: centralized M_PI, removed dead code, fixed memory management,
deduplicated magnetization face assembly, consolidated 4 NS assembler overloads into
unified function via NSForceData struct. All MMS tests verified passing after changes.

### 5.6 Known Working Reference

The Decoupled project uses a different Kelvin force formulation (3 separate terms instead
of the DG skew form) and produces correct Rosensweig instability patterns. A Decoupled
rosensweig-nonuniform test is currently running on 4 ranks for comparison.

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
