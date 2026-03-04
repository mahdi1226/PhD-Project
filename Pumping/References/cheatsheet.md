# Two-Phase Ferrofluid Literature Cheatsheet

Comprehensive reference for 13 papers on ferrofluid modeling, compiled for Phase B
(Cahn-Hilliard + FHD droplet transport in pumping channel).

---

## Table of Contents

1. [Nochetto, Salgado & Tomas (2016) -- arXiv 1601.06824v1](#paper-1)
2. [Nochetto, Salgado & Tomas (2016) -- CMAME](#paper-2)
3. [Zhang, He & Yang (2021) -- SIAM J. Sci. Comput.](#paper-3)
4. [Yang (2021) -- JCP 448 (MHD)](#paper-4)
5. [Wu, Yang et al. (2024) -- arXiv 2410.11129v4](#paper-5)
6. [Yang et al. (2025) -- JCP 539 (BDF2 SAV)](#paper-6)
7. [Chen, Li, Li & He (2025) -- JCP 539 (different densities)](#paper-7)
8. [Zhang, Zhou, Huang, He & Yang (2026) -- JCP 548 (porous media)](#paper-8)
9. [Hu, Li & Niu (2018) -- Phys. Rev. E (LBM)](#paper-9)
10. [Afkhami & Renardy (2017) -- J. Eng. Math. (review)](#paper-10)
11. [Afkhami, Renardy et al. (2008) -- J. Fluid Mech. (droplet motion)](#paper-11)
12. [Afkhami, Tyler et al. (2010) -- J. Fluid Mech. (droplet deformation)](#paper-12)
13. [Mao, Elborai, He, Zahn & Koser (2011) -- Phys. Rev. B (pumping experiment)](#paper-13)
14. [Comparison Tables](#comparison-tables)

---

<a name="paper-1"></a>
## 1. Nochetto, Salgado & Tomas (2016) -- arXiv

**Citation**: R.H. Nochetto, A.J. Salgado, I. Tomas. "A diffuse interface model for
two-phase ferrofluid flows." arXiv:1601.06824v1, 2016.

**Problem**: Two-phase ferrofluid flow (ferrofluid + non-magnetic fluid) with diffuse
interface. Matching density (Boussinesq approximation for gravity). Foundational
model for the field.

**Formulation** (Eqs. 14a--14f):

- **Cahn-Hilliard** (phase field theta, chemical potential psi):
  - theta_t + div(u*theta) + gamma*Delta*psi = 0
  - psi - epsilon*Delta*theta + (1/epsilon)*f(theta) = 0
  - F(theta) = (1/4)(theta^2 - 1)^2 (truncated outside [-1,1])

- **Chemical potential**: psi = epsilon*Delta*theta + (1/epsilon)*f(theta)
  - **CRITICAL**: No explicit magnetic term in the chemical potential.
    The magnetic coupling enters through the Kelvin force in NS and through
    the capillary force f_c = (lambda/epsilon)*theta*grad(psi).

- **Magnetization** (Shliomis relaxation, simplified):
  - m_t + (u . grad)m = -(1/T)(m - kappa_theta * h)
  - T = relaxation time, kappa_theta = kappa_0 * H(theta/epsilon)
  - Susceptibility: kappa_theta varies with phase via sigmoid H(x) = 1/(1+e^{-x})

- **Magnetostatics** (scalar potential):
  - -Delta*phi = div(m - h_a)
  - h = grad(phi) is the effective magnetic field
  - h = h_a + h_d (applied + demagnetizing)

- **Navier-Stokes**:
  - u_t + (u . grad)u - div(nu_theta * T(u)) + grad(p)
    = mu_0*(m . grad)h + (lambda/epsilon)*theta*grad(psi)
  - div(u) = 0
  - T(u) = (1/2)(grad(u) + grad(u)^T) symmetric gradient
  - **Kelvin force**: mu_0*(m . grad)h
  - **Capillary force**: (lambda/epsilon)*theta*grad(psi)

- **Phase-dependent properties**:
  - nu_theta = nu_w + (nu_f - nu_w)*H(theta/epsilon)
  - kappa_theta = kappa_0 * H(theta/epsilon)
  - Gravity (optional): f_g = (1 + r*H(theta/epsilon))*g

- **Boundary conditions**: dn(theta) = dn(psi) = 0, u = 0, dn(phi) = (h_a - m).n

**Numerical scheme**:

- Time: BDF1 (backward Euler), semi-implicit
- Coupling: **Fully coupled** (nonlinear system each time step)
- FE spaces: Taylor-Hood P2/P1 for NS, P1 for CH and phi, Nedelec H(curl) for h
- Energy stability: Unconditionally energy-stable (proven)
- Solver: UMFPACK direct

**Benchmarks**:

- Rosensweig instability (uniform vertical field, spike formation)
- Droplet deformation under uniform applied field
- Open pattern of spikes (non-uniform applied field)

**Key results**:

- First time-dependent two-phase ferrofluid model with energy stability
- Established the model that all subsequent papers build upon
- Capillary force in simplified form (lambda/epsilon)*theta*grad(psi) after
  pressure redefinition (Remark 2.1)
- Energy identity B(u,m,h) = -B(m,h,u) is crucial (Lemma 3.1)

---

<a name="paper-2"></a>
## 2. Nochetto, Salgado & Tomas (2016) -- CMAME

**Citation**: R.H. Nochetto, A.J. Salgado, I. Tomas. "A diffuse interface model for
two-phase ferrofluid flows." CMAME 309 (2016) 497--531.

**Problem**: Same as Paper 1 (journal version of arXiv paper).

**Formulation**: Identical to Paper 1. Same Eqs. 14a--14f.

**Numerical scheme**: Same as Paper 1.
- BDF1, semi-implicit, fully coupled
- Taylor-Hood + Nedelec
- Algorithm 1 described in detail

**Benchmarks**: Same as Paper 1 with additional rigor.

**Key results**:

- More rigorous well-posedness proofs and convergence analysis
- Stability + convergence under simplifications (Section 5)
- Detailed Algorithm 1 description

---

<a name="paper-3"></a>
## 3. Zhang, He & Yang (2021) -- SIAM J. Sci. Comput.

**Citation**: G.-D. Zhang, X. He, X. Yang. "Decoupled, linear, and unconditionally
energy stable fully discrete finite element numerical scheme for a two-phase
ferrohydrodynamics model." SIAM J. Sci. Comput. 43(1), B167--B193, 2021.

**Problem**: Two-phase ferrofluid flow. Extends Nochetto et al. model with
full Shliomis terms restored + energy-stable decoupled scheme.

**Formulation** (Eqs. 2.4--2.11):

- **Cahn-Hilliard**:
  - Phi_t + div(u*Phi) = M*Delta*W
  - W = -lambda*epsilon*Delta*Phi + lambda*f(Phi)
  - F(Phi) = (1/4)*Phi^2*(Phi-1)^2 (truncated, note: Phi in [0,1] convention)
  - **CRITICAL**: Chemical potential W does NOT contain a magnetic term.
    W = -lambda*epsilon*Delta*Phi + lambda*f(Phi) only.

- **Navier-Stokes** (Eq. 2.6):
  - u_t - div(nu(Phi)*D(u)) + (u . grad)u + grad(p) + Phi*grad(W)
    = mu*(m . grad)h + (mu/2)*curl(m x h)
  - **Kelvin force**: mu*(m . grad)h
  - **Capillary (elastic stress)**: Phi*grad(W) (the induced elastic stress)
  - **Spin torque**: (mu/2)*curl(m x h)

- **Magnetization** (Eq. 2.8, full Shliomis):
  - m_t + (u . grad)m - (1/2)*curl(u) x m + beta*m x (m x h)
    = -(1/tau)*(m - chi(Phi)*h)
  - Includes convection, spin coupling, phenomenological beta term
  - chi(Phi) = chi_0 / (1 + e^{-(2Phi-1)/epsilon})

- **Magnetostatics** (Eq. 2.9):
  - -Delta*phi = div(m - h_a)
  - h = grad(phi)

- **Phase-dependent properties**:
  - nu(Phi) = nu_w + (nu_f - nu_w) / (1 + e^{-(2Phi-1)/epsilon})
  - chi(Phi) = chi_0 / (1 + e^{-(2Phi-1)/epsilon})

- **Energy law** (Theorem 2.1):
  - E(Phi,u,h,m) = lambda*(epsilon/2 ||grad Phi||^2 + (F(Phi),1))
    + (1/2)||u||^2 + (mu/2)||h||^2 + (mu/(2*chi_0))||m||^2

**Numerical scheme**:

- Time: BDF1 (first-order backward Euler)
- Coupling: **Fully decoupled** -- solve CH, NS, magnetization, magnetostatics separately
- Decoupling technique: **IEQ** (Invariant Energy Quadratization)
  - U = sqrt(F(Phi) + B) as auxiliary variable
  - **ZEC** (Zero-Energy-Contribution) for maintaining energy stability after decoupling
- FE spaces: P_{l+1}/P_l for NS (Taylor-Hood), P_l for CH/Phi, P_l for magnetization
- Energy stability: Unconditionally energy-stable (proven, Theorem 3.2)
- All linear systems, constant coefficient matrices

**Benchmarks**:

- Convergence test (first-order in time confirmed)
- Ferrofluid droplet deformation under uniform field
- Rosensweig instability

**Key results**:

- First fully decoupled, energy-stable scheme for two-phase FHD
- Only linear systems to solve at each time step
- IEQ + ZEC technique enables decoupling while preserving stability
- Restored terms dropped by Nochetto et al. (spin torque, beta term)

---

<a name="paper-4"></a>
## 4. Zhang, He & Yang (2022) -- JCP 448 (MHD)

**Citation**: G.-D. Zhang, X. He, X. Yang. "A fully decoupled linearized finite
element method with second-order temporal accuracy and unconditional energy
stability for incompressible MHD equations." JCP 448 (2022) 110752.

**Problem**: Incompressible MHD (magnetohydrodynamics). NOT ferrofluid/phase-field.
Relevant for decoupling techniques that are reused in ferrofluid papers.

**Formulation** (Eqs. 2.11--2.14):

- **Navier-Stokes**:
  - u_t - nu*Delta*u + (u . grad)u + grad(p) + kappa*B x curl(B) = 0
  - div(u) = 0

- **Magnetic induction**:
  - B_t + eta*curl(curl(B)) - curl(u x B) + grad(r) = 0
  - div(B) = 0

- NO Cahn-Hilliard, NO phase field, NO ferrofluid magnetization

**Numerical scheme**:

- Time: **BDF2** (second-order)
- Coupling: **Fully decoupled** via nonlocal variable Q(t)
  - Q_t = integral[(u.grad)u . u dx] + kappa*integral[B x curl(B) . u dx]
           - kappa*integral[u x B . curl(B) dx]
  - Q(t) = 1 at continuous level (ZEC property)
- **Pressure projection** (second-order)
- FE: Lagrange P_{l+1}/P_l for NS, first Nedelec N_l for B, Lagrange P_l for r
- Energy stability: Unconditionally energy-stable (Theorem 3.1)

**Benchmarks**:

- Convergence test (second-order confirmed)
- Kelvin-Helmholtz instability
- Driven cavity with magnetic field

**Key results**:

- Introduced the nonlocal variable Q(t) technique for ZEC in MHD context
- BDF2 fully decoupled + energy-stable: first "desired type" scheme for MHD
- Technique later adapted to ferrofluid papers (Papers 5, 6, 7, 8)

---

<a name="paper-5"></a>
## 5. Wu, Yang et al. (2024) -- arXiv 2410.11129v4

**Citation**: Y. Wu, X. Yang et al. "Efficient and unconditionally energy stable fully
discrete numerical schemes for the two-phase ferrohydrodynamics model."
arXiv:2410.11129v4, 2024.

**Problem**: Two-phase ferrofluid flow with matched density. Two schemes:
Scheme 1 (BDF1, first-order) and Scheme 2 (BDF2, second-order).

**Formulation**: Based on Zhang/He/Yang (2021) model, same PDE system.

- **Cahn-Hilliard**:
  - Phi_t + div(u*Phi) = M*Delta*W
  - W = -epsilon*sigma*Delta*Phi + (sigma/epsilon)*W'(Phi)
  - **CRITICAL**: Chemical potential W does NOT contain a magnetic term.
    Same as Paper 3.

- **NS**: Same force terms as Paper 3 (Kelvin + capillary + spin torque)

- **Magnetization**: Full Shliomis with chi(Phi) phase-dependent susceptibility

- **Magnetostatics**: Scalar potential, same as Papers 1--3

**Numerical scheme**:

- **Scheme 1**: BDF1, fully decoupled, SAV-based
- **Scheme 2**: BDF2, fully decoupled, SAV-based
  - SAV: r(t) = sqrt(E_1(Phi) + C_0) where E_1 contains double-well energy
  - ZEC for decoupling
  - Pressure-correction projection
- FE: P_{l+1}/P_l for NS, P_l for CH/magnetics
- Both schemes unconditionally energy-stable
- All constant-coefficient linear systems

**Benchmarks**:

- Convergence: Scheme 1 = O(dt), Scheme 2 = O(dt^2) confirmed
- Ferrofluid droplet deformation
- Rosensweig instability
- Comparison of first vs second order accuracy

**Key results**:

- Unified SAV-ZEC framework for both first and second order schemes
- Demonstrates clear advantage of BDF2 over BDF1 in accuracy
- Constant coefficient linear systems -- very efficient

---

<a name="paper-6"></a>
## 6. Yang et al. (2025) -- JCP 539 (BDF2 SAV)

**Citation**: X. Yang et al. "A fully-decoupled, second-order accurate, and
unconditionally energy stable numerical scheme for the two-phase ferrofluid model."
JCP 539 (2025).

**Problem**: Two-phase ferrofluid, matched density. Second-order in time.

**Formulation**: Same PDE model as Papers 3, 5 (Zhang/He/Yang family).

- **Cahn-Hilliard**:
  - Phi_t + div(u*Phi) = M*Delta*W
  - W = -lambda*Delta*Phi + (lambda/epsilon^2)*f(Phi)
  - **CRITICAL**: Chemical potential W does NOT contain a magnetic term.

- **NS with forces**: Kelvin mu*(m.grad)h + capillary Phi*grad(W)

- **Magnetization**: Full Shliomis, phase-dependent chi(Phi)

- **Magnetostatics**: Scalar potential

**Numerical scheme**:

- Time: **BDF2**
- Coupling: **Fully decoupled**
- SAV: r(t) = sqrt(E_1(Phi) + C_0) for the double-well potential
- Pressure-correction projection for NS velocity-pressure decoupling
- ZEC terms for energy stability after decoupling
- BDF2 extrapolation: 2*Phi^n - Phi^{n-1} for explicit treatment
- FE: P2/P1 + P1
- Unconditionally energy-stable (proven)

**Benchmarks**:

- Convergence: second-order in time confirmed for all variables
- Ferrofluid droplet deformation
- Rosensweig instability

**Key results**:

- Second-order in time, fully decoupled, unconditionally energy-stable
- SAV + ZEC + pressure-correction = complete decoupling toolkit
- All linear systems with constant coefficients

---

<a name="paper-7"></a>
## 7. Chen, Li, Li & He (2025) -- JCP 539 (different densities/viscosities)

**Citation**: X. Chen, R. Li, J. Li, X. He. "A decoupled, linear, unconditionally
stable, and fully discrete finite element scheme for two-phase ferrofluid flows
with different densities and viscosities." JCP 539 (2025) 114209.

**Problem**: Two-phase ferrofluid flow with **different densities and viscosities**
(rho_f != rho_w, nu_f != nu_w). Key extension from matched-density models.

**Formulation** (Eqs. 5a--5f, then rewritten as 13a--13f):

- **Cahn-Hilliard** (Eq. 5a--5b, note: Phi in {-1, +1}):
  - Phi_t + div(u*Phi) = M*Delta*W
  - W = -epsilon*Delta*Phi + f(Phi)
  - f(Phi) = F'(Phi), F(Phi) = (1/4)*Phi^2*(Phi-1)^2 (truncated)
  - **CRITICAL**: Chemical potential W does NOT contain a magnetic term.
    W = -epsilon*Delta*Phi + f(Phi) only.

- **Navier-Stokes** (Eq. 5c, rewritten as 13c with variable density):
  - sigma(sigma*u)_t + (1/2)*div(rho(Phi)*u)*u + rho(Phi)*(u . grad)u
    - div(nu(Phi)*D(u)) + grad(p) + (lambda/epsilon)*Phi*grad(W)
    = mu*(m . grad)h
  - sigma = sqrt(rho(Phi)) -- new variable for variable density
  - **Kelvin force**: mu*(m . grad)h
  - **Capillary**: (lambda/epsilon)*Phi*grad(W)
  - No spin torque term (dropped under linear magnetization assumption)

- **Magnetization** (Eq. 5e):
  - m_t + (u . grad)m = -(1/tau)*(m - chi(Phi)*h)
  - Simplified Shliomis (no spin coupling, no beta term)

- **Magnetostatics** (Eq. 5f):
  - -Delta*phi = div(m - h_a)
  - h = grad(phi)

- **Phase-dependent properties** (Eq. 15):
  - rho(Phi) = (rho_f - rho_w)*Phi + rho_w (linear interpolation)
  - nu(Phi) = (nu_f - nu_w)*Phi + nu_w
  - chi(Phi) = chi_0*Phi (linear in phase field)

- **Energy law** (Theorem 2.1, Eq. 17):
  - d/dt E(Phi,u,h,m) + D(w,u,m,h) <= (mu/tau)||h_a||^2 + tau*mu||(h_a)_t||^2

- **Reformulated magnetostatics** (Eq. 30 -- key innovation):
  - (1/tau)(grad(phi), grad(psi)) + (phi_t, grad(psi))
    + (1/tau)(m, grad(psi)) + (m_t, grad(psi))
    = (1/tau)(h_a, grad(psi)) + ((h_a)_t, grad(psi))
  - Avoids taking two test functions in energy proof

**Numerical scheme**:

- Time: **BDF1** (first-order)
- Coupling: **Fully decoupled** (linear systems)
- Artificial compressibility method for NS velocity-pressure decoupling
  (no artificial pressure BC needed)
- Implicit-explicit treatment for nonlinear coupling terms
- Stabilization terms added for energy stability
- Reformulated magnetostatic equation (Eq. 30) for decoupling phi and m
- FE (Eq. 34): P_{l_1} for Phi, P_{l+1}/P_{l-1} for u/p (Taylor-Hood),
  L^2 for magnetization, P_l for phi
- AMR (adaptive mesh refinement) applied
- Unconditionally energy-stable (Theorem 4.1)

**Benchmarks**:

- Convergence test: first-order in time, optimal spatial rates
- Ferrofluid droplet deformation
- One or two air bubbles rising in ferrofluid
- Controllable ferrofluid droplet in Y-domain
- Rosensweig instability under uniform and non-uniform fields

**Key results**:

- First decoupled scheme handling different densities AND viscosities
- High density ratio (up to 1000) and viscosity ratio demonstrated
- Artificial compressibility -- no artificial BC on pressure
- AMR captures diffuse interface efficiently
- Y-domain droplet: demonstrates controllable routing via magnetic field

---

<a name="paper-8"></a>
## 8. Zhang, Zhou, Huang, He & Yang (2026) -- JCP 548 (porous media)

**Citation**: G.-D. Zhang, S. Zhou, Y. Huang, X. He, X. Yang. "A diffuse interface
model and fully decoupled, energy-stable scheme for the two-phase ferrofluid flows
in porous media." JCP 548 (2026) 114561.

**Problem**: Two-phase ferrofluid flow in **porous media**. Replaces Navier-Stokes
with Darcy's law. Relevant for subsurface ferrofluid applications.

**Formulation** (Eqs. 2.6--2.13):

- **Cahn-Hilliard** (Eq. 2.6--2.7):
  - phi_t + div(u*phi) - M*Delta*w = 0
  - w = -lambda*Delta*phi + (lambda/epsilon^2)*f(phi)
  - **CRITICAL**: Chemical potential w does NOT contain a magnetic term.

- **Darcy equation** (replacing NS, Eq. 2.8):
  - u_t + K^{-1}*nu(phi)*u + grad(p) + phi*grad(w) = mu*(m . grad)h
  - K = permeability tensor
  - **Kelvin force**: mu*(m . grad)h
  - **Capillary**: phi*grad(w)

- **Magnetization** (Eq. 2.10):
  - m_t + (u . grad)m + beta*m x (m x h) = -(1/tau)*(m - kappa(phi)*h)
  - Full Shliomis with beta phenomenological term

- **Magnetostatics** (Eq. 2.11):
  - -Delta*phi_mag = div(m - h_a)

- **Phase-dependent properties**:
  - nu(phi) = nu_w + (nu_f - nu_w) / (1 + e^{-phi/epsilon})
  - kappa(phi) = kappa_w + (kappa_f - kappa_w) / (1 + e^{-phi/epsilon})

- **Energy law** (Theorem 1, Eq. 2.16--2.17)

- **Reformulated magnetostatics** (Eq. 2.31):
  - Same reformulation technique as Paper 7 to decouple phi and m

**Numerical scheme**:

- Time: **BDF2** (second-order) with second-order pressure projection
- Coupling: **Fully decoupled**
- SAV-ZEC for nonlinearities and decoupling
- L^2 projection for Kelvin force gradient (handles internal boundary jumps)
- Reformulated magnetostatic equation for phi-m decoupling
- FE: P_{l_1} for phase, P_{l+1}/P_{l-1} for velocity/pressure,
  CG for magnetization, CG for magnetic potential
- Unconditionally energy-stable (Theorem 3.1)

**Benchmarks**:

- Convergence test (2D and 3D): second-order in time confirmed
- Saffman-Taylor fingering instability
- Suppression of fingering by tangential magnetic fields (first numerical
  validation of 1980 experimental observation)

**Key results**:

- First energy-law-consistent and computable two-phase FHD model in porous media
- Darcy replaces NS: different function space challenges (natural Neumann for pressure)
- L^2 projection handles Kelvin force gradient that is inadmissible in continuous FE space
- Saffman-Taylor fingering suppression by magnetic field: novel numerical validation

---

<a name="paper-9"></a>
## 9. Hu, Li & Niu (2018) -- Phys. Rev. E (LBM)

**Citation**: Y. Hu, D. Li, X. Niu. "Phase-field-based lattice Boltzmann model
for multiphase ferrofluid flows." Phys. Rev. E 98, 033301 (2018).

**Problem**: Multiphase ferrofluid flows using Lattice Boltzmann Method (LBM).
Conservative Allen-Cahn (not Cahn-Hilliard) for interface tracking.

**Formulation**:

- **Conservative Allen-Cahn** (Eq. 12):
  - d(phi)/dt + div(u*phi) = div(M_phi * [grad(phi) + (4/xi)*phi*(phi-1)*n_hat])
  - n_hat = grad(phi)/|grad(phi)| (interface normal)
  - NOT Cahn-Hilliard. No split into phi + mu.

- **Chemical potential** (Eq. 15):
  - mu_phi = 4*beta*phi*(phi-1)*(phi - 1/2) - kappa*laplacian(phi)
  - beta = 12*sigma/D, kappa = 3*D*sigma/2
  - **CRITICAL**: No magnetic term in chemical potential.

- **Navier-Stokes** (Eq. 7):
  - rho*(du/dt + div(uu)) = -grad(p) + eta*laplacian(u) + div(tau_m) + f_s + f_b
  - **Magnetic stress tensor** (Cowley-Rosensweig, Eq. 8):
    tau_m = -(mu_0/2)*|H|^2*I + H*B
  - **Kelvin force** (Eq. 10):
    f_m = div(tau_m) = (mu_0*chi/2)*grad(|H|^2)
    (for linear magnetization M = chi*H)
  - **Surface tension**: f_s = mu_phi * grad(phi)

- **Magnetostatics** (Eq. 5):
  - div(mu*grad(psi)) = 0
  - H = -grad(psi)
  - mu = mu_0*(1 + chi) with chi from Langevin law (Eq. 11)

- **Nonlinear Langevin magnetization** (Eq. 11):
  - chi = M/H = (M_s/H)*[coth(3*chi_0*H/M_s) - M_s/(3*chi_0*H)]

- **Variable density**: rho = rho_l + phi*(rho_h - rho_l)

**Numerical scheme**:

- **Lattice Boltzmann Method** (NOT FEM):
  - D2Q9 lattice for Allen-Cahn and NS
  - Separate LB equation for magnetic potential
  - MRT (multiple-relaxation-time) collision operator
- FDM for magnetic potential equation
- Handles density ratios up to 850.7

**Benchmarks**:

- Circular cylinder in uniform applied field (analytical comparison)
- Ferrofluid droplet deformation under uniform field
- Two bubbles merging in ferrofluid
- Ferrofluid droplets moving/merging near permanent magnet

**Key results**:

- Unified LBM framework for multiphase ferrofluid
- Handles high density ratios (up to 850.7)
- Nonlinear Langevin magnetization law (not just linear chi)
- Conservative Allen-Cahn instead of Cahn-Hilliard
- Magnetic stress tensor approach (not Kelvin body force decomposition)

---

<a name="paper-10"></a>
## 10. Afkhami & Renardy (2017) -- J. Eng. Math. (review)

**Citation**: S. Afkhami, Y. Renardy. "Ferrofluids and magnetically guided
superparamagnetic particles in flows: a review of simulations and modeling."
J. Eng. Math. 107 (2017) 231--251.

**Problem**: Review paper covering three topics:
1. Drug targeting with superparamagnetic nanoparticles
2. Ferrofluid drop motion and deformation
3. Thin film ferrofluid breakup

**Formulation** (review of multiple approaches):

- **Magnetic drug targeting** (Section 2):
  - Point dipole model: phi = -(1/4pi)*(m.r/r^3)
  - Force on particle: F_m = integral[mu_0*(M . grad)*H_e dV]
  - Stokes drag: F_v = -D*(dx/dt - u) where D = 6*pi*eta_0*a
  - Particle trajectory equation of motion

- **Ferrofluid drop deformation** (Section 3):
  - VOF (Volume-of-Fluid) sharp-interface method
  - NS + Maxwell equations
  - Magnetic stress tensor: tau_m = -(mu_0/2)*|H|^2*I + H*B
  - Normal stress jump at interface: [tau_m . n] . n
  - Interface conditions: [B.n] = 0, [H x n] = 0
  - CSF (continuum surface force) for surface tension

- **Thin film** (Section 3.4):
  - Lubrication theory + magnetic pressure
  - Film breakup and satellite droplet formation

**Numerical scheme**: VOF with CSF (for drop simulations),
analytical/ODE integration (for particle tracking),
lubrication theory (for thin films).

**Benchmarks**:

- Particle capture efficiency vs flow rate
- PDMS ferrofluid drop deformation (comparison with experiments)
- Thin film breakup patterns

**Key results**:

- Comprehensive review of sharp-interface approaches
- Experimental validation of drop shapes (PDMS ferrofluid)
- Magnetic force F_m = mu_0*(M . grad)*H_e (body force formulation)
- Nonlinear magnetization effects at high fields
- Apparent interfacial tension varies with magnetic field strength

---

<a name="paper-11"></a>
## 11. Afkhami, Renardy et al. (2008) -- J. Fluid Mech. (droplet motion)

**Citation**: S. Afkhami, Y. Renardy, M. Renardy, J.S. Riffle, T. St Pierre.
"Field-induced motion of ferrofluid droplets through immiscible viscous media."
J. Fluid Mech. 610 (2008) 363--380.

**Problem**: Motion of a hydrophobic ferrofluid droplet driven by an external
magnetic field (permanent magnet), through an immiscible viscous medium.
Axisymmetric geometry.

**Formulation**:

- **Sharp interface** (VOF framework):
  - Two-phase NS with volume fraction tracking
  - Magnetic field from scalar potential: H = grad(psi)
  - div(mu*grad(psi)) = 0 (Eq. 2.2)
  - Permeability: mu_1 = mu_0*(1 + chi_m) in ferrofluid, mu_0 outside

- **Forces**:
  - Magnetic body force from Maxwell stress tensor
  - Interfacial tension via CSF (continuum surface force)
  - Viscous drag

- **Boundary conditions**:
  - Far-field H from experimental polynomial fit (6th degree, Eq. 2.3)
  - Axisymmetric computational domain

**Numerical scheme**:

- VOF (Volume-of-Fluid) for interface tracking
- Finite difference / projection method for NS
- Axisymmetric coordinates
- Adaptive mesh refinement

**Benchmarks**:

- Three regimes studied:
  1. Inertia dominant, magnetic Laplace number varied
  2. Inertia negligible, Laplace number varied
  3. Both small (magnetic force ~ viscous drag)
- Transit time comparison with experiments (Mefford et al. 2007)
- Drop shape deformation (sphere to teardrop)

**Key results**:

- Numerical ferrofluid droplet transit times compare favorably with experiments
- Drop deformation from sphere to teardrop as it approaches magnet
- Tail separation observed at larger drop sizes
- Numerical model superior to solid-sphere analytical approximation

---

<a name="paper-12"></a>
## 12. Afkhami, Tyler et al. (2010) -- J. Fluid Mech. (droplet deformation)

**Citation**: S. Afkhami, A.J. Tyler, Y. Renardy, M. Renardy, T.G. St. Pierre,
R.C. Woodward, J.S. Riffle. "Deformation of a hydrophobic ferrofluid droplet
suspended in a viscous medium under uniform magnetic fields."
J. Fluid Mech. 663 (2010) 358--384.

**Problem**: Equilibrium deformation of a hydrophobic PDMS-ferrofluid droplet
in a viscous medium under uniform applied magnetic field. Comparison with
experiments.

**Formulation**:

- **Sharp interface** (VOF framework):
  - Same as Paper 11
  - div(mu*grad(psi)) = 0
  - Two immiscible fluids with different permeabilities

- **Forces**:
  - Magnetic normal stress jump at interface
  - Surface tension (CSF)
  - Analytical solutions for ellipsoidal/spheroidal shapes

- **Magnetization**:
  - Linear: M = chi_m * H (low field)
  - Nonlinear effects at high fields (apparent interfacial tension varies)

**Numerical scheme**:

- VOF with CSF
- Finite difference / projection
- Axisymmetric and 3D simulations
- Comparison with analytical ellipsoidal solutions

**Benchmarks**:

- Drop aspect ratio vs magnetic field strength
- Comparison with PDMS ferrofluid experiments in glycerol
- Low field: good agreement with constant interfacial tension model
- High field: interfacial tension appears to depend on magnetic field

**Key results**:

- At low fields, drop shape follows small deformation theory
- At high fields, apparent interfacial tension varies with field (microstructural effects)
- Numerical-experimental comparison provides interfacial tension estimates
- S-shaped bifurcation: above critical permeability ratio, three equilibrium shapes exist

---

<a name="paper-13"></a>
## 13. Mao, Elborai, He, Zahn & Koser (2011) -- Phys. Rev. B (pumping)

**Citation**: L. Mao, S. Elborai, X. He, M. Zahn, H. Koser. "Direct observation of
closed-loop ferrohydrodynamic pumping under traveling magnetic fields."
Phys. Rev. B 84, 104431 (2011).

**Problem**: **Experimental** demonstration of closed-loop ferrofluid pumping using
spatially traveling, sinusoidally time-varying magnetic fields. No numerical PDE
scheme (pure experiment + analytical model fitting).

**Formulation** (analytical/experimental):

- Traveling magnetic field created by multiphase electromagnetic coils
  (4 balanced phases: 0, 90, 180, 270 degrees)
- Ferrofluid: EFH1 (oil-based, magnetite nanoparticles in mineral oil)
  - Density: 1.22 g/mL, viscosity: 11.1 cP
  - Initial susceptibility: chi_0 = 1.56
  - Saturation magnetization: M_s = 34.4 kA/m
  - Relaxation mechanism: Neel (core diameter ~ 7.5 nm)

- **Pumping mechanism**: Spatially non-uniform rotating magnetic field creates
  radial gradient in nanoparticle rotation velocity, producing radial shear
  that drives ferrofluid flow. Based on spin viscosity / body couple theory.

- **Key physics**: Even with vanishing spin viscosity, pumping occurs because
  the alternating field is spatially non-uniform within the pumping region.

**Numerical scheme**: None (experimental paper). FEA simulations mentioned
for field distribution only.

**Benchmarks / Experimental Results**:

- Maximum flow velocity: ~6 mm/s (at 1600 Hz, 12 A excitation)
- Maximum volumetric flow rate: 0.69 mL/s
- Optimal pumping frequency related to Neel relaxation time
- Flow velocity proportional to current amplitude squared
- Laminar Poiseuille profile outside pump section (Re < 0.01)
- Pressure difference: up to ~12 Pa at 12 A

**Key results**:

- First experimental demonstration of direct body-force ferrofluid pumping
  at macroscopic scale (not surface-tension-based plug pumping)
- Compact, scalable, no moving parts
- Moderate field amplitudes (~10 mT) sufficient
- Optimal frequency ~ kHz range (matches Neel relaxation)
- Dimer formation model: small percentage of rotating dimers sufficient
  to drive macroscopic flow
- Directly relevant to user's PhD on ferrofluid pumping channels

---

<a name="comparison-tables"></a>
## Comparison Tables

### Table 1: Magnetic Coupling in Chemical Potential

This is the critical question: does the chemical potential W (or mu/psi)
contain a term -0.5*mu_0*chi'(phi)*|H|^2 coupling the magnetic field
to the phase-field evolution?

| # | Paper | Magnetic term in W? | Chemical potential formula |
|---|-------|--------------------|-----------------------------|
| 1 | Nochetto (2016, arXiv) | **NO** | psi = eps*Delta*theta + (1/eps)*f(theta) |
| 2 | Nochetto (2016, CMAME) | **NO** | Same as #1 |
| 3 | Zhang/He/Yang (2021) | **NO** | W = -lam*eps*Delta*Phi + lam*f(Phi) |
| 4 | Zhang/He/Yang (2022, MHD) | N/A | No CH equation |
| 5 | Wu/Yang (2024) | **NO** | W = -eps*sig*Delta*Phi + (sig/eps)*W'(Phi) |
| 6 | Yang (2025, JCP 539) | **NO** | W = -lam*Delta*Phi + (lam/eps^2)*f(Phi) |
| 7 | Chen/Li/Li/He (2025) | **NO** | W = -eps*Delta*Phi + f(Phi) |
| 8 | Zhang/Zhou et al. (2026) | **NO** | w = -lam*Delta*phi + (lam/eps^2)*f(phi) |
| 9 | Hu/Li/Niu (2018, LBM) | **NO** | mu_phi = 4*beta*phi*(phi-1)*(phi-0.5) - kappa*lap(phi) |
| 10 | Afkhami/Renardy (2017) | N/A | Review (no CH) |
| 11 | Afkhami et al. (2008) | N/A | Sharp interface (no CH) |
| 12 | Afkhami et al. (2010) | N/A | Sharp interface (no CH) |
| 13 | Mao et al. (2011) | N/A | Experimental |

**Conclusion**: NONE of the papers reviewed include a direct magnetic energy
term in the chemical potential. The magnetic-phase coupling enters through:
1. Phase-dependent susceptibility chi(phi) in the magnetization equation
2. Phase-dependent Kelvin force mu_0*(m . grad)h in the NS equation
3. Phase-dependent elastic/capillary stress phi*grad(W) in NS

**Important note**: Although the total free energy E includes a magnetic term
(e.g., -(mu_0/2)*integral[chi(phi)*|H|^2 dx] in Nochetto), the variational
derivative delta_E/delta_phi that produces the chemical potential is handled
through the coupled system's energy estimate, not by adding an explicit
-0.5*mu_0*chi'(phi)*|H|^2 to the chemical potential equation. The energy
stability is proven for the full coupled system, with the magnetic contribution
entering through the force balance in NS and magnetization equations.

### Table 2: Numerical Scheme Summary

| # | Paper | Time order | Coupling | Decoupling method | FE spaces |
|---|-------|-----------|----------|-------------------|-----------|
| 1 | Nochetto (arXiv) | BDF1 | Coupled | None | P2/P1 + P1 + Nedelec |
| 2 | Nochetto (CMAME) | BDF1 | Coupled | None | P2/P1 + P1 + Nedelec |
| 3 | Zhang/He/Yang | BDF1 | Decoupled | IEQ + ZEC | P_{l+1}/P_l + P_l |
| 4 | Zhang/He/Yang (MHD) | BDF2 | Decoupled | Nonlocal Q + ZEC | P_{l+1}/P_l + Nedelec |
| 5 | Wu/Yang | BDF1 + BDF2 | Decoupled | SAV + ZEC | P_{l+1}/P_l + P_l |
| 6 | Yang (JCP 539) | BDF2 | Decoupled | SAV + ZEC + proj. | P2/P1 + P1 |
| 7 | Chen/Li/Li/He | BDF1 | Decoupled | Artif. compress. + stab. | P_{l+1}/P_{l-1} + P_l |
| 8 | Zhang/Zhou et al. | BDF2 | Decoupled | SAV + ZEC + L2 proj. | P_{l+1}/P_{l-1} + P_l |
| 9 | Hu/Li/Niu (LBM) | -- | Coupled | None | D2Q9 LBM + FDM |
| 10--13 | Others | -- | -- | -- | VOF / experimental |

### Table 3: NS Force Terms

| # | Paper | Kelvin force | Capillary/elastic stress | Spin torque | Gravity |
|---|-------|-------------|------------------------|------------|---------|
| 1 | Nochetto | mu_0*(m.grad)h | (lam/eps)*theta*grad(psi) | No | Boussinesq |
| 2 | Nochetto | mu_0*(m.grad)h | (lam/eps)*theta*grad(psi) | No | Boussinesq |
| 3 | Zhang/He/Yang | mu*(m.grad)h | Phi*grad(W) | (mu/2)*curl(mxh) | No |
| 5 | Wu/Yang | mu*(m.grad)h | Phi*grad(W) | (mu/2)*curl(mxh) | No |
| 6 | Yang (JCP) | mu*(m.grad)h | Phi*grad(W) | (mu/2)*curl(mxh) | No |
| 7 | Chen/Li/Li/He | mu*(m.grad)h | (lam/eps)*Phi*grad(W) | No | No |
| 8 | Zhang/Zhou | mu*(m.grad)h | phi*grad(w) | No | No |
| 9 | Hu/Li/Niu | (mu_0*chi/2)*grad(|H|^2) | mu_phi*grad(phi) | No | Yes |

### Table 4: Magnetization Model

| # | Paper | Magnetization equation | chi(phi) interpolation |
|---|-------|----------------------|----------------------|
| 1 | Nochetto | m_t + (u.grad)m = -(1/T)(m - chi_theta*h) | chi_0*H(theta/eps) |
| 3 | Zhang/He/Yang | m_t + (u.grad)m - 0.5*curl(u)xm + beta*mx(mxh) = -(1/tau)(m - chi*h) | chi_0/(1+e^{-(2Phi-1)/eps}) |
| 5 | Wu/Yang | Same as #3 | Same as #3 |
| 6 | Yang (JCP) | Same as #3 | Same as #3 |
| 7 | Chen/Li/Li/He | m_t + (u.grad)m = -(1/tau)(m - chi(Phi)*h) | chi_0*Phi (linear) |
| 8 | Zhang/Zhou | m_t + (u.grad)m + beta*mx(mxh) = -(1/tau)(m - kappa(phi)*h) | sigmoid |
| 9 | Hu/Li/Niu | Equilibrium: M = chi*H (Langevin) | Langevin nonlinear |

### Table 5: Benchmark Comparison

| # | Paper | Convergence | Droplet | Rosensweig | Other |
|---|-------|-------------|---------|------------|-------|
| 1 | Nochetto (arXiv) | -- | Yes | Yes | Open spikes |
| 3 | Zhang/He/Yang | 1st order | Yes | Yes | -- |
| 5 | Wu/Yang | 1st + 2nd | Yes | Yes | -- |
| 6 | Yang (JCP) | 2nd order | Yes | Yes | -- |
| 7 | Chen/Li/Li/He | 1st order | Yes | Yes | Bubbles, Y-domain |
| 8 | Zhang/Zhou | 2nd order | -- | -- | Saffman-Taylor |
| 9 | Hu/Li/Niu | -- | Yes | -- | Cylinder, bubbles, merging |
| 11 | Afkhami (2008) | -- | Yes (motion) | -- | Transit time |
| 12 | Afkhami (2010) | -- | Yes (deform) | -- | Exp. comparison |
| 13 | Mao (2011) | -- | -- | -- | Pumping (exp.) |

---

## Quick Reference: Key Equations

### Cahn-Hilliard (standard form, all phase-field papers)

```
phi_t + div(u*phi) = M * Delta(W)
W = -epsilon*Delta(phi) + (1/epsilon)*f(phi)
F(phi) = (1/4)*(phi^2 - 1)^2    (double well)
f(phi) = F'(phi) = phi^3 - phi   (truncated derivative)
```

### Navier-Stokes (two-phase ferrofluid)

```
rho(phi) * [u_t + (u.grad)u] - div(nu(phi)*D(u)) + grad(p)
  = mu_0*(m.grad)h              (Kelvin force)
  + (lambda/epsilon)*phi*grad(W) (capillary stress)
div(u) = 0
```

### Magnetization (Shliomis relaxation)

```
m_t + (u.grad)m = -(1/tau)*(m - chi(phi)*h)

Full form (Zhang/He/Yang):
m_t + (u.grad)m - (1/2)*curl(u) x m + beta*m x (m x h)
  = -(1/tau)*(m - chi(phi)*h)
```

### Magnetostatics (scalar potential)

```
-Delta(phi_mag) = div(m - h_a)
h = grad(phi_mag)     (effective magnetic field)
h = h_a + h_d         (applied + demagnetizing)
BC: dn(phi_mag) = (h_a - m) . n
```

### Phase-dependent properties

```
rho(phi) = (rho_f + rho_w)/2 + (rho_f - rho_w)/2 * phi     (linear)
nu(phi) = nu_w + (nu_f - nu_w) / (1 + e^{-(2*phi-1)/epsilon})  (sigmoid)
chi(phi) = chi_0 / (1 + e^{-(2*phi-1)/epsilon})               (sigmoid)
```

### Energy (typical form)

```
E(phi,u,h,m) = lambda*(epsilon/2*||grad(phi)||^2 + (F(phi),1))
             + (1/2)*||u||^2
             + (mu/2)*||h||^2
             + (mu/(2*chi_0))*||m||^2

d/dt E + D <= C*(||h_a||^2 + ||(h_a)_t||^2)
```

---

## Implementation Notes for Phase B

1. **Chemical potential**: No magnetic term needed in W. The magnetic coupling
   enters through chi(phi) in the magnetization equation and Kelvin force in NS.

2. **Capillary force**: Use the simplified form (lambda/epsilon)*phi*grad(W),
   which arises from pressure redefinition (Nochetto Remark 2.1).

3. **Energy identity**: B(u,m,h) = -B(m,h,u) where B(m,h,u) = sum_ij integral[m^i * h^j_{x_i} * u^j dx].
   This is crucial for the energy estimate and must be preserved in discretization.

4. **Decoupling toolkit** (for efficiency):
   - SAV or IEQ for double-well nonlinearity
   - ZEC for coupling terms
   - Pressure projection for velocity-pressure
   - Reformulated magnetostatics for phi-m coupling

5. **Variable density**: Use sigma = sqrt(rho(phi)) reformulation (Chen et al. 2025)
   to handle non-constant density while maintaining energy stability.

6. **BDF2 advantage**: Second-order schemes (Papers 5, 6, 8) are clearly superior
   for accuracy. BDF2 + SAV-ZEC is the state-of-the-art.

7. **Kelvin force**: mu_0*(m.grad)h form. For linear magnetization m ~ chi*h,
   this reduces to (mu_0*chi/2)*grad(|H|^2).

8. **Susceptibility interpolation**: Sigmoid H(x) = 1/(1+e^{-x}) is standard.
   Some papers use linear interpolation chi = chi_0*phi for simplicity.
