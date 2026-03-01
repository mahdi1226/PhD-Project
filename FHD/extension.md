# FHD Extension Roadmap: Novel Coupled PDE Systems

Extending the Nochetto-Salgado-Tomas ferrohydrodynamics framework with additional
physics. Each tier is ranked by **novelty** (what's publishable) and **feasibility**
(reuses existing FEM infrastructure).

Current system: NS + Angular Momentum + Magnetization + Poisson (+ passive scalar).

---

## Tier 1: High Impact, Directly Feasible

### 1A. Heat Equation (Thermomagnetic Convection) **[RECOMMENDED FIRST]**

**PDE:** T_t + u . nabla T - kappa_T laplacian T = Q(M, H)

**Coupling:**
- Temperature T enters the magnetic susceptibility: chi(T) = chi_0 * T_C / T
  (Curie-Weiss law), so hotter regions become less magnetic
- Buoyancy via Boussinesq: f_buoy = -rho_0 * beta * (T - T_ref) * g added to NS
- Joule-like heating Q = mu_0 * tau^{-1} |M - chi(T) H|^2 from magnetization relaxation
- One-way coupling from velocity (advection of T) already proven with passive scalar

**Why novel:**
- Thermomagnetic convection in ferrofluids is an active research area
- The Curie-Weiss feedback (chi depends on T) creates Benard-like instabilities
  that are NOT in the Nochetto paper
- Combined magnetic + thermal + micropolar effects are rarely done with rigorous
  FEM discretization

**Implementation:**
- Copy passive scalar subsystem, add reaction term Q(M,H)
- Modify magnetization assembler: chi_0 -> chi(T_q) at each quadrature point
- Add Boussinesq force to NS assembler (similar to Kelvin force)
- New parameters: kappa_T (thermal diffusivity), beta (expansion coefficient),
  T_ref, T_C (Curie temperature)
- ~3-5 days of work

**Validation experiments:**
- Thermomagnetic convection in a cavity with vertical temperature gradient
  and horizontal magnetic field (compare with Ganguly et al., 2004)
- Heated ferrofluid droplet under non-uniform field

### 1B. Cahn-Hilliard (Two-Phase Ferrofluid) **[Phase B of project]**

**PDE:** c_t + u . nabla c = nabla . (M_CH * nabla mu_CH)
         mu_CH = Psi'(c) - epsilon^2 laplacian c

**Coupling:**
- Phase field c determines material properties: nu(c), rho(c), chi(c)
- Surface tension via capillary stress tensor in NS
- Kelvin force modulated by chi(c) -> magnetic wetting/dewetting

**Why novel:**
- Two-phase ferrofluid with diffuse interface + micropolar effects
- Magnetic Rosensweig instability with interface tracking
- Log potential (Flory-Huggins) more physical than polynomial double-well

**Implementation:**
- Already planned as Phase B; 4th-order system needs mixed formulation
- Requires careful energy-stable time stepping (convex-concave splitting)
- ~2-3 weeks of work

---

## Tier 2: Moderate Novelty, Cross-Disciplinary

### 2A. Linear Elasticity (Magnetostriction)

**PDE:** rho u_tt - nabla . sigma(u) = f_Maxwell + f_body
         sigma = C : epsilon(u), epsilon = (nabla u + nabla u^T) / 2

**Coupling:**
- Maxwell stress tensor sigma_M = mu_0 (H otimes H - |H|^2 I / 2) acts on
  deformable magnetic body
- Deformation changes domain geometry (fluid-structure interaction)
- Magnetic body force from gradient of |H|^2

**Why novel:**
- Magnetostriction (deformation of magnetic materials under field) is
  important for MEMS/actuators
- Coupling ferrofluid flow + elastic deformation is rare in FEM literature
- Could model magnetic elastomers (composite materials)

**Implementation:**
- Vector-valued CG system (similar to NS but hyperbolic)
- ALE (Arbitrary Lagrangian-Eulerian) for moving mesh, or immersed boundary
- Coupling through Maxwell stress on fluid-solid interface
- ~3-4 weeks of work (ALE is the main challenge)

**Validation:**
- Deformation of a soft magnetic sphere in uniform field (analytical solution exists)
- Vibration of ferrofluid-filled elastic membrane

### 2B. Poroelasticity (Biot's Equations)

**PDE:** -nabla . sigma_eff + alpha nabla p_f = f_body     (momentum)
         d/dt(S p_f + alpha nabla . u) - nabla . (K nabla p_f) = Q  (fluid mass)

**Coupling:**
- Ferrofluid flows through deformable porous medium
- Magnetic body forces on pore fluid affect consolidation
- Could model magnetic drug delivery through tissue

**Why novel:**
- Magnetically-driven flow in deformable porous media is virtually unstudied
  with rigorous FEM
- Biomedical applications: targeted drug delivery, tissue engineering
- Combines three-field saddle-point problem with ferrofluid physics

**Implementation:**
- Mixed formulation: displacement u (CG Q2) + pore pressure p_f (CG Q1)
- Biot coupling similar to Stokes: inf-sup stable pair
- Magnetic body force enters as source term
- ~4-5 weeks of work

---

## Tier 3: High Novelty, More Challenging

### 3A. Multi-Species Transport (Reaction-Diffusion)

**PDE:** c_i,t + u . nabla c_i - D_i laplacian c_i = R_i(c_1, ..., c_N)

**Coupling:**
- Multiple species advected by ferrofluid flow
- Reactions R_i model chemical kinetics (e.g., nanoparticle aggregation)
- Species concentrations can affect fluid viscosity and susceptibility

**Why novel:**
- Magnetically-controlled microreactors (lab-on-chip applications)
- Particle aggregation/disaggregation under magnetic fields
- Extension of passive scalar to reactive multi-component system

**Implementation:**
- N copies of passive scalar subsystem with reaction coupling
- Operator splitting: transport step + reaction step (Strang splitting)
- Main challenge: stiff reaction terms may need implicit treatment
- ~2-3 weeks of work

### 3B. Full Maxwell Equations (Time-Harmonic or Transient)

**PDE:** curl E = -dB/dt, curl H = J + dD/dt
         B = mu_0(H + M), D = epsilon E

**Coupling:**
- Replaces Poisson (magnetostatic) with full electromagnetic model
- Enables AC magnetic fields, eddy currents, electromagnetic induction
- Lorentz force J x B replaces Kelvin force in NS

**Why novel:**
- Full EM + ferrofluid + micropolar is essentially unexplored territory
- Enables modeling of inductive heating, AC field-driven flows
- Could handle high-frequency field applications

**Implementation:**
- Nedelec (edge) elements for H, Raviart-Thomas for B
- Saddle-point structure similar to NS but in EM context
- Time-harmonic formulation simpler than full transient
- ~6-8 weeks of work (new element types, complex arithmetic)

---

## Recommended Path

**Phase 1 (Current):** Complete Sections 7.1-7.3 of Nochetto paper
  - [done] 7.1 Spinning magnet
  - [done] 7.2 Pumping
  - [in progress] 7.3 Stirring with passive scalar

**Phase 2 (Next - 1-2 weeks):** Heat equation (Tier 1A)
  - Fastest path to a novel result
  - Reuses passive scalar infrastructure (just add reaction + feedback)
  - Thermomagnetic convection experiments are physically compelling
  - Publishable as: "Thermomagnetic convection in micropolar ferrofluids:
    a fully-coupled FEM approach"

**Phase 3 (Following - 2-3 weeks):** Cahn-Hilliard (Tier 1B)
  - Natural extension of the current framework
  - Two-phase ferrofluid is the stated Phase B goal
  - Rosensweig instability with interface tracking is a showcase result

**Phase 4 (If time permits):** One of Tier 2 depending on research direction
  - Biomedical focus -> Poroelasticity (drug delivery)
  - Materials/MEMS focus -> Magnetostriction
  - Microfluidics focus -> Multi-species transport
