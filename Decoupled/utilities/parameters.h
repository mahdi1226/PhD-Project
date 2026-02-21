// ============================================================================
// utilities/parameters.h - Runtime Configuration
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
//            Zhang, He & Yang, CMAME 371 (2020) — β-term extension
//
// Includes parameters for all subsystems:
//   Poisson (Eq. 42d), Magnetization (Eq. 42c),
//   Cahn-Hilliard (Eq. 42a-b), Navier-Stokes (Eq. 42e-f)
// ============================================================================
#ifndef PARAMETERS_H
#define PARAMETERS_H

#include "utilities/solver_info.h"

#include <deal.II/base/point.h>

#include <vector>
#include <string>

struct Parameters
{
    // ========================================================================
    // Domain geometry (Section 6.2, p.522)
    // ========================================================================
    struct Domain
    {
        double x_min = 0.0;
        double x_max = 1.0;
        double y_min = 0.0;
        double y_max = 0.6;
        unsigned int initial_cells_x = 10;
        unsigned int initial_cells_y = 6;
    } domain;

    // ========================================================================
    // Applied magnetic field h_a
    //
    // Two modes:
    //   1. Uniform field: h_a = direction * intensity * ramp(t)
    //      Used by Zhang, He & Yang (2020) for all validation tests.
    //   2. Dipole sources: h_a from 2D line dipoles (Nochetto Eq. 97-98)
    //
    // The uniform field is checked first. If uniform_field.enabled = true,
    // dipoles are ignored.
    // ========================================================================
    struct UniformField
    {
        bool enabled = false;
        std::vector<double> direction = {0.0, 1.0};  // field direction (unit)
        double intensity_max = 0.0;                   // max field magnitude
        double ramp_time = 0.0;                       // 0 = instant
        double ramp_slope = 0.0;                      // α(t) = slope*t (0 = use ramp_time mode)
    } uniform_field;

    struct Dipoles
    {
        std::vector<dealii::Point<2>> positions;     // dipole locations (2D only)
        std::vector<double> direction = {0.0, 1.0};  // dipole moment direction
        double intensity_max = 6000.0;               // maximum field intensity
        double ramp_time = 1.6;                      // ramp-up time (0 = instant)
        double ramp_slope = 0.0;                     // α(t) = slope*t (0 = use ramp_time mode)
    } dipoles;

    // ========================================================================
    // Time-stepping
    //
    // Used by all time-dependent subsystems:
    //   Magnetization (Eq. 42c), Cahn-Hilliard (Eq. 42a-b),
    //   Navier-Stokes (Eq. 42e-f), and the production time loop.
    //   Poisson standalone is steady-state.
    // ========================================================================
    struct Time
    {
        double dt = 5e-4;               // time step
        double t_final = 2.0;           // final simulation time
        unsigned int max_steps = 4000;  // safety limit
    } time;

    // ========================================================================
    // Physical parameters (Paper Section 6.2-6.3)
    // ========================================================================
    struct Physics
    {
        double epsilon = 0.01;    // interface thickness (Eq. 17-18)
        double chi_0 = 0.5;      // max susceptibility in ferrofluid (Eq. 18)
        double mu_0 = 1.0;       // permeability of free space

        // -- Magnetization (Eq. 42c, Section 6.2) --
        double tau_M = 1e-6;     // magnetization relaxation time

        // Shliomis extension: Landau-Lifshitz damping (Zhang-He-Yang 2021)
        //   β M×(M×H) = β[M(M·H) - H|M|²]  (2D identity)
        double beta = 0.0;              // β coefficient (0 = base Nochetto)
        bool enable_beta_term = false;  // explicit enable flag

        // -- Cahn-Hilliard (Eq. 42a-b, Section 6.2) --
        //
        // mobility γ:  controls diffusion rate of the interface
        //   Eq. 42a RHS: γ(∇ψ, ∇χ)  (called "Pe = 1/γ" in some references)
        //
        // lambda λ:  capillary/surface tension coefficient
        //   Appears in the CH energy: E_CH = λ ∫ [ε/2|∇θ|² + (1/ε)F(θ)] dΩ
        //   Couples CH to NS via capillary force: λ ψ∇θ
        //
        // Rosensweig (§6.2): γ = 0.0002, λ = 0.05
        // Hedgehog  (§6.3): γ = 0.0002, λ = 0.025
        double mobility = 0.0002;       // γ (Cahn-Hilliard mobility)
        double lambda = 0.05;           // λ (surface tension / capillary coefficient)

        // -- Navier-Stokes (Eq. 42e-f, Section 6.2) --
        //
        // Viscosity ν(θ) = ν_w + (ν_f - ν_w) H(θ/ε)      (Eq. 17, p.501)
        //   Interpolates between non-magnetic and ferrofluid phases.
        //   Used in: (ν(θ) T(U), T(V)) where T = ½(∇U + ∇U^T)
        //
        // Rosensweig (§6.2, p.520): ν_w = 1.0, ν_f = 2.0
        double nu_water = 1.0;          // ν_w: non-magnetic phase viscosity
        double nu_ferro = 2.0;          // ν_f: ferrofluid phase viscosity

        // Density ρ(θ) = 1 + r H(θ/ε)                     (Eq. 19, p.502)
        //   r = (ρ_ferro - ρ_water) / ρ_water is the density ratio.
        //   Note: ρ is implicitly taken as unity (p.520), so Re = O(‖u‖).
        //   The density ratio only enters the gravity body force.
        //
        // Rosensweig (§6.2, p.520): r = 0.1
        double r = 0.1;                 // density ratio (Eq. 19)

        // Gravity body force:  f_g = ρ(θ) g                (Eq. 19, p.502)
        //   gravity_magnitude * gravity_direction gives g vector.
        //   The paper uses non-dimensional gravity with |g| = 30000.
        //
        // Rosensweig (§6.2): |g| = 30000, direction = (0, -1)
        double gravity_magnitude = 30000.0;  // |g| (non-dimensional)
        std::vector<double> gravity_direction = {0.0, -1.0};  // unit vector
    } physics;

    // ========================================================================
    // Mesh
    // ========================================================================
    struct Mesh
    {
        unsigned int initial_refinement = 5;

        // -- AMR parameters will be added here --
    } mesh;

    // ========================================================================
    // Finite element degrees
    //
    // Poisson:        Q2 continuous    (degree_potential = 2)
    // Magnetization:  DG-Q1 discontinuous  (degree_magnetization = 1)
    // CH (θ, ψ):     Q2 continuous    (degree_phase = 2)
    // Velocity:       Q2 continuous    (degree_velocity = 2)
    // Pressure:       DG-Q1 discontinuous  (degree_pressure = 1)
    //
    // degree_velocity is included because the Magnetization and CH MMS tests
    // need CG velocity DoFHandlers for the advection input U.
    //
    // DG pressure (FE_DGQ) enforces local incompressibility per element,
    // which is critical for Kelvin force stability (Section 4.1).
    //
    // NOTE: φ MUST be Q2 (not Q1) because the Kelvin force μ₀(M·∇)H
    // requires Hess(φ). With Q1 (bilinear) elements, the Hessian has
    // ZERO diagonal entries (∂²φ/∂x²=0, ∂²φ/∂y²=0), so (M·∇)H only
    // produces cross-derivative forces — vertical M gives only horizontal
    // force, which is physically wrong. Q2 (biquadratic) elements have
    // non-zero ∂²φ/∂x² and ∂²φ/∂y², giving correct force directions.
    // This matches the Round1/Parallel code (degree_potential = 2).
    // ========================================================================
    struct FE
    {
        unsigned int degree_potential = 2;       // Poisson φ: CG Q2 (MUST be ≥2 for Hess)
        unsigned int degree_magnetization = 1;   // Magnetization M: DG Q1 (Paper: M_h)
        unsigned int degree_velocity = 2;        // Velocity U: CG Q2 (Paper: V_h)
        unsigned int degree_phase = 2;           // CH θ, ψ: CG Q2 (Paper: Θ_h)
        unsigned int degree_pressure = 1;        // Pressure p: DG Q1 (Paper: Q_h)
    } fe;

    // ========================================================================
    // Output
    // ========================================================================
    struct Output
    {
        std::string folder = "../Results";
        bool verbose = false;
        unsigned int vtk_interval = 20;
    } output;

    // ========================================================================
    // Solver parameters
    // ========================================================================
    struct Solvers
    {
        LinearSolverParams poisson = {
            LinearSolverParams::Type::CG,
            LinearSolverParams::Preconditioner::AMG,
            /*rel_tolerance=*/1e-8,
            /*abs_tolerance=*/1e-12,
            /*max_iterations=*/2000,
            /*gmres_restart=*/50,
            /*ssor_omega=*/1.2,
            /*ilu_strengthen=*/1.0,
            /*use_iterative=*/true,
            /*fallback_to_direct=*/true,
            /*verbose=*/true
        };

        // Magnetization: Direct solver (MUMPS) for small DG systems.
        // Falls back to GMRES+ILU for large problems.
        // ILU(0) exploits DG block-diagonal structure effectively.
        LinearSolverParams magnetization = {
            LinearSolverParams::Type::Direct,
            LinearSolverParams::Preconditioner::ILU,
            /*rel_tolerance=*/1e-8,
            /*abs_tolerance=*/1e-12,
            /*max_iterations=*/1000,
            /*gmres_restart=*/50,
            /*ssor_omega=*/1.2,
            /*ilu_strengthen=*/1.0,
            /*use_iterative=*/false,
            /*fallback_to_direct=*/true,
            /*verbose=*/false
        };

        // CH: Direct solver (MUMPS) for coupled θ-ψ system.
        // The coupled system is indefinite (saddle-point-like), so
        // direct solvers are preferred. Falls back through MUMPS →
        // SuperLU_DIST → KLU.  For large problems, GMRES+AMG
        // fallback is available but less reliable for indefinite systems.
        LinearSolverParams ch = {
            LinearSolverParams::Type::Direct,
            LinearSolverParams::Preconditioner::AMG,
            /*rel_tolerance=*/1e-8,
            /*abs_tolerance=*/1e-12,
            /*max_iterations=*/2000,
            /*gmres_restart=*/50,
            /*ssor_omega=*/1.2,
            /*ilu_strengthen=*/1.0,
            /*use_iterative=*/false,
            /*fallback_to_direct=*/true,
            /*verbose=*/false
        };

        // NS: Direct solver (MUMPS) for saddle-point system.
        //
        // The coupled velocity-pressure system has structure:
        //   [A  B^T] [U]   [f]
        //   [B   0 ] [p] = [0]
        //
        // Direct solver is preferred for refinement levels 3-6
        // (~1K to ~500K DoFs). For larger problems, Block Schur
        // complement preconditioner with FGMRES:
        //   - A block: AMG-preconditioned CG
        //   - S block: pressure mass matrix scaled by (ν + 1/dt)^{-1}
        //
        // Falls back through: MUMPS → BlockSchur FGMRES → SuperLU_DIST
        LinearSolverParams ns = {
            LinearSolverParams::Type::Direct,
            LinearSolverParams::Preconditioner::BlockSchur,
            /*rel_tolerance=*/1e-6,
            /*abs_tolerance=*/1e-10,
            /*max_iterations=*/500,
            /*gmres_restart=*/100,
            /*ssor_omega=*/1.2,
            /*ilu_strengthen=*/1.0,
            /*use_iterative=*/false,
            /*fallback_to_direct=*/true,
            /*verbose=*/false,
            // Block Schur settings
            /*schur_inner_tolerance=*/1e-3,
            /*schur_max_inner_iters=*/20,
            /*schur_gmres_restart=*/30,
            /*direct_dof_threshold=*/2000000
        };
    } solvers;

    // ========================================================================
    // Picard iteration (Poisson ↔ Magnetization coupling)
    // ========================================================================
    unsigned int picard_iterations = 7;
    double picard_tolerance = 0.01;

    // ========================================================================
    // Validation test mode
    //
    //   --validation droplet   Circle IC, no dipoles, surface tension only
    //   --validation square    Diamond IC, dipoles active, with gravity
    // ========================================================================
    std::string validation_test;  // "" = production, "droplet", "square"

    // ========================================================================
    // Subsystem enables
    // ========================================================================
    bool enable_magnetic = true;               // enable applied field h_a
    bool enable_mms = false;                   // MMS verification mode
    bool use_reduced_magnetic_field = false;    // Dome: H = h_a only (no ∇φ)
    bool enable_ns = true;                     // enable Navier-Stokes solve
    bool enable_gravity = true;                // enable gravity body force

    // ========================================================================
    // Zhang's SAV+ZEC energy stabilization framework
    //
    // Reference: Zhang, He & Yang, SIAM J. Sci. Comput. 43(1) B167-B193 (2021)
    //
    // SAV: Scalar Auxiliary Variable r(t) = sqrt(E1(theta) + C0)
    //   Enables unconditionally energy-stable time stepping by replacing
    //   the nonlinear f(theta)/eps with (r/sqrt(E1+C0)) * f(theta)/eps
    //
    // ZEC: Zero Energy Contribution property
    //   -mu0(m.grad h, u) - mu0(u.grad m, h) - mu0(u x m, curl h) = 0
    //   Allows explicit Kelvin force treatment while maintaining energy balance
    //
    // Algebraic magnetization: m = chi(theta) * h  (no PDE)
    //   Eliminates magnetization subsystem entirely.
    //   Makes Poisson nonlinear: ((1+chi(theta)) grad phi, grad X) = ((1-chi(theta)) h_a, grad X)
    // ========================================================================
    bool use_algebraic_magnetization = false;  // true = m=chi(theta)h, false = mag PDE
    bool use_sav = false;                      // true = SAV energy stabilization

    struct SAV
    {
        double C0 = 1.0;       // positivity constant: r(t) = sqrt(E1(theta) + C0)
        double S1 = 0.0;       // CH stabilization: S1 >= 1/eps (auto-computed if 0)
        double S2 = 0.0;       // NS stabilization: S2 >= mu0^2 C_M^2 / (4 nu_min)
    } sav;

    // ========================================================================
    // Run configuration (CLI mode dispatch)
    //
    // Shared by all 4 subsystem drivers:
    //   --mode mms|2d|3d|temporal
    //   --ref 2 3 4 5 6          (multi-value refinement list)
    //   --steps N                 (override time steps)
    // ========================================================================
    struct Run
    {
        std::string mode = "mms";
        std::vector<unsigned int> refs = {2, 3, 4, 5, 6};
        int steps = -1;   // -1 = subsystem default
    } run;

    // ========================================================================
    // Preset configurations
    // ========================================================================
    void setup_rosensweig();

    // -- Other presets (hedgehog, droplet, dome) added later --

    // ========================================================================
    // Command line parsing
    // ========================================================================
    static Parameters parse_command_line(int argc, char* argv[]);
};

#endif // PARAMETERS_H