// ============================================================================
// utilities/parameters.h - Runtime Configuration for FHD
//
// Phase A: Reproduce Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015)
//          Single-phase ferrofluid: NS + Angular Momentum + Magnetization + Poisson
//
// Phase B: Two-phase extension with Cahn-Hilliard (log potential)
//          Adds diffuse interface with c-dependent material properties
//
// Subsystems (Phase A):
//   Navier-Stokes (Eq 1a-1b), Angular Momentum (Eq 1c),
//   Magnetization (Eq 1d), Poisson (Eq 20)
//
// Subsystem (Phase B):
//   Cahn-Hilliard with Flory-Huggins logarithmic free energy
// ============================================================================
#ifndef FHD_PARAMETERS_H
#define FHD_PARAMETERS_H

#include "utilities/solver_info.h"

#include <deal.II/base/point.h>

#include <vector>
#include <string>

struct Parameters
{
    // ========================================================================
    // Domain geometry
    // ========================================================================
    struct Domain
    {
        double x_min = 0.0;
        double x_max = 1.0;
        double y_min = 0.0;
        double y_max = 1.0;
        unsigned int initial_cells_x = 1;
        unsigned int initial_cells_y = 1;
    } domain;

    // ========================================================================
    // Applied magnetic field h_a (Nochetto Eq 101-103)
    //
    // Point dipole potential: φ_s(x) = d·(x_s - x) / |x_s - x|^2  (2D)
    // Applied field: h_a = Σ α_s ∇φ_s
    //
    // Also supports uniform field: h_a = direction * intensity * ramp(t)
    // ========================================================================
    struct UniformField
    {
        bool enabled = false;
        std::vector<double> direction = {0.0, 1.0};
        double intensity_max = 0.0;
        double ramp_time = 0.0;
        double ramp_slope = 0.0;
    } uniform_field;

    struct Dipoles
    {
        std::vector<dealii::Point<2>> positions;
        std::vector<double> direction = {0.0, 1.0};   // shared direction
        double intensity_max = 10.0;
        double ramp_time = 1.0;
        double ramp_slope = 0.0;

        // Per-dipole overrides (when non-empty, override shared values)
        std::vector<std::vector<double>> directions;   // per-dipole d_s
        std::vector<double> intensities;               // per-dipole α_s

        double frequency = 20.0;   // oscillation frequency f (Hz)
    } dipoles;

    // Experiment identifier (set by presets for time-dependent logic)
    std::string experiment_name;

    // ========================================================================
    // Time-stepping
    // ========================================================================
    struct Time
    {
        double dt = 1e-2;
        double t_final = 4.0;
        unsigned int max_steps = 10000;
    } time;

    // ========================================================================
    // Physical parameters (Nochetto Section 2)
    //
    // All constitutive constants from the Rosensweig model:
    //   ν      kinematic viscosity
    //   ν_r    vortex viscosity (micropolar coupling)
    //   μ₀     magnetic permeability of free space
    //   ȷ      microinertia density
    //   c_1    angular viscosity c_a + c_d
    //   c_2    angular viscosity c_a - c_d
    //   σ      magnetic diffusion (0 for pure transport)
    //   T      Debye relaxation time (𝒯 in paper)
    //   κ₀     magnetic susceptibility
    // ========================================================================
    struct Physics
    {
        // -- Shared --
        double mu_0 = 1.0;           // permeability of free space
        double kappa_0 = 1.0;        // magnetic susceptibility χ₀

        // -- Navier-Stokes (Eq 1a-1b) --
        double nu = 1.0;             // kinematic viscosity ν
        double nu_r = 1.0;           // vortex viscosity ν_r

        // -- Angular Momentum (Eq 1c) --
        double j_micro = 1.0;        // microinertia ȷ
        double c_1 = 1.0;            // c_a + c_d (angular viscosity)
        double c_2 = 1.0;            // c_a - c_d (angular viscosity)

        // -- Magnetization (Eq 1d) --
        double sigma = 0.0;          // magnetic diffusion (0 = pure transport)
        double T_relax = 1e-4;       // Debye relaxation time 𝒯

        // -- Phase B: Two-phase extension --
        double nu_carrier = 1.0;     // carrier fluid viscosity
        double nu_ferro = 1.0;       // ferrofluid viscosity (Phase A: same as nu)
        double rho_carrier = 1.0;    // carrier fluid density
        double rho_ferro = 1.0;      // ferrofluid density
        double chi_ferro = 1.0;      // susceptibility of ferrofluid phase
    } physics;

    // ========================================================================
    // Mesh
    // ========================================================================
    struct Mesh
    {
        unsigned int initial_refinement = 4;
    } mesh;

    // ========================================================================
    // Finite element degrees
    //
    // Nochetto Section 4.3 / 6:
    //   Velocity U:        CG Q_ℓ enriched (Taylor-Hood pair with P)
    //   Pressure P:        CG Q_{ℓ-1}
    //   Angular velocity W: CG Q_ℓ (same as velocity)
    //   Magnetization M:   DG [Q_ℓ]^d (discontinuous)
    //   Potential φ:       CG Q_ℓ (continuous, ∇X ⊂ M required)
    //
    // With ℓ=2: Q2/Q1 Taylor-Hood, DG-Q2 magnetization, CG-Q2 potential
    // With deal.II quadrilaterals, use Q_ℓ (not P_ℓ)
    // ========================================================================
    struct FE
    {
        unsigned int degree_velocity = 2;        // U: CG Q2
        unsigned int degree_pressure = 1;        // P: CG Q1
        unsigned int degree_angular = 2;         // W: CG Q2
        unsigned int degree_magnetization = 2;   // M: DG [Q2]^d
        unsigned int degree_potential = 2;        // φ: CG Q2 (∇X ⊂ M)
        unsigned int degree_scalar = 2;          // c: CG Q2 (passive scalar)
    } fe;

    // ========================================================================
    // DG interior penalty (magnetization, Nochetto Eq 63-65)
    // ========================================================================
    struct DG
    {
        double penalty_parameter = 10.0;    // η for face penalty terms
    } dg;

    // ========================================================================
    // Output
    // ========================================================================
    struct Output
    {
        std::string folder = "Results";
        bool verbose = false;
        unsigned int vtk_interval = 10;
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
            /*verbose=*/false
        };

        LinearSolverParams magnetization = {
            LinearSolverParams::Type::GMRES,
            LinearSolverParams::Preconditioner::ILU,
            /*rel_tolerance=*/1e-8,
            /*abs_tolerance=*/1e-12,
            /*max_iterations=*/1000,
            /*gmres_restart=*/50,
            /*ssor_omega=*/1.2,
            /*ilu_strengthen=*/1.0,
            /*use_iterative=*/true,
            /*fallback_to_direct=*/true,
            /*verbose=*/false
        };

        LinearSolverParams navier_stokes = {
            LinearSolverParams::Type::Direct,
            LinearSolverParams::Preconditioner::ILU,
            /*rel_tolerance=*/1e-6,
            /*abs_tolerance=*/1e-10,
            /*max_iterations=*/500,
            /*gmres_restart=*/100,
            /*ssor_omega=*/1.2,
            /*ilu_strengthen=*/1.0,
            /*use_iterative=*/false,
            /*fallback_to_direct=*/true,
            /*verbose=*/false
        };

        LinearSolverParams angular_momentum = {
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
            /*verbose=*/false
        };

        LinearSolverParams passive_scalar = {
            LinearSolverParams::Type::GMRES,
            LinearSolverParams::Preconditioner::ILU,
            /*rel_tolerance=*/1e-8,
            /*abs_tolerance=*/1e-12,
            /*max_iterations=*/2000,
            /*gmres_restart=*/100,
            /*ssor_omega=*/1.2,
            /*ilu_strengthen=*/1.0,
            /*use_iterative=*/true,
            /*fallback_to_direct=*/true,
            /*verbose=*/false
        };
    } solvers;

    // ========================================================================
    // Picard iteration (coupling between subsystems)
    // ========================================================================
    unsigned int picard_iterations = 5;
    double picard_tolerance = 1e-6;
    double picard_relaxation = 0.5;

    // ========================================================================
    // Subsystem enables
    // ========================================================================
    bool enable_mms = false;
    bool use_simplified_model = false;  // true = h := h_a (no Poisson, scheme 78)

    // Passive scalar (Eq. 104, Section 7.3)
    struct PassiveScalar
    {
        double alpha = 0.001;   // diffusion coefficient

        // SUPG stabilization (Codina 1998)
        bool use_supg = true;         // enable SUPG for convection-dominated transport
        double supg_factor = 1.0;     // scaling factor for tau_SUPG
    } passive_scalar;
    bool enable_passive_scalar = false;

    // Dipole y-position override (for systematic y-position testing)
    double dipole_y_override = -0.1;
    bool has_dipole_y_override = false;

    // Phase B
    bool enable_cahn_hilliard = false;  // master switch for two-phase

    // ========================================================================
    // Run configuration
    // ========================================================================
    struct Run
    {
        std::string mode = "mms";
        std::vector<unsigned int> refs = {2, 3, 4, 5, 6};
        int steps = -1;
    } run;

    // ========================================================================
    // Presets (Nochetto Section 7)
    // ========================================================================
    void setup_spinning_magnet();
    void setup_pumping();
    void setup_stirring();           // common base for 7.3
    void setup_stirring_approach1(); // Section 7.3, Eq. 105
    void setup_stirring_approach2(); // Section 7.3, Eq. 106
    void setup_stirring_approach2_enhanced(); // Figure 19: f=40Hz, ν=0.1
    void setup_mms_validation();     // Section 6 MMS test

    // ========================================================================
    // Command line parsing
    // ========================================================================
    static Parameters parse_command_line(int argc, char* argv[]);
};

#endif // FHD_PARAMETERS_H
