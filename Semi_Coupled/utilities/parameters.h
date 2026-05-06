// ============================================================================
// utilities/parameters.h - Runtime Configuration (Numerics Only)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
//
// Physics constants are in physics/material_properties.h
// ============================================================================
#ifndef PARAMETERS_H
#define PARAMETERS_H

#include "solvers/solver_info.h"  // LinearSolverParams

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <vector>
#include <string>
#include <iostream>

// ============================================================================
// Runtime Parameters (Numerics / Configuration)
// ============================================================================
struct Parameters
{
    Parameters()
    {
        gravity_direction[0] = 0.0;
        gravity_direction[1] = -1.0;
    }

    // ========================================================================
    // Current simulation time (updated by time-stepping loop)
    // ========================================================================
    double current_time = 0.0;

    // ========================================================================
    // Domain geometry
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

    struct Dipoles
    {
        std::vector<dealii::Point<2>> positions;
        std::vector<double> direction = {0.0, 1.0};
        double intensity_max = 6000.0;
        double ramp_time = 1.6;
        double regularization = 0.01;   // δ = regularization * min_dipole_dist

        // Uniform field mode: constant h_a everywhere (bypasses dipole computation)
        bool use_uniform_field = false;
        dealii::Tensor<1, 2> uniform_field_value;  // constant h_a when use_uniform_field=true
    } dipoles;

    // ========================================================================
    // Time-stepping
    // ========================================================================
    struct Time
    {
        double dt = 5e-4;
        double t_final = 2.0;
        unsigned int max_steps = 4000;
        bool use_adaptive_dt = true;
    } time;

    struct IC
    {
        int type = 0;              // 0 = flat pool, 1 = circular droplet, 2 = diamond droplet
        double pool_depth = 0.2;
        double perturbation = 0.0;
        int perturbation_modes = 0;
        // Droplet parameters (used when type = 1 or 2)
        double droplet_center_x = 0.5;
        double droplet_center_y = 0.5;
        double droplet_radius = 0.25;
    } ic;

    // ========================================================================
    // Mesh / AMR
    // ========================================================================
    struct Mesh
    {
        unsigned int initial_refinement = 5;
        bool use_amr = true;
        unsigned int amr_min_level = 0;
        unsigned int amr_max_level = 0;
        unsigned int amr_interval = 5;
        double amr_upper_fraction = 0.3;
        double amr_lower_fraction = 0.7;
        double amr_refine_threshold = 0.0;  // Min Kelly error to allow refinement (0 = disabled)
        double interface_coarsen_threshold = 0.9;
    } mesh;

    // ========================================================================
    // Physical parameters (Paper Section 6.2-6.3)
    // These vary between Rosensweig and Hedgehog!
    // ========================================================================
    struct Physics
    {
        // Cahn-Hilliard (Eq. 14a-14b)
        double epsilon = 0.01;      // interface thickness (Rosensweig: 0.01, Hedgehog: 0.005)
        double mobility = 0.0002;   // γ mobility coefficient
        double lambda = 0.05;       // capillary coefficient (Rosensweig: 0.05, Hedgehog: 0.025)

        // Viscosity (Eq. 17)
        double nu_water = 1.0;      // non-magnetic phase viscosity
        double nu_ferro = 2.0;      // ferrofluid phase viscosity

        // Magnetic (Section 6.2)
        double chi_0 = 0.5;         // susceptibility (Rosensweig: 0.5, Hedgehog: 0.9)
        double mu_0 = 1.0;          // permeability of free space
        double tau_M = 1e-6;        // magnetization relaxation time

        // Density / Gravity (Eq. 19)
        double rho = 1.0;           // reference density
        double r = 0.1;             // density ratio
        double gravity = 30000.0;   // non-dimensionalized gravity
    } physics;

    // ========================================================================
    // Finite element degrees
    // ========================================================================
    struct FE
    {
        unsigned int degree_phase = 2;
        unsigned int degree_velocity = 2;
        unsigned int degree_pressure = 1;
        unsigned int degree_potential = 2;
        unsigned int degree_magnetization = 1;
    } fe;

    // ========================================================================
    // Output
    // ========================================================================
    struct Output
    {
        std::string folder = "./Results";
        std::string run_name = "";   // Auto-generated if empty: preset-rN[-amr]

        // VTK output cadence — TWO modes, in order of precedence:
        //
        //  1) Time-based (NEW DEFAULT). If dt_output > 0, write a VTU when
        //     simulation time crosses an integer multiple of dt_output.
        //     File count = round(t_final / dt_output) — independent of dt.
        //     Recommended for production runs and cross-dt comparisons.
        //
        //  2) Step-based (LEGACY). If dt_output <= 0, write every
        //     `frequency` steps. Kept for backward compat with --vtk_interval.
        //
        // The CLI flag --vtk_interval N forces mode (2) by setting dt_output = 0.
        // The CLI flag --output_dt T forces mode (1) by setting dt_output = T.
        double dt_output = 0.01;     // physics-time spacing; <=0 disables time mode
        unsigned int frequency = 10; // step-based fallback (mode 2)
        bool verbose = false;
    } output;

    // ========================================================================
    // Preset name (for auto-generating run_name)
    // ========================================================================
    std::string preset_name = "custom";

    // ========================================================================
    // Subsystem enables (for debugging)
    // ========================================================================
    bool enable_magnetic = true;
    bool enable_ns = true;
    bool enable_gravity = true;
    bool enable_mms = false;
    bool use_h_a_only = false;  // H = h_a (no demagnetizing field, paper Section 5)

    // MMS
    double mms_t_init = 0.0;

    // ------------------------------------------------------------------------
    // MMS time-derivative source convention (added 2026-05-05).
    //
    // false (default): MMS sources use **discrete** time differences,
    //   e.g.  f = (theta_new - theta_old)/dt + ... .  This matches the
    //   discrete scheme exactly in time and yields tau-independent error
    //   in the temporal convergence tests — useful for verifying that the
    //   *implementation* matches the discrete equations.
    //
    // true: MMS sources use **analytical** time derivatives,
    //   e.g.  f = dtheta_dt(t) + ... .  This exposes the BE truncation
    //   error so the temporal tests can measure the formal scheme order
    //   (expected ~1.0 for backward Euler).
    //
    // Toggle with `--mms-analytical` in the MMS test driver.
    // ------------------------------------------------------------------------
    bool mms_analytical_dt = false;

    // Gravity direction (unit vector)
    dealii::Tensor<1, 2> gravity_direction;

    // ========================================================================
    // Block-Gauss-Seidel global iteration (Paper CMAME 2016, Section 6, p.520)
    // Single pass per time step: [CH] -> [Mag+Poisson] -> [NS]
    // Paper describes BGS structure but does not specify iteration count.
    // Testing shows iterating to convergence diverges at strong coupling.
    // ========================================================================
    unsigned int bgs_max_iterations = 1;   // Single BGS pass per time step
    double bgs_tolerance = 1e-2;           // Relative change tolerance for convergence

    // ========================================================================
    // Solver parameters
    // ========================================================================
    struct Solvers
    {
        LinearSolverParams ch = {
            LinearSolverParams::Type::GMRES,
            LinearSolverParams::Preconditioner::AMG,
            1e-8, 1e-12, 2000, 50, 1.2, 1.2,  // ssor_omega, ilu_strengthen
            false, true, false
        };

        // NS solver: Direct (MUMPS) is 10-50x faster than iterative for ref 3-5
        // Auto-fallback to iterative (Block Schur) if direct fails or for large problems
        LinearSolverParams ns = {
            LinearSolverParams::Type::Direct,  // Changed: MUMPS much faster than FGMRES
            LinearSolverParams::Preconditioner::BlockSchur,  // Used if fallback to iterative
            1e-6, 1e-9,
            1500, 100, 1.2, 1.2,
            false, true, true  // use_iterative=false, fallback=true, verbose=true
        };

        // Magnetic solver: Direct (MUMPS) by default; iterative path = GMRES
        // with cached ILU on the full monolithic [Mx | My | phi] system.
        // - For h_a-only mode (dome): phi block is trivial, ilu_fill=0 works.
        // - For full Poisson (hedgehog/Rosensweig): phi is a Laplacian, needs
        //   higher fill. ilu_fill=4, tolerance 1e-7 gives ~10x speedup at L5.
        // Note: tolerance 1e-7 is slightly looser than CH/NS (1e-8) — allows
        // the magnetic GMRES to terminate before the residual stalls. Errors
        // accumulate as O(tol * n_steps); 1e-7 over 60k steps is acceptable.
        LinearSolverParams magnetic = {
            LinearSolverParams::Type::GMRES,
            LinearSolverParams::Preconditioner::ILU,
            1e-7, 1e-12,           // looser tol than CH/NS — see comment
            1000, 50, 1.2, 1.2,
            false, true, false
        };
    } solvers;

    // ========================================================================
    // Preset configurations (numerics only)
    // ========================================================================
    void setup_rosensweig();
    void setup_hedgehog();
    void setup_droplet();
    void setup_droplet_uniform_B();                  // Droplet + uniform magnetic field
    void setup_elongation();                          // Ferrofluid droplet elongation in uniform field
    void setup_square();                             // Square relaxation test
    void setup_dome();                               // Dome configuration

    // ========================================================================
    // Parallel diagnostics (--parallel-diag)
    // Records assembly/solve timing breakdown, sparsity, load balance
    // ========================================================================
    bool enable_parallel_diagnostics = false;       // Write parallel_diagnostics.csv
    bool parallel_diag_all_ranks = false;           // Also write per-rank CSV files

    // ========================================================================
    // DoF renumbering (--renumber-dofs)
    // Cuthill-McKee reduces matrix bandwidth → faster direct solvers
    // ========================================================================
    bool renumber_dofs = false;                     // Apply Cuthill-McKee to CG DoFHandlers

    // ========================================================================
    // Sparsity pattern export (--dump-sparsity)
    // Exports SVG/gnuplot of sparsity patterns + bandwidth + per-row nnz
    // ========================================================================
    bool dump_sparsity = false;                     // Export sparsity patterns at step 0

    // ========================================================================
    // Diagnostics frequency (--diagnostics_frequency N)
    // 0 = disabled, 1 = every step (default), N = every N steps
    // ========================================================================
    unsigned int diagnostics_frequency = 1;

    // ========================================================================
    // Build run_name from preset + refinement + amr (call after parsing)
    // ========================================================================
    void finalize_run_name();

    // ========================================================================
    // Command line parsing
    // ========================================================================
    static Parameters parse_command_line(int argc, char* argv[]);
};

#endif // PARAMETERS_H