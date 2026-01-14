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
        int type = 0;              // 0 = flat pool, 1 = circular droplet
        double pool_depth = 0.2;
        double perturbation = 0.0;
        int perturbation_modes = 0;
        // Droplet parameters (used when type = 1)
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
        double amr_lower_fraction = 0.01;
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
        double grad_div_stabilization = 0.0;  // Optional stabilization
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
        std::string folder = "../Results";
        std::string run_name = "";   // Auto-generated if empty: preset-rN[-amr]
        unsigned int frequency = 10;
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
    bool use_dg_transport = true;

    // MMS
    double mms_t_init = 0.0;

    // Gravity direction (unit vector)
    dealii::Tensor<1, 2> gravity_direction;

    // ========================================================================
    // Picard iteration settings
    // ========================================================================
    unsigned int picard_iterations = 7;
    double picard_tolerance = 0.01;

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

        LinearSolverParams poisson = {
            LinearSolverParams::Type::CG,
            LinearSolverParams::Preconditioner::AMG,
            1e-8, 1e-12, 2000, 50, 1.2, 1.0,  // ssor_omega, ilu_strengthen (unused for SSOR)
            true, true, true
        };

        LinearSolverParams magnetization = {
            LinearSolverParams::Type::Direct,
            LinearSolverParams::Preconditioner::None,
            1e-8, 1e-12, 1000, 50, 1.2, 1.0,  // ssor_omega, ilu_strengthen (unused for Direct)
            false, true, false  // Direct solver for small DG system
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
    } solvers;

    // ========================================================================
    // Preset configurations (numerics only)
    // ========================================================================
    void setup_rosensweig();
    void setup_hedgehog();
    void setup_droplet();
    void setup_dome();                               // Dome configuration
    bool use_reduced_magnetic_field = false;        // Dome set-up h = h_a only

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