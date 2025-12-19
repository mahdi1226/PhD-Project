// ============================================================================
// utilities/parameters.h - Runtime Configuration (Numerics Only)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
//
// Physics constants are in physics/material_properties.h
// ============================================================================
#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <vector>
#include <string>
#include <iostream>

// ============================================================================
// Linear Solver Configuration
// ============================================================================
struct LinearSolverParams
{
    enum class Type { CG, GMRES, FGMRES, Direct };
    Type type = Type::GMRES;

    enum class Preconditioner { None, Jacobi, SSOR, ILU, BlockSchur };
    Preconditioner preconditioner = Preconditioner::ILU;

    double rel_tolerance = 1e-8;
    double abs_tolerance = 1e-12;
    unsigned int max_iterations = 2000;
    unsigned int gmres_restart = 50;
    double ssor_omega = 1.2;

    bool use_iterative = true;
    bool fallback_to_direct = true;
    bool verbose = false;
};

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
    } time;

    struct IC
    {
        int type = 0;
        double pool_depth = 0.2;
        double perturbation = 0.0;
        int perturbation_modes = 0;
    } ic;


    // ========================================================================
    // Mesh / AMR
    // ========================================================================
    struct Mesh
    {
        unsigned int initial_refinement = 5;
        bool use_amr = true;
        unsigned int amr_min_level = 4;
        unsigned int amr_max_level = 7;
        unsigned int amr_interval = 5;
        double amr_upper_fraction = 0.3;
        double amr_lower_fraction = 0.1;
    } mesh;

    // ========================================================================
    // Finite element degrees
    // ========================================================================
    struct FE
    {
        unsigned int degree_phase = 2;
        unsigned int degree_velocity = 2;
        unsigned int degree_pressure = 1;
        unsigned int degree_potential = 2;
        unsigned int degree_magnetization = 0;
    } fe;

    // ========================================================================
    // Output
    // ========================================================================
    struct Output
    {
        std::string folder = "../Results";
        unsigned int frequency = 20;
        bool verbose = false;
    } output;

    // ========================================================================
    // Subsystem enables (for debugging)
    // ========================================================================
    bool enable_magnetic = true;
    bool enable_ns = true;
    bool enable_gravity = true;
    bool enable_mms = false;
    bool use_dg_transport = false;

    // MMS
    double mms_t_init = 0.0;

    // Gravity direction (unit vector)
    dealii::Tensor<1, 2> gravity_direction;

    // ========================================================================
    // Solver parameters
    // ========================================================================
    struct Solvers
    {
        LinearSolverParams ch = {
            LinearSolverParams::Type::GMRES,
            LinearSolverParams::Preconditioner::ILU,
            1e-8, 1e-12, 2000, 50, 1.2,
            true, true, false
        };

        LinearSolverParams poisson = {
            LinearSolverParams::Type::CG,
            LinearSolverParams::Preconditioner::SSOR,
            1e-8, 1e-12, 2000, 50, 1.2,
            true, true, false
        };

        LinearSolverParams ns = {
            LinearSolverParams::Type::FGMRES,
            LinearSolverParams::Preconditioner::BlockSchur,
            1e-6, 1e-10, 1500, 100, 1.2,
            true, true, false
        };
    } solvers;

    // ========================================================================
    // Preset configurations (numerics only)
    // ========================================================================
    void setup_rosensweig();
    void setup_hedgehog();

    // ========================================================================
    // Command line parsing
    // ========================================================================
    static Parameters parse_command_line(int argc, char* argv[]);
};

#endif // PARAMETERS_H