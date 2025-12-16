// ============================================================================
// utilities/parameters.h - Simulation Parameters
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
//
// Test cases:
//   - Rosensweig (Section 6.2): 5 dipoles at y=-15, χ₀=0.5, ε=0.01
//   - Hedgehog (Section 6.3): 42 dipoles at y=-0.5 to -1.0, χ₀=0.9, ε=0.005
// ============================================================================
#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <vector>
#include <string>
#include <cmath>
#include <iostream>

struct Parameters
{
    // ========================================================================
    // Default constructor
    // ========================================================================
    Parameters()
    {
        gravity.direction[0] = 0.0;
        gravity.direction[1] = -1.0;
    }

    // ========================================================================
    // Current simulation time (updated by time-stepping loop)
    // ========================================================================
    double current_time = 0.0;

    // ========================================================================
    // Domain parameters
    // ========================================================================
    struct Domain
    {
        double x_min = 0.0;
        double x_max = 1.0;
        double y_min = 0.0;
        double y_max = 0.6;
        double layer_height = 0.2;
        unsigned int initial_refinement = 5;
        unsigned int initial_cells_x = 10;
        unsigned int initial_cells_y = 6;
    } domain;

    // ========================================================================
    // Initial condition: FLAT LAYER ONLY (type=0)
    // ========================================================================
    struct IC
    {
        int type = 0;                    // ALWAYS 0 for flat layer
        double pool_depth = 0.2;
        double perturbation = 0.0;       // No perturbation
        int perturbation_modes = 0;      // No modes
    } ic;

    // ========================================================================
    // MMS parameters
    // ========================================================================
    struct MMS
    {
        bool enabled = false;
        double t_init = 0.0;
    } mms;

    // ========================================================================
    // Cahn-Hilliard parameters (Eq. 14a-14b)
    // ========================================================================
    struct CH
    {
        double epsilon = 0.01;
        double lambda = 0.05;
        double gamma = 0.0002;
        double eta = 0.01;
    } ch;

    // ========================================================================
    // Magnetization parameters (Eq. 14c)
    // tau_M = 0 means quasi-equilibrium M = χH
    // ========================================================================
    struct Magnetization
    {
        double chi_0 = 0.5;
        double tau_M = 0.0;
        double T_relax = 0.0;  // Alias for tau_M
    } magnetization;

    // ========================================================================
    // Navier-Stokes parameters (Eq. 14e-14f)
    // ========================================================================
    struct NS
    {
        bool enabled = true;
        double nu_water = 1.0;
        double nu_ferro = 2.0;
        double mu_0 = 1.0;
        double rho = 1.0;
        double r = 0.1;
        double grad_div = 0.0;
    } ns;

    // ========================================================================
    // Gravity parameters (Eq. 19, 103)
    // ========================================================================
    struct Gravity
    {
        bool enabled = true;
        double magnitude = 30000.0;
        dealii::Tensor<1, 2> direction;
    } gravity;

    // ========================================================================
    // Dipole parameters (Eq. 96-98)
    // ========================================================================
    struct Dipoles
    {
        std::vector<dealii::Point<2>> positions;
        std::vector<double> direction = {0.0, 1.0};
        double intensity_max = 6000.0;
        double ramp_time = 1.6;
    } dipoles;

    // ========================================================================
    // Magnetic model
    // use_dg_transport = false means quasi-equilibrium M = χH
    // ========================================================================
    struct Magnetic
    {
        bool enabled = true;
        bool use_simplified = false;
        bool use_dg_transport = false;  // false = quasi-equilibrium (paper default)
    } magnetic;

    // ========================================================================
    // Time-stepping
    // ========================================================================
    struct Time
    {
        double dt = 5e-4;
        double t_final = 2.0;
        unsigned int max_steps = 4000;
        double theta = 1.0;
    } time;

    // ========================================================================
    // Finite element
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
    // Mesh
    // ========================================================================
    struct Mesh
    {
        unsigned int initial_refinement = 5;
        bool use_amr = false;
        unsigned int amr_min_level = 4;
        unsigned int amr_max_level = 7;
        unsigned int amr_interval = 5;
        double amr_upper_fraction = 0.3;
        double amr_lower_fraction = 0.1;
    } mesh;

    // ========================================================================
    // Output
    // ========================================================================
    struct Output
    {
        std::string folder = "../Results";
        unsigned int frequency = 100;
        bool verbose = false;
    } output;

    // ========================================================================
    // Solver
    // ========================================================================
    struct Solver
    {
        unsigned int max_iterations = 1000;
        double tolerance = 1e-10;
        bool use_direct = true;
    } solver;

    // ========================================================================
    // Parse command line
    // ========================================================================
    static Parameters parse_command_line(int argc, char* argv[]);

    // ========================================================================
    // Rosensweig test case (Section 6.2)
    // ========================================================================
    void setup_rosensweig()
    {
        // Domain: [0,1] x [0,0.6], base mesh 10×6
        domain.x_min = 0.0;
        domain.x_max = 1.0;
        domain.y_min = 0.0;
        domain.y_max = 0.6;
        domain.layer_height = 0.2;
        domain.initial_refinement = 5;
        domain.initial_cells_x = 10;
        domain.initial_cells_y = 6;

        // Flat pool at y = 0.2
        ic.type = 0;
        ic.pool_depth = 0.2;
        ic.perturbation = 0.0;
        ic.perturbation_modes = 0;

        // Cahn-Hilliard
        ch.epsilon = 0.01;
        ch.lambda = 0.05;
        ch.gamma = 0.0002;
        ch.eta = 0.01;

        // Magnetization (quasi-equilibrium)
        magnetization.chi_0 = 0.5;
        magnetization.tau_M = 0.0;
        magnetization.T_relax = 0.0;

        // NS + Magnetic enabled
        ns.enabled = true;
        magnetic.enabled = true;
        magnetic.use_dg_transport = false;

        // Gravity
        gravity.enabled = true;
        gravity.magnitude = 30000.0;

        // 5 dipoles at y = -15
        dipoles.positions = {
            dealii::Point<2>(-0.5, -15.0),
            dealii::Point<2>( 0.0, -15.0),
            dealii::Point<2>( 0.5, -15.0),
            dealii::Point<2>( 1.0, -15.0),
            dealii::Point<2>( 1.5, -15.0)
        };
        dipoles.direction = {0.0, 1.0};
        dipoles.intensity_max = 6000.0;
        dipoles.ramp_time = 1.6;

        // Time
        time.dt = 5e-4;
        time.t_final = 2.0;
        time.max_steps = 4000;

        // Mesh
        mesh.initial_refinement = 5;
        mesh.use_amr = true;
        mesh.amr_interval = 5;

        // Output
        output.frequency = 100;
    }

    // ========================================================================
    // Hedgehog test case (Section 6.3)
    // ========================================================================
    void setup_hedgehog()
    {
        // Domain: [0,1] x [0,0.6], base mesh 15×9
        domain.x_min = 0.0;
        domain.x_max = 1.0;
        domain.y_min = 0.0;
        domain.y_max = 0.6;
        domain.layer_height = 0.11;
        domain.initial_refinement = 6;
        domain.initial_cells_x = 15;
        domain.initial_cells_y = 9;

        // Flat pool at y = 0.11
        ic.type = 0;
        ic.pool_depth = 0.11;
        ic.perturbation = 0.0;
        ic.perturbation_modes = 0;

        // Cahn-Hilliard (sharper interface)
        ch.epsilon = 0.005;
        ch.lambda = 0.025;
        ch.gamma = 0.0002;
        ch.eta = 0.005;

        // Magnetization (quasi-equilibrium, higher susceptibility)
        magnetization.chi_0 = 0.9;
        magnetization.tau_M = 0.0;
        magnetization.T_relax = 0.0;

        // NS + Magnetic enabled
        ns.enabled = true;
        magnetic.enabled = true;
        magnetic.use_dg_transport = false;

        // Gravity
        gravity.enabled = true;
        gravity.magnitude = 30000.0;

        // 42 dipoles (3 rows × 14 dipoles)
        dipoles.positions.clear();
        const double y_rows[3] = {-0.5, -0.75, -1.0};
        const int n_per_row = 14;
        const double x_start = 0.3;
        const double x_end = 0.7;
        const double dx = (x_end - x_start) / (n_per_row - 1);

        for (int row = 0; row < 3; ++row)
        {
            for (int i = 0; i < n_per_row; ++i)
            {
                double x = x_start + i * dx;
                dipoles.positions.push_back(dealii::Point<2>(x, y_rows[row]));
            }
        }
        dipoles.direction = {0.0, 1.0};
        dipoles.intensity_max = 4.3;
        dipoles.ramp_time = 4.2;

        // Time
        time.dt = 0.00025;
        time.t_final = 6.0;
        time.max_steps = 24000;

        // Mesh
        mesh.initial_refinement = 6;
        mesh.use_amr = true;
        mesh.amr_interval = 5;

        // Output
        output.frequency = 100;
    }
};

#endif // PARAMETERS_H