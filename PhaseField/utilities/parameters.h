// ============================================================================
// utilities/parameters.h - Simulation Parameters
//
// All parameters for ferrofluid simulation based on:
// Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
//
// Two test cases:
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
    // Default constructor - initializes non-trivial members
    // ========================================================================
    Parameters()
    {
        gravity.direction[0] = 0.0;
        gravity.direction[1] = -1.0;
    }

    // ========================================================================
    // Current simulation time (updated by time-stepping loop)
    // Used by poisson_assembler for dipole ramping
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
        double layer_height = 0.2;            // Initial ferrofluid pool depth
        unsigned int initial_refinement = 5;  // Global refinement level
        unsigned int initial_cells_x = 10;    // Base mesh cells in x (before refinement)
        unsigned int initial_cells_y = 6;     // Base mesh cells in y (before refinement)
    } domain;

    // ========================================================================
    // Initial condition parameters
    // IC types:
    //   0 = Flat ferrofluid layer (DEFAULT for Rosensweig/Hedgehog)
    //   1 = Perturbed layer (with cosine modes)
    //   2 = Circular droplet (for testing only)
    // ========================================================================
    struct IC
    {
        int type = 0;                     // 0=flat (DEFAULT), 1=perturbed, 2=circle
        double pool_depth = 0.2;          // Interface y-position (absolute, not fraction!)
        double perturbation = 0.0;        // Perturbation amplitude (for type=1)
        int perturbation_modes = 4;       // Number of Fourier modes (for type=1)
    } ic;

    // ========================================================================
    // Method of Manufactured Solutions (MMS) parameters
    // ========================================================================
    struct MMS
    {
        bool enabled = false;
        double t_init = 0.0;              // Initial time for MMS
    } mms;

    // ========================================================================
    // Cahn-Hilliard parameters (Eq. 14a-14b, p.499)
    // ========================================================================
    struct CH
    {
        double epsilon = 0.01;    // Interface thickness ε
        double lambda = 0.05;     // Capillary coefficient λ
        double gamma = 0.0002;    // Mobility γ
        double eta = 0.01;        // Stabilization parameter η ≤ ε
    } ch;

    // ========================================================================
    // Magnetization parameters (Eq. 14c, p.499)
    //
    // Paper Eq. 42d: ∂M/∂t + (u·∇)M = (1/τ_M)(χ(θ)H - M)
    //
    // τ_M = 0: quasi-equilibrium M = χ(θ)H (instantaneous relaxation)
    // τ_M > 0: finite relaxation time (DG transport equation)
    // ========================================================================
    struct Magnetization
    {
        double chi_0 = 0.5;       // Susceptibility χ₀ ≤ 4
        double tau_M = 0.0;       // Relaxation time τ_M (0 = quasi-equilibrium)
        double T_relax = 0.0;     // Alias for tau_M (backward compatibility)
    } magnetization;

    // ========================================================================
    // Navier-Stokes parameters (Eq. 14e-14f, p.500)
    // ========================================================================
    struct NS
    {
        bool enabled = false;     // Enable Navier-Stokes solve
        double nu_water = 1.0;    // ν_w (non-magnetic phase)
        double nu_ferro = 2.0;    // ν_f (ferrofluid phase)
        double mu_0 = 1.0;        // Magnetic permeability μ₀
        double rho = 1.0;         // Density ρ
        double r = 0.1;           // Density ratio for Boussinesq (Eq. 19)
        double density_ratio = 0.1;  // Backward compatibility alias for r
        double grad_div = 0.0;    // Grad-div stabilization γ_gd
    } ns;

    // ========================================================================
    // Gravity parameters (Eq. 19, 103)
    //
    // WARNING: Gravity is NOT part of paper Eq. 42e and breaks the discrete
    // energy law. Enable only for physical simulations, not paper replication.
    // ========================================================================
    struct Gravity
    {
        bool enabled = false;     // DEFAULT OFF for paper compliance
        double magnitude = 30000.0;  // |g| from Eq. 103
        dealii::Tensor<1, 2> direction;  // Default: (0, -1) downward, set in Parameters()
    } gravity;

    // ========================================================================
    // Dipole parameters (Eq. 96-98, p.519)
    // Rosensweig: 5 dipoles at y=-15
    // Hedgehog: 42 dipoles at y=-0.5, -0.75, -1.0
    // ========================================================================
    struct Dipoles
    {
        std::vector<dealii::Point<2>> positions = {
            dealii::Point<2>(-0.5, -15.0),
            dealii::Point<2>( 0.0, -15.0),
            dealii::Point<2>( 0.5, -15.0),
            dealii::Point<2>( 1.0, -15.0),
            dealii::Point<2>( 1.5, -15.0)
        };
        std::vector<double> direction = {0.0, 1.0};  // d = (0, 1)^T upward
        double intensity_max = 6000.0;               // α_max
        double ramp_time = 1.6;                      // Linear ramp [0, ramp_time]
    } dipoles;

    // ========================================================================
    // Magnetic model options
    // ========================================================================
    struct Magnetic
    {
        bool enabled = false;     // Enable magnetic coupling (Poisson + M transport)
        bool use_simplified = false;  // Section 5: h := h_a (skip Poisson)
        bool use_dg_transport = true; // Use DG transport for M (vs quasi-equilibrium)
    } magnetic;

    // ========================================================================
    // Time-stepping parameters
    // ========================================================================
    struct Time
    {
        double dt = 5e-4;             // Time step τ
        double t_final = 2.0;         // Final time t_F
        unsigned int max_steps = 4000;
        double theta = 1.0;           // Time stepping parameter (1.0 = backward Euler)
    } time;

    // ========================================================================
    // Finite element parameters
    // ========================================================================
    struct FE
    {
        unsigned int degree_phase = 2;       // θ, ψ polynomial degree
        unsigned int degree_velocity = 2;    // u polynomial degree
        unsigned int degree_pressure = 1;    // p polynomial degree
        unsigned int degree_potential = 2;   // φ polynomial degree
        unsigned int degree_magnetization = 0;  // M polynomial degree (DG0 default)
    } fe;

    // ========================================================================
    // Mesh parameters
    // ========================================================================
    struct Mesh
    {
        unsigned int initial_refinement = 5;
        bool use_amr = false;         // AMR disabled by default
        unsigned int amr_min_level = 4;
        unsigned int amr_max_level = 7;
        unsigned int amr_interval = 5;       // Refine every N steps
        double amr_upper_fraction = 0.3;
        double amr_lower_fraction = 0.1;
    } mesh;

    // ========================================================================
    // Output parameters
    // ========================================================================
    struct Output
    {
        std::string folder = "../Results";   // Output directory (relative to build folder)
        std::string output_dir = "../Results";   // Alias for folder
        unsigned int frequency = 100;        // Output every N steps
        unsigned int output_interval = 100;  // Alias for frequency
        bool verbose = false;
    } output;

    // ========================================================================
    // Solver parameters
    // ========================================================================
    struct Solver
    {
        unsigned int max_iterations = 1000;
        double tolerance = 1e-10;
        bool use_direct = true;   // Use direct solver (UMFPACK) - default for robustness
    } solver;

    // ========================================================================
    // Static method: Parse command line arguments
    // ========================================================================
    static Parameters parse_command_line(int argc, char* argv[]);

    // ========================================================================
    // Helper: Setup for Rosensweig test case (Section 6.2)
    // ========================================================================
    void setup_rosensweig()
    {
        // Domain
        domain.x_min = 0.0;
        domain.x_max = 1.0;
        domain.y_min = 0.0;
        domain.y_max = 0.6;
        domain.layer_height = 0.2;
        domain.initial_refinement = 5;
        domain.initial_cells_x = 10;  // Base mesh 10×6 (Section 6.2)
        domain.initial_cells_y = 6;

        // IC
        ic.type = 0;  // Flat interface
        ic.pool_depth = 0.2;
        ic.perturbation = 0.0;

        // Cahn-Hilliard
        ch.epsilon = 0.01;
        ch.lambda = 0.05;
        ch.gamma = 0.0002;
        ch.eta = 0.01;

        // Magnetization (quasi-equilibrium: τ_M = 0)
        magnetization.chi_0 = 0.5;
        magnetization.tau_M = 0.0;  // Quasi-equilibrium M = χH

        // Enable magnetic and NS
        magnetic.enabled = true;
        ns.enabled = true;

        // NS
        ns.nu_water = 1.0;
        ns.nu_ferro = 2.0;
        ns.mu_0 = 1.0;
        ns.r = 0.1;
        ns.density_ratio = 0.1;

        // Gravity (paper includes it for physical realism)
        gravity.enabled = true;
        gravity.magnitude = 30000.0;
        gravity.direction[0] = 0.0;
        gravity.direction[1] = -1.0;

        // Dipoles: 5 at y = -15 (far below, nearly uniform field)
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
        time.theta = 1.0;

        // Mesh
        mesh.initial_refinement = 5;
        mesh.amr_interval = 5;
    }

    // ========================================================================
    // Helper: Setup for Hedgehog test case (Section 6.3)
    // ========================================================================
    void setup_hedgehog()
    {
        // Domain: same as Section 6.2
        domain.x_min = 0.0;
        domain.x_max = 1.0;
        domain.y_min = 0.0;
        domain.y_max = 0.6;
        domain.layer_height = 0.11;       // Pool depth for Section 6.3
        domain.initial_refinement = 6;
        domain.initial_cells_x = 15;      // Base mesh 15×9 (Section 6.3)
        domain.initial_cells_y = 9;

        // IC: Flat pool at y = 0.11
        ic.type = 0;  // Flat interface
        ic.pool_depth = 0.11;
        ic.perturbation = 0.0;

        // Cahn-Hilliard (sharper interface than 6.2)
        ch.epsilon = 0.005;
        ch.lambda = 0.025;
        ch.gamma = 0.0002;
        ch.eta = 0.005;

        // Magnetization (quasi-equilibrium: τ_M = 0, higher susceptibility)
        magnetization.chi_0 = 0.9;
        magnetization.tau_M = 0.0;  // Quasi-equilibrium M = χH

        // Enable magnetic and NS
        magnetic.enabled = true;
        ns.enabled = true;

        // NS
        ns.nu_water = 1.0;
        ns.nu_ferro = 2.0;
        ns.mu_0 = 1.0;
        ns.r = 0.1;
        ns.density_ratio = 0.1;

        // Gravity
        gravity.enabled = true;
        gravity.magnitude = 30000.0;
        gravity.direction[0] = 0.0;
        gravity.direction[1] = -1.0;

        // ================================================================
        // Dipoles: 42 dipoles simulating bar magnet (Section 6.3)
        // ================================================================
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
        time.theta = 1.0;

        // Mesh
        mesh.initial_refinement = 6;
        mesh.use_amr = true;
        mesh.amr_interval = 5;

        // Output
        output.frequency = 100;
        output.output_interval = 100;

        // DEFAULT: Full model h = h_a + h_d (solve Poisson)
        magnetic.use_simplified = false;
    }
};

#endif // PARAMETERS_H