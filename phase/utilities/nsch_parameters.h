// ============================================================================
// utilities/nsch_parameters.h - Parameters for ferrofluid NS-CH solver
//
// Based on Nochetto, Salgado & Tomas (2016):
// "A diffuse interface model for two-phase ferrofluid flows"
// Comput. Methods Appl. Mech. Engrg. 309 (2016) 497-531
//
// Default parameters from Section 6.2:
//   Domain: (0, 1) x (0, 0.6)
//   nu_w = 1.0, nu_f = 2.0
//   mu_0 = 1, kappa_0 = 0.5
//   gamma = 0.0002, lambda = 0.05, epsilon = 0.01
//   r = 0.1, g = 30000
//   Dipole intensity: 0 -> 6000 over t in [0, 1.6]
// ============================================================================
#ifndef NSCH_PARAMETERS_H
#define NSCH_PARAMETERS_H

#include <deal.II/base/parameter_handler.h>
#include <string>

struct NSCHParameters
{
    // ========================================================================
    // Navier-Stokes: Phase-dependent viscosity [Eq 17]
    //   nu_theta = nu_w + (nu_f - nu_w) H(theta/epsilon)
    // ========================================================================
    double nu_water  = 1.0;    // nu_w: water/air viscosity [Nochetto: 1.0]
    double nu_ferro  = 2.0;    // nu_f: ferrofluid viscosity [Nochetto: 2.0]
    double viscosity = 1.0;    // Constant viscosity (MMS mode only)

    // ========================================================================
    // Cahn-Hilliard [Eq 14a-14b]
    //   theta_t + div(u*theta) + gamma * Laplacian(psi) = 0
    //   psi - epsilon * Laplacian(theta) + (1/epsilon) f(theta) = 0
    // ========================================================================
    double epsilon  = 0.01;    // epsilon: interface thickness [Nochetto: 0.01]
    double lambda   = 0.05;    // lambda: capillary coefficient [Nochetto: 0.05]
    double mobility = 0.0002;  // gamma: mobility [Nochetto: 0.0002]

    // ========================================================================
    // Magnetostatics [Eq 14c-14d]
    // ========================================================================
    bool   enable_magnetic = false;
    double chi_m = 0.5;        // kappa_0: susceptibility [Nochetto: 0.5]
    double mu_0  = 1.0;        // mu_0: permeability [Nochetto: 1.0]

    // Dipole field (Section 6.2)
    double dipole_intensity  = 6000.0;  // Max intensity [Nochetto: 6000]
    double dipole_ramp_time  = 1.6;     // Ramp time [Nochetto: 1.6]
    double dipole_y_position = -15.0;   // Dipole y-position [Nochetto: -15]

    // Dipole direction (default: upward for Rosensweig)
    double dipole_dir_x = 0.0;
    double dipole_dir_y = 1.0;

    // Uniform vertical magnetic field (simpler Rosensweig model)
    double uniform_magnetic_field = 0.0;  // B_0: uniform vertical field strength
    bool   use_uniform_field = false;     // Use uniform field instead of dipoles

    // ========================================================================
    // Gravity (Boussinesq) [Eq 19]
    //   f_g = (1 + r H(theta/epsilon)) g
    // ========================================================================
    bool   enable_gravity = true;       // Essential for Rosensweig
    double gravity = 30000.0;           // g magnitude [Nochetto: ~30000]
    double gravity_angle = -90.0;       // -90 deg = downward
    double density_ratio = 0.1;         // r [Nochetto: 0.1]

    // ========================================================================
    // Numerical stabilization
    // ========================================================================
    double grad_div_gamma = 0.0;        // Grad-div stabilization

    // ========================================================================
    // Domain [Section 6.2]
    // ========================================================================
    double x_min = 0.0;
    double x_max = 1.0;   // Width = 1
    double y_min = 0.0;
    double y_max = 0.6;   // Height = 0.6 [Nochetto]

    // ========================================================================
    // Time stepping
    // ========================================================================
    double dt      = 5.0e-4;   // tau [Nochetto: t_F/4000 ~ 0.0005]
    double t_final = 2.0;      // t_F [Nochetto: 2.0]
    double theta   = 1.0;      // Time discretization (1.0 = Backward Euler)
    bool   use_adaptive_dt = false;  // Adaptive time stepping

    // ========================================================================
    // Coupling (Picard iteration)
    // ========================================================================
    bool         use_picard      = false;
    unsigned int picard_max_iter = 5;
    double       picard_tol      = 1.0e-6;

    // ========================================================================
    // Finite elements [Section 6]
    // ========================================================================
    unsigned int fe_degree_velocity  = 2;  // Q2 velocity (Taylor-Hood)
    unsigned int fe_degree_pressure  = 1;  // Q1 pressure (Taylor-Hood)
    unsigned int fe_degree_phase     = 2;  // Q2 phase field [Nochetto: l=2]
    unsigned int fe_degree_potential = 2;  // Q2 magnetic potential
    unsigned int n_refinements       = 4;  // Mesh refinement level

    // ========================================================================
    // Initial condition
    // ========================================================================
    int ic_type = 1;  // 0=droplet, 1=flat layer, 2=perturbed layer
    double rosensweig_layer_height      = 0.2;   // Pool depth [Nochetto: 0.2]
    double rosensweig_perturbation      = 0.01;  // Perturbation amplitude
    int    rosensweig_perturbation_modes = 5;    // Number of cosine modes

    // ========================================================================
    // Output
    // ========================================================================
    unsigned int output_interval = 10;
    std::string  output_dir      = "results";
    bool         verbose         = true;

    // ========================================================================
    // Adaptive Mesh Refinement (AMR)
    // ========================================================================
    bool         use_amr              = false;
    unsigned int amr_interval         = 5;
    unsigned int amr_min_level        = 3;
    unsigned int amr_max_level        = 7;
    double       amr_refine_fraction  = 0.3;
    double       amr_coarsen_fraction = 0.1;
    int          amr_indicator_type   = 0;  // 0=gradient, 1=Kelly, 2=interface

    // ========================================================================
    // MMS Verification
    // ========================================================================
    bool   mms_mode  = false;
    double mms_alpha = 1.0;
    double mms_beta  = 1.0;
    double mms_delta = 1.0;

    // ========================================================================
    // Convergence study
    // ========================================================================
    bool         eoc_mode       = false;
    unsigned int min_refinement = 3;
    unsigned int max_refinement = 6;

    // ========================================================================
    // Runtime state (for passing to assemblers)
    // ========================================================================
    mutable double current_time = 0.0;

    // ========================================================================
    // Convenience accessors (aliases for consistency)
    // ========================================================================
    double dipole_y() const { return dipole_y_position; }

    // ========================================================================
    // Parameter file I/O
    // ========================================================================
    static void declare(dealii::ParameterHandler& prm);
    void parse(dealii::ParameterHandler& prm);
};

NSCHParameters parse_command_line(int argc, char* argv[]);

#endif // NSCH_PARAMETERS_H