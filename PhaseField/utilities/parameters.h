// ============================================================================
// utilities/parameters.h - Simulation Parameters
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Section 6.2, p.520-522
// ============================================================================
#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <string>
#include <vector>

/**
 * @brief All simulation parameters organized by subsystem
 */
struct Parameters
{
    // ========================================================================
    // Domain and mesh parameters
    // ========================================================================
    struct Domain
    {
        double x_min = 0.0;
        double x_max = 1.0;
        double y_min = 0.0;
        double y_max = 0.6;
        unsigned int initial_refinement = 5;
    } domain;

    // ========================================================================
    // Finite element parameters
    // ========================================================================
    struct FE
    {
        unsigned int degree_velocity = 2;
        unsigned int degree_pressure = 1;
        unsigned int degree_phase = 2;
        unsigned int degree_potential = 2;
        unsigned int degree_magnetization = 2;
    } fe;

    // ========================================================================
    // Time stepping parameters
    // ========================================================================
    struct Time
    {
        double dt = 5e-4;
        double t_final = 2.0;
        double theta = 1.0;
        bool adaptive = false;
        double dt_min = 1e-8;
        double dt_max = 1e-3;
    } time;

    // ========================================================================
    // Cahn-Hilliard parameters (Eq. 14a-14b, p.499)
    // ========================================================================
    struct CH
    {
        double epsilon = 0.01; // Interface thickness
        double gamma = 0.0002; // Mobility
        double lambda = 0.05; // Capillary coefficient
        double eta = 0.005; // Stabilization (eta <= epsilon)
    } ch;

    // ========================================================================
    // Magnetization parameters (Eq. 14c, p.499)
    // ========================================================================
    struct Magnetization
    {
        double chi_0 = 0.5; // Susceptibility (chi_0 <= 4)
        double T_relax = 0.0; // Relaxation time
    } magnetization;

    // ========================================================================
    // Navier-Stokes parameters (Eq. 14e-14f, p.500)
    // ========================================================================
    struct NS
    {
        bool enabled = false; // Enable Navier-Stokes solve
        double nu_water = 1.0;
        double nu_ferro = 2.0;
        double mu_0 = 1.0;
        double rho = 1.0;
        double r = 0.1;
        double grad_div = 0.0;
    } ns;

    // ========================================================================
    // Dipole parameters (Eq. 96-98, p.519)
    // ========================================================================
    struct Dipoles
    {
        std::vector<dealii::Point<2>> positions = {
            dealii::Point<2>(-0.5, -15),
            dealii::Point<2>(0.0, -15),
            dealii::Point<2>(0.5, -1.5),
            dealii::Point<2>(1.0, -15),
            dealii::Point<2>(1.5, -15)
        };
        dealii::Tensor<1, 2> direction = dealii::Tensor<1, 2>({0.0, 1.0});
        double intensity_max = 6000.0;
        double ramp_time = 1.6;
    } dipoles;

    // ========================================================================
    // Gravity parameters
    // ========================================================================
    struct Gravity
    {
        bool enabled = true;
        double magnitude = 30000.0;
        dealii::Tensor<1, 2> direction = dealii::Tensor<1, 2>({0.0, -1.0});
    } gravity;

    // ========================================================================
    // Magnetic field parameters (Poisson solve enable)
    // ========================================================================
    struct Magnetic
    {
        bool enabled = false; // Enable magnetostatic Poisson solve
    } magnetic;

    // ========================================================================
    // AMR parameters
    // ========================================================================
    struct AMR
    {
        bool enabled = true;
        unsigned int min_level = 4;
        unsigned int max_level = 7;
        unsigned int interval = 5;
        double refine_fraction = 0.3;
        double coarsen_fraction = 0.0;
        int indicator_type = 0;
    } amr;

    // ========================================================================
    // Initial condition parameters
    // ========================================================================
    struct IC
    {
        int type = 1; // 0=droplet, 1=flat, 2=perturbed
        double pool_depth = 0.2;
        double perturbation = 0.01;
        int perturbation_modes = 5;
    } ic;

    // ========================================================================
    // Coupling parameters
    // ========================================================================
    struct Coupling
    {
        bool use_picard = false;
        unsigned int max_iterations = 5;
        double tolerance = 1e-6;
    } coupling;

    // ========================================================================
    // Output parameters
    // ========================================================================
    struct Output
    {
        std::string folder = "../Results";
        unsigned int frequency = 100;
        bool verbose = true;
    } output;

    // ========================================================================
    // MMS parameters (Method of Manufactured Solutions)
    // ========================================================================
    struct MMS
    {
        bool enabled = false;
        double t_init = 0.1; // Initial time for MMS (avoid t=0)
        double alpha = 1.0; // Reserved for coupled MMS
        double beta = 1.0; // Reserved for coupled MMS
        double delta = 1.0; // Reserved for coupled MMS
    } mms;

    // ========================================================================
    // Runtime state
    // ========================================================================
    mutable double current_time = 0.0;

    // ========================================================================
    // Parameter validation
    // ========================================================================
    bool validate() const
    {
        bool valid = true;
        if (magnetization.chi_0 > 4.0) valid = false;
        if (ch.eta > ch.epsilon) valid = false;
        return valid;
    }

    // ========================================================================
    // Parse command line arguments
    // ========================================================================
    static Parameters parse_command_line(int argc, char* argv[]);
};

#endif // PARAMETERS_H
