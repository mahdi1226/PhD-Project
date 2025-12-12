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
 *
 * Parameters from Section 6.2, p.520-522 with page references
 */
struct Parameters
{
    // ========================================================================
    // Domain and mesh parameters
    // ========================================================================
    struct Domain
    {
        double x_min = 0.0;           // p.520: Ω = [0,1] × [0,0.6]
        double x_max = 1.0;           // p.520
        double y_min = 0.0;           // p.520
        double y_max = 0.6;           // p.520
        unsigned int initial_refinement = 5;  // p.522: 6 levels for main results
    } domain;
    
    // ========================================================================
    // Time stepping parameters
    // ========================================================================
    struct Time
    {
        double dt = 5e-4;             // p.522: τ = 5×10⁻⁴ (K=4000 steps)
        double t_final = 2.0;         // p.522: t_F = 2.0
    } time;
    
    // ========================================================================
    // Cahn-Hilliard parameters
    // ========================================================================
    struct CH
    {
        double epsilon = 0.01;        // p.522: ε = 0.01 (interface thickness)
        double gamma = 0.0002;        // p.522: γ = 0.0002 (mobility)
        double lambda = 0.05;         // p.522: λ = 0.05 (capillary coefficient)

        // ASSUMPTION: η = 0.5ε (stabilization parameter)
        // BASIS: Paper states η ≤ ε (Theorem 4.1, p.505; Proposition 5.1)
        //        but does not specify exact value. We choose η = 0.5ε as conservative.
        // QUESTION: What value of η was used in the paper's numerical experiments?
        double eta = 0.005;           // η = 0.5 * ε = 0.5 * 0.01 = 0.005
    } ch;

    // ========================================================================
    // Magnetization parameters
    // ========================================================================
    struct Magnetization
    {
        double chi_0 = 0.5;           // p.520: χ₀ = 0.5 (susceptibility)
                                      // Constraint: χ₀ ≤ 4 (Proposition 3.1, p.502)
        double T_relax = 0.0;         // Relaxation time T (not specified in paper)
                                      // Range: 10⁻⁵–10⁻⁹ s (p.500)
                                      // T = 0 means quasi-equilibrium m = χ_θ h
    } magnetization;

    // ========================================================================
    // Navier-Stokes parameters
    // ========================================================================
    struct NS
    {
        double nu_water = 1.0;        // p.520: ν_w = 1.0 (water viscosity)
        double nu_ferro = 2.0;        // p.520: ν_f = 2.0 (ferrofluid viscosity)
        double mu_0 = 1.0;            // p.520: μ₀ = 1 (magnetic permeability)
        double rho = 1.0;             // p.520: ρ = 1 (unitary density)
        double r = 0.1;               // p.520: r = 0.1 (density ratio for gravity)
        double grad_div = 1.0;        // Grad-div stabilization (not in paper)
    } ns;

    // ========================================================================
    // Dipole parameters for applied field h_a
    // ========================================================================
    struct Dipoles
    {
        // Positions (p.522): (-0.5,-1.5), (0,-1.5), (0.5,-1.5), (1,-1.5), (1.5,-1.5)
        // NOTE: y = -1.5, NOT -15
        std::vector<dealii::Point<2>> positions = {
            dealii::Point<2>(-0.5, -1.5),
            dealii::Point<2>(0.0, -1.5),
            dealii::Point<2>(0.5, -1.5),
            dealii::Point<2>(1.0, -1.5),
            dealii::Point<2>(1.5, -1.5)
        };

        dealii::Tensor<1, 2> direction = dealii::Tensor<1, 2>({0.0, 1.0});  // d = (0,1)^T upward

        double intensity_max = 6000.0;  // p.522: α_s max = 6000
        double ramp_time = 1.6;         // p.522: ramp over t ∈ [0, 1.6]
    } dipoles;

    // ========================================================================
    // Gravity parameters (optional, Eq. 19)
    // ========================================================================
    struct Gravity
    {
        bool enabled = true;          // Gravity is optional supplement (p.501)
        double magnitude = 30000.0;   // p.522, Eq.103: g ≈ 3×10⁴
        dealii::Tensor<1, 2> direction = dealii::Tensor<1, 2>({0.0, -1.0});  // downward
    } gravity;

    // ========================================================================
    // AMR parameters
    // ========================================================================
    struct AMR
    {
        bool enabled = true;          // p.522: mesh refined-coarsened every 5 steps
        unsigned int min_level = 4;   // Minimum refinement level
        unsigned int max_level = 7;   // Maximum refinement level (Fig. 3)
        unsigned int interval = 5;    // p.522: "once every 5 time steps"
    } amr;

    // ========================================================================
    // Initial condition parameters
    // ========================================================================
    struct IC
    {
        double pool_depth = 0.2;      // p.522: "ferrofluid pool of 0.2 units of depth"
    } ic;

    // ========================================================================
    // Output parameters
    // ========================================================================
    struct Output
    {
        std::string folder = "../Results";  // Output in project root, not build folder
        unsigned int frequency = 100;    // Output every N steps
    } output;

    // ========================================================================
    // Parse command line arguments
    // ========================================================================
    static Parameters parse_command_line(int argc, char* argv[]);
};

#endif // PARAMETERS_H