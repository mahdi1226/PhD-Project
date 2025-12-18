// ============================================================================
// diagnostics/poisson_diagnostics.h - Magnetostatic Poisson Diagnostics
//
// Computes diagnostic quantities for the Poisson subsystem:
//   - φ bounds (min, max)
//   - H field magnitude |∇φ| (L², max)
//   - Magnetic energy: ½∫ μ(θ)|∇φ|² dx
//   - M magnitude (quasi-equilibrium): |χ(θ)∇φ|
//   - Applied field contribution: ∫ h_a · ∇φ dx
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef POISSON_DIAGNOSTICS_H
#define POISSON_DIAGNOSTICS_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>

#include "utilities/parameters.h"

#include <string>

/**
 * @brief Container for Poisson diagnostic quantities
 */
struct PoissonDiagnostics
{
    // Potential bounds
    double phi_min = 0.0;
    double phi_max = 0.0;

    // H field = ∇φ
    double H_L2_norm = 0.0;      // ||∇φ||_{L²}
    double H_max = 0.0;          // max |∇φ|

    // Magnetization (quasi-equilibrium: M = χ(θ)∇φ)
    double M_L2_norm = 0.0;      // ||M||_{L²}
    double M_max = 0.0;          // max |M|

    // Applied field
    double h_a_max = 0.0;        // max |h_a|
    double h_a_dot_grad_phi = 0.0;  // ∫ h_a · ∇φ dx

    // Magnetic energy: ½∫ μ(θ)|∇φ|² dx
    double magnetic_energy = 0.0;

    // Permeability range
    double mu_min = 1.0;
    double mu_max = 1.0;

    /**
     * @brief Print diagnostics to console (single line)
     */
    void print(unsigned int step, double time) const;

    /**
     * @brief Get header string for output
     */
    static std::string header();

    /**
     * @brief Get CSV-formatted data string
     */
    std::string to_csv(unsigned int step, double time) const;
};

/**
 * @brief Compute Poisson diagnostics for quasi-equilibrium model
 *
 * For quasi-equilibrium: M = χ(θ)∇φ, μ(θ) = 1 + χ(θ)
 *
 * @param phi_dof_handler   DoFHandler for potential φ
 * @param phi_solution      Potential solution vector
 * @param theta_dof_handler DoFHandler for phase field θ
 * @param theta_solution    Phase field solution vector
 * @param params            Simulation parameters
 * @param current_time      Current simulation time
 * @return PoissonDiagnostics struct with computed values
 */
template <int dim>
PoissonDiagnostics compute_poisson_diagnostics(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::Vector<double>& phi_solution,
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::Vector<double>& theta_solution,
    const Parameters& params,
    double current_time);

/**
 * @brief Compute Poisson diagnostics for simplified model (M = 0)
 *
 * For simplified model: M = 0, μ = 1
 */
template <int dim>
PoissonDiagnostics compute_poisson_diagnostics_simplified(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::Vector<double>& phi_solution,
    const Parameters& params,
    double current_time);

#endif // POISSON_DIAGNOSTICS_H