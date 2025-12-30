// ============================================================================
// diagnostics/force_diagnostics.h - Force Magnitude Diagnostics
//
// Computes L2 norms of the three force terms in Navier-Stokes:
//   - Capillary force: F_cap = (λ/ε) θ ∇ψ
//   - Kelvin force:    F_mag = (μ₀/2) ∇(χ(θ)|H|²)  [Paper Eq. 36]
//   - Gravity force:   F_grav = ρ(θ) g
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef FORCE_DIAGNOSTICS_H
#define FORCE_DIAGNOSTICS_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include "utilities/parameters.h"
#include "physics/material_properties.h"

#include <cmath>

/**
 * @brief Smooth Heaviside function H(x) = 1/(1 + e^(-x))
 *
 * Used for phase-dependent material properties:
 *   χ(θ) = χ₀ * H(θ/ε)
 *   ρ(θ) = 1 + r * H(θ/ε)
 */
static inline double sigmoid_force(double x)
{
    if (x > 20.0) return 1.0;
    if (x < -20.0) return 0.0;
    return 1.0 / (1.0 + std::exp(-x));
}

/**
 * @brief Force diagnostic data
 */
struct ForceDiagnostics
{
    double F_cap_L2 = 0.0;    // ||F_cap||_L2
    double F_mag_L2 = 0.0;    // ||F_mag||_L2
    double F_grav_L2 = 0.0;   // ||F_grav||_L2

    double F_cap_max = 0.0;   // max|F_cap|
    double F_mag_max = 0.0;   // max|F_mag|
    double F_grav_max = 0.0;  // max|F_grav|
};

/**
 * @brief Compute force magnitudes (L2 norms and max values)
 *
 * Forces computed per Paper Section 3:
 *   - F_cap = (λ/ε) θ ∇ψ                    [Capillary, Eq. 35]
 *   - F_mag = (μ₀/2) χ'(θ) |H|² ∇θ          [Kelvin body force approximation]
 *   - F_grav = ρ(θ) g                       [Gravity/buoyancy]
 *
 * Note: The full Kelvin force is F = (μ₀/2)∇(χ|H|²) = (μ₀/2)[χ'|H|²∇θ + 2χ(H·∇)H]
 * We compute the dominant term χ'|H|²∇θ which drives interface deformation.
 *
 * @param theta_dof_handler  DoFHandler for θ
 * @param psi_dof_handler    DoFHandler for ψ (unused, for interface consistency)
 * @param phi_dof_handler    DoFHandler for φ (magnetic potential)
 * @param theta_solution     Phase field solution
 * @param psi_solution       Chemical potential solution
 * @param phi_solution       Magnetic potential solution
 * @param params             Simulation parameters
 * @return ForceDiagnostics struct with L2 norms and max values
 */
template <int dim>
ForceDiagnostics compute_force_diagnostics(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::Vector<double>& theta_solution,
    const dealii::Vector<double>& psi_solution,
    const dealii::Vector<double>* phi_solution,
    const Parameters& params)
{
    ForceDiagnostics result;

    const double eps = params.physics.epsilon;
    const double lam = params.physics.lambda;
    const double chi0 = params.physics.chi_0;
    const double g_val = params.enable_gravity ? gravity_dimensionless : 0.0;

    const auto& fe = theta_dof_handler.get_fe();
    dealii::QGauss<dim> quadrature(fe.degree + 1);
    const unsigned int n_q_points = quadrature.size();

    dealii::FEValues<dim> fe_values(fe, quadrature,
        dealii::update_values |
        dealii::update_gradients |
        dealii::update_JxW_values);

    std::vector<double> theta_values(n_q_points);
    std::vector<dealii::Tensor<1, dim>> theta_gradients(n_q_points);
    std::vector<dealii::Tensor<1, dim>> psi_gradients(n_q_points);
    std::vector<dealii::Tensor<1, dim>> phi_gradients(n_q_points);

    double F_cap_sq = 0.0;
    double F_mag_sq = 0.0;
    double F_grav_sq = 0.0;

    for (const auto& cell : theta_dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        fe_values.get_function_values(theta_solution, theta_values);
        fe_values.get_function_gradients(theta_solution, theta_gradients);
        fe_values.get_function_gradients(psi_solution, psi_gradients);

        if (phi_solution != nullptr)
            fe_values.get_function_gradients(*phi_solution, phi_gradients);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double theta = theta_values[q];
            const double JxW = fe_values.JxW(q);

            // ================================================================
            // Capillary force: F_cap = (λ/ε) θ ∇ψ
            // ================================================================
            dealii::Tensor<1, dim> F_cap;
            for (unsigned int d = 0; d < dim; ++d)
                F_cap[d] = (lam / eps) * theta * psi_gradients[q][d];

            const double F_cap_mag = F_cap.norm();
            F_cap_sq += F_cap_mag * F_cap_mag * JxW;
            result.F_cap_max = std::max(result.F_cap_max, F_cap_mag);

            // ================================================================
            // Kelvin force: F_mag ≈ (μ₀/2) χ'(θ) |H|² ∇θ
            //
            // Full expression: F = (μ₀/2) ∇(χ|H|²)
            //                    = (μ₀/2) [χ'|H|² ∇θ + 2χ (H·∇)H]
            //
            // The first term (χ'|H|² ∇θ) dominates at the interface where
            // χ' is large. This term drives the Rosensweig instability.
            // ================================================================
            if (phi_solution != nullptr)
            {
                // H = ∇φ (we use H = ∇φ, not -∇φ, per our convention)
                dealii::Tensor<1, dim> H = phi_gradients[q];
                const double H_sq = H.norm_square();

                // χ(θ) = χ₀ * H(θ/ε) where H is sigmoid
                // χ'(θ) = (χ₀/ε) * H'(θ/ε) = (χ₀/ε) * H(1-H)
                const double H_sigmoid = sigmoid_force(theta / eps);
                const double chi_prime = (chi0 / eps) * H_sigmoid * (1.0 - H_sigmoid);

                // Dominant Kelvin force term at interface
                dealii::Tensor<1, dim> F_mag;
                for (unsigned int d = 0; d < dim; ++d)
                    F_mag[d] = 0.5 * mu_0 * chi_prime * H_sq * theta_gradients[q][d];

                const double F_mag_mag = F_mag.norm();
                F_mag_sq += F_mag_mag * F_mag_mag * JxW;
                result.F_mag_max = std::max(result.F_mag_max, F_mag_mag);
            }

            // ================================================================
            // Gravity force: F_grav = ρ(θ) g
            // ρ(θ) = ρ_water + (ρ_ferro - ρ_water) * H(θ/ε)
            //      = 1 + r * H(θ/ε)  [dimensionless, r = density ratio - 1]
            // ================================================================
            if (params.enable_gravity)
            {
                const double H_val = sigmoid_force(theta / eps);
                const double rho = 1.0 + r * H_val;

                dealii::Tensor<1, dim> F_grav;
                F_grav[0] = 0.0;
                F_grav[1] = -rho * g_val;  // g points downward

                const double F_grav_mag = F_grav.norm();
                F_grav_sq += F_grav_mag * F_grav_mag * JxW;
                result.F_grav_max = std::max(result.F_grav_max, F_grav_mag);
            }
        }
    }

    result.F_cap_L2 = std::sqrt(F_cap_sq);
    result.F_mag_L2 = std::sqrt(F_mag_sq);
    result.F_grav_L2 = std::sqrt(F_grav_sq);

    return result;
}

#endif // FORCE_DIAGNOSTICS_H