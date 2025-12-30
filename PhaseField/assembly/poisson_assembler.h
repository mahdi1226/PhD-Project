// ============================================================================
// assembly/poisson_assembler.h - Magnetostatic Poisson Assembly
//
// PAPER EQUATION 42d:
//
//   Variational form: (∇φ, ∇χ) = (h_a - M^k, ∇χ)  ∀χ ∈ X_h
//
//   Strong form:      -Δφ = ∇·(M - h_a)         in Ω
//                     ∂φ/∂n = (h_a - M)·n       on Γ (Neumann BC)
//
// For QUASI-EQUILIBRIUM (M = χH = χ∇φ):
//   ((1 + χ(θ))∇φ, ∇χ) = (h_a, ∇χ)
//   (μ(θ)∇φ, ∇χ) = (h_a, ∇χ)  where μ(θ) = 1 + χ(θ)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef POISSON_ASSEMBLER_H
#define POISSON_ASSEMBLER_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include "utilities/parameters.h"

#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Sigmoid function H(x) = 1/(1 + e^(-x)) [Eq. 18]
// ============================================================================
inline double sigmoid(double x)
{
    if (x > 20.0) return 1.0;
    if (x < -20.0) return 0.0;
    return 1.0 / (1.0 + std::exp(-x));
}

// ============================================================================
// Phase-dependent magnetic susceptibility χ_θ = χ₀ H(θ/ε) [Eq. 17]
// ============================================================================
inline double compute_susceptibility(double theta, double epsilon, double chi_0)
{
    return chi_0 * sigmoid(theta / epsilon);
}

// ============================================================================
// Phase-dependent magnetic permeability μ(θ) = 1 + χ_θ [Eq. 17]
// ============================================================================
inline double compute_permeability(double theta, double epsilon, double chi_0)
{
    return 1.0 + compute_susceptibility(theta, epsilon, chi_0);
}

// ============================================================================
// Compute applied magnetic field h_a at a point (2D) [Eq. 97-98]
//
// h_a = Σ_s α_s ∇_x φ_s
//
// where φ_s(x) = d·(x_s - x) / |x_s - x|²  (2D dipole potential)
//
// Let r = x_s - x. Then φ = d·r / |r|²
// Taking gradient with respect to evaluation point x:
//   ∇_x φ = ∇_x [d·r / |r|²]
//         = d · ∇_x(r)/|r|² + (d·r) · ∇_x(1/|r|²)
//         = d · (-I)/|r|² + (d·r) · (-2r/|r|⁴)
//         = -d/|r|² + 2(d·r)r/|r|⁴
//
// SIGN FIX: Original code had opposite sign!
// ============================================================================
template <int dim>
dealii::Tensor<1, dim> compute_applied_field(
    const dealii::Point<dim>& p,
    const Parameters& params,
    double current_time)
{
    static_assert(dim == 2, "Applied field only implemented for 2D");

    // Time ramping factor
    const double ramp_factor = (params.dipoles.ramp_time > 0.0)
        ? std::min(current_time / params.dipoles.ramp_time, 1.0)
        : 1.0;

    const double alpha = params.dipoles.intensity_max * ramp_factor;

    // Dipole direction (unit vector)
    const double d_x = params.dipoles.direction[0];
    const double d_y = params.dipoles.direction[1];

    dealii::Tensor<1, dim> h_a;
    h_a[0] = 0.0;
    h_a[1] = 0.0;

    // Sum contributions from all dipoles: h_a = Σ α ∇_x φ_s
    // φ_s = d·r / |r|²  where r = x_s - x
    // ∇_x φ_s = -d/|r|² + 2(d·r)r/|r|⁴
    // Softening parameter: larger = more uniform field spread
    // Try values: 0.01 (minimal), 0.04 (moderate), 0.1 (strong smoothing)
    const double delta_sq = 0.04;  // Add to params.dipoles for easy tuning

    for (const auto& dipole_pos : params.dipoles.positions)
    {
        const double rx = dipole_pos[0] - p[0];  // r = x_s - x
        const double ry = dipole_pos[1] - p[1];
        const double r_sq = rx * rx + ry * ry;

        // Regularized distance (softening prevents sharp 1/r² decay)
        const double r_eff_sq = r_sq + delta_sq;

        // Avoid singularity (now less critical due to softening)
        if (r_eff_sq < 1e-12)
            continue;

        const double r_eff_sq_sq = r_eff_sq * r_eff_sq;
        const double d_dot_r = d_x * rx + d_y * ry;

        // ∇_x φ_s = -d/|r_eff|² + 2(d·r)r/|r_eff|⁴
        h_a[0] += alpha * (-d_x / r_eff_sq + 2.0 * d_dot_r * rx / r_eff_sq_sq);
        h_a[1] += alpha * (-d_y / r_eff_sq + 2.0 * d_dot_r * ry / r_eff_sq_sq);
    }

    return h_a;
}

// ============================================================================
// Compute dipole potential φ_dipole at a point [Eq. 97]
// ============================================================================
template <int dim>
double compute_dipole_potential(
    const dealii::Point<dim>& p,
    const Parameters& params,
    double current_time)
{
    static_assert(dim == 2, "Dipole potential only implemented for 2D");

    const double ramp_factor = (params.dipoles.ramp_time > 0.0)
        ? std::min(current_time / params.dipoles.ramp_time, 1.0)
        : 1.0;

    const double intensity = params.dipoles.intensity_max * ramp_factor;

    const double dir_x = params.dipoles.direction[0];
    const double dir_y = params.dipoles.direction[1];

    double phi_total = 0.0;

    for (const auto& dipole_pos : params.dipoles.positions)
    {
        const double rx = dipole_pos[0] - p[0];
        const double ry = dipole_pos[1] - p[1];
        const double r_sq = rx * rx + ry * ry;

        if (r_sq < 1e-12)
            continue;

        const double d_dot_r = dir_x * rx + dir_y * ry;
        phi_total += intensity * d_dot_r / r_sq;
    }

    return phi_total;
}

/**
 * @brief Setup constraints for pure Neumann Poisson problem
 */
template <int dim>
void setup_poisson_neumann_constraints(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    dealii::AffineConstraints<double>& phi_constraints);

/**
 * @brief Assemble the magnetostatic Poisson system (FULL MODEL with DG M)
 *
 * Paper Eq. 42d:
 *   (∇φ, ∇χ) = (h_a - M^k, ∇χ)  ∀χ ∈ X_h
 */
template <int dim>
void assemble_poisson_system(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::DoFHandler<dim>& M_dof_handler,
    const dealii::Vector<double>& mx_solution,
    const dealii::Vector<double>& my_solution,
    const Parameters& params,
    double current_time,
    dealii::SparseMatrix<double>& phi_matrix,
    dealii::Vector<double>& phi_rhs,
    const dealii::AffineConstraints<double>& phi_constraints);

/**
 * @brief Assemble Poisson for QUASI-EQUILIBRIUM (M = χ(θ)H)
 *
 * For quasi-equilibrium, M = χ(θ)∇φ. The equation becomes:
 *   (μ(θ)∇φ, ∇χ) = (h_a, ∇χ)  where μ(θ) = 1 + χ(θ)
 *
 * This is the CORRECT formulation for ferrofluid without DG transport.
 */
template <int dim>
void assemble_poisson_system_quasi_equilibrium(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::Vector<double>& theta_solution,
    const Parameters& params,
    double current_time,
    dealii::SparseMatrix<double>& phi_matrix,
    dealii::Vector<double>& phi_rhs,
    const dealii::AffineConstraints<double>& phi_constraints);

/**
 * @brief Assemble the magnetostatic Poisson system (SIMPLIFIED MODEL)
 *
 * Simplified model (Section 5): M = 0, μ = 1
 *   (∇φ, ∇χ) = (h_a, ∇χ)
 *
 * Use this only for comparison/debugging, not for physical simulations!
 */
template <int dim>
void assemble_poisson_system_simplified(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const Parameters& params,
    double current_time,
    dealii::SparseMatrix<double>& phi_matrix,
    dealii::Vector<double>& phi_rhs,
    const dealii::AffineConstraints<double>& phi_constraints);

// ============================================================================
// BACKWARD COMPATIBILITY WRAPPERS
// ============================================================================

/**
 * @brief Old interface with optional M vectors
 */
template <int dim>
void assemble_poisson_system(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::Vector<double>& theta_solution,
    const dealii::Vector<double>* mx_solution,
    const dealii::Vector<double>* my_solution,
    const Parameters& params,
    double current_time,
    dealii::SparseMatrix<double>& phi_matrix,
    dealii::Vector<double>& phi_rhs,
    const dealii::AffineConstraints<double>& phi_constraints,
    bool use_simplified);

/**
 * @brief 7-argument interface (no time parameter)
 */
template <int dim>
void assemble_poisson_system(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::Vector<double>& theta_solution,
    const Parameters& params,
    dealii::SparseMatrix<double>& phi_matrix,
    dealii::Vector<double>& phi_rhs,
    const dealii::AffineConstraints<double>& phi_constraints);

/**
 * @brief 8-argument interface with explicit time (FIXED to use quasi-equilibrium)
 */
template <int dim>
void assemble_poisson_system(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::Vector<double>& theta_solution,
    const Parameters& params,
    double current_time,
    dealii::SparseMatrix<double>& phi_matrix,
    dealii::Vector<double>& phi_rhs,
    const dealii::AffineConstraints<double>& phi_constraints);

/**
 * @brief [DEPRECATED] Dirichlet BC setup - now a no-op
 */
template <int dim>
[[deprecated("Use setup_poisson_neumann_constraints instead")]]
void apply_poisson_dirichlet_bcs(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const Parameters& params,
    double current_time,
    dealii::AffineConstraints<double>& phi_constraints);

#endif // POISSON_ASSEMBLER_H