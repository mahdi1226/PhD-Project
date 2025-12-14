// ============================================================================
// assembly/poisson_assembler.h - Magnetostatic Poisson Assembly
//
// CORRECTED EQUATION (Paper Eq. 42d, 63):
//
//   Variational form: (∇φ, ∇χ) = (h_a - m, ∇χ)  ∀χ ∈ H¹(Ω)/ℝ
//
//   Strong form:      -Δφ = ∇·(m - h_a)         in Ω
//                     ∂φ/∂n = (h_a - m)·n       on Γ (Neumann BC)
//
// The magnetic field is then: h = ∇φ
//
// NOTE: Pure Neumann problem - solution unique up to constant.
//       We enforce ∫_Ω φ dx = 0 via a constraint.
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
// DEPRECATED: The paper uses simple Laplacian, not μ-weighted.
// Kept for backward compatibility only.
// ============================================================================
inline double compute_permeability(double theta, double epsilon, double kappa_0)
{
    return 1.0 + kappa_0 * sigmoid(theta / epsilon);
}

// ============================================================================
// Compute applied magnetic field h_a at a point (2D) [Eq. 97-98]
//
// h_a = Σ_s α_s ∇φ_s
//
// where φ_s(x) = d·(x_s - x) / |x_s - x|²  (2D dipole potential)
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

    // Sum contributions from all dipoles: h_a = Σ α ∇φ_s
    // φ_s = d·r / |r|²  where r = x_s - x
    // ∇φ_s = d/|r|² - 2(d·r)r/|r|⁴
    for (const auto& dipole_pos : params.dipoles.positions)
    {
        const double rx = dipole_pos[0] - p[0];  // x_s - x
        const double ry = dipole_pos[1] - p[1];
        const double r_sq = rx * rx + ry * ry;

        // Avoid singularity
        if (r_sq < 1e-12)
            continue;

        const double r_sq_sq = r_sq * r_sq;
        const double d_dot_r = d_x * rx + d_y * ry;

        // ∇φ_s = d/|r|² - 2(d·r)r/|r|⁴
        h_a[0] += alpha * (d_x / r_sq - 2.0 * d_dot_r * rx / r_sq_sq);
        h_a[1] += alpha * (d_y / r_sq - 2.0 * d_dot_r * ry / r_sq_sq);
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
 *
 * For pure Neumann, we need to fix the constant. We do this by
 * constraining one DoF (e.g., DoF 0) to zero.
 *
 * @param phi_dof_handler   DoFHandler for magnetic potential φ
 * @param phi_constraints   [OUT] Constraints with mean-zero or pinned DoF
 */
template <int dim>
void setup_poisson_neumann_constraints(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    dealii::AffineConstraints<double>& phi_constraints);

/**
 * @brief Assemble the magnetostatic Poisson system (CORRECTED)
 *
 * Variational form (Eq. 42d):
 *   (∇φ, ∇χ) = (h_a - m, ∇χ)  ∀χ ∈ X_h
 *
 * @param phi_dof_handler   DoFHandler for magnetic potential φ
 * @param theta_dof_handler DoFHandler for phase field θ
 * @param theta_solution    Current phase field solution
 * @param mx_solution       Magnetization x-component (nullptr for simplified)
 * @param my_solution       Magnetization y-component (nullptr for simplified)
 * @param params            Physical parameters
 * @param current_time      Current time (for dipole ramping)
 * @param phi_matrix        [OUT] Assembled system matrix
 * @param phi_rhs           [OUT] Assembled RHS
 * @param phi_constraints   Constraints (hanging nodes + mean-zero)
 * @param use_simplified    If true, ignore magnetization (h := h_a)
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
    bool use_simplified = false);

// ============================================================================
// BACKWARD COMPATIBILITY WRAPPERS
// These allow old code to compile while we transition to the new interface
// ============================================================================

/**
 * @brief [DEPRECATED] Old 7-argument interface
 *
 * Calls new implementation with defaults:
 * - No magnetization vectors (uses quasi-equilibrium m ≈ χ_θ h_a)
 * - current_time = 0 (uses params for ramping)
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
 * @brief Old interface with explicit time parameter (RECOMMENDED)
 *
 * Use this instead of relying on params.current_time.
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
 *
 * The corrected implementation uses Neumann BC, so this function
 * does nothing. Constraints are set up by setup_poisson_neumann_constraints().
 */
template <int dim>
void apply_poisson_dirichlet_bcs(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const Parameters& params,
    double current_time,
    dealii::AffineConstraints<double>& phi_constraints);

#endif // POISSON_ASSEMBLER_H