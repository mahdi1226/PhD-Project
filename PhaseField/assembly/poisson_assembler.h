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
// The magnetic field is then: H = ∇φ
//
// NOTE: Pure Neumann problem - solution unique up to constant.
//       We fix the constant by pinning DoF 0 to zero.
//
// IMPORTANT: M is on a DG DoFHandler (FE_DGP), separate from θ (CG).
//            The assembler requires the M DoFHandler to properly extract
//            magnetization values at quadrature points.
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
 * pinning the first unconstrained DoF to zero.
 *
 * NOTE: The Neumann BC ∂φ/∂n = (h_a - m)·n is a NATURAL BC, enforced
 *       implicitly by the weak form - not by constraints. This function
 *       only handles hanging nodes and the nullspace fix.
 *
 * @param phi_dof_handler   DoFHandler for magnetic potential φ
 * @param phi_constraints   [OUT] Constraints with pinned DoF
 */
template <int dim>
void setup_poisson_neumann_constraints(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    dealii::AffineConstraints<double>& phi_constraints);

/**
 * @brief Assemble the magnetostatic Poisson system (FULL MODEL)
 *
 * Paper Eq. 42d:
 *   (∇φ, ∇χ) = (h_a - M^k, ∇χ)  ∀χ ∈ X_h
 *
 * This is the FULL MODEL where M^k comes from the magnetization transport
 * equation. The resulting H^k = ∇φ^k includes the demagnetizing field.
 *
 * IMPORTANT: M is on a DG DoFHandler (FE_DGP), separate from φ and θ.
 *            This function creates appropriate FEValues for M.
 *
 * @param phi_dof_handler   DoFHandler for magnetic potential φ (CG)
 * @param M_dof_handler     DoFHandler for magnetization M (DG)
 * @param mx_solution       Magnetization x-component M_x^k
 * @param my_solution       Magnetization y-component M_y^k
 * @param params            Physical parameters
 * @param current_time      Current time (for dipole ramping)
 * @param phi_matrix        [OUT] Assembled system matrix
 * @param phi_rhs           [OUT] Assembled RHS
 * @param phi_constraints   Constraints (hanging nodes + pinned DoF)
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
 * @brief Assemble the magnetostatic Poisson system (SIMPLIFIED MODEL)
 *
 * Simplified model (Section 5): M = 0, so RHS = (h_a, ∇χ)
 *
 * This is primarily for comparison/debugging. In the true simplified model
 * (Section 5), you wouldn't solve Poisson at all - you'd just set H = h_a.
 * But this function is useful for testing the demagnetizing field effect.
 *
 * @param phi_dof_handler   DoFHandler for magnetic potential φ (CG)
 * @param params            Physical parameters
 * @param current_time      Current time (for dipole ramping)
 * @param phi_matrix        [OUT] Assembled system matrix
 * @param phi_rhs           [OUT] Assembled RHS
 * @param phi_constraints   Constraints (hanging nodes + pinned DoF)
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
// These interfaces are valid for QUASI-EQUILIBRIUM model (M = χH).
// For full DG transport model, use the 9-arg version with explicit M_dof_handler.
// ============================================================================

/**
 * @brief Old interface with optional M vectors
 *
 * For quasi-equilibrium: pass mx_solution = nullptr, my_solution = nullptr
 * The permeability μ(θ) is computed from θ.
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
 *
 * Uses params.current_time internally.
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
 * @brief Assemble Poisson system for QUASI-EQUILIBRIUM model
 *
 * This interface is CORRECT for quasi-equilibrium (τ_M = 0, M = χH).
 * The M term is absorbed into permeability μ(θ) = 1 + χ(θ).
 *
 * Use the 9-arg version with explicit M_dof_handler only for full
 * DG transport model (Eq. 42c), which requires separate M storage.
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
[[deprecated("Use setup_poisson_neumann_constraints instead")]]
void apply_poisson_dirichlet_bcs(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const Parameters& params,
    double current_time,
    dealii::AffineConstraints<double>& phi_constraints);

#endif // POISSON_ASSEMBLER_H