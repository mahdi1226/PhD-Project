// ============================================================================
// assembly/poisson_assembler.h - Magnetostatic Poisson Assembly
//
// Equation: -∇·(μ(θ)∇φ) = 0 in Ω, φ = φ_dipole on ∂Ω [Eq. 14d, p.499]
//
// where μ(θ) = 1 + κ₀ H(θ/ε) is the phase-dependent permeability [Eq. 17]
// and H(x) = 1/(1 + e^(-x)) is the sigmoid function [Eq. 18]
//
// Physical interpretation:
//   - θ = +1 (ferrofluid): μ ≈ 1 + κ₀ (magnetic)
//   - θ = -1 (water):      μ ≈ 1 (non-magnetic)
//
// Boundary conditions: φ = φ_dipole from external dipole sources [Eq. 96-98]
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
// Phase-dependent magnetic permeability [Eq. 17]
//   μ(θ) = 1 + κ₀ H(θ/ε)
//
// @param theta    Phase field value
// @param epsilon  Interface thickness
// @param kappa_0  Magnetic susceptibility (χ₀)
// ============================================================================
inline double compute_permeability(double theta, double epsilon, double kappa_0)
{
    return 1.0 + kappa_0 * sigmoid(theta / epsilon);
}

// ============================================================================
// Dipole magnetic potential (2D) [Eq. 96-98, p.519]
//
// For a magnetic dipole at (x₀, y₀) with moment m pointing in direction d:
//   φ = α × (d · r) / |r|²
//
// Nochetto's setup (Section 6.2, p.520):
//   - 5 dipoles at x = -0.5, 0, 0.5, 1, 1.5 at y = -1.5
//   - All pointing upward: direction = (0, 1)
//   - Time ramping: α(t) = α_max * min(t / t_ramp, 1)
//
// @param p             Point at which to evaluate
// @param params        Parameters (contains dipole configuration)
// @param current_time  Current time (for ramping)
// ============================================================================
template <int dim>
double compute_dipole_potential(
    const dealii::Point<dim>& p,
    const Parameters& params,
    double current_time)
{
    static_assert(dim == 2, "Dipole potential only implemented for 2D");

    // Time ramping factor
    const double ramp_factor = (params.dipoles.ramp_time > 0.0)
        ? std::min(current_time / params.dipoles.ramp_time, 1.0)
        : 1.0;

    const double intensity = params.dipoles.intensity_max * ramp_factor;

    // Dipole direction
    const double dir_x = params.dipoles.direction[0];
    const double dir_y = params.dipoles.direction[1];

    double phi_total = 0.0;

    // Sum contributions from all dipoles
    for (const auto& dipole_pos : params.dipoles.positions)
    {
        const double rx = dipole_pos[0] - p[0];
        const double ry = dipole_pos[1] - p[1];
        const double r_sq = rx * rx + ry * ry;

        // Avoid singularity at dipole location
        if (r_sq < 1e-12)
            continue;

        // φ = α × (d · r) / |r|² [Eq. 97]
        const double d_dot_r = dir_x * rx + dir_y * ry;
        phi_total += intensity * d_dot_r / r_sq;
    }

    return phi_total;
}

/**
 * @brief Apply Dirichlet BCs for Poisson from dipole field
 *
 * Sets φ = φ_dipole on all boundaries.
 *
 * @param phi_dof_handler   DoFHandler for magnetic potential φ
 * @param params            Parameters (contains dipole configuration)
 * @param current_time      Current simulation time (for ramping)
 * @param phi_constraints   [OUT] Constraints with Dirichlet BCs added
 */
template <int dim>
void apply_poisson_dirichlet_bcs(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const Parameters& params,
    double current_time,
    dealii::AffineConstraints<double>& phi_constraints);

/**
 * @brief Assemble the magnetostatic Poisson system
 *
 * Weak form: (μ(θ)∇φ, ∇ψ) = 0
 * BCs: φ = φ_dipole on ∂Ω (applied via constraints)
 *
 * @param phi_dof_handler   DoFHandler for magnetic potential φ
 * @param theta_dof_handler DoFHandler for phase field θ
 * @param theta_solution    Current phase field solution
 * @param params            Physical parameters
 * @param phi_matrix        [OUT] Assembled system matrix
 * @param phi_rhs           [OUT] Assembled RHS (zero for Poisson)
 * @param phi_constraints   Dirichlet BCs
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

#endif // POISSON_ASSEMBLER_H