// ============================================================================
// assembly/poisson_assembler.h - Magnetostatic Poisson assembly
//
// CORRECTED to match Nochetto, Salgado & Tomas (2016):
// "A diffuse interface model for two-phase ferrofluid flows"
//
// Solves: -∇·(μ(θ)∇φ) = 0  in Ω
//         φ = φ_dipole      on ∂Ω
//
// where μ(θ) = 1 + κ₀ H(θ/ε) is the phase-dependent permeability [Eq 17]
//
// Boundary conditions: φ = φ_dipole on ∂Ω (from external dipole sources)
//
// Physical interpretation:
//   - θ = +1 (ferrofluid phase): μ ≈ 1 + κ₀ (magnetic)
//   - θ = -1 (non-magnetic phase): μ ≈ 1
// ============================================================================
#ifndef POISSON_ASSEMBLER_H
#define POISSON_ASSEMBLER_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/affine_constraints.h>

#include "utilities/nsch_parameters.h"

#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Sigmoid function H(x) = 1/(1 + e^(-x))  [Eq 18]
// ============================================================================
inline double sigmoid_poisson(double x)
{
    if (x > 20.0) return 1.0;
    if (x < -20.0) return 0.0;
    return 1.0 / (1.0 + std::exp(-x));
}

// ============================================================================
// Phase-dependent magnetic permeability [Eq 17]
//   μ(θ) = 1 + κ₀ H(θ/ε)
// ============================================================================
inline double compute_permeability_nochetto(double theta, double epsilon, double kappa_0)
{
    return 1.0 + kappa_0 * sigmoid_poisson(theta / epsilon);
}

// ============================================================================
// Dipole magnetic potential (2D)
//
// For a magnetic dipole at (x₀, y₀) with moment m pointing in direction d:
//   φ = (m / 2π) * (d · r) / |r|²
//
// For Nochetto's setup: 5 dipoles at x = -0.5, 0, 0.5, 1, 1.5
// at y = dipole_y, all pointing upward (direction = (0, 1))
//
// With time ramping: m(t) = m_max * min(t / t_ramp, 1)
// ============================================================================
inline double compute_dipole_potential(
    double x, double y,
    const NSCHParameters& params,
    double current_time)
{
    // Time ramping factor
    const double ramp_factor = (params.dipole_ramp_time > 0.0)
        ? std::min(current_time / params.dipole_ramp_time, 1.0)
        : 1.0;

    const double intensity = params.dipole_intensity * ramp_factor;

    // Dipole positions (Nochetto Section 6.2)
    // 5 dipoles at x = -0.5, 0, 0.5, 1, 1.5 at y = dipole_y
    const std::vector<double> dipole_x = {-0.5, 0.0, 0.5, 1.0, 1.5};
    const double dipole_y_pos = params.dipole_y_position;

    // Dipole direction (default: upward (0, 1))
    const double dir_x = params.dipole_dir_x;
    const double dir_y = params.dipole_dir_y;

    double phi_total = 0.0;

    for (const double x0 : dipole_x)
    {
        const double rx = x0 - x;
        const double ry = dipole_y_pos - y;
        const double r_sq = rx * rx + ry * ry;

        // Avoid singularity at dipole location
        if (r_sq < 1e-12)
            continue;

        // Nochetto Eq. (97): φ = α × (d · r) / |r|²
        // Note: No 1/(2π) factor - this matches the paper exactly
        const double d_dot_r = dir_x * rx + dir_y * ry;
        phi_total += intensity * d_dot_r / r_sq;
    }

    return phi_total;
}

/**
 * @brief Set up Dirichlet boundary conditions for Poisson from dipole field
 *
 * Applies φ = φ_dipole on all boundaries. This is the key fix - previously
 * we were solving with φ=0 BCs which gave zero field inside!
 *
 * @param phi_dof_handler   DoFHandler for magnetic potential φ
 * @param params            Physical parameters (dipole configuration)
 * @param current_time      Current simulation time (for ramping)
 * @param phi_constraints   Output: constraints with Dirichlet BCs
 */
template <int dim>
void setup_poisson_dirichlet_bcs(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const NSCHParameters&          params,
    double                         current_time,
    dealii::AffineConstraints<double>& phi_constraints);

/**
 * @brief Assemble the magnetostatic Poisson system using scalar DoFHandlers
 *
 * Weak form: (μ(θ)∇φ, ∇ψ) = 0
 * BCs: φ = φ_dipole on ∂Ω (applied via constraints)
 *
 * @param phi_dof_handler   DoFHandler for magnetic potential φ (Q2)
 * @param c_dof_handler     DoFHandler for concentration (Q2)
 * @param c_solution        Current concentration solution
 * @param params            Physical and numerical parameters
 * @param phi_matrix        Output: assembled system matrix
 * @param phi_rhs           Output: assembled RHS
 * @param phi_constraints   Dirichlet BCs for applied field
 */
template <int dim>
void assemble_poisson_system_scalar(
    const dealii::DoFHandler<dim>&           phi_dof_handler,
    const dealii::DoFHandler<dim>&           c_dof_handler,
    const dealii::Vector<double>&            c_solution,
    const NSCHParameters&                    params,
    dealii::SparseMatrix<double>&            phi_matrix,
    dealii::Vector<double>&                  phi_rhs,
    const dealii::AffineConstraints<double>& phi_constraints);

#endif // POISSON_ASSEMBLER_H