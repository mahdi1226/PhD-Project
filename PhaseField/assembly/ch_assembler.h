// ============================================================================
// assembly/ch_assembler.h - Cahn-Hilliard System Assembler
//
// Free function interface - no circular dependencies.
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Equations 14a-14b (continuous), 42a-b (discrete), p.499, 505
//
// Weak form:
//   Eq 14a: (1/τ)(θ - θ_old, φ) - (θu, ∇φ) + γ(∇ψ, ∇φ) = 0
//   Eq 14b: (ψ, χ) + ε(∇θ, ∇χ) + (1/ε)(f'(θ_old)θ, χ) = (1/ε)(f'(θ_old)θ_old - f(θ_old), χ)
//
// where f(θ) = θ³ - θ, f'(θ) = 3θ² - 1
//
// MMS MODE: When params.mms.enabled is true, adds manufactured source terms
// to verify implementation correctness. See diagnostics/ch_mms.h for details.
// ============================================================================
#ifndef CH_ASSEMBLER_H
#define CH_ASSEMBLER_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <vector>

// Forward declaration
struct Parameters;

/**
 * @brief Assemble the coupled Cahn-Hilliard system
 *
 * Assembles into a monolithic matrix with block structure:
 *   [θ-θ  θ-ψ] [θ]   [rhs_θ]
 *   [ψ-θ  ψ-ψ] [ψ] = [rhs_ψ]
 *
 * @param theta_dof_handler  DoFHandler for phase field θ
 * @param psi_dof_handler    DoFHandler for chemical potential ψ
 * @param theta_old          Previous time step θ solution
 * @param ux_solution        Velocity x-component (can be zero for standalone CH)
 * @param uy_solution        Velocity y-component (can be zero for standalone CH)
 * @param params             Simulation parameters
 * @param dt                 Time step size τ
 * @param current_time       Current simulation time (used for MMS source terms)
 * @param theta_to_ch_map    Index mapping: θ DoF → coupled system index
 * @param psi_to_ch_map      Index mapping: ψ DoF → coupled system index
 * @param matrix             Output: assembled system matrix
 * @param rhs                Output: assembled RHS vector
 */
template <int dim>
void assemble_ch_system(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::DoFHandler<dim>& psi_dof_handler,
    const dealii::Vector<double>&  theta_old,
    const dealii::Vector<double>&  ux_solution,
    const dealii::Vector<double>&  uy_solution,
    const Parameters&              params,
    double                         dt,
    double                         current_time,
    const std::vector<dealii::types::global_dof_index>& theta_to_ch_map,
    const std::vector<dealii::types::global_dof_index>& psi_to_ch_map,
    dealii::SparseMatrix<double>&  matrix,
    dealii::Vector<double>&        rhs);

#endif // CH_ASSEMBLER_H