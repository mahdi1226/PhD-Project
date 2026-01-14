// ============================================================================
// assembly/ch_assembler.h - Cahn-Hilliard System Assembler
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Equations 42a-42b (discrete scheme), p.505
//
// Paper's discrete scheme:
//
//   Eq 42a: (δθ^k/τ, Λ) - (U^k θ^{k-1}, ∇Λ) - γ(∇ψ^k, ∇Λ) = 0
//
//   Eq 42b: (ψ^k, Υ) + ε(∇θ^k, ∇Υ) + (1/ε)(f(θ^{k-1}), Υ) + (1/η)(δθ^k, Υ) = 0
//
// where:
//   - δθ^k = θ^k - θ^{k-1}
//   - f(θ) = θ³ - θ  (double-well derivative)
//   - η > 0 is the stabilization parameter (η ~ ε for stability)
//   - θ^{k-1} is LAGGED (explicit), θ^k and ψ^k are implicit
//
// KEY DIFFERENCES FROM NEWTON LINEARIZATION:
//   - Paper uses LAGGED f(θ^{k-1}), NOT linearized f'(θ_old)θ
//   - Advection uses LAGGED θ^{k-1}, NOT implicit θ^k
//   - Stabilization term (1/η)(δθ^k, Υ) is REQUIRED for energy stability
//
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
 * @brief Assemble the coupled Cahn-Hilliard system (Paper Eq. 42a-42b)
 *
 * Assembles into a monolithic matrix with block structure:
 *   [θ-θ  θ-ψ] [θ^k]   [rhs_θ]
 *   [ψ-θ  ψ-ψ] [ψ^k] = [rhs_ψ]
 *
 * Matrix blocks (LHS):
 *   θ-θ: (1/τ)(θ, Λ)
 *   θ-ψ: -γ(∇ψ, ∇Λ)
 *   ψ-θ: ε(∇θ, ∇Υ) + (1/η)(θ, Υ)
 *   ψ-ψ: (ψ, Υ)
 *
 * RHS:
 *   rhs_θ: (1/τ)(θ^{k-1}, Λ) + (U^k θ^{k-1}, ∇Λ)
 *   rhs_ψ: -(1/ε)(f(θ^{k-1}), Υ) + (1/η)(θ^{k-1}, Υ)
 *
 * @param theta_dof_handler  DoFHandler for phase field θ
 * @param psi_dof_handler    DoFHandler for chemical potential ψ
 * @param theta_old          Previous time step θ^{k-1}
 * @param ux_solution        Velocity x-component U^k_x
 * @param uy_solution        Velocity y-component U^k_y
 * @param params             Simulation parameters (includes η = params.ch.eta)
 * @param dt                 Time step size τ
 * @param current_time       Current simulation time (for MMS)
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