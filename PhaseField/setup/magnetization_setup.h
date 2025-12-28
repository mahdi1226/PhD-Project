// ============================================================================
// setup/magnetization_setup.h - Magnetization DG System Setup (PAPER_MATCH v2)
//
// Free functions for magnetization system setup:
//   - Sparsity pattern (DG flux pattern for upwind)
//   - Initialization (L² projection of M⁰ = χ(θ⁰)H⁰)
//
// FIX: Added Parameters to initialize_magnetization_equilibrium() to use
//      params.physics.epsilon and params.physics.chi_0 instead of globals.
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Section 5, Eq. 56: M_h = {M ∈ L²(Ω) | M|_T ∈ [P_{ℓ-1}(T)]^d, ∀T ∈ T_h}
// Section 5.1, Eq. 41: Initial condition M⁰ = I_{M_h}(χ(θ⁰) H⁰)
// ============================================================================
#ifndef MAGNETIZATION_SETUP_H
#define MAGNETIZATION_SETUP_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include "utilities/parameters.h"

/**
 * @brief Set up sparsity pattern for magnetization DG system
 *
 * Creates flux sparsity pattern for DG elements, which includes
 * face coupling needed for upwind fluxes in the transport equation.
 *
 * Note: Mx and My share the same sparsity pattern since they use
 * the same DG space and transport operator structure.
 *
 * @param M_dof_handler    DoFHandler for M (DG, scalar - same for Mx and My)
 * @param M_sparsity       [OUT] Sparsity pattern (flux pattern for DG)
 * @param verbose          Print setup info
 */
template <int dim>
void setup_magnetization_sparsity(
    const dealii::DoFHandler<dim>& M_dof_handler,
    dealii::SparsityPattern& M_sparsity,
    bool verbose = false);

/**
 * @brief Initialize magnetization M⁰ = χ(θ⁰)H⁰ via L² projection
 *
 * Paper Eq. 41: Initial condition for magnetization at quasi-static equilibrium.
 *
 * For DG, this is a cell-local L² projection:
 *   Find M_h such that (M_h, W)_T = (χ(θ⁰) H⁰, W)_T  ∀W ∈ P_{ℓ-1}(T), ∀T
 *
 * Since DG mass matrix is block-diagonal over cells, this reduces to
 * independent cell-wise solves (no global system needed).
 *
 * IMPORTANT: Uses params.physics.epsilon and params.physics.chi_0
 * to compute susceptibility correctly for each preset.
 *
 * @param M_dof_handler     DoFHandler for M (DG, scalar - same for Mx and My)
 * @param theta_dof_handler DoFHandler for θ (CG)
 * @param phi_dof_handler   DoFHandler for φ (CG)
 * @param theta_solution    Initial phase field θ⁰
 * @param phi_solution      Initial magnetic potential φ⁰
 * @param params            Simulation parameters (for physics.epsilon, physics.chi_0)
 * @param Mx_solution       [OUT] Initial Mx⁰
 * @param My_solution       [OUT] Initial My⁰
 */
template <int dim>
void initialize_magnetization_equilibrium(
    const dealii::DoFHandler<dim>& M_dof_handler,
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::Vector<double>& theta_solution,
    const dealii::Vector<double>& phi_solution,
    const Parameters& params,
    dealii::Vector<double>& Mx_solution,
    dealii::Vector<double>& My_solution);

#endif // MAGNETIZATION_SETUP_H