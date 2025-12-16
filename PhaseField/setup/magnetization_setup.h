// ============================================================================
// setup/magnetization_setup.h - Magnetization Initialization (DG)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Section 5, Eq. 56: M_h = {M ∈ L²(Ω) | M|_T ∈ [P_{ℓ-1}(T)]^d, ∀T ∈ T_h}
// Section 5.1, Eq. 41: Initial condition M⁰ = I_{M_h}(χ(θ⁰) H⁰)
//
// ARCHITECTURE:
//   - ONE scalar DoFHandler for DG space (same for Mx and My)
//   - FE_DGP(ℓ-1) where ℓ = velocity degree (typically FE_DGP(1) for Q2 velocity)
//   - Sparsity pattern and system matrix created by MagnetizationAssembler
//   - This file provides ONLY the initialization (L² projection for M⁰)
//
// ============================================================================
#ifndef MAGNETIZATION_SETUP_H
#define MAGNETIZATION_SETUP_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>

/**
 * @brief Initialize magnetization M⁰ = χ(θ⁰) H⁰ via L² projection
 *
 * Paper Eq. 41: Initial condition for magnetization at quasi-static equilibrium.
 *
 * For DG, this is a cell-local L² projection:
 *   Find M_h such that (M_h, W)_T = (χ(θ⁰) H⁰, W)_T  ∀W ∈ P_{ℓ-1}(T), ∀T
 *
 * Since DG mass matrix is block-diagonal over cells, this reduces to
 * independent cell-wise solves (no global system needed).
 *
 * @param M_dof_handler     DoFHandler for M (DG, scalar - same for Mx and My)
 * @param theta_dof_handler DoFHandler for θ (CG)
 * @param phi_dof_handler   DoFHandler for φ (CG)
 * @param theta_solution    Initial phase field θ⁰
 * @param phi_solution      Initial magnetic potential φ⁰
 * @param chi_0             Susceptibility parameter χ₀
 * @param Mx_solution       [OUT] Initial Mx⁰
 * @param My_solution       [OUT] Initial My⁰
 */
template <int dim>
void initialize_magnetization_dg(
    const dealii::DoFHandler<dim>& M_dof_handler,
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::Vector<double>& theta_solution,
    const dealii::Vector<double>& phi_solution,
    double chi_0,
    dealii::Vector<double>& Mx_solution,
    dealii::Vector<double>& My_solution);

/**
 * @brief Compute χ(θ) = χ₀(1+θ)/2 at a point
 *
 * θ ∈ [-1, 1]:
 *   θ = +1 (ferrofluid): χ = χ₀
 *   θ = -1 (non-magnetic): χ = 0
 */
inline double susceptibility(double theta, double chi_0)
{
    return chi_0 * (1.0 + theta) / 2.0;
}

#endif // MAGNETIZATION_SETUP_H