// ============================================================================
// assembly/ns_assembler.h - Navier-Stokes Assembly (CORRECTED)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Equation 42e (discrete scheme), p.505
//
// Paper's discrete NS scheme:
//
//   (δU^k/τ, V) + B_h(U^{k-1}, U^k, V) + 2(ν(θ^{k-1})D(U^k), D(V))
//     - (P^k, div V) + (div U^k, Q) = (F^{k-1}, V)
//
// where:
//   - δU^k = U^k - U^{k-1}
//   - D(U) = (∇U + ∇U^T)/2  (symmetric strain rate)
//   - B_h(w,u,v) = skew convection form (Eq. 37)
//   - ν(θ^{k-1}) = viscosity at LAGGED θ
//   - F^{k-1} = F_cap + F_mag + F_grav with LAGGED fields
//
// Forces (all using lagged θ^{k-1}):
//   - Capillary: F_cap = (λ/ε)θ^{k-1}∇ψ^k        [Eq. 10, lagged θ]
//   - Kelvin:    F_mag = μ₀ B_h^m(V, H^k, M^k)   [Eq. 38, FULL model]
//   - Gravity:   F_grav = (1 + r·H(θ^{k-1}/ε))g  [Eq. 19]
//
// CRITICAL CHANGES from old code:
//   1. θ^{k-1} (lagged) for capillary, viscosity, gravity
//   2. Symmetric strain D(U), not full gradient ∇U
//   3. Skew convection B_h(U^{k-1}, U^k, V)
//   4. Full Kelvin force with transported M^k (not equilibrium χH)
//
// ============================================================================
#ifndef NS_ASSEMBLER_H
#define NS_ASSEMBLER_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include "utilities/parameters.h"

#include <vector>

/**
 * @brief Assemble the Navier-Stokes system (Paper Eq. 42e)
 *
 * Assembles the coupled (ux, uy, p) system with all forces.
 * Constraints are applied symmetrically to preserve saddle-point structure.
 *
 * @param ux_dof_handler    DoFHandler for velocity x (Q2)
 * @param uy_dof_handler    DoFHandler for velocity y (Q2)
 * @param p_dof_handler     DoFHandler for pressure (Q1)
 * @param theta_dof_handler DoFHandler for phase field θ (Q2)
 * @param psi_dof_handler   DoFHandler for chemical potential ψ (Q2)
 * @param phi_dof_handler   DoFHandler for magnetic potential φ (Q2, can be nullptr)
 * @param M_dof_handler     DoFHandler for magnetization M (DG, can be nullptr)
 * @param ux_old            Previous velocity U^{k-1}_x
 * @param uy_old            Previous velocity U^{k-1}_y
 * @param theta_old         Previous phase field θ^{k-1} (LAGGED for energy stability)
 * @param psi_solution      Current chemical potential ψ^k
 * @param phi_solution      Current magnetic potential φ^k (can be nullptr)
 * @param mx_solution       Current magnetization M^k_x (can be nullptr)
 * @param my_solution       Current magnetization M^k_y (can be nullptr)
 * @param params            Physical parameters
 * @param dt                Time step τ
 * @param current_time      Current time (for MMS)
 * @param ux_to_ns_map      Index map: ux DoF → coupled index
 * @param uy_to_ns_map      Index map: uy DoF → coupled index
 * @param p_to_ns_map       Index map: p DoF → coupled index
 * @param ns_constraints    Combined constraints (hanging nodes + BCs)
 * @param ns_matrix         [OUT] Assembled system matrix
 * @param ns_rhs            [OUT] Assembled RHS vector
 */
template <int dim>
void assemble_ns_system(
    const dealii::DoFHandler<dim>& ux_dof_handler,
    const dealii::DoFHandler<dim>& uy_dof_handler,
    const dealii::DoFHandler<dim>& p_dof_handler,
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::DoFHandler<dim>& psi_dof_handler,
    const dealii::DoFHandler<dim>* phi_dof_handler,
    const dealii::DoFHandler<dim>* M_dof_handler,
    const dealii::Vector<double>& ux_old,
    const dealii::Vector<double>& uy_old,
    const dealii::Vector<double>& theta_old,
    const dealii::Vector<double>& psi_solution,
    const dealii::Vector<double>* phi_solution,
    const dealii::Vector<double>* mx_solution,
    const dealii::Vector<double>* my_solution,
    const Parameters& params,
    double dt,
    double current_time,
    const std::vector<dealii::types::global_dof_index>& ux_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& uy_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& p_to_ns_map,
    const dealii::AffineConstraints<double>& ns_constraints,
    dealii::SparseMatrix<double>& ns_matrix,
    dealii::Vector<double>& ns_rhs);

struct NSAssemblyInfo
{
    double F_cap_max = 0.0;     // max|F_capillary|
    double F_mag_max = 0.0;     // max|F_magnetic|
    double F_grav_max = 0.0;    // max|F_gravity|

    void reset()
    {
        F_cap_max = 0.0;
        F_mag_max = 0.0;
        F_grav_max = 0.0;
    }
};

#endif // NS_ASSEMBLER_H