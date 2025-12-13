// ============================================================================
// assembly/ns_assembler.h - Navier-Stokes Assembly
//
// Equation (Nochetto 14e-14f, p.500):
//   u_t + (u·∇)u - div(ν_θ ∇u) + ∇p = F_cap + F_mag + F_grav
//   div(u) = 0
//
// Forces:
//   - Capillary: F_cap = (λ/ε)θ∇ψ           [surface tension, Eq. 10]
//   - Kelvin:    F_mag = μ₀χ(θ)(H·∇)H       [magnetic, Eq. 14f]
//   - Gravity:   F_grav = (1 + r·H(θ/ε))g   [Boussinesq, Eq. 19]
//
// Time stepping (semi-implicit):
//   - Mass:       (1/dt)(u, v)              implicit
//   - Viscosity:  θν(∇u, ∇v)                θ-implicit
//   - Convection: ((u_old·∇)u_old, v)       explicit
//   - Pressure:   -(p, div(v))              implicit
//   - Forces:     (F, v)                    explicit
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef NS_ASSEMBLER_H
#define NS_ASSEMBLER_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include "utilities/parameters.h"

#include <vector>

/**
 * @brief Assemble the Navier-Stokes system
 *
 * Assembles the coupled (ux, uy, p) system with all forces.
 *
 * @param ux_dof_handler    DoFHandler for velocity x (Q2)
 * @param uy_dof_handler    DoFHandler for velocity y (Q2)
 * @param p_dof_handler     DoFHandler for pressure (Q1)
 * @param theta_dof_handler DoFHandler for phase field θ (Q2)
 * @param psi_dof_handler   DoFHandler for chemical potential ψ (Q2)
 * @param phi_dof_handler   DoFHandler for magnetic potential φ (Q2, can be nullptr)
 * @param ux_old            Previous velocity x
 * @param uy_old            Previous velocity y
 * @param theta_solution    Current phase field θ
 * @param psi_solution      Current chemical potential ψ
 * @param phi_solution      Current magnetic potential φ (can be nullptr)
 * @param params            Physical parameters
 * @param dt                Time step
 * @param current_time      Current time (for MMS)
 * @param ux_to_ns_map      Index map: ux DoF → coupled index
 * @param uy_to_ns_map      Index map: uy DoF → coupled index
 * @param p_to_ns_map       Index map: p DoF → coupled index
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
    const dealii::Vector<double>& ux_old,
    const dealii::Vector<double>& uy_old,
    const dealii::Vector<double>& theta_solution,
    const dealii::Vector<double>& psi_solution,
    const dealii::Vector<double>* phi_solution,
    const Parameters& params,
    double dt,
    double current_time,
    const std::vector<dealii::types::global_dof_index>& ux_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& uy_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& p_to_ns_map,
    dealii::SparseMatrix<double>& ns_matrix,
    dealii::Vector<double>& ns_rhs);

#endif // NS_ASSEMBLER_H