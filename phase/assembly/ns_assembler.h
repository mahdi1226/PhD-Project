// ============================================================================
// assembly/ns_assembler_scalar.h - Navier-Stokes assembly for scalar DoFHandlers
//
// REFACTORED VERSION: Accepts separate DoFHandlers and Vector<double> for each field
// This replaces the BlockVector-based ns_assembler.h
//
// Equation (Nochetto 14e):
//   u_t + (u·∇)u - div(ν_θ T(u)) + ∇p = μ₀(m·∇)h + (λ/ε)θ∇ψ + f_g
//
// Forces:
//   - Capillary: F_cap = (λ/ε)θ∇ψ  [surface tension]
//   - Magnetic:  F_mag = μ₀κ_θ(h·∇)h  [Kelvin force]
//   - Gravity:   F_g = (1 + r·H(θ/ε))g  [Boussinesq]
// ============================================================================
#ifndef NS_ASSEMBLER_H
#define NS_ASSEMBLER_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include "utilities/nsch_parameters.h"

#include <vector>

/**
 * @brief Assemble the Navier-Stokes system using separate scalar DoFHandlers
 *
 * This assembler works with the refactored architecture where each field
 * has its own DoFHandler on the shared triangulation.
 *
 * The coupled NS system matrix has structure:
 *   [ A_uxux  A_uxuy  B_uxp ]   [ ux ]   [ rhs_ux ]
 *   [ A_uyux  A_uyuy  B_uyp ] × [ uy ] = [ rhs_uy ]
 *   [ B_pux   B_puy   0     ]   [ p  ]   [ rhs_p  ]
 *
 * @param ux_dof_handler    DoFHandler for velocity x-component (Q2)
 * @param uy_dof_handler    DoFHandler for velocity y-component (Q2)
 * @param p_dof_handler     DoFHandler for pressure (Q1)
 * @param c_dof_handler     DoFHandler for concentration (Q2)
 * @param mu_dof_handler    DoFHandler for chemical potential (Q2)
 * @param phi_dof_handler   DoFHandler for magnetic potential (Q2, can be nullptr)
 * @param ux_old_solution   Previous time step velocity x
 * @param uy_old_solution   Previous time step velocity y
 * @param c_solution        Current concentration
 * @param mu_solution       Current chemical potential
 * @param phi_solution      Current magnetic potential (ignored if magnetic disabled)
 * @param params            Physical and numerical parameters
 * @param dt                Time step size
 * @param current_time      Current simulation time
 * @param ns_matrix         Output: coupled system matrix
 * @param ns_rhs            Output: coupled RHS vector
 * @param ux_to_ns_map      Index mapping from ux DoFs to coupled system
 * @param uy_to_ns_map      Index mapping from uy DoFs to coupled system
 * @param p_to_ns_map       Index mapping from p DoFs to coupled system
 */
template <int dim>
void assemble_ns_system_scalar(
    const dealii::DoFHandler<dim>& ux_dof_handler,
    const dealii::DoFHandler<dim>& uy_dof_handler,
    const dealii::DoFHandler<dim>& p_dof_handler,
    const dealii::DoFHandler<dim>& c_dof_handler,
    const dealii::DoFHandler<dim>& mu_dof_handler,
    const dealii::DoFHandler<dim>* phi_dof_handler,  // Pointer, can be nullptr
    const dealii::Vector<double>&  ux_old_solution,
    const dealii::Vector<double>&  uy_old_solution,
    const dealii::Vector<double>&  c_solution,
    const dealii::Vector<double>&  mu_solution,
    const dealii::Vector<double>*  phi_solution,     // Pointer, can be nullptr
    const NSCHParameters&          params,
    double                         dt,
    double                         current_time,
    dealii::SparseMatrix<double>&  ns_matrix,
    dealii::Vector<double>&        ns_rhs,
    const std::vector<dealii::types::global_dof_index>& ux_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& uy_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& p_to_ns_map);

#endif // NS_ASSEMBLER_H