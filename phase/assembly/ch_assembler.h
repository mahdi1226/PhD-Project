// ============================================================================
// assembly/ch_assembler_scalar.h - Cahn-Hilliard assembly for scalar DoFHandlers
//
// REFACTORED VERSION: Accepts separate DoFHandlers and Vector<double> for each field
// This replaces the BlockVector-based ch_assembler.h
//
// Equations (Nochetto 14a-14b):
//   θ_t + div(uθ) + γ Δψ = 0        (phase field evolution)
//   ψ - ε Δθ + (1/ε) f(θ) = 0       (chemical potential)
//
// where f(θ) = θ³ - θ is the double-well potential derivative
// ============================================================================
#ifndef CH_ASSEMBLER_H
#define CH_ASSEMBLER_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/affine_constraints.h>

#include "utilities/nsch_parameters.h"

#include <vector>

/**
 * @brief Assemble the Cahn-Hilliard system using separate scalar DoFHandlers
 *
 * This assembler works with the refactored architecture where each field
 * (c, μ, ux, uy) has its own DoFHandler on the shared triangulation.
 *
 * The coupled CH system matrix has structure:
 *   [ A_cc   A_cμ  ]   [ c  ]   [ rhs_c  ]
 *   [ A_μc   A_μμ  ] × [ μ  ] = [ rhs_μ  ]
 *
 * @param c_dof_handler   DoFHandler for concentration c (Q2)
 * @param mu_dof_handler  DoFHandler for chemical potential μ (Q2)
 * @param ux_dof_handler  DoFHandler for velocity x-component (Q2)
 * @param uy_dof_handler  DoFHandler for velocity y-component (Q2)
 * @param c_old_solution  Previous time step concentration
 * @param ux_solution     Current velocity x-component (for advection)
 * @param uy_solution     Current velocity y-component (for advection)
 * @param params          Physical and numerical parameters
 * @param dt              Time step size
 * @param current_time    Current simulation time
 * @param ch_matrix       Output: coupled system matrix
 * @param ch_rhs          Output: coupled RHS vector
 * @param c_to_ch_map     Index mapping from c DoFs to coupled system
 * @param mu_to_ch_map    Index mapping from μ DoFs to coupled system
 */
template <int dim>
void assemble_ch_system_scalar(
    const dealii::DoFHandler<dim>& c_dof_handler,
    const dealii::DoFHandler<dim>& mu_dof_handler,
    const dealii::DoFHandler<dim>& ux_dof_handler,
    const dealii::DoFHandler<dim>& uy_dof_handler,
    const dealii::Vector<double>&  c_old_solution,
    const dealii::Vector<double>&  ux_solution,
    const dealii::Vector<double>&  uy_solution,
    const NSCHParameters&          params,
    double                         dt,
    double                         current_time,
    dealii::SparseMatrix<double>&  ch_matrix,
    dealii::Vector<double>&        ch_rhs,
    const std::vector<dealii::types::global_dof_index>& c_to_ch_map,
    const std::vector<dealii::types::global_dof_index>& mu_to_ch_map);

#endif // CH_ASSEMBLER_H