// ============================================================================
// assembly/ch_assembler.h - Parallel Cahn-Hilliard Assembler
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Equations 42a-42b (discrete scheme), p.505
//
// Parallel version using TrilinosWrappers.
// ============================================================================
#ifndef CH_ASSEMBLER_H
#define CH_ASSEMBLER_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/affine_constraints.h>

#include <vector>
#include <mpi.h>

// Forward declaration
struct Parameters;

/**
 * @brief Assemble the coupled Cahn-Hilliard system (Paper Eq. 42a-42b)
 *
 * @param theta_dof_handler  DoFHandler for phase field θ
 * @param psi_dof_handler    DoFHandler for chemical potential ψ
 * @param theta_old          Previous time step θ^{k-1} (ghosted)
 * @param ux_dof_handler     DoFHandler for velocity x-component
 * @param uy_dof_handler     DoFHandler for velocity y-component
 * @param ux_solution        Velocity x-component U^k_x (ghosted)
 * @param uy_solution        Velocity y-component U^k_y (ghosted)
 * @param params             Simulation parameters
 * @param dt                 Time step size τ
 * @param current_time       Current simulation time (for MMS)
 * @param theta_to_ch_map    Index mapping: θ DoF → coupled system index
 * @param psi_to_ch_map      Index mapping: ψ DoF → coupled system index
 * @param ch_constraints     Combined constraints for coupled system
 * @param matrix             Output: assembled Trilinos system matrix
 * @param rhs                Output: assembled Trilinos RHS vector
 */
template <int dim>
void assemble_ch_system(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::DoFHandler<dim>& psi_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& theta_old,
    const dealii::DoFHandler<dim>& ux_dof_handler,
    const dealii::DoFHandler<dim>& uy_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& ux_solution,
    const dealii::TrilinosWrappers::MPI::Vector& uy_solution,
    const Parameters& params,
    double dt,
    double current_time,
    const std::vector<dealii::types::global_dof_index>& theta_to_ch_map,
    const std::vector<dealii::types::global_dof_index>& psi_to_ch_map,
    const dealii::AffineConstraints<double>& ch_constraints,
    dealii::TrilinosWrappers::SparseMatrix& matrix,
    dealii::TrilinosWrappers::MPI::Vector& rhs);

#endif // CH_ASSEMBLER_H