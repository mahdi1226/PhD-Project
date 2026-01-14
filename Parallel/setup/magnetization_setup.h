// ============================================================================
// setup/magnetization_setup.h - Magnetization DG System Setup (PARALLEL)
//
// PARALLEL VERSION:
//   - Uses TrilinosWrappers::SparsityPattern
//   - Uses TrilinosWrappers::MPI::Vector
//   - Only processes locally owned cells
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Section 5, Eq. 56: M_h DG space
// Section 5.1, Eq. 41: M⁰ = I_{M_h}(χ(θ⁰) H⁰)
// ============================================================================
#ifndef MAGNETIZATION_SETUP_H
#define MAGNETIZATION_SETUP_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include "utilities/parameters.h"

#include <mpi.h>

/**
 * @brief Set up sparsity pattern for magnetization DG system (PARALLEL)
 *
 * Creates flux sparsity pattern for DG elements, which includes
 * face coupling needed for upwind fluxes in the transport equation.
 *
 * @param M_dof_handler       DoFHandler for M (DG scalar)
 * @param M_locally_owned     IndexSet of locally owned DoFs
 * @param M_locally_relevant  IndexSet of locally relevant DoFs
 * @param M_matrix            [OUT] Trilinos sparse matrix (initialized with sparsity)
 * @param mpi_communicator    MPI communicator
 * @param pcout               Conditional output stream
 */
template <int dim>
void setup_magnetization_sparsity(
    const dealii::DoFHandler<dim>& M_dof_handler,
    const dealii::IndexSet& M_locally_owned,
    const dealii::IndexSet& M_locally_relevant,
    dealii::TrilinosWrappers::SparseMatrix& M_matrix,
    MPI_Comm mpi_communicator,
    dealii::ConditionalOStream& pcout);

/**
 * @brief Initialize magnetization M⁰ = χ(θ⁰)H⁰ via L² projection (PARALLEL)
 *
 * For DG, this is a cell-local L² projection - no global solve needed.
 * Each cell inverts its local mass matrix independently.
 *
 * @param M_dof_handler     DoFHandler for M (DG scalar)
 * @param theta_dof_handler DoFHandler for θ (CG)
 * @param phi_dof_handler   DoFHandler for φ (CG)
 * @param theta_solution    Initial phase field θ⁰ (ghosted)
 * @param phi_solution      Initial magnetic potential φ⁰ (ghosted)
 * @param params            Simulation parameters
 * @param Mx_solution       [OUT] Initial Mx⁰ (owned)
 * @param My_solution       [OUT] Initial My⁰ (owned)
 */
template <int dim>
void initialize_magnetization_equilibrium(
    const dealii::DoFHandler<dim>& M_dof_handler,
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& theta_solution,
    const dealii::TrilinosWrappers::MPI::Vector& phi_solution,
    const Parameters& params,
    dealii::TrilinosWrappers::MPI::Vector& Mx_solution,
    dealii::TrilinosWrappers::MPI::Vector& My_solution);

#endif // MAGNETIZATION_SETUP_H