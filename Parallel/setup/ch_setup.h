// ============================================================================
// setup/ch_setup.h - Cahn-Hilliard Coupled System Setup (Parallel Version)
//
// Free function to build coupled θ-ψ system:
//   - Index maps (θ → coupled, ψ → coupled)
//   - Combined constraints (hanging nodes + BCs mapped to coupled indices)
//   - Trilinos sparse matrix with distributed sparsity pattern
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef CH_SETUP_H
#define CH_SETUP_H

#include <deal.II/base/mpi.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <vector>

/**
 * @brief Set up the parallel coupled Cahn-Hilliard system
 *
 * Creates the data structures needed for the coupled θ-ψ system:
 *   - Index maps: field DoF → coupled system index
 *   - Combined constraints: hanging nodes + BCs mapped to coupled indices
 *   - Trilinos sparse matrix with distributed sparsity
 *
 * Data layout in coupled system:
 *   θ occupies indices [0, n_theta)
 *   ψ occupies indices [n_theta, n_theta + n_psi)
 *
 * @param theta_dof_handler       DoFHandler for phase field θ
 * @param psi_dof_handler         DoFHandler for chemical potential ψ
 * @param theta_constraints       Individual constraints for θ (hanging nodes + BCs)
 * @param psi_constraints         Individual constraints for ψ (hanging nodes + BCs)
 * @param ch_locally_owned        IndexSet of owned DoFs in coupled system
 * @param ch_locally_relevant     IndexSet of relevant DoFs in coupled system
 * @param theta_to_ch_map         [OUT] Index map: θ DoF → coupled index
 * @param psi_to_ch_map           [OUT] Index map: ψ DoF → coupled index
 * @param ch_combined_constraints [OUT] Combined constraints for coupled system
 * @param ch_matrix               [OUT] Trilinos sparse matrix for coupled system
 * @param mpi_communicator        MPI communicator
 * @param pcout                   Conditional output stream (rank 0 only)
 */
template <int dim>
void setup_ch_coupled_system(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::DoFHandler<dim>& psi_dof_handler,
    const dealii::AffineConstraints<double>& theta_constraints,
    const dealii::AffineConstraints<double>& psi_constraints,
    const dealii::IndexSet& ch_locally_owned,
    const dealii::IndexSet& ch_locally_relevant,
    std::vector<dealii::types::global_dof_index>& theta_to_ch_map,
    std::vector<dealii::types::global_dof_index>& psi_to_ch_map,
    dealii::AffineConstraints<double>& ch_combined_constraints,
    dealii::TrilinosWrappers::SparseMatrix& ch_matrix,
    MPI_Comm mpi_communicator,
    dealii::ConditionalOStream& pcout);

#endif // CH_SETUP_H