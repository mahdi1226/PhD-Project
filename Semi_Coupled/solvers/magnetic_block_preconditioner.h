// ============================================================================
// solvers/magnetic_block_preconditioner.h - Block preconditioner for the
// monolithic magnetic system [Mx | My | phi].
//
// Component_wise renumbering (set in core/phase_field_setup.cc) gives:
//   [0, n_Mx + n_My)   -> M-block  (DG, mass-coefficient ~1e6 dominated)
//   [n_M, n_total)     -> phi-block (CG Laplacian, possibly Neumann pinned)
//
// Coupling C_Mphi ~ chi/tau_M ~ 1e-7 is negligible, so block-diagonal
// preconditioning is essentially exact:
//
//   M^{-1} ≈ [ A_M^{-1}    0       ]
//            [ 0           A_phi^{-1} ]
//
// We use ILU(0) for the M block (mass-dominated) and AMG for the phi block
// (Laplacian).
//
// IMPLEMENTATION NOTE: We extract sub-blocks via Epetra_CrsMatrix's
// ExtractMyRowView rather than deal.II's SparseMatrix::begin(row) iterator.
// The latter throws at the M/phi block boundary in deal.II 9.7.1 due to a
// known iterator bug. This pattern is the same one used in
// solvers/ns_block_preconditioner.cc.
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef MAGNETIC_BLOCK_PRECONDITIONER_H
#define MAGNETIC_BLOCK_PRECONDITIONER_H

#include <deal.II/base/index_set.h>
#include <deal.II/base/types.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_precondition.h>

#include <Epetra_CrsMatrix.h>
#include <Epetra_Map.h>

#include <memory>
#include <mpi.h>

class MagneticBlockPreconditioner
{
public:
    /**
     * @brief Build the preconditioner from the monolithic [M | phi] matrix.
     *
     * @param system_matrix      The assembled monolithic magnetic matrix.
     * @param mag_locally_owned  IndexSet of locally-owned monolithic DoFs.
     * @param n_M_dofs           Number of M DoFs (boundary between blocks).
     * @param mpi_comm           MPI communicator.
     */
    MagneticBlockPreconditioner(
        const dealii::TrilinosWrappers::SparseMatrix& system_matrix,
        const dealii::IndexSet& mag_locally_owned,
        dealii::types::global_dof_index n_M_dofs,
        MPI_Comm mpi_comm);

    /// dst = M^{-1} src  (block-diagonal: ILU on M block, AMG on phi block)
    void vmult(
        dealii::TrilinosWrappers::MPI::Vector& dst,
        const dealii::TrilinosWrappers::MPI::Vector& src) const;

private:
    void extract_M_block(const Epetra_CrsMatrix& epetra_mat);
    void extract_phi_block(const Epetra_CrsMatrix& epetra_mat);

    // Source metadata
    dealii::IndexSet mag_locally_owned_;
    dealii::types::global_dof_index n_M_dofs_;
    dealii::types::global_dof_index n_phi_dofs_;
    dealii::types::global_dof_index n_total_;
    MPI_Comm mpi_comm_;
    int rank_;

    // Sub-block IndexSets (in their own LOCAL numbering, [0, n_block))
    dealii::IndexSet M_owned_;     // GIDs in [0, n_M_dofs_), original numbering
    dealii::IndexSet phi_owned_;   // phi GIDs reindexed: GID = original - n_M

    // Sub-block matrices (deal.II wrappers around Epetra)
    dealii::TrilinosWrappers::SparseMatrix M_block_matrix_;
    dealii::TrilinosWrappers::SparseMatrix phi_block_matrix_;

    // Per-block preconditioners
    dealii::TrilinosWrappers::PreconditionILU M_prec_;
    dealii::TrilinosWrappers::PreconditionAMG phi_prec_;

    // Reusable vectors for vmult (mutable for const correctness)
    mutable dealii::TrilinosWrappers::MPI::Vector r_M_, r_phi_, z_M_, z_phi_;
};

#endif // MAGNETIC_BLOCK_PRECONDITIONER_H
