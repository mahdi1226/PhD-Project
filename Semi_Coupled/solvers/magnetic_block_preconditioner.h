// ============================================================================
// solvers/magnetic_block_preconditioner.h - Block Preconditioner for M+φ
//
// Block preconditioner for the monolithic magnetics system:
//   | A_M      C_Mφ  | | M |   | f_M |
//   | C_φM     A_φ   | | φ | = | f_φ |
//
// Upper block-triangular preconditioner:
//   P = | A_M    C_Mφ |       P^{-1} applied as:
//       | 0      A_φ  |
//
// Application (back-substitution):
//   1. Solve A_φ · z_φ = r_φ         (CG + AMG, elliptic Laplacian)
//   2. rhs_M = r_M - C_Mφ · z_φ
//   3. Solve A_M · z_M = rhs_M       (GMRES + ILU, DG mass-dominated)
//
// Key insight: With component_wise renumbering, DoFs are contiguous:
//   [Mx(0..n_Mx-1) | My(n_Mx..n_M-1) | φ(n_M..n_total-1)]
// This makes block extraction trivial (index ranges, no GID remapping).
//
// A_M is DG mass + transport (well-conditioned, ILU(0) sufficient)
// A_φ is Laplacian (SPD, AMG ideal with elliptic=true)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef MAGNETIC_BLOCK_PRECONDITIONER_H
#define MAGNETIC_BLOCK_PRECONDITIONER_H

#include <deal.II/base/subscriptor.h>

#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_control.h>

#include <memory>

// ============================================================================
// Magnetic Block Preconditioner (Parallel)
// ============================================================================
class MagneticBlockPreconditioner : public dealii::EnableObserverPointer
{
public:
    /**
     * @brief Constructor
     *
     * @param system_matrix     Full monolithic mag matrix [A_M, C_Mφ; C_φM, A_φ]
     * @param mag_owned         Locally owned DoFs for full system
     * @param n_M_dofs          Total M DoFs (Mx + My combined)
     * @param n_phi_dofs        Total φ DoFs
     * @param mpi_comm          MPI communicator
     * @param use_ilu           Use ILU for both blocks (HPC fallback)
     */
    MagneticBlockPreconditioner(
        const dealii::TrilinosWrappers::SparseMatrix& system_matrix,
        const dealii::IndexSet& mag_owned,
        dealii::types::global_dof_index n_M_dofs,
        dealii::types::global_dof_index n_phi_dofs,
        MPI_Comm mpi_comm,
        bool use_ilu = false);

    /**
     * @brief Apply preconditioner: dst = P^{-1} src
     */
    void vmult(dealii::TrilinosWrappers::MPI::Vector& dst,
               const dealii::TrilinosWrappers::MPI::Vector& src) const;

    // ------------------------------------------------------------------------
    // Diagnostics (accumulated per outer solve)
    // ------------------------------------------------------------------------
    mutable unsigned int n_iterations_M;   // M block inner iterations
    mutable unsigned int n_iterations_phi; // φ block inner iterations

    // ------------------------------------------------------------------------
    // Tuning parameters (public by design)
    // ------------------------------------------------------------------------
    double inner_tolerance;
    unsigned int max_inner_iterations;
    bool verbose_ = false;

private:
    // ------------------------------------------------------------------------
    // Helper: extract/insert sub-vectors by index range
    // ------------------------------------------------------------------------
    void extract_M(const dealii::TrilinosWrappers::MPI::Vector& src,
                   dealii::TrilinosWrappers::MPI::Vector& M_vec) const;

    void extract_phi(const dealii::TrilinosWrappers::MPI::Vector& src,
                     dealii::TrilinosWrappers::MPI::Vector& phi_vec) const;

    void insert_M(const dealii::TrilinosWrappers::MPI::Vector& M_vec,
                  dealii::TrilinosWrappers::MPI::Vector& dst) const;

    void insert_phi(const dealii::TrilinosWrappers::MPI::Vector& phi_vec,
                    dealii::TrilinosWrappers::MPI::Vector& dst) const;

    // Helper: apply C_Mφ (off-diagonal coupling block) to φ vector
    void apply_C_M_phi(const dealii::TrilinosWrappers::MPI::Vector& phi_vec,
                       dealii::TrilinosWrappers::MPI::Vector& M_vec) const;

    // ------------------------------------------------------------------------
    // System matrix pointer
    // ------------------------------------------------------------------------
    const dealii::TrilinosWrappers::SparseMatrix* system_matrix_ptr_;

    // ------------------------------------------------------------------------
    // Block dimensions
    // ------------------------------------------------------------------------
    dealii::types::global_dof_index n_M_;
    dealii::types::global_dof_index n_phi_;
    dealii::types::global_dof_index n_total_;

    // ------------------------------------------------------------------------
    // Owned index sets for sub-blocks
    // ------------------------------------------------------------------------
    dealii::IndexSet mag_owned_;     // Full system
    dealii::IndexSet M_owned_;       // [0, n_M) ∩ locally_owned
    dealii::IndexSet phi_owned_;     // [n_M, n_total) ∩ locally_owned

    // Ghost indices: phi DoFs needed by local M rows for C_Mφ product
    // Includes locally owned + off-rank phi entries in the coupling stencil
    dealii::IndexSet phi_relevant_for_coupling_;

    // ------------------------------------------------------------------------
    // Sub-block matrices (extracted from system)
    // ------------------------------------------------------------------------
    dealii::TrilinosWrappers::SparseMatrix M_block_;    // A_M
    dealii::TrilinosWrappers::SparseMatrix phi_block_;  // A_φ

    // ------------------------------------------------------------------------
    // Preconditioners
    // ------------------------------------------------------------------------
    std::unique_ptr<dealii::TrilinosWrappers::PreconditionBase> M_preconditioner_;
    std::unique_ptr<dealii::TrilinosWrappers::PreconditionBase> phi_preconditioner_;

    // ------------------------------------------------------------------------
    // MPI
    // ------------------------------------------------------------------------
    MPI_Comm mpi_comm_;
    int rank_;

    // ------------------------------------------------------------------------
    // Cached temporary vectors (avoid re-allocation per vmult call)
    // Mutable because vmult is const but we reuse these across calls.
    // ------------------------------------------------------------------------
    mutable dealii::TrilinosWrappers::MPI::Vector tmp_r_M_;
    mutable dealii::TrilinosWrappers::MPI::Vector tmp_r_phi_;
    mutable dealii::TrilinosWrappers::MPI::Vector tmp_z_M_;
    mutable dealii::TrilinosWrappers::MPI::Vector tmp_z_phi_;
    mutable dealii::TrilinosWrappers::MPI::Vector tmp_C_zphi_;
    mutable dealii::TrilinosWrappers::MPI::Vector tmp_rhs_M_;
    mutable bool tmp_initialized_;
};

#endif // MAGNETIC_BLOCK_PRECONDITIONER_H
