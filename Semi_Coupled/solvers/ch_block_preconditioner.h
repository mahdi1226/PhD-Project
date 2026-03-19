// ============================================================================
// solvers/ch_block_preconditioner.h - Block Preconditioner for CH (θ+ψ)
//
// Block preconditioner for the coupled Cahn-Hilliard system:
//   | A_θθ   A_θψ | | θ |   | f_θ |
//   | A_ψθ   A_ψψ | | ψ | = | f_ψ |
//
// Lower block-triangular preconditioner:
//   P = | A_θθ   0   |       P^{-1} applied as:
//       | A_ψθ   A_ψψ|
//
// Application (forward substitution):
//   1. Solve A_ψψ · z_ψ = r_ψ         (CG + Jacobi, mass matrix ~1-2 iters)
//   2. rhs_θ = r_θ - A_θψ · z_ψ
//   3. Solve A_θθ · z_θ = rhs_θ       (GMRES + ILU, mass+convection)
//
// Key insight: With the CH setup, DoFs are contiguous:
//   [θ(0..n_theta-1) | ψ(n_theta..n_total-1)]
//
// A_ψψ ≈ mass matrix (SPD, well-conditioned, Jacobi sufficient)
// A_θθ = mass + convection (non-symmetric but dominant, ILU works)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef CH_BLOCK_PRECONDITIONER_H
#define CH_BLOCK_PRECONDITIONER_H

#include <deal.II/base/subscriptor.h>

#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_control.h>

#include <memory>

// ============================================================================
// CH Block Preconditioner (Parallel)
// ============================================================================
class CHBlockPreconditioner : public dealii::EnableObserverPointer
{
public:
    /**
     * @brief Constructor
     *
     * @param system_matrix     Full monolithic CH matrix [A_θθ, A_θψ; A_ψθ, A_ψψ]
     * @param ch_owned          Locally owned DoFs for full system
     * @param n_theta_dofs      Total θ DoFs
     * @param n_psi_dofs        Total ψ DoFs
     * @param mpi_comm          MPI communicator
     * @param use_ilu           Use ILU for both blocks (HPC fallback)
     */
    CHBlockPreconditioner(
        const dealii::TrilinosWrappers::SparseMatrix& system_matrix,
        const dealii::IndexSet& ch_owned,
        dealii::types::global_dof_index n_theta_dofs,
        dealii::types::global_dof_index n_psi_dofs,
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
    mutable unsigned int n_iterations_theta;
    mutable unsigned int n_iterations_psi;

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
    void extract_theta(const dealii::TrilinosWrappers::MPI::Vector& src,
                       dealii::TrilinosWrappers::MPI::Vector& theta_vec) const;

    void extract_psi(const dealii::TrilinosWrappers::MPI::Vector& src,
                     dealii::TrilinosWrappers::MPI::Vector& psi_vec) const;

    void insert_theta(const dealii::TrilinosWrappers::MPI::Vector& theta_vec,
                      dealii::TrilinosWrappers::MPI::Vector& dst) const;

    void insert_psi(const dealii::TrilinosWrappers::MPI::Vector& psi_vec,
                    dealii::TrilinosWrappers::MPI::Vector& dst) const;

    // Helper: apply A_θψ (off-diagonal coupling block) to ψ vector
    void apply_A_theta_psi(const dealii::TrilinosWrappers::MPI::Vector& psi_vec,
                           dealii::TrilinosWrappers::MPI::Vector& theta_vec) const;

    // ------------------------------------------------------------------------
    // System matrix pointer
    // ------------------------------------------------------------------------
    const dealii::TrilinosWrappers::SparseMatrix* system_matrix_ptr_;

    // ------------------------------------------------------------------------
    // Block dimensions
    // ------------------------------------------------------------------------
    dealii::types::global_dof_index n_theta_;
    dealii::types::global_dof_index n_psi_;
    dealii::types::global_dof_index n_total_;

    // ------------------------------------------------------------------------
    // Owned index sets for sub-blocks
    // ------------------------------------------------------------------------
    dealii::IndexSet ch_owned_;       // Full system
    dealii::IndexSet theta_owned_;    // [0, n_theta) ∩ locally_owned
    dealii::IndexSet psi_owned_;      // [n_theta, n_total) ∩ locally_owned

    // Ghost indices: ψ DoFs needed by local θ rows for A_θψ product
    dealii::IndexSet psi_relevant_for_coupling_;

    // ------------------------------------------------------------------------
    // Sub-block matrices (extracted from system)
    // ------------------------------------------------------------------------
    dealii::TrilinosWrappers::SparseMatrix theta_block_;  // A_θθ
    dealii::TrilinosWrappers::SparseMatrix psi_block_;    // A_ψψ

    // ------------------------------------------------------------------------
    // Preconditioners
    // ------------------------------------------------------------------------
    std::unique_ptr<dealii::TrilinosWrappers::PreconditionBase> theta_preconditioner_;
    std::unique_ptr<dealii::TrilinosWrappers::PreconditionBase> psi_preconditioner_;

    // ------------------------------------------------------------------------
    // MPI
    // ------------------------------------------------------------------------
    MPI_Comm mpi_comm_;
    int rank_;

    // ------------------------------------------------------------------------
    // Cached temporary vectors
    // ------------------------------------------------------------------------
    mutable dealii::TrilinosWrappers::MPI::Vector tmp_r_theta_;
    mutable dealii::TrilinosWrappers::MPI::Vector tmp_r_psi_;
    mutable dealii::TrilinosWrappers::MPI::Vector tmp_z_theta_;
    mutable dealii::TrilinosWrappers::MPI::Vector tmp_z_psi_;
    mutable dealii::TrilinosWrappers::MPI::Vector tmp_C_zpsi_;
    mutable dealii::TrilinosWrappers::MPI::Vector tmp_rhs_theta_;
    mutable dealii::TrilinosWrappers::MPI::Vector tmp_psi_ghosted_;
    mutable bool tmp_initialized_;
};

#endif // CH_BLOCK_PRECONDITIONER_H
