// ============================================================================
// navier_stokes/ns_block_schur_preconditioner.h - Block-Schur Preconditioner
//
// Right preconditioner for the NS saddle-point system:
//
//   [A   B^T] [u]   [f]
//   [B   0  ] [p] = [0]
//
// Block-triangular preconditioner:
//
//   P = [A   B^T]     P^{-1} = [A^{-1}  -A^{-1} B^T S^{-1}]
//       [0   -S ]               [0        -S^{-1}            ]
//
// where S = B A^{-1} B^T ≈ (1/ν_eff) M_p (pressure mass matrix).
//
// Inner solves:
//   A^{-1}: AMG-preconditioned CG on diagonal velocity blocks
//   S^{-1}: Jacobi-preconditioned CG on pressure mass matrix (scaled)
//
// Used with FGMRES (flexible, since inner solves are inexact).
//
// Reference: Elman, Silvester & Wathen, "Finite Elements and Fast Iterative
//            Solvers", 2nd ed., Chapter 8
// ============================================================================
#ifndef FHD_NS_BLOCK_SCHUR_PRECONDITIONER_H
#define FHD_NS_BLOCK_SCHUR_PRECONDITIONER_H

#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/base/index_set.h>

#include "utilities/solver_info.h"

#include <mpi.h>

template <int dim>
class NSBlockSchurPreconditioner
{
public:
    NSBlockSchurPreconditioner(
        const dealii::TrilinosWrappers::SparseMatrix& A_ux_ux,
        const dealii::TrilinosWrappers::SparseMatrix& A_uy_uy,
        const dealii::TrilinosWrappers::SparseMatrix& Bt_ux,
        const dealii::TrilinosWrappers::SparseMatrix& Bt_uy,
        const dealii::TrilinosWrappers::SparseMatrix& B_ux,
        const dealii::TrilinosWrappers::SparseMatrix& B_uy,
        const dealii::TrilinosWrappers::SparseMatrix& M_p,
        double nu_eff,
        const dealii::IndexSet& ux_owned,
        const dealii::IndexSet& uy_owned,
        const dealii::IndexSet& p_owned,
        dealii::types::global_dof_index n_ux,
        dealii::types::global_dof_index n_uy,
        const LinearSolverParams& params,
        MPI_Comm mpi_comm);

    void initialize();

    void vmult(dealii::TrilinosWrappers::MPI::Vector& dst,
               const dealii::TrilinosWrappers::MPI::Vector& src) const;

private:
    // Block matrices (references, not owned)
    const dealii::TrilinosWrappers::SparseMatrix& A_ux_ux_;
    const dealii::TrilinosWrappers::SparseMatrix& A_uy_uy_;
    const dealii::TrilinosWrappers::SparseMatrix& Bt_ux_;   // B^T for ux
    const dealii::TrilinosWrappers::SparseMatrix& Bt_uy_;   // B^T for uy
    const dealii::TrilinosWrappers::SparseMatrix& B_ux_;    // B for ux
    const dealii::TrilinosWrappers::SparseMatrix& B_uy_;    // B for uy
    const dealii::TrilinosWrappers::SparseMatrix& M_p_;     // Pressure mass matrix

    double nu_eff_;

    // Component index sets
    const dealii::IndexSet& ux_owned_;
    const dealii::IndexSet& uy_owned_;
    const dealii::IndexSet& p_owned_;

    // Offsets for coupled system
    dealii::types::global_dof_index n_ux_;
    dealii::types::global_dof_index n_uy_;

    // Solver parameters
    const LinearSolverParams& params_;
    MPI_Comm mpi_comm_;

    // Preconditioners (built in initialize())
    dealii::TrilinosWrappers::PreconditionAMG amg_ux_;
    dealii::TrilinosWrappers::PreconditionAMG amg_uy_;
    dealii::TrilinosWrappers::PreconditionJacobi jacobi_p_;

    // Workspace vectors (mutable for const vmult)
    mutable dealii::TrilinosWrappers::MPI::Vector tmp_ux_;
    mutable dealii::TrilinosWrappers::MPI::Vector tmp_uy_;
    mutable dealii::TrilinosWrappers::MPI::Vector tmp_p_;
    mutable dealii::TrilinosWrappers::MPI::Vector rhs_ux_;
    mutable dealii::TrilinosWrappers::MPI::Vector rhs_uy_;
    mutable dealii::TrilinosWrappers::MPI::Vector rhs_p_;

    bool initialized_ = false;
};

extern template class NSBlockSchurPreconditioner<2>;
extern template class NSBlockSchurPreconditioner<3>;

#endif // FHD_NS_BLOCK_SCHUR_PRECONDITIONER_H
