// ============================================================================
// solvers/ns_block_preconditioner.h - Schur Complement Preconditioner
//
// Block preconditioner for NS saddle-point system following deal.II step-22/56.
//
// System structure:
//   [A   B^T] [u]   [f]
//   [B   0  ] [p] = [g]
//
// Block triangular preconditioner P such that P^{-1} applied via:
//   1. Solve S̃ z_p = r_p  (pressure, S̃ ≈ pressure mass matrix)
//   2. z_u = Â^{-1} (r_u - B^T z_p)  (velocity)
//
// References:
//   - deal.II step-22, step-56
//   - Elman, Silvester & Wathen, "Finite Elements and Fast Iterative Solvers"
// ============================================================================
#ifndef NS_BLOCK_PRECONDITIONER_H
#define NS_BLOCK_PRECONDITIONER_H

#include <deal.II/base/subscriptor.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/precondition.h>

#include <vector>
#include <memory>

// ============================================================================
// InverseMatrix - Wraps iterative solve as matrix-vector product
// (Following deal.II step-22 pattern)
// ============================================================================
template <typename MatrixType, typename PreconditionerType>
class InverseMatrix : public dealii::EnableObserverPointer
{
public:
    InverseMatrix(const MatrixType& matrix,
                  const PreconditionerType& preconditioner,
                  double solver_tolerance = 1e-6,
                  unsigned int max_iterations = 1000);

    void vmult(dealii::Vector<double>& dst,
               const dealii::Vector<double>& src) const;

    mutable unsigned int last_iterations;

private:
    const dealii::ObserverPointer<const MatrixType> matrix_;
    const dealii::ObserverPointer<const PreconditionerType> preconditioner_;
    double solver_tolerance_;
    unsigned int max_iterations_;
};

// ============================================================================
// BlockSchurPreconditioner - Main block preconditioner for NS system
// (Following deal.II step-56 pattern, adapted for monolithic matrix)
// ============================================================================
class BlockSchurPreconditioner : public dealii::EnableObserverPointer
{
public:
    BlockSchurPreconditioner(
        const dealii::SparseMatrix<double>& system_matrix,
        const dealii::SparseMatrix<double>& pressure_mass,
        const std::vector<dealii::types::global_dof_index>& ux_to_ns_map,
        const std::vector<dealii::types::global_dof_index>& uy_to_ns_map,
        const std::vector<dealii::types::global_dof_index>& p_to_ns_map,
        bool do_solve_A = true);

    void vmult(dealii::Vector<double>& dst,
               const dealii::Vector<double>& src) const;

    // Statistics
    mutable unsigned int n_iterations_A;
    mutable unsigned int n_iterations_S;

private:
    void extract_velocity(const dealii::Vector<double>& src,
                          dealii::Vector<double>& vel) const;

    void extract_pressure(const dealii::Vector<double>& src,
                          dealii::Vector<double>& p) const;

    void insert_velocity(const dealii::Vector<double>& vel,
                         dealii::Vector<double>& dst) const;

    void insert_pressure(const dealii::Vector<double>& p,
                         dealii::Vector<double>& dst) const;

    void apply_BT(const dealii::Vector<double>& p,
                  dealii::Vector<double>& vel) const;

    // Matrix references
    const dealii::SparseMatrix<double>& system_matrix_;
    const dealii::SparseMatrix<double>& pressure_mass_;

    // Index maps
    const std::vector<dealii::types::global_dof_index>& ux_map_;
    const std::vector<dealii::types::global_dof_index>& uy_map_;
    const std::vector<dealii::types::global_dof_index>& p_map_;

    // Sizes
    unsigned int n_ux_, n_uy_, n_p_, n_vel_, n_total_;

    // Reverse mapping for efficient lookups
    std::vector<int> global_to_vel_;
    std::vector<int> global_to_p_;

    // Velocity block matrix (extracted)
    mutable dealii::SparsityPattern velocity_sparsity_;
    mutable dealii::SparseMatrix<double> velocity_block_;

    // Preconditioners
    mutable dealii::SparseILU<double> A_preconditioner_;
    mutable dealii::SparseILU<double> S_preconditioner_;
    mutable bool preconditioners_initialized_;

    bool do_solve_A_;
};

#endif // NS_BLOCK_PRECONDITIONER_H