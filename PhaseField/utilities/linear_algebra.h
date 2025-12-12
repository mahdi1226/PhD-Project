// ============================================================================
// utilities/linear_algebra.h - Linear Algebra Type Definitions
// ============================================================================
#ifndef LINEAR_ALGEBRA_H
#define LINEAR_ALGEBRA_H

#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/block_sparsity_pattern.h>

/**
 * @brief Type aliases for linear algebra objects
 */
namespace LinearAlgebra
{
    // Scalar vectors
    using Vector = dealii::Vector<double>;
    
    // Block vectors for coupled systems
    using BlockVector = dealii::BlockVector<double>;
    
    // Sparse matrices
    using SparseMatrix = dealii::SparseMatrix<double>;
    using SparsityPattern = dealii::SparsityPattern;
    
    // Block matrices (for fully coupled systems)
    using BlockSparseMatrix = dealii::BlockSparseMatrix<double>;
    using BlockSparsityPattern = dealii::BlockSparsityPattern;
    
} // namespace LinearAlgebra

#endif // LINEAR_ALGEBRA_H
