// ============================================================================
// linear_algebra.h - Linear algebra types for coupled NS-CH solver
// ============================================================================
#ifndef NSCH_LINEAR_ALGEBRA_H
#define NSCH_LINEAR_ALGEBRA_H

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/affine_constraints.h>

// ============================================================================
// Type aliases for NS system (velocity + pressure)
// ============================================================================
using NSVector          = dealii::BlockVector<double>;
using NSMatrix          = dealii::BlockSparseMatrix<double>;
using NSSparsityPattern = dealii::BlockSparsityPattern;
using NSConstraints     = dealii::AffineConstraints<double>;

// ============================================================================
// Type aliases for CH system (c + μ)
// ============================================================================
using CHVector          = dealii::BlockVector<double>;
using CHMatrix          = dealii::BlockSparseMatrix<double>;
using CHSparsityPattern = dealii::BlockSparsityPattern;
using CHConstraints     = dealii::AffineConstraints<double>;

#endif // NSCH_LINEAR_ALGEBRA_H