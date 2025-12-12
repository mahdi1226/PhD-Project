// ============================================================================
// utilities/tensor_operations.h - Tensor Operations
// ============================================================================
#ifndef TENSOR_OPERATIONS_H
#define TENSOR_OPERATIONS_H

#include <deal.II/base/tensor.h>

/**
 * @brief Utility functions for tensor operations
 */
namespace TensorOps
{

/**
 * @brief Compute (a ⊗ b) : C = Σ_{ij} a_i b_j C_{ij}
 */
template <int dim>
double outer_product_contract(const dealii::Tensor<1, dim>& a,
                               const dealii::Tensor<1, dim>& b,
                               const dealii::Tensor<2, dim>& C)
{
    double result = 0.0;
    for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
            result += a[i] * b[j] * C[i][j];
    return result;
}

/**
 * @brief Compute trace of tensor
 */
template <int dim>
double trace(const dealii::Tensor<2, dim>& A)
{
    double result = 0.0;
    for (unsigned int i = 0; i < dim; ++i)
        result += A[i][i];
    return result;
}

/**
 * @brief Compute Frobenius norm squared: ||A||_F^2 = Σ_{ij} A_{ij}^2
 */
template <int dim>
double frobenius_norm_squared(const dealii::Tensor<2, dim>& A)
{
    double result = 0.0;
    for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
            result += A[i][j] * A[i][j];
    return result;
}

} // namespace TensorOps

#endif // TENSOR_OPERATIONS_H
