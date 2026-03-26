// ============================================================================
// physics/kelvin_force.h — Kelvin Force: μ₀(M·∇)H
//
// Zhang's decoupled scheme uses the simple Kelvin body force μ₀(M·∇)H
// in the NS momentum equation. Since H = ∇φ (CG), the gradient ∇H is
// obtained from the Hessian of the Poisson solution:
//
//   (M·∇)H[i] = Σ_j M[j] ∂²φ/∂x_i∂x_j
//
// This is the ONLY Kelvin term needed — no DG skew form, no face
// integrals, no curl corrections (∇×∇φ = 0 identically).
// ============================================================================
#ifndef KELVIN_FORCE_H
#define KELVIN_FORCE_H

#include <deal.II/base/tensor.h>

namespace KelvinForce
{

// ============================================================================
// Compute (M·∇)H from M and Hessian of φ
//
// Since H = ∇φ, we have:
//   ∂H_i/∂x_j = ∂²φ/∂x_i∂x_j = hess_phi[i][j]
//
// (M·∇)H[i] = Σ_j M[j] * ∂H_i/∂x_j = Σ_j M[j] * hess_phi[i][j]
//
// @param M          Magnetization vector at quadrature point
// @param hess_phi   Hessian of φ (∂²φ/∂x_i∂x_j) at quadrature point
// @return (M·∇)H vector
// ============================================================================
template <int dim>
inline dealii::Tensor<1, dim> compute_M_grad_H(
    const dealii::Tensor<1, dim>& M,
    const dealii::Tensor<2, dim>& hess_phi)
{
    dealii::Tensor<1, dim> result;
    for (unsigned int i = 0; i < dim; ++i)
    {
        result[i] = 0.0;
        for (unsigned int j = 0; j < dim; ++j)
            result[i] += M[j] * hess_phi[i][j];
    }
    return result;
}

} // namespace KelvinForce

#endif // KELVIN_FORCE_H
