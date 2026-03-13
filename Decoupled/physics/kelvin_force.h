// ============================================================================
// physics/kelvin_force.h — Kelvin Force Utility Functions
//
// Reference: Zhang, He & Yang, SIAM J. Sci. Comput. 43(1), 2021, B167-B193
//            Equation 3.11 (velocity predictor with Kelvin body force)
//
// Provides helper functions for computing the Kelvin body force terms
// in the Navier-Stokes velocity predictor:
//
//   F_kelvin = μ₀ [(M·∇)H + ½(∇·M)H]   (Zhang Eq 3.11, terms 1-2)
//
// where H = ∇φ is the total magnetic field (Poisson encodes h_a via BCs).
//
// Note: The third Kelvin term μ₀(M × ∇×H) ≡ 0 because H = ∇φ and
// curl(grad) = 0 for CG fields.
//
// Functions:
//   compute_M_grad_H  — (M·∇)H from M and Hessian of φ
//   compute_div_M     — ∇·M from component gradients
//
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

// ============================================================================
// Compute div(M) from component gradients
//
// div(M) = ∂Mx/∂x + ∂My/∂y   (2D)
//
// @param grad_Mx   Gradient of Mx component
// @param grad_My   Gradient of My component
// @return div(M) scalar
// ============================================================================
template <int dim>
inline double compute_div_M(
    const dealii::Tensor<1, dim>& grad_Mx,
    const dealii::Tensor<1, dim>& grad_My)
{
    static_assert(dim >= 2, "Kelvin force requires dim >= 2");
    return grad_Mx[0] + grad_My[1];
}

} // namespace KelvinForce

#endif // KELVIN_FORCE_H
