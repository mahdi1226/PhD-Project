// ============================================================================
// utilities/trilinear_forms.cc - Trilinear Forms Implementation
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Equations 20-21, 37-38, p.502-504
// ============================================================================

#include "trilinear_forms.h"

namespace TrilinearForms
{

// ============================================================================
// B_magnetic()
//
// Magnetic trilinear form (Eq. 20, p.502):
//   B(m, h, u) = Σ_{i,j} m_i (∂h_i/∂x_j) u_j
//
// In component form:
//   B = m₀ (∂h₀/∂x) u_x + m₀ (∂h₀/∂y) u_y
//     + m₁ (∂h₁/∂x) u_x + m₁ (∂h₁/∂y) u_y  (2D)
//
// Key identity (Lemma 3.1, Eq. 21):
//   B(u, m, h) = -B(m, h, u)
// ============================================================================
template <int dim>
double B_magnetic(const dealii::Tensor<1, dim>& m,
                  const dealii::Tensor<2, dim>& grad_h,
                  const dealii::Tensor<1, dim>& u)
{
    double result = 0.0;
    
    // B(m, h, u) = Σ_{i,j} m_i (∂h_i/∂x_j) u_j
    for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
            result += m[i] * grad_h[i][j] * u[j];
    
    return result;
}

// ============================================================================
// b_convection_skew()
//
// Skew-symmetric convection form:
//   b(u, v, w) = ½[(u·∇v, w) - (u·∇w, v)]
//              = ½[Σ_{i,j} u_j (∂v_i/∂x_j) w_i - u_j (∂w_i/∂x_j) v_i]
//
// Property: b(u, v, v) = 0 when div(u) = 0
//
// Discrete form (Eq. 37, p.504):
//   B_h(U, V, W) = -B_h(U, W, V)
// ============================================================================
template <int dim>
double b_convection_skew(const dealii::Tensor<1, dim>& u,
                         const dealii::Tensor<2, dim>& grad_v,
                         const dealii::Tensor<2, dim>& grad_w,
                         const dealii::Tensor<1, dim>& v,
                         const dealii::Tensor<1, dim>& w)
{
    double term1 = 0.0;  // (u·∇v, w)
    double term2 = 0.0;  // (u·∇w, v)
    
    for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
        {
            term1 += u[j] * grad_v[i][j] * w[i];
            term2 += u[j] * grad_w[i][j] * v[i];
        }
    
    return 0.5 * (term1 - term2);
}

// ============================================================================
// symmetric_gradient()
//
// T(u) = ½(∇u + ∇u^T)
//
// Rate-of-strain tensor (Eq. 14e, p.501)
// ============================================================================
template <int dim>
dealii::Tensor<2, dim> symmetric_gradient(const dealii::Tensor<2, dim>& grad_u)
{
    dealii::Tensor<2, dim> T;
    
    for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
            T[i][j] = 0.5 * (grad_u[i][j] + grad_u[j][i]);
    
    return T;
}

// ============================================================================
// symmetric_gradient_inner_product()
//
// (T(u), T(v)) = Σ_{i,j} T_u[i][j] * T_v[i][j]
//
// Used in viscous stress: (ν_θ T(u), T(v))
// ============================================================================
template <int dim>
double symmetric_gradient_inner_product(const dealii::Tensor<2, dim>& T_u,
                                         const dealii::Tensor<2, dim>& T_v)
{
    double result = 0.0;
    
    for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
            result += T_u[i][j] * T_v[i][j];
    
    return result;
}

// Explicit instantiations
template double B_magnetic<2>(const dealii::Tensor<1, 2>&,
                               const dealii::Tensor<2, 2>&,
                               const dealii::Tensor<1, 2>&);
template double B_magnetic<3>(const dealii::Tensor<1, 3>&,
                               const dealii::Tensor<2, 3>&,
                               const dealii::Tensor<1, 3>&);

template double b_convection_skew<2>(const dealii::Tensor<1, 2>&,
                                      const dealii::Tensor<2, 2>&,
                                      const dealii::Tensor<2, 2>&,
                                      const dealii::Tensor<1, 2>&,
                                      const dealii::Tensor<1, 2>&);
template double b_convection_skew<3>(const dealii::Tensor<1, 3>&,
                                      const dealii::Tensor<2, 3>&,
                                      const dealii::Tensor<2, 3>&,
                                      const dealii::Tensor<1, 3>&,
                                      const dealii::Tensor<1, 3>&);

template dealii::Tensor<2, 2> symmetric_gradient<2>(const dealii::Tensor<2, 2>&);
template dealii::Tensor<2, 3> symmetric_gradient<3>(const dealii::Tensor<2, 3>&);

template double symmetric_gradient_inner_product<2>(const dealii::Tensor<2, 2>&,
                                                     const dealii::Tensor<2, 2>&);
template double symmetric_gradient_inner_product<3>(const dealii::Tensor<2, 3>&,
                                                     const dealii::Tensor<2, 3>&);

} // namespace TrilinearForms
