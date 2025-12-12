// ============================================================================
// utilities/trilinear_forms.h - Trilinear Forms for NS and Magnetization
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Equations 20-21, 37-38, p.502-504
// ============================================================================
#ifndef TRILINEAR_FORMS_H
#define TRILINEAR_FORMS_H

#include <deal.II/base/tensor.h>

/**
 * @brief Trilinear forms used in NS and magnetization equations
 *
 * Magnetic trilinear form (Eq. 20, p.502):
 *   B(m, h, u) = Σ_{i,j} ∫_Ω m_i (∂h_i/∂x_j) u_j dx
 *
 * Key identity (Lemma 3.1, Eq. 21, p.502):
 *   B(u, m, h) = -B(m, h, u)
 *
 * Discrete forms (Eq. 37-38, p.504):
 *   B_h(U, V, W) = -B_h(U, W, V)      (skew-symmetric)
 *   B_h^m(U, V, W) = -B_h^m(U, W, V)  (skew-symmetric)
 *
 * These skew-symmetry properties are crucial for energy stability.
 */
namespace TrilinearForms
{

/**
 * @brief Compute B(m, h, u) at a quadrature point
 *
 * B(m, h, u) = Σ_{i,j} m_i (∂h_i/∂x_j) u_j
 *
 * Eq. 20, p.502
 *
 * @param m Magnetization at point
 * @param grad_h Gradient of magnetic field at point
 * @param u Velocity at point
 */
template <int dim>
double B_magnetic(const dealii::Tensor<1, dim>& m,
                  const dealii::Tensor<2, dim>& grad_h,
                  const dealii::Tensor<1, dim>& u);

/**
 * @brief Compute skew-symmetric convection form b(u, v, w)
 *
 * Standard skew-symmetric form:
 *   b(u, v, w) = ½[(u·∇)v, w) - ((u·∇)w, v)]
 *
 * Satisfies: b(u, v, v) = 0 for div(u) = 0
 *
 * @param u Convecting velocity at point
 * @param grad_v Gradient of convected velocity
 * @param grad_w Gradient of test function
 * @param v Convected velocity at point (for skew-symmetric part)
 * @param w Test velocity at point (for skew-symmetric part)
 */
template <int dim>
double b_convection_skew(const dealii::Tensor<1, dim>& u,
                         const dealii::Tensor<2, dim>& grad_v,
                         const dealii::Tensor<2, dim>& grad_w,
                         const dealii::Tensor<1, dim>& v,
                         const dealii::Tensor<1, dim>& w);

/**
 * @brief Compute symmetric gradient T(u) = ½(∇u + ∇u^T)
 *
 * Rate-of-strain tensor (Eq. 14e, p.501)
 *
 * @param grad_u Gradient of velocity
 */
template <int dim>
dealii::Tensor<2, dim> symmetric_gradient(const dealii::Tensor<2, dim>& grad_u);

/**
 * @brief Compute (T(u), T(v)) inner product
 *
 * @param T_u Symmetric gradient of u
 * @param T_v Symmetric gradient of v
 */
template <int dim>
double symmetric_gradient_inner_product(const dealii::Tensor<2, dim>& T_u,
                                         const dealii::Tensor<2, dim>& T_v);

} // namespace TrilinearForms

#endif // TRILINEAR_FORMS_H
