// ============================================================================
// physics/kelvin_force.h — Kelvin Force DG Skew Form B_h^m
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015)
//            Equation 38, Lemma 3.1
//
// Paper's DG skew form for magnetic body force (Eq. 38):
//
//   B_h^m(V, H, M) = Σ_T ∫_T [(M·∇)H · V + ½(∇·M)(H·V)] dx
//                   − Σ_F ∫_F (V·n⁻) [[H]] · {M} ds
//
// where:
//   V     = velocity test function (first slot)
//   H     = total magnetizing field h_a + ∇φ (second slot)
//   M     = magnetization, DG (third slot)
//   [[H]] = H⁻ − H⁺        (jump, minus-side normal convention)
//   {M}   = ½(M⁻ + M⁺)     (average)
//   n⁻    = outward normal of minus cell
//
// Energy identity (Lemma 3.1):
//   B_h^m(H, H, M) = 0  (magnetic energy cancellation)
//
// THREE INVARIANTS FOR ENERGY IDENTITY:
//
// 1. V·n on faces: Use the full dot product V·n.
//    For split test V = (φ_ux, 0): V·n = φ_ux * n[0].
//
// 2. Single minus-side normal + jump: Defined ONCE per face with FIXED
//    orientation. Never flip the normal or jump per cell.
//
// 3. Elementwise div(M) + face repair: M is DG, so ∇·M is elementwise.
//    The face term restores global integration-by-parts.
//
// Usage in NS momentum assembly (Eq. 42e RHS):
//   Cell loop:  rhs[i] += μ₀ * cell_kernel(...) * JxW
//   Face loop:  rhs[i] += μ₀ * face_kernel(...) * JxW_face
//
// CRITICAL: H = h_a + ∇φ. For ∇φ, use hessian to get (M·∇)(∇φ).
//           For h_a, use ∇h_a from applied_field.h.
// ============================================================================
#ifndef FHD_KELVIN_FORCE_H
#define FHD_KELVIN_FORCE_H

#include <deal.II/base/tensor.h>

namespace KelvinForce
{

// ============================================================================
// Compute (M·∇)H from M and ∇H (gradient of total field)
//
// (M·∇)H[i] = Σ_j M[j] * ∂H_i/∂x_j = Σ_j M[j] * grad_H[i][j]
//
// When H = ∇φ:   grad_H[i][j] = ∂²φ/∂x_i∂x_j = hess_phi[i][j]
// When H = h_a:   grad_H = ∇h_a from applied_field.h
// Total:          grad_H = hess_phi + grad_h_a
// ============================================================================
template <int dim>
inline dealii::Tensor<1, dim> compute_M_grad_H(
    const dealii::Tensor<1, dim>& M,
    const dealii::Tensor<2, dim>& grad_H)
{
    dealii::Tensor<1, dim> result;
    for (unsigned int i = 0; i < dim; ++i)
    {
        result[i] = 0.0;
        for (unsigned int j = 0; j < dim; ++j)
            result[i] += M[j] * grad_H[i][j];
    }
    return result;
}

// ============================================================================
// Compute div(M) from DG component gradients
//
// div(M) = ∂M_x/∂x + ∂M_y/∂y   (2D)
// ============================================================================
template <int dim>
inline double compute_div_M(
    const dealii::Tensor<1, dim>& grad_Mx,
    const dealii::Tensor<1, dim>& grad_My)
{
    static_assert(dim >= 2, "Kelvin force requires dim >= 2");
    return grad_Mx[0] + grad_My[1];
}

// 3D overload
template <int dim>
inline double compute_div_M(
    const dealii::Tensor<1, dim>& grad_Mx,
    const dealii::Tensor<1, dim>& grad_My,
    const dealii::Tensor<1, dim>& grad_Mz)
{
    static_assert(dim == 3, "3-component div(M) requires dim == 3");
    return grad_Mx[0] + grad_My[1] + grad_Mz[2];
}

// ============================================================================
// CELL KERNEL — Eq. 38, first line
//
// Integrand: (M·∇)H · V + ½(∇·M)(H·V)
//
// For split velocity test functions V = (φ_ux, 0) and V = (0, φ_uy):
//   ux: [(M·∇)H][0] * φ_ux + ½ div(M) * H[0] * φ_ux
//   uy: [(M·∇)H][1] * φ_uy + ½ div(M) * H[1] * φ_uy
// ============================================================================
template <int dim>
inline void cell_kernel(
    const dealii::Tensor<1, dim>& M_grad_H,
    double div_M,
    const dealii::Tensor<1, dim>& H,
    double phi_ux,
    double phi_uy,
    double& kelvin_ux,
    double& kelvin_uy)
{
    kelvin_ux = M_grad_H[0] * phi_ux + 0.5 * div_M * H[0] * phi_ux;
    kelvin_uy = M_grad_H[1] * phi_uy + 0.5 * div_M * H[1] * phi_uy;
}

// ============================================================================
// FACE KERNEL — Eq. 38, second line
//
// Integrand: −(V·n⁻) [[H]] · {M}
//
// Evaluated ONCE per interior face with minus-side normal.
// For split test functions:
//   V = (φ_ux, 0): V·n = φ_ux * n[0]
//   V = (0, φ_uy): V·n = φ_uy * n[1]
// ============================================================================
template <int dim>
inline void face_kernel(
    double phi_ux,
    double phi_uy,
    const dealii::Tensor<1, dim>& normal,
    const dealii::Tensor<1, dim>& jump_H,
    const dealii::Tensor<1, dim>& avg_M,
    double& kelvin_ux,
    double& kelvin_uy)
{
    double jump_H_dot_avg_M = 0.0;
    for (unsigned int d = 0; d < dim; ++d)
        jump_H_dot_avg_M += jump_H[d] * avg_M[d];

    kelvin_ux = -(phi_ux * normal[0]) * jump_H_dot_avg_M;
    kelvin_uy = -(phi_uy * normal[1]) * jump_H_dot_avg_M;
}

// ============================================================================
// CONVENIENCE: Combined cell kernel with inline M_grad_H and div_M
// ============================================================================
template <int dim>
inline void cell_kernel_full(
    const dealii::Tensor<1, dim>& M,
    const dealii::Tensor<2, dim>& grad_H,
    const dealii::Tensor<1, dim>& grad_Mx,
    const dealii::Tensor<1, dim>& grad_My,
    const dealii::Tensor<1, dim>& H,
    double phi_ux,
    double phi_uy,
    double& kelvin_ux,
    double& kelvin_uy)
{
    const dealii::Tensor<1, dim> M_grad_H = compute_M_grad_H<dim>(M, grad_H);
    const double div_M = compute_div_M<dim>(grad_Mx, grad_My);
    cell_kernel<dim>(M_grad_H, div_M, H, phi_ux, phi_uy, kelvin_ux, kelvin_uy);
}

// ============================================================================
// CONVENIENCE: Compute jump and average for face integrals
// ============================================================================
template <int dim>
inline void compute_jump_and_average(
    const dealii::Tensor<1, dim>& V_minus,
    const dealii::Tensor<1, dim>& V_plus,
    dealii::Tensor<1, dim>& jump,
    dealii::Tensor<1, dim>& avg)
{
    for (unsigned int d = 0; d < dim; ++d)
    {
        jump[d] = V_minus[d] - V_plus[d];
        avg[d]  = 0.5 * (V_minus[d] + V_plus[d]);
    }
}

// ============================================================================
// Magnetic torque: m × h (for angular momentum equation)
//
// In 2D, the cross product m × h is a scalar (pseudo-vector):
//   m × h = m_x * h_y − m_y * h_x
//
// Used in Eq. 42f: μ₀(m × h, ξ)
// ============================================================================
inline double magnetic_torque_2d(
    double m_x, double m_y,
    double h_x, double h_y)
{
    return m_x * h_y - m_y * h_x;
}

} // namespace KelvinForce

#endif // FHD_KELVIN_FORCE_H
