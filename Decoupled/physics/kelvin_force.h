// ============================================================================
// physics/kelvin_force.h — Kelvin Force DG Skew Form B_h^m
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
//            Equation 38, Lemma 3.1
//
// Paper's DG skew form for magnetic body force (Eq. 38):
//
//   B_h^m(V, H, M) = Σ_T ∫_T [(M·∇)H · V + ½(∇·M)(H·V)] dx
//                   - Σ_F ∫_F (V·n⁻) [[H]] · {M} ds
//
// where:
//   V     = velocity test function (first slot)
//   H     = effective magnetizing field = ∇φ (second slot)
//   M     = magnetization, DG (third slot)
//   [[H]] = H⁻ − H⁺        (jump, minus-side normal convention)
//   {M}   = ½(M⁻ + M⁺)     (average)
//   n⁻    = outward normal of minus cell
//
// Energy identity (Lemma 3.1, p.502):
//   B_h^m(H, H, M) = 0  (magnetic energy cancellation)
//
// This is the DISCRETE version of the continuous trilinear form
//   B(m, h, u) = Σ_{i,j} ∫ m_i ∂h_j/∂x_i u_j dx    (Eq. 20)
// rewritten to exploit the skew form in M (using div(M)), NOT in V.
//
// ============================================================================
// THREE INVARIANTS FOR ENERGY IDENTITY:
//
// 1. V·n on faces: Use the full dot product V·n as the invariant quantity.
//    For split test functions V = (φ_ux, 0): V·n = φ_ux * n[0].
//    Component-wise is equivalent but the dot-product form is the invariant.
//
// 2. Single minus-side normal + jump: The DG skew form is defined ONCE per
//    interior face with a FIXED orientation (minus-side normal). Never flip
//    the normal or jump per cell — that breaks sign cancellation.
//
// 3. Elementwise div(M) + face repair: M is DG, so ∇·M is only elementwise.
//    The face term restores the missing integration-by-parts terms globally.
//    This is why the skew form works and why CG gradients would be WRONG
//    for computing div(M).
//
// ============================================================================
// Usage in NS momentum assembly (Eq. 42e RHS):
//
//   Cell loop:
//     rhs_ux[i] += μ₀ * cell_kernel_ux(...) * JxW
//     rhs_uy[i] += μ₀ * cell_kernel_uy(...) * JxW
//
//   Face loop (interior faces only):
//     rhs_ux[i] += μ₀ * face_kernel_ux(...) * JxW_face
//     rhs_uy[i] += μ₀ * face_kernel_uy(...) * JxW_face
//
// CRITICAL: div(M) must be computed from DG cell gradients (update_gradients
// on DG FEValues). Do NOT use CG gradients or projected fields.
//
// CRITICAL: H = ∇φ requires update_hessians on Poisson FEValues to get
// ∂H_i/∂x_j = ∂²φ/∂x_i∂x_j.
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
// Compute div(M) from DG component gradients
//
// CRITICAL (Invariant 3): M is DG, so ∇·M is only elementwise.
// The face term in B_h^m restores the global integration-by-parts.
//
// div(M) = ∂Mx/∂x + ∂My/∂y   (2D)
//        = ∂Mx/∂x + ∂My/∂y + ∂Mz/∂z  (3D)
//
// @param grad_Mx   Gradient of Mx component (from DG FEValues)
// @param grad_My   Gradient of My component (from DG FEValues)
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

// 3D overload: includes ∂Mz/∂z
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
// For split velocity test functions V = (φ_ux, 0) and V = (0, φ_uy),
// the kernel splits into separate ux and uy contributions:
//
//   ux: [(M·∇)H][0] * φ_ux + ½ div(M) * H[0] * φ_ux
//   uy: [(M·∇)H][1] * φ_uy + ½ div(M) * H[1] * φ_uy
//
// @param M_grad_H   (M·∇)H vector (from compute_M_grad_H)
// @param div_M      ∇·M scalar (from compute_div_M)
// @param H          Magnetic field H = ∇φ at quadrature point
// @param phi_ux     ux test function value
// @param phi_uy     uy test function value
// @param kelvin_ux  [OUT] ux contribution (before μ₀ * JxW)
// @param kelvin_uy  [OUT] uy contribution (before μ₀ * JxW)
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
    // (M·∇)H · V + ½(∇·M)(H·V) for split test functions
    kelvin_ux = M_grad_H[0] * phi_ux + 0.5 * div_M * H[0] * phi_ux;
    kelvin_uy = M_grad_H[1] * phi_uy + 0.5 * div_M * H[1] * phi_uy;
}

// ============================================================================
// FACE KERNEL — Eq. 38, second line
//
// Integrand: −(V·n⁻) [[H]] · {M}
//
// This is evaluated ONCE per interior face with a single minus-side normal.
// The kernel contributes to BOTH cells sharing the face (via the test
// functions from both sides).
//
// For split test functions:
//   V = (φ_ux, 0):  V·n = φ_ux * n[0]
//   V = (0, φ_uy):  V·n = φ_uy * n[1]
//
// CRITICAL (Invariant 2): jump_H and avg_M use the SAME minus-side
// orientation. Never recompute with flipped normal.
//
// @param phi_ux     ux test function value at face quadrature point
// @param phi_uy     uy test function value at face quadrature point
// @param normal     Outward normal of minus cell (n⁻)
// @param jump_H     [[H]] = H⁻ − H⁺ (always computed this way!)
// @param avg_M      {M} = ½(M⁻ + M⁺)
// @param kelvin_ux  [OUT] ux contribution (before μ₀ * JxW_face)
// @param kelvin_uy  [OUT] uy contribution (before μ₀ * JxW_face)
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
    // [[H]] · {M}
    double jump_H_dot_avg_M = 0.0;
    for (unsigned int d = 0; d < dim; ++d)
        jump_H_dot_avg_M += jump_H[d] * avg_M[d];

    // −(V·n) [[H]]·{M}
    // For V = (φ_ux, 0): V·n = φ_ux * n[0]
    // For V = (0, φ_uy): V·n = φ_uy * n[1]
    kelvin_ux = -(phi_ux * normal[0]) * jump_H_dot_avg_M;
    kelvin_uy = -(phi_uy * normal[1]) * jump_H_dot_avg_M;
}

// ============================================================================
// CONVENIENCE: Combined cell kernel with inline M_grad_H and div_M
//
// Computes everything from raw inputs (no precomputation needed).
// Useful for simple/test code. Production assembler should precompute
// M_grad_H and div_M once per quadrature point, then call cell_kernel
// for each test function.
//
// @param M          Magnetization vector
// @param hess_phi   Hessian of φ
// @param grad_Mx    DG gradient of Mx component
// @param grad_My    DG gradient of My component
// @param H          Magnetic field H = ∇φ
// @param phi_ux     ux test function value
// @param phi_uy     uy test function value
// @param kelvin_ux  [OUT] ux contribution (before μ₀ * JxW)
// @param kelvin_uy  [OUT] uy contribution (before μ₀ * JxW)
// ============================================================================
template <int dim>
inline void cell_kernel_full(
    const dealii::Tensor<1, dim>& M,
    const dealii::Tensor<2, dim>& hess_phi,
    const dealii::Tensor<1, dim>& grad_Mx,
    const dealii::Tensor<1, dim>& grad_My,
    const dealii::Tensor<1, dim>& H,
    double phi_ux,
    double phi_uy,
    double& kelvin_ux,
    double& kelvin_uy)
{
    const dealii::Tensor<1, dim> M_grad_H = compute_M_grad_H<dim>(M, hess_phi);
    const double div_M = compute_div_M<dim>(grad_Mx, grad_My);
    cell_kernel<dim>(M_grad_H, div_M, H, phi_ux, phi_uy, kelvin_ux, kelvin_uy);
}

// ============================================================================
// CONVENIENCE: Compute jump and average for face integrals
//
// Given field values on minus and plus sides, compute:
//   jump = V⁻ − V⁺
//   avg  = ½(V⁻ + V⁺)
//
// @param V_minus    Value on minus cell (current cell)
// @param V_plus     Value on plus cell (neighbor)
// @param jump       [OUT] V⁻ − V⁺
// @param avg        [OUT] ½(V⁻ + V⁺)
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

} // namespace KelvinForce

#endif // KELVIN_FORCE_H
