// ============================================================================
// physics/kelvin_force.h - Kelvin Force DG Skew Form B_h^m
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Equation 38, Lemma 3.1
//
// Paper's DG skew form for magnetic body force:
//
//   B_h^m(V, H, M) = Σ_T ∫_T [(M·∇)H · V + ½(∇·M)(H·V)] dx
//                  - Σ_F ∫_F (V·n⁻) [[H]] · {M} ds
//
// where:
//   [[H]] = H⁻ - H⁺        (jump, using minus-side normal convention)
//   {M}  = ½(M⁻ + M⁺)      (average)
//   n⁻   = outward normal of minus cell
//
// Energy identity (Lemma 3.1):
//   B_h^m(H, H, M) = 0  (magnetic energy cancellation)
//
// This requires the skew form in M (using div(M)), NOT in V!
//
// ============================================================================
// THREE INVARIANTS FOR ENERGY IDENTITY:
//
// 1. V·n on faces: Use the full dot product V·n as the invariant quantity.
//    This guarantees correctness under refactoring and preserves the energy
//    identity. Component-wise is equivalent for split test functions.
//
// 2. Single minus-side normal + jump: The DG skew form is defined once per
//    interior face with a FIXED orientation. Never flip normal or jump per
//    cell - that breaks sign cancellation. One normal, one jump, once.
//
// 3. Elementwise div(M) + face repair: M is DG, so ∇·M is only elementwise.
//    The face term restores the missing integration-by-parts terms globally.
//    This is why the skew form works and why CG gradients would be WRONG.
//
// ============================================================================
// Usage in NS momentum (Eq. 42e RHS):
//   rhs += μ₀ B_h^m(V, H^k, M^k)
//
// CRITICAL: div(M) must be computed from DG cell gradients (update_gradients
// on DG FEValues). Do NOT use CG gradients or projected fields. The face
// term "repairs" the global integration-by-parts structure via jumps.
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
// H = ∇φ, so ∂H/∂x_j = ∂²φ/∂x_i∂x_j = hess_phi[i][j]
//
// (M·∇)H[i] = Σ_j M[j] * ∂H[i]/∂x_j = Σ_j M[j] * hess_phi[i][j]
//
// @param M          Magnetization vector
// @param hess_phi   Hessian of φ (∂²φ/∂x_i∂x_j)
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
// The face term in B_h^m restores the missing integration-by-parts globally.
//
// grad_Mx and grad_My MUST come from:
//   - DG FEValues with update_gradients
//   - The SAME cell as the cell integral
//
// Do NOT use:
//   - CG gradients (wrong space)
//   - Projected or reconstructed fields
//   - Gradients from a different cell
//
// @param grad_Mx    ∇M_x from DG FEValues on current cell
// @param grad_My    ∇M_y from DG FEValues on current cell
// @return ∇·M = ∂M_x/∂x + ∂M_y/∂y (elementwise)
// ============================================================================
template <int dim>
inline double compute_div_M(
    const dealii::Tensor<1, dim>& grad_Mx,
    const dealii::Tensor<1, dim>& grad_My)
{
    static_assert(dim == 2, "Only 2D implemented");
    return grad_Mx[0] + grad_My[1];
}

// ============================================================================
// Compute jump [[H]] = H⁻ - H⁺
// ============================================================================
template <int dim>
inline dealii::Tensor<1, dim> compute_jump_H(
    const dealii::Tensor<1, dim>& H_minus,
    const dealii::Tensor<1, dim>& H_plus)
{
    return H_minus - H_plus;
}

// ============================================================================
// Compute average {M} = ½(M⁻ + M⁺)
// ============================================================================
template <int dim>
inline dealii::Tensor<1, dim> compute_avg_M(
    const dealii::Tensor<1, dim>& M_minus,
    const dealii::Tensor<1, dim>& M_plus)
{
    dealii::Tensor<1, dim> result;
    for (unsigned int d = 0; d < dim; ++d)
        result[d] = 0.5 * (M_minus[d] + M_plus[d]);
    return result;
}

// ============================================================================
// Build M vector from scalar DG components
// ============================================================================
template <int dim>
inline dealii::Tensor<1, dim> make_M_vector(double Mx, double My)
{
    static_assert(dim == 2, "Only 2D implemented");
    dealii::Tensor<1, dim> M;
    M[0] = Mx;
    M[1] = My;
    return M;
}

// ============================================================================
// Build H vector from scalar components (for symmetry with make_M_vector)
// ============================================================================
template <int dim>
inline dealii::Tensor<1, dim> make_H_vector(double Hx, double Hy)
{
    static_assert(dim == 2, "Only 2D implemented");
    dealii::Tensor<1, dim> H;
    H[0] = Hx;
    H[1] = Hy;
    return H;
}

// ============================================================================
// Cell kernel: (M·∇)H · V + ½(∇·M)(H·V)
//
// Computes both ux and uy contributions for a single test function.
//
// @param M_grad_H   (M·∇)H vector, from compute_M_grad_H()
// @param div_M      ∇·M, from compute_div_M() using DG gradients
// @param H          Magnetic field H = ∇φ
// @param phi_ux     Test function value for ux component
// @param phi_uy     Test function value for uy component
// @param mu_0       Permeability constant
// @param kelvin_ux  [OUT] Contribution to ux RHS (before *JxW)
// @param kelvin_uy  [OUT] Contribution to uy RHS (before *JxW)
// ============================================================================
template <int dim>
inline void cell_kernel(
    const dealii::Tensor<1, dim>& M_grad_H,
    double div_M,
    const dealii::Tensor<1, dim>& H,
    double phi_ux,
    double phi_uy,
    double mu_0,
    double& kelvin_ux,
    double& kelvin_uy)
{
    // (M·∇)H · V + ½(∇·M)(H·V)
    // For V = (φ_ux, 0): contribution is M_grad_H[0]*φ_ux + 0.5*div_M*H[0]*φ_ux
    // For V = (0, φ_uy): contribution is M_grad_H[1]*φ_uy + 0.5*div_M*H[1]*φ_uy
    kelvin_ux = mu_0 * (M_grad_H[0] * phi_ux + 0.5 * div_M * H[0] * phi_ux);
    kelvin_uy = mu_0 * (M_grad_H[1] * phi_uy + 0.5 * div_M * H[1] * phi_uy);
}

// ============================================================================
// Face kernel: -(V·n⁻) [[H]] · {M}
//
// Computes the face contribution for a SINGLE test function.
// Call this for both minus and plus cell test functions, always using
// the minus-side normal and the same jump_H = H⁻ - H⁺.
//
// INVARIANTS (preserve B_h^m(H,H,M) = 0):
//
//   1. Use V·n as the invariant quantity, not component-wise products.
//      For V = (φ_ux, 0): V·n = φ_ux * n[0]
//      For V = (0, φ_uy): V·n = φ_uy * n[1]
//      This guarantees correctness under refactoring.
//
//   2. Use SINGLE minus-side normal for both cells.
//      Never flip the normal or jump per cell - that breaks sign cancellation.
//      One normal, one jump, assembled once.
//
//   3. Elementwise div(M) + face repair.
//      M is DG, so ∇·M is only elementwise (cell term).
//      This face term restores the missing integration-by-parts terms globally.
//      Using CG gradients for div(M) would be WRONG.
//
// Assembly pattern:
//   for each interior face (processed once):
//     normal_minus = outward normal of minus cell
//     jump_H = H⁻ - H⁺
//     avg_M = ½(M⁻ + M⁺)
//     for each test function on minus cell:
//       face_kernel(..., phi_minus, normal_minus, jump_H, avg_M, ...)
//     for each test function on plus cell:
//       face_kernel(..., phi_plus, normal_minus, jump_H, avg_M, ...)
//
// @param phi_ux     Test function value for ux (from either cell)
// @param phi_uy     Test function value for uy (from either cell)
// @param normal     Outward normal of MINUS cell (always use this!)
// @param jump_H     [[H]] = H⁻ - H⁺ (always computed this way!)
// @param avg_M      {M} = ½(M⁻ + M⁺)
// @param mu_0       Permeability constant
// @param kelvin_ux  [OUT] Contribution to ux RHS (before *JxW_face)
// @param kelvin_uy  [OUT] Contribution to uy RHS (before *JxW_face)
// ============================================================================
template <int dim>
inline void face_kernel(
    double phi_ux,
    double phi_uy,
    const dealii::Tensor<1, dim>& normal,
    const dealii::Tensor<1, dim>& jump_H,
    const dealii::Tensor<1, dim>& avg_M,
    double mu_0,
    double& kelvin_ux,
    double& kelvin_uy)
{
    // [[H]] · {M}
    double jump_H_dot_avg_M = 0.0;
    for (unsigned int d = 0; d < dim; ++d)
        jump_H_dot_avg_M += jump_H[d] * avg_M[d];

    // -(V·n) [[H]]·{M}
    // For V = (φ_ux, 0): V·n = φ_ux * n[0]
    // For V = (0, φ_uy): V·n = φ_uy * n[1]
    // This is the invariant form; component-wise is equivalent for split test functions
    kelvin_ux = -mu_0 * (phi_ux * normal[0]) * jump_H_dot_avg_M;
    kelvin_uy = -mu_0 * (phi_uy * normal[1]) * jump_H_dot_avg_M;
}

} // namespace KelvinForce

#endif // KELVIN_FORCE_H