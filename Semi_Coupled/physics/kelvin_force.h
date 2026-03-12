// ============================================================================
// physics/kelvin_force.h - Kelvin Force DG Skew Form B_h^m
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Equations 38, 42e, 57
//
// Paper's trilinear form (Eq. 57):
//
//   B_h^m(U, V, W) = Σ_T ∫_T [(U·∇)V·W + ½ div(U) V·W] dx
//                   - Σ_F ∫_F ([[V]]·{W})(U·n_F) dS
//
// Skew symmetry (Eq. 38): B_h^m(U, V, W) = -B_h^m(U, W, V)
//
// In the NS equation (42e), the Kelvin force is:
//
//   μ₀ B_h^m(V, H^k, M^k)
//
// where V is the velocity TEST FUNCTION (first argument!). Expanding:
//
//   Cell: (V·∇)H·M + ½ div(V)(H·M)
//   Face: -([[H]]·{M})(V·n_F)
//
// NOTE: The face term is an O(h) correction (Nochetto Remark 5.4).
// Although φ is CG, ∇φ is discontinuous across faces, so [[H]]≠0.
// However, [[H]] = O(h) for smooth φ, making this a higher-order term.
// We omit it here — the cell term alone provides the leading-order force.
//
// Since H=∇φ, ∇H is symmetric (Hessian), so (V·∇)H·M = (M·∇)H·V.
// The cell term is thus: (M·∇)H·V + ½ div(V)(H·M)
//
// CRITICAL: The stabilization uses div(V) (test function), NOT div(M).
// This is essential for energy stability (Proposition 4.1) because when
// V = U^k, the term B_h^m(U^k, H, M) has U^k as first argument, matching
// the magnetization transport B_h^m(U^k, Z, M^k) for cancellation.
//
// ============================================================================
// Usage in NS momentum (Eq. 42e RHS):
//   rhs += μ₀ B_h^m(V, H^k, M^k)  [cell term only]
//
// Requires: velocity test function gradients (update_gradients on vel FEValues)
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
// Cell kernel: (M·∇)H · V + ½ div(V)(H·M)     [paper's B_h^m(V, H, M)]
//
// Computes both ux and uy contributions for a single test function.
//
// For V = (φ_ux, 0):  div(V) = ∂φ_ux/∂x = grad_phi_ux[0]
// For V = (0, φ_uy):  div(V) = ∂φ_uy/∂y = grad_phi_uy[1]
//
// @param M_grad_H     (M·∇)H vector, from compute_M_grad_H()
// @param H            Magnetic field H = ∇φ
// @param M            Magnetization vector
// @param phi_ux       Test function value for ux component
// @param phi_uy       Test function value for uy component
// @param grad_phi_ux  Gradient of ux test function (∂φ_ux/∂x, ∂φ_ux/∂y)
// @param grad_phi_uy  Gradient of uy test function (∂φ_uy/∂x, ∂φ_uy/∂y)
// @param mu_0         Permeability constant
// @param kelvin_ux    [OUT] Contribution to ux RHS (before *JxW)
// @param kelvin_uy    [OUT] Contribution to uy RHS (before *JxW)
// ============================================================================
template <int dim>
inline void cell_kernel(
    const dealii::Tensor<1, dim>& M_grad_H,
    const dealii::Tensor<1, dim>& H,
    const dealii::Tensor<1, dim>& M,
    double phi_ux,
    double phi_uy,
    const dealii::Tensor<1, dim>& grad_phi_ux,
    const dealii::Tensor<1, dim>& grad_phi_uy,
    double mu_0,
    double& kelvin_ux,
    double& kelvin_uy)
{
    // H · M scalar product (same for both components)
    double H_dot_M = 0.0;
    for (unsigned int d = 0; d < dim; ++d)
        H_dot_M += H[d] * M[d];

    // (M·∇)H · V + ½ div(V)(H·M)
    // For V = (φ_ux, 0): div(V) = ∂φ_ux/∂x = grad_phi_ux[0]
    kelvin_ux = mu_0 * (M_grad_H[0] * phi_ux + 0.5 * grad_phi_ux[0] * H_dot_M);
    // For V = (0, φ_uy): div(V) = ∂φ_uy/∂y = grad_phi_uy[1]
    kelvin_uy = mu_0 * (M_grad_H[1] * phi_uy + 0.5 * grad_phi_uy[1] * H_dot_M);
}

} // namespace KelvinForce

#endif // KELVIN_FORCE_H