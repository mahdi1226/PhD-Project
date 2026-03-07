// ============================================================================
// physics/skew_forms.h — Skew-Symmetric Transport Forms
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015)
//
// Two skew-symmetric trilinear forms used throughout the FHD system:
//
// 1. NS convection b_h (Eq. 46, CG):
//    b_h(u, v, w) = Σ_T ∫_T [(u·∇)v · w + ½(∇·u)(v·w)] dx
//
// 2. DG magnetic transport B_h^m (Eq. 62):
//    B_h^m(U, M, Z) = Σ_T ∫_T [(U·∇)M·Z + ½(∇·U)(M·Z)] dx
//                     − Σ_F ∫_F (U·n⁻) [[M]]·{Z} dS
//
// Energy neutrality:
//   b_h(u, v, v)   = 0   (Prop. 3.3, CG: no face terms needed)
//   B_h^m(U, M, M) = 0   (Prop. 3.5, globally, not cell-local)
//
// Conventions:
//   "⁻" side = current cell ("here"), "⁺" side = neighbor ("there")
//   n = outward normal of "⁻" side
//   [[V]] = V⁻ − V⁺  (jump)
//   {W}  = ½(W⁻ + W⁺) (average)
// ============================================================================
#ifndef FHD_SKEW_FORMS_H
#define FHD_SKEW_FORMS_H

#include <deal.II/base/tensor.h>

// ============================================================================
//                    SCALAR VERSIONS (componentwise DG)
//
// Magnetization is assembled component-by-component (M_x, M_y separately).
// These operate on scalar fields V, W ∈ ℝ.
// ============================================================================

// ----------------------------------------------------------------------------
// Cell integrand: (U·∇V)W + ½(∇·U)(V·W)              [Eq. 62, line 1]
//
// U      — velocity at quadrature point
// div_U  — ∇·U at quadrature point
// V      — trial function value (M: differentiated/jumped)
// grad_V — ∇V (trial function gradient)
// W      — test function value (Z: averaged)
// ----------------------------------------------------------------------------
template <int dim>
inline double
skew_magnetic_cell_value_scalar(const dealii::Tensor<1, dim>& U,
                                double div_U,
                                double V,
                                const dealii::Tensor<1, dim>& grad_V,
                                double W)
{
    return (U * grad_V) * W + 0.5 * div_U * V * W;
}

// ----------------------------------------------------------------------------
// Face integrand: −(U·n) [[M]] {Z}                     [Eq. 62, line 2]
//
// Uses paper sign convention with minus/plus = ⁻/⁺.
// ----------------------------------------------------------------------------
inline double
skew_magnetic_face_value_scalar(double U_dot_n,
                                double V_minus, double V_plus,
                                double W_minus, double W_plus)
{
    const double jump_V = V_minus - V_plus;
    const double avg_W  = 0.5 * (W_minus + W_plus);
    return -U_dot_n * jump_V * avg_W;
}

// Alias with FEInterfaceValues naming (here = ⁻, there = ⁺)
inline double
skew_magnetic_face_value_scalar_interface(double U_dot_n,
                                          double V_here, double V_there,
                                          double W_here, double W_there)
{
    return skew_magnetic_face_value_scalar(U_dot_n,
                                           V_here, V_there,
                                           W_here, W_there);
}

// ----------------------------------------------------------------------------
// Upwind penalty: +½|U·n| [[V]]·[[W]]                 (stabilization)
//
// Adds numerical diffusion to the DG face flux. For a continuous exact
// solution M*, [[M*]]=0 so this term vanishes — does not affect MMS or
// formal accuracy, but is essential for DG transport convergence.
//
// Central flux:  −(U·n) [[V]] {W}
// Upwind flux:   −(U·n) [[V]] {W}  +  ½|U·n| [[V]] [[W]]
// ----------------------------------------------------------------------------
inline double
upwind_penalty_scalar(double abs_U_dot_n,
                      double V_here, double V_there,
                      double W_here, double W_there)
{
    const double jump_V = V_here - V_there;
    const double jump_W = W_here - W_there;
    return 0.5 * abs_U_dot_n * jump_V * jump_W;
}

// ============================================================================
//                    VECTOR VERSIONS (V, W ∈ ℝ^d)
//
// Used by Kelvin force computation and for reference/validation.
// ============================================================================

// ----------------------------------------------------------------------------
// Cell integrand: (U·∇V)·W + ½(∇·U)(V·W)              [Eq. 62, line 1]
// ----------------------------------------------------------------------------
template <int dim>
inline double
skew_magnetic_cell_value(const dealii::Tensor<1, dim>& U,
                         const dealii::Tensor<2, dim>& grad_U,
                         const dealii::Tensor<1, dim>& V,
                         const dealii::Tensor<2, dim>& grad_V,
                         const dealii::Tensor<1, dim>& W)
{
    // (U·∇V)[i] = Σ_j U[j] ∂V[i]/∂x_j
    dealii::Tensor<1, dim> U_dot_grad_V;
    for (unsigned int i = 0; i < dim; ++i)
    {
        U_dot_grad_V[i] = 0.0;
        for (unsigned int j = 0; j < dim; ++j)
            U_dot_grad_V[i] += U[j] * grad_V[i][j];
    }

    double div_U = 0.0;
    for (unsigned int d = 0; d < dim; ++d)
        div_U += grad_U[d][d];

    return (U_dot_grad_V * W) + 0.5 * div_U * (V * W);
}

// ----------------------------------------------------------------------------
// Face integrand: −(U·n) ([[V]] · {W})                  [Eq. 62, line 2]
// ----------------------------------------------------------------------------
template <int dim>
inline double
skew_magnetic_face_value(double U_dot_n,
                         const dealii::Tensor<1, dim>& V_minus,
                         const dealii::Tensor<1, dim>& V_plus,
                         const dealii::Tensor<1, dim>& W_minus,
                         const dealii::Tensor<1, dim>& W_plus)
{
    dealii::Tensor<1, dim> jump_V;
    dealii::Tensor<1, dim> avg_W;
    for (unsigned int d = 0; d < dim; ++d)
    {
        jump_V[d] = V_minus[d] - V_plus[d];
        avg_W[d]  = 0.5 * (W_minus[d] + W_plus[d]);
    }
    return -U_dot_n * (jump_V * avg_W);
}

// Alias with FEInterfaceValues naming
template <int dim>
inline double
skew_magnetic_face_value_interface(double U_dot_n,
                                   const dealii::Tensor<1, dim>& V_here,
                                   const dealii::Tensor<1, dim>& V_there,
                                   const dealii::Tensor<1, dim>& W_here,
                                   const dealii::Tensor<1, dim>& W_there)
{
    return skew_magnetic_face_value<dim>(U_dot_n,
                                         V_here, V_there,
                                         W_here, W_there);
}

// ============================================================================
//                    NS CONVECTION (Eq. 46, CG only)
//
// b_h(u, v, w) = Σ_T ∫_T [(u·∇)v · w + ½(∇·u)(v·w)] dx
//
// CG spaces have H¹ regularity so no face integrals are needed.
// Skew-symmetry gives b_h(u, v, v) = 0 without any CFL condition.
// ============================================================================
template <int dim>
inline double
skew_convection_cell_value(const dealii::Tensor<1, dim>& U,
                           const dealii::Tensor<2, dim>& grad_U,
                           const dealii::Tensor<1, dim>& V,
                           const dealii::Tensor<2, dim>& grad_V,
                           const dealii::Tensor<1, dim>& W)
{
    return skew_magnetic_cell_value<dim>(U, grad_U, V, grad_V, W);
}

// ============================================================================
//                    ANGULAR MOMENTUM / SCALAR CONVECTION
//
// Scalar skew-symmetric form for angular velocity, passive scalar, and
// Cahn-Hilliard convection (CG, no face terms).
// b_h(u, w, ξ) = Σ_T ∫_T [(u·∇)w · ξ + ½(∇·u)(w·ξ)] dx
//
// Algebraically identical to skew_magnetic_cell_value_scalar — single
// implementation avoids code duplication.
// ============================================================================
template <int dim>
inline double
skew_angular_convection_scalar(const dealii::Tensor<1, dim>& U,
                               double div_U,
                               double V,
                               const dealii::Tensor<1, dim>& grad_V,
                               double W)
{
    return skew_magnetic_cell_value_scalar<dim>(U, div_U, V, grad_V, W);
}

#endif // FHD_SKEW_FORMS_H
