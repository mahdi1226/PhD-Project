// ============================================================================
// physics/skew_forms.h — Skew-Symmetric Transport Forms
//
// Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
//
// Eq. 57 — Magnetization DG transport:
//   B_h^m(U, M, Z) = Σ_T ∫_T [(U·∇)M·Z + ½(∇·U)(M·Z)] dx
//                    - Σ_F ∫_F (U·n⁻) [[M]]·{Z} dS
//
// Eq. 37 — NS convection (CG, Temam/skew form):
//   B_h(U, V, W) = Σ_T ∫_T [(U·∇)V·W + ½(∇·U)(V·W)] dx
//
// Conventions:
//   "⁻" side = current cell ("here"), "⁺" side = neighbor ("there")
//   n = outward normal of "⁻" side
//   [[V]] = V⁻ - V⁺  (jump)
//   {W}  = ½(W⁻ + W⁺) (average)
//
// Energy neutrality: B_h^m(U, M, M) = 0 globally (not cell-local).
// ============================================================================
#ifndef SKEW_FORMS_H
#define SKEW_FORMS_H

#include <deal.II/base/tensor.h>

// ============================================================================
//                    SCALAR VERSIONS (componentwise DG)
//
// Magnetization is assembled component-by-component (Mx, My separately).
// These operate on scalar fields V, W ∈ ℝ.
// ============================================================================

// ----------------------------------------------------------------------------
// Cell integrand: (U·∇V)W + ½(∇·U)(V·W)              [Eq. 57, line 1]
//
// U      — velocity at quadrature point
// div_U  — ∇·U at quadrature point
// V      — trial function value (second slot of B_h^m, M — differentiated/jumped)
// grad_V — ∇V (trial function gradient)
// W      — test function value (third slot of B_h^m, Z — averaged)
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
// Face integrand: -(U·n) [[M]] {Z}                    [Eq. 57, line 2]
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

// ============================================================================
//                    VECTOR VERSIONS (V, W ∈ ℝ^d)
//
// Used by Kelvin force computation and for reference/validation.
// ============================================================================

// ----------------------------------------------------------------------------
// Cell integrand: (U·∇V)·W + ½(∇·U)(V·W)             [Eq. 57, line 1]
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
// Face integrand: -(U·n) ([[V]] · {W})                 [Eq. 57, line 2]
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
//                    NS CONVECTION (Eq. 37, CG only)
//
// Same formula as vector magnetization cell term. CG spaces have
// H¹ regularity so no face integrals are needed.
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

#endif // SKEW_FORMS_H