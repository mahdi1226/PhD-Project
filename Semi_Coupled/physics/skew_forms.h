// ============================================================================
// physics/skew_forms.h - Skew-Symmetric Forms (DG, paper-faithful)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Eq. 37: NS convection B_h(U, V, W)
// Eq. 57: Magnetization transport B_h^m(U, V, W)
//
// PAPER EQ. 57:
//   B_h^m(U,V,W) = Σ_T ∫_T [(U·∇)V·W + (1/2)(∇·U)(V·W)] dx
//                 - Σ_F ∫_F (U·n) [[V]]·{W} dS
//
// where:
//   [[V]] = V⁻ - V⁺  (jump)
//   {W}  = (W⁻ + W⁺)/2  (average)
//   n = outward normal of "-" side
//
// KEY PROPERTY: B_h^m(U, M, M) = 0 (energy neutrality, GLOBAL not local)
//
// The cancellation mechanism:
//   - Volume term produces boundary flux: +½∫_{∂T} (U·n)|M|² ds
//   - Face term produces: -(U·n)[[M]]·{M} = -½(U·n)(|M⁻|² - |M⁺|²)
//   - These cancel GLOBALLY when summed over all cells
//
// This file provides SCALAR versions: V, W ∈ R (for componentwise assembly)
//
// ============================================================================
#ifndef SKEW_FORMS_H
#define SKEW_FORMS_H

#include <deal.II/base/tensor.h>

// ============================================================================
// PAPER-NATIVE CONVENTIONS (Eq. 57)
//
// Orientation:
//   "-" side  = current cell ("here")
//   "+" side  = neighbor ("there")
//   n         = outward normal of "-" side (FEInterfaceValues::normal())
//
// Paper definitions:
//   [[V]] = V⁻ - V⁺
//   {W}   = 0.5*(W⁻ + W⁺)
//
// Face integrand:  -(U·n) * ([[V]] · {W})
// ============================================================================

// ============================================================================
//                         SCALAR VERSIONS
//              (for componentwise magnetization assembly)
// ============================================================================

/**
 * @brief Cell contribution for scalar DG transport (Eq. 57, first line)
 *
 * Integrand: (U·∇V)W + (1/2)(∇·U)(V W)
 *
 * @param U       Velocity vector at quadrature point
 * @param div_U   Divergence of U at quadrature point
 * @param V       Scalar field value (second slot in B_h^m)
 * @param grad_V  Gradient of V
 * @param W       Scalar field value (third slot in B_h^m)
 * @return        Integrand value (multiply by JxW)
 */
template <int dim>
inline double
skew_magnetic_cell_value_scalar(const dealii::Tensor<1, dim>& U,
                                double div_U,
                                double V,
                                const dealii::Tensor<1, dim>& grad_V,
                                double W)
{
    // (U·∇V)W + (1/2)(∇·U)(V W)
    const double U_dot_grad_V = U * grad_V;
    return U_dot_grad_V * W + 0.5 * div_U * V * W;
}

/**
 * @brief Face contribution for scalar DG transport (Eq. 57, second line)
 *
 * Paper Eq. 57: -(U·n) [[V]] {W}
 *
 * Convention: V = transported quantity (trial), W = test function
 *
 * where [[V]] = V⁻ - V⁺ and {W} = 0.5*(W⁻ + W⁺)
 *
 * @param U_dot_n_minus   U·n at face quadrature point
 * @param V_minus   V on "-" side (here)
 * @param V_plus    V on "+" side (there)
 * @param W_minus   W on "-" side (here)
 * @param W_plus    W on "+" side (there)
 * @return          Integrand value (multiply by JxW)
 */
inline double
skew_magnetic_face_value_scalar(double U_dot_n_minus,
                                double V_minus,
                                double V_plus,
                                double W_minus,
                                double W_plus)
{
    const double jump_V = V_minus - V_plus;           // [[V]] = V⁻ - V⁺
    const double avg_W  = 0.5 * (W_minus + W_plus);   // {W}
    return -U_dot_n_minus * jump_V * avg_W;
}

/**
 * @brief Face contribution using FEInterfaceValues naming (here/there)
 *
 * FEInterfaceValues convention:
 *   here  → "-" side (current cell)
 *   there → "+" side (neighbor)
 */
inline double
skew_magnetic_face_value_scalar_interface(double U_dot_n_minus,
                                          double V_here,
                                          double V_there,
                                          double W_here,
                                          double W_there)
{
    return skew_magnetic_face_value_scalar(U_dot_n_minus,
                                           /*V⁻*/ V_here,
                                           /*V⁺*/ V_there,
                                           /*W⁻*/ W_here,
                                           /*W⁺*/ W_there);
}

#endif // SKEW_FORMS_H