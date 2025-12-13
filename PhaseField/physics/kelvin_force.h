// ============================================================================
// physics/kelvin_force.h - Kelvin Force Computation (Header-Only)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Equation 14f, p.500; Section 3.1, p.500-501
//
// In the quasi-static limit, magnetization is instantaneous:
//   M = χ(θ) H
//
// The Kelvin force is:
//   F_mag = μ₀ χ(θ) (H·∇)H
//
// where:
//   H = -∇φ          (magnetic field from Poisson solve)
//   χ(θ) = χ₀ H(θ/ε) (phase-dependent susceptibility)
//
// Note: The paper uses κ_θ = χ_θ for susceptibility (same thing)
// ============================================================================
#ifndef KELVIN_FORCE_H
#define KELVIN_FORCE_H

#include <deal.II/base/tensor.h>

/**
 * @brief Compute magnetic field H = -∇φ
 *
 * @param grad_phi Gradient of magnetic potential ∇φ
 * @return H = -∇φ
 */
template <int dim>
inline dealii::Tensor<1, dim> compute_magnetic_field(
    const dealii::Tensor<1, dim>& grad_phi)
{
    dealii::Tensor<1, dim> H;
    for (unsigned int d = 0; d < dim; ++d)
        H[d] = -grad_phi[d];
    return H;
}

/**
 * @brief Compute gradient of magnetic field ∇H = -Hess(φ)
 *
 * @param hess_phi Hessian of magnetic potential
 * @return ∇H = -Hess(φ)
 */
template <int dim>
inline dealii::Tensor<2, dim> compute_grad_H(
    const dealii::Tensor<2, dim>& hess_phi)
{
    dealii::Tensor<2, dim> grad_H;
    for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
            grad_H[i][j] = -hess_phi[i][j];
    return grad_H;
}

/**
 * @brief Compute Kelvin force F_mag = μ₀ χ(θ) (H·∇)H
 *
 * This is the magnetic body force in the Navier-Stokes equation.
 *
 * Components: F_mag[i] = μ₀ χ_θ Σⱼ H[j] ∂H[i]/∂x[j]
 *
 * @param H Magnetic field H = -∇φ
 * @param grad_H Gradient of magnetic field ∇H = -Hess(φ)
 * @param chi_theta Phase-dependent susceptibility χ(θ) = χ₀ H(θ/ε)
 * @param mu_0 Magnetic permeability of free space (typically 1.0 in normalized units)
 * @return Kelvin force vector
 */
template <int dim>
inline dealii::Tensor<1, dim> compute_kelvin_force(
    const dealii::Tensor<1, dim>& H,
    const dealii::Tensor<2, dim>& grad_H,
    double chi_theta,
    double mu_0 = 1.0)
{
    dealii::Tensor<1, dim> F_mag;
    const double coeff = mu_0 * chi_theta;

    for (unsigned int i = 0; i < dim; ++i)
    {
        F_mag[i] = 0.0;
        for (unsigned int j = 0; j < dim; ++j)
            F_mag[i] += H[j] * grad_H[i][j];
        F_mag[i] *= coeff;
    }

    return F_mag;
}

/**
 * @brief Compute magnetization M = χ(θ) H (quasi-static limit)
 *
 * In the quasi-static limit (no relaxation dynamics), magnetization
 * responds instantaneously to the magnetic field.
 *
 * @param H Magnetic field H = -∇φ
 * @param chi_theta Phase-dependent susceptibility χ(θ)
 * @return Magnetization vector M
 */
template <int dim>
inline dealii::Tensor<1, dim> compute_magnetization(
    const dealii::Tensor<1, dim>& H,
    double chi_theta)
{
    dealii::Tensor<1, dim> M;
    for (unsigned int d = 0; d < dim; ++d)
        M[d] = chi_theta * H[d];
    return M;
}

/**
 * @brief Compute |H| (magnitude of magnetic field)
 *
 * @param H Magnetic field vector
 * @return |H|
 */
template <int dim>
inline double compute_H_magnitude(const dealii::Tensor<1, dim>& H)
{
    return H.norm();
}

#endif // KELVIN_FORCE_H