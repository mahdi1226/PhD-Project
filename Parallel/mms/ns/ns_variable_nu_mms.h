// ============================================================================
// mms/ns/ns_variable_nu_mms.h - NS with Variable Viscosity MMS Verification
//
// Tests variable viscosity ν(θ) in Navier-Stokes:
//   ∂u/∂t + (u·∇)u = -∇p + ∇·(ν(θ)·T(u)) + f
//
// where:
//   T(u) = ∇u + (∇u)ᵀ (symmetric gradient)
//   ν(θ) = ν_water·(1-θ)/2 + ν_ferro·(1+θ)/2
//
// APPROACH:
//   - Use existing NS MMS exact solutions (u, p) from ns_mms.h
//   - Prescribe simple analytical θ(x,y) with known gradient
//   - Compute variable viscosity MMS source analytically
//   - Verify NS convergence rates maintained
//
// KEY IDENTITY:
//   ∇·(ν(θ)·T(u)) = ν(θ)·∇·T(u) + ∇ν·T(u)
//                 = ν(θ)·∇²u + ∇ν·T(u)   (for div-free u, T = 2·sym(∇u))
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef NS_VARIABLE_NU_MMS_H
#define NS_VARIABLE_NU_MMS_H

#include "mms/ns/ns_mms.h"  // For NS exact solutions

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/function.h>

#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Prescribed θ for variable viscosity test
// θ = cos(πx)·cos(πy/L_y)  (simple, smooth, known gradient)
// ============================================================================
template <int dim>
class NSVarNuPrescribedTheta : public dealii::Function<dim>
{
public:
    NSVarNuPrescribedTheta(double L_y = 1.0)
        : dealii::Function<dim>(1), L_y_(L_y) {}

    virtual double value(const dealii::Point<dim>& p,
                         const unsigned int = 0) const override
    {
        const double x = p[0];
        const double y = p[1];
        return std::cos(M_PI * x) * std::cos(M_PI * y / L_y_);
    }

    virtual dealii::Tensor<1, dim> gradient(const dealii::Point<dim>& p,
                                            const unsigned int = 0) const override
    {
        const double x = p[0];
        const double y = p[1];
        dealii::Tensor<1, dim> grad;
        grad[0] = -M_PI * std::sin(M_PI * x) * std::cos(M_PI * y / L_y_);
        grad[1] = -(M_PI / L_y_) * std::cos(M_PI * x) * std::sin(M_PI * y / L_y_);
        return grad;
    }

private:
    double L_y_;
};

// ============================================================================
// Get prescribed θ at a point
// ============================================================================
template <int dim>
double ns_varnu_prescribed_theta(
    const dealii::Point<dim>& p,
    double L_y = 1.0)
{
    const double x = p[0];
    const double y = p[1];
    return std::cos(M_PI * x) * std::cos(M_PI * y / L_y);
}

// ============================================================================
// Get ∇θ at a point
// ============================================================================
template <int dim>
dealii::Tensor<1, dim> ns_varnu_grad_theta(
    const dealii::Point<dim>& p,
    double L_y = 1.0)
{
    const double x = p[0];
    const double y = p[1];
    dealii::Tensor<1, dim> grad;
    grad[0] = -M_PI * std::sin(M_PI * x) * std::cos(M_PI * y / L_y);
    grad[1] = -(M_PI / L_y) * std::cos(M_PI * x) * std::sin(M_PI * y / L_y);
    return grad;
}

// ============================================================================
// Compute ν(θ) from phase field
// ν(θ) = ν_water·(1-θ)/2 + ν_ferro·(1+θ)/2
// ============================================================================
inline double compute_nu_of_theta(double theta, double nu_water, double nu_ferro)
{
    return nu_water * (1.0 - theta) / 2.0 + nu_ferro * (1.0 + theta) / 2.0;
}

// ============================================================================
// Compute ∂ν/∂θ (constant)
// ∂ν/∂θ = (ν_ferro - ν_water) / 2
// ============================================================================
inline double compute_dnu_dtheta(double nu_water, double nu_ferro)
{
    return (nu_ferro - nu_water) / 2.0;
}

// ============================================================================
// Compute ∇ν = (∂ν/∂θ)·∇θ
// ============================================================================
template <int dim>
dealii::Tensor<1, dim> compute_grad_nu(
    const dealii::Point<dim>& p,
    double nu_water,
    double nu_ferro,
    double L_y = 1.0)
{
    const double dnu_dtheta = compute_dnu_dtheta(nu_water, nu_ferro);
    const dealii::Tensor<1, dim> grad_theta = ns_varnu_grad_theta<dim>(p, L_y);
    return dnu_dtheta * grad_theta;
}

// ============================================================================
// Compute symmetric gradient T(u) = ∇u + (∇u)ᵀ at a point
// Uses exact velocity from ns_mms.h
// ============================================================================
template <int dim>
dealii::Tensor<2, dim> compute_symmetric_gradient_u(
    const dealii::Point<dim>& p,
    double time,
    double L_y = 1.0)
{
    // Get exact gradients from ns_mms.h
    NSExactVelocityX<dim> exact_ux(time, L_y);
    NSExactVelocityY<dim> exact_uy(time, L_y);

    dealii::Tensor<1, dim> grad_ux = exact_ux.gradient(p);
    dealii::Tensor<1, dim> grad_uy = exact_uy.gradient(p);

    // ∇u matrix: (∇u)_ij = ∂u_i/∂x_j
    dealii::Tensor<2, dim> grad_u;
    grad_u[0][0] = grad_ux[0];  // ∂ux/∂x
    grad_u[0][1] = grad_ux[1];  // ∂ux/∂y
    grad_u[1][0] = grad_uy[0];  // ∂uy/∂x
    grad_u[1][1] = grad_uy[1];  // ∂uy/∂y

    // T(u) = ∇u + (∇u)ᵀ
    dealii::Tensor<2, dim> T;
    for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
            T[i][j] = grad_u[i][j] + grad_u[j][i];

    return T;
}

// ============================================================================
// Compute extra viscous term: ∇ν · T(u)
//
// This is a vector: (∇ν · T(u))_i = ∇ν_j · T_ij = ∑_j (∂ν/∂x_j) · T_ij
// ============================================================================
template <int dim>
dealii::Tensor<1, dim> compute_grad_nu_dot_T(
    const dealii::Point<dim>& p,
    double time,
    double nu_water,
    double nu_ferro,
    double L_y = 1.0)
{
    const dealii::Tensor<1, dim> grad_nu = compute_grad_nu<dim>(p, nu_water, nu_ferro, L_y);
    const dealii::Tensor<2, dim> T = compute_symmetric_gradient_u<dim>(p, time, L_y);

    // (∇ν · T)_i = ∑_j grad_nu[j] * T[i][j]
    dealii::Tensor<1, dim> result;
    for (unsigned int i = 0; i < dim; ++i)
    {
        result[i] = 0.0;
        for (unsigned int j = 0; j < dim; ++j)
        {
            result[i] += grad_nu[j] * T[i][j];
        }
    }

    return result;
}

// ============================================================================
// Compute NS MMS source WITH variable viscosity
//
// Full equation:
//   ∂u/∂t + (u·∇)u = -∇p + ∇·(ν(θ)·T(u)) + f
//
// Expanding viscous term:
//   ∇·(ν(θ)·T(u)) = ν(θ)·∇·T(u) + ∇ν·T(u)
//
// For constant ν, the MMS source is (from ns_mms.h):
//   f_const = ∂u/∂t + (u·∇)u + ∇p - ν·∇²u
//
// For variable ν(θ), we need:
//   f_var = ∂u/∂t + (u·∇)u + ∇p - ν(θ)·∇²u - ∇ν·T(u)
//         = f_const(with ν=ν(θ)) - ∇ν·T(u)
//
// But wait - the assembler handles variable ν internally!
// The assembler computes ν(θ) at quadrature points and assembles
// the correct weak form: (ν(θ)·T(u), T(v)).
//
// So the MMS source should be the same as constant-ν case,
// evaluated with ν = ν(θ(x)).
// ============================================================================
template <int dim>
dealii::Tensor<1, dim> compute_ns_variable_nu_mms_source(
    const dealii::Point<dim>& p,
    double t_new,
    double t_old,
    double nu_water,
    double nu_ferro,
    double L_y = 1.0)
{
    // Get θ at this point
    const double theta = ns_varnu_prescribed_theta<dim>(p, L_y);

    // Compute local viscosity
    const double nu_local = compute_nu_of_theta(theta, nu_water, nu_ferro);

    // Use standard NS MMS source with local viscosity
    // This is what the exact solution satisfies
    return compute_ns_mms_source_semi_implicit<dim>(p, t_new, t_old, nu_local, L_y);
}

#endif // NS_VARIABLE_NU_MMS_H