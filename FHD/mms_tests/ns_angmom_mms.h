// ============================================================================
// mms_tests/ns_angmom_mms.h — Coupled NS + Angular Momentum MMS
//
// PAPER EQUATIONS 42e + 42f (Nochetto, Salgado & Tomas, arXiv:1511.04381):
//
// NS (Eq. 42e, micropolar coupling):
//   (u^k/τ, v) + ν_eff(D(u^k), D(v)) - (p^k, ∇·v) + (∇·u^k, q)
//     = (u^{k-1}/τ, v) + 2ν_r(w^{k-1}, ∇×v) + (f_u, v)
//
// Angular momentum (Eq. 42f, curl coupling):
//   j(w^k/τ, z) + c₁(∇w^k, ∇z) + 4ν_r(w^k, z)
//     = j(w^{k-1}/τ, z) + 2ν_r(∇×u^k, z) + (f_w, z)
//
// ALGORITHM (sequential, no Picard needed):
//   1. Solve NS for (u^k, p^k) using w^{k-1} from previous time step
//   2. Solve AngMom for w^k using u^k from step 1
//
// EXACT SOLUTIONS (reusing standalone solutions):
//   ux*(x,y,t) = t·π·sin²(πx)·sin(2πy)
//   uy*(x,y,t) = -t·π·sin(2πx)·sin²(πy)
//   p*(x,y,t)  = t·cos(πx)·cos(πy)
//   w*(x,y,t)  = t·sin(πx)·sin(πy)
//
// MMS SOURCES (backward Euler, analytical coupling):
//   f_u = (u* - u*_old)/τ - (ν_eff/2)Δu* + ∇p* - 2ν_r·curl_vec(w*)
//   f_w = j(w* - w*_old)/τ - c₁Δw* + 4ν_r w* - 2ν_r·curl_scalar(u*)
//
// where:
//   curl_vec(w) = (∂w/∂y, -∂w/∂x) in 2D  [scalar → vector]
//   curl_scalar(u) = ∂uy/∂x - ∂ux/∂y     [vector → scalar]
//
// NOTE: The MMS sources use ANALYTICAL coupling (exact w*, u*) rather than
// discrete values. The pollution error is:
//   NS:    O(h³) in dual norm → preserves O(h³) L2, O(h²) H1 for velocity
//   AngMom: O(h²) in dual norm → by Aubin-Nitsche gives O(h³) L2, O(h²) H1
//
// EXPECTED CONVERGENCE:
//   U_L2: O(h³) — rate ≈ 3.0    (CG Q₂)
//   U_H1: O(h²) — rate ≈ 2.0
//   p_L2: O(h²) — rate ≈ 2.0    (DG P₁)
//   w_L2: O(h³) — rate ≈ 3.0    (CG Q₂)
//   w_H1: O(h²) — rate ≈ 2.0
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015)
// ============================================================================
#ifndef FHD_NS_ANGMOM_MMS_H
#define FHD_NS_ANGMOM_MMS_H

#include "navier_stokes/tests/navier_stokes_mms.h"
#include "angular_momentum/tests/angular_momentum_mms.h"

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Curl of scalar w in 2D: curl_vec(w) = (∂w/∂y, -∂w/∂x)
//
// For w* = t·sin(πx)·sin(πy):
//   ∂w*/∂x = t·π·cos(πx)·sin(πy)
//   ∂w*/∂y = t·π·sin(πx)·cos(πy)
//   curl_vec(w*) = (t·π·sin(πx)·cos(πy), -t·π·cos(πx)·sin(πy))
// ============================================================================
template <int dim>
dealii::Tensor<1, dim> curl_vec_w_exact(
    const dealii::Point<dim>& p, double time)
{
    const double x = p[0], y = p[1];
    dealii::Tensor<1, dim> curl;
    curl[0] = time * M_PI * std::sin(M_PI * x) * std::cos(M_PI * y);
    curl[1] = -time * M_PI * std::cos(M_PI * x) * std::sin(M_PI * y);
    return curl;
}

// ============================================================================
// Curl of vector u in 2D: curl_scalar(u) = ∂uy/∂x - ∂ux/∂y
//
// For the divergence-free velocity:
//   ux* = t·π·sin²(πx)·sin(2πy)
//   uy* = -t·π·sin(2πx)·sin²(πy)
//
//   ∂ux*/∂y = 2π²t·sin²(πx)·cos(2πy)
//   ∂uy*/∂x = -2π²t·cos(2πx)·sin²(πy)
//
//   curl(u*) = -2π²t·cos(2πx)·sin²(πy) - 2π²t·sin²(πx)·cos(2πy)
// ============================================================================
template <int dim>
double curl_scalar_u_exact(
    const dealii::Point<dim>& p, double time)
{
    const double x = p[0], y = p[1];
    const double sx = std::sin(M_PI * x);
    const double sy = std::sin(M_PI * y);
    const double c2x = std::cos(2.0 * M_PI * x);
    const double c2y = std::cos(2.0 * M_PI * y);

    // ∂uy*/∂x - ∂ux*/∂y
    return -2.0 * M_PI * M_PI * time * c2x * sy * sy
           - 2.0 * M_PI * M_PI * time * sx * sx * c2y;
}

// ============================================================================
// Coupled NS MMS source (backward Euler + micropolar coupling)
//
// f_u = (u*(t_new) - u*(t_old))/τ - (ν_eff/2)Δu*(t_new) + ∇p*(t_new)
//       - 2ν_r · curl_vec(w*(t_old))
//
// The assembly adds +2ν_r(w_discrete, ∇×v) to the RHS. The MMS source
// provides everything else. Since we use w*(t_old) analytically, the
// residual 2ν_r(w_disc - w*, ∇×v) is O(h³) — preserves convergence.
//
// Note: nu_r is captured by the lambda, not in the callback signature.
// ============================================================================
template <int dim>
dealii::Tensor<1, dim> compute_ns_mms_source_coupled(
    const dealii::Point<dim>& p,
    double t_new, double t_old,
    double nu_eff, double nu_r,
    const dealii::Tensor<1, dim>& /*U_old_disc*/,
    double /*div_U_old_disc*/ = 0.0,
    bool /*include_convection*/ = false)
{
    const double tau = t_new - t_old;

    // Time derivative: (u*(t_new) - u*(t_old))/τ
    const auto U_new = ns_exact_velocity<dim>(p, t_new);
    const auto U_old = ns_exact_velocity<dim>(p, t_old);

    // Viscous: -(ν_eff/2) Δu*(t_new)
    const auto lap = ns_exact_laplacian<dim>(p, t_new);

    // Pressure gradient: ∇p*(t_new)
    const auto grad_p = ns_exact_pressure_gradient<dim>(p, t_new);

    // Micropolar coupling: -2ν_r · curl_vec(w*(t_old))
    const auto curl_w = curl_vec_w_exact<dim>(p, t_old);

    dealii::Tensor<1, dim> f;
    for (unsigned int d = 0; d < dim; ++d)
        f[d] = (U_new[d] - U_old[d]) / tau
               - (nu_eff / 2.0) * lap[d]
               + grad_p[d]
               - 2.0 * nu_r * curl_w[d];

    return f;
}

// ============================================================================
// Coupled AngMom MMS source (backward Euler + curl coupling)
//
// f_w = j(w*(t_new) - w*(t_old))/τ - c₁Δw*(t_new) + 4ν_r w*(t_new)
//       - 2ν_r · curl_scalar(u*(t_new))
//
// The assembly adds +2ν_r(∇×u_discrete, z) to the RHS. The MMS source
// provides everything else. Since we use u*(t_new) analytically, the
// residual 2ν_r(curl(u_disc) - curl(u*), z) is O(h²) in dual norm,
// which by Aubin-Nitsche gives O(h³) in L2 — preserves convergence.
// ============================================================================
template <int dim>
double compute_angmom_mms_source_coupled(
    const dealii::Point<dim>& p,
    double t_new, double t_old,
    double j_micro, double c1, double nu_r,
    double /*w_old_disc*/)
{
    const double tau = t_new - t_old;
    const double w_new = angular_momentum_exact<dim>(p, t_new);
    const double w_old = angular_momentum_exact<dim>(p, t_old);

    // Time derivative: j(w* - w*_old)/τ
    double f = j_micro * (w_new - w_old) / tau;

    // Diffusion: -c₁ Δw* = +2π²c₁ · w*(t_new)
    f += 2.0 * M_PI * M_PI * c1 * w_new;

    // Reaction: 4ν_r · w*(t_new)
    f += 4.0 * nu_r * w_new;

    // Curl coupling: -2ν_r · curl(u*(t_new))
    f -= 2.0 * nu_r * curl_scalar_u_exact<dim>(p, t_new);

    return f;
}

// ============================================================================
// Result structure
// ============================================================================
struct NSAngMomMMSResult
{
    unsigned int refinement = 0;
    unsigned int n_dofs = 0;
    double h = 0.0;

    // NS errors
    double U_L2 = 0.0;
    double U_H1 = 0.0;
    double U_Linf = 0.0;
    double p_L2 = 0.0;

    // AngMom errors
    double w_L2 = 0.0;
    double w_H1 = 0.0;
    double w_Linf = 0.0;

    unsigned int n_steps = 0;
    double walltime = 0.0;
};

#endif // FHD_NS_ANGMOM_MMS_H
