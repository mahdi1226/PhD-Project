// ============================================================================
// mms_tests/ch_ns_mms.h — Coupled Cahn-Hilliard + Navier-Stokes MMS
//
// TWO-PHASE COUPLING:
//   NS ← CH:  Capillary force σ μ ∇φ on RHS of momentum equation
//   CH ← NS:  Convection u · ∇φ in Cahn-Hilliard transport
//
// ALGORITHM (sequential, single Picard per step):
//   1. Solve NS for (u^k, p^k) using capillary force from (φ^{k-1}, μ^{k-1})
//   2. Solve CH for (φ^k, μ^k) using velocity u^k for convection
//
// EXACT SOLUTIONS (reusing standalone solutions):
//   ux*(x,y,t)  =  t·π·sin²(πx)·sin(2πy)         [NS, div-free, Dirichlet]
//   uy*(x,y,t)  = −t·π·sin(2πx)·sin²(πy)
//   p*(x,y,t)   =  t·cos(πx)·cos(πy)
//   phi*(x,y,t) =  A·t·cos(πx)·cos(πy)            [CH, Neumann]
//   mu*(x,y,t)  =  B·t·cos(2πx)·cos(2πy)
//
// MMS SOURCES:
//   f_ns = (u* − U_old)/τ − (ν_eff/2)Δu* + ∇p* − σ μ*(t_old) ∇φ*(t_old)
//   f_phi = (φ* − φ_old_disc)/τ + u*·∇φ* − γ Δμ*
//   f_mu  = μ* − S(φ* − φ_old_disc) + ε²Δφ* − Ψ'(φ_old_disc)
//
// NOTE: NS source uses ANALYTICAL coupling (exact μ*, ∇φ* at t_old).
//   The assembler applies discrete CH values. Pollution is O(h³) in L2
//   for mu, O(h²) in L2 for ∇φ → by Aubin-Nitsche O(h³) velocity L2.
//   CH source uses ANALYTICAL u*(t_new) for convection. Pollution O(h³).
//
// EXPECTED CONVERGENCE (CG Q2 throughout):
//   U_L2:   O(h³) — rate ≈ 3.0
//   U_H1:   O(h²) — rate ≈ 2.0
//   p_L2:   O(h²) — rate ≈ 2.0
//   phi_L2: O(h³) — rate ≈ 3.0
//   phi_H1: O(h²) — rate ≈ 2.0
//   mu_L2:  O(h³) — rate ≈ 3.0
//   mu_H1:  O(h²) — rate ≈ 2.0
//
// Reference: Nochetto, Salgado & Tomas, CMAME 2016, arXiv:1601.06824
// ============================================================================
#ifndef FHD_CH_NS_MMS_H
#define FHD_CH_NS_MMS_H

#include "navier_stokes/tests/navier_stokes_mms.h"
#include "cahn_hilliard/tests/cahn_hilliard_mms.h"
#include "physics/material_properties.h"

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <cmath>
#include <utility>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Gradient of exact phi* at given time
//
// phi*(x,y,t) = A·t·cos(πx)·cos(πy)
// ∇φ* = (−A·t·π·sin(πx)·cos(πy), −A·t·π·cos(πx)·sin(πy))
// ============================================================================
template <int dim>
dealii::Tensor<1, dim> ch_exact_grad_phi(
    const dealii::Point<dim>& p, double time)
{
    dealii::Tensor<1, dim> grad;
    const double cx = std::cos(M_PI * p[0]);
    const double sx = std::sin(M_PI * p[0]);
    const double cy = (dim >= 2) ? std::cos(M_PI * p[1]) : 1.0;
    const double sy = (dim >= 2) ? std::sin(M_PI * p[1]) : 0.0;

    grad[0] = -CHmms::A * time * M_PI * sx * cy;
    if constexpr (dim >= 2)
        grad[1] = -CHmms::A * time * M_PI * cx * sy;
    return grad;
}

// ============================================================================
// Exact mu* value at given time
//
// mu*(x,y,t) = B·t·cos(2πx)·cos(2πy)
// ============================================================================
template <int dim>
double ch_exact_mu(const dealii::Point<dim>& p, double time)
{
    double val = CHmms::B * time * std::cos(2.0 * M_PI * p[0]);
    if constexpr (dim >= 2)
        val *= std::cos(2.0 * M_PI * p[1]);
    return val;
}

// ============================================================================
// Coupled NS MMS source (backward Euler + capillary force)
//
// f_ns = (u*(t_new) − U_old_disc)/τ − (ν_eff/2)Δu*(t_new) + ∇p*(t_new)
//        − σ · μ*(t_old) · ∇φ*(t_old)
//
// The assembler adds +σ μ_disc ∇φ_disc · v to the RHS.
// The source subtracts the exact coupling analytically.
// Pollution: σ(μ*∇φ* − μ_disc∇φ_disc) → O(h²) in L2 → O(h³) vel L2.
// ============================================================================
template <int dim>
dealii::Tensor<1, dim> compute_ns_ch_coupled_source(
    const dealii::Point<dim>& p,
    double t_new, double t_old,
    double nu_eff, double sigma_cap,
    const dealii::Tensor<1, dim>& U_old_disc,
    double /*div_U_old_disc*/ = 0.0,
    bool /*include_convection*/ = false)
{
    const double tau = t_new - t_old;

    // Time derivative: (u*(t_new) − U_old_disc)/τ
    const auto U_new = ns_exact_velocity<dim>(p, t_new);

    // Viscous: −(ν_eff/2) Δu*(t_new)
    const auto lap = ns_exact_laplacian<dim>(p, t_new);

    // Pressure gradient: ∇p*(t_new)
    const auto grad_p = ns_exact_pressure_gradient<dim>(p, t_new);

    // Capillary force: −σ μ*(t_old) ∇φ*(t_old)
    const double mu_star = ch_exact_mu<dim>(p, t_old);
    const auto grad_phi_star = ch_exact_grad_phi<dim>(p, t_old);

    dealii::Tensor<1, dim> f;
    for (unsigned int d = 0; d < dim; ++d)
        f[d] = (U_new[d] - U_old_disc[d]) / tau
               - (nu_eff / 2.0) * lap[d]
               + grad_p[d]
               - sigma_cap * mu_star * grad_phi_star[d];

    return f;
}

// ============================================================================
// Coupled CH MMS source (backward Euler + convection from NS)
//
// phi equation:
//   f_phi = (1/dt)(φ*(t_new) − φ_old_disc) + u*(t_new)·∇φ*(t_new)
//           − γ Δμ*(t_new)
//
// Note: u* is divergence-free so the skew term ½(∇·u*)φ* vanishes.
//
// mu equation (same as standalone — no velocity coupling):
//   f_mu = μ*(t_new) − S(φ*(t_new) − φ_old_disc) + ε²Δφ*(t_new)
//          − Ψ'(φ_old_disc)
//
// The assembler adds +b_h(u_disc; φ, v) to the LHS.
// The source includes the exact convection analytically.
// Pollution: (u* − u_disc)·∇φ* → O(h³) in L2 → preserves φ L2 rate.
// ============================================================================
template <int dim>
std::pair<double, double> compute_ch_ns_coupled_source(
    const dealii::Point<dim>& p,
    double t_new,
    double dt,
    double phi_old_disc,
    double epsilon,
    double gamma)
{
    const double S    = 1.0 / (epsilon * epsilon);
    const double eps2 = epsilon * epsilon;

    // Exact phi and mu at t_new
    double cos_phi = std::cos(M_PI * p[0]);
    double cos_mu  = std::cos(2.0 * M_PI * p[0]);
    for (unsigned int d = 1; d < dim; ++d)
    {
        cos_phi *= std::cos(M_PI * p[d]);
        cos_mu  *= std::cos(2.0 * M_PI * p[d]);
    }
    const double phi_star_new = CHmms::A * t_new * cos_phi;
    const double mu_star_new  = CHmms::B * t_new * cos_mu;

    // Laplacians at t_new
    const double lap_phi = -CHmms::A * t_new * dim * M_PI * M_PI * cos_phi;
    const double lap_mu  = -CHmms::B * t_new * dim * 4.0 * M_PI * M_PI * cos_mu;

    // Convection: u*(t_new) · ∇φ*(t_new)
    // u* is divergence-free, so skew term ½(∇·u*)φ* = 0
    const auto U_star = ns_exact_velocity<dim>(p, t_new);
    const auto grad_phi_star = ch_exact_grad_phi<dim>(p, t_new);
    const double conv = U_star * grad_phi_star;

    // f_phi = (1/dt)(φ*_new − φ_old_disc) + u*·∇φ* − γ Δμ*
    const double f_phi = (1.0 / dt) * (phi_star_new - phi_old_disc)
                         + conv
                         - gamma * lap_mu;

    // f_mu = μ*_new − S(φ*_new − φ_old_disc) + ε²Δφ*_new − Ψ'(φ_old_disc)
    const double psi_prime = double_well_derivative(phi_old_disc);
    const double f_mu = mu_star_new
                        - S * (phi_star_new - phi_old_disc)
                        + eps2 * lap_phi
                        - psi_prime;

    return {f_phi, f_mu};
}

// ============================================================================
// Result structure
// ============================================================================
struct CHNSMMSResult
{
    unsigned int refinement = 0;
    unsigned int n_dofs = 0;
    double h = 0.0;

    // NS errors
    double U_L2 = 0.0, U_H1 = 0.0;
    double p_L2 = 0.0;

    // CH errors
    double phi_L2 = 0.0, phi_H1 = 0.0;
    double mu_L2 = 0.0, mu_H1 = 0.0;

    unsigned int n_steps = 0;
    double walltime = 0.0;
};

#endif // FHD_CH_NS_MMS_H
