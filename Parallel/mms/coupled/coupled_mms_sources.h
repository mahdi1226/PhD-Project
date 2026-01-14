// ============================================================================
// mms/coupled/coupled_mms_sources.h - Coupled MMS Source Terms (PARALLEL)
//
// CRITICAL: These source terms account for coupling between subsystems.
//
// The key insight is that when subsystems are coupled, the manufactured
// solution from one subsystem appears in the equations of another.
// The MMS source must account for these additional terms.
//
// Supports three coupled tests:
//   1. CH_NS: Phase advection by velocity
//   2. POISSON_MAGNETIZATION: φ ↔ M Picard iteration
//   3. FULL_SYSTEM: All couplings (uses PhaseFieldProblem directly)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef COUPLED_MMS_SOURCES_H
#define COUPLED_MMS_SOURCES_H

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/function.h>

#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// EXACT SOLUTIONS (collected from individual MMS files for reference)
// ============================================================================
// CH:  θ = t⁴·cos(πx)·cos(πy)
//      ψ = t⁴·sin(πx)·sin(πy)
//
// NS:  ux = t·(π/L_y)·sin²(πx)·sin(2πy/L_y)
//      uy = -t·π·sin(2πx)·sin²(πy/L_y)
//      p  = t·cos(πx)·cos(πy/L_y)
//
// Poisson: φ = t·cos(πx)·cos(πy/L_y)
//
// Magnetization: Mx = t·sin(πx)·sin(πy/L_y)
//                My = t·cos(πx)·cos(πy/L_y)
// ============================================================================

// ============================================================================
// POISSON + MAGNETIZATION COUPLING (for POISSON_MAGNETIZATION test)
//
// Poisson equation: -Δφ = -∇·M
// Magnetization:    ∂M/∂t + M/τ_M = χ·H/τ_M, where H = -∇φ
//
// Source for Poisson WITH M coupling:
//   f_φ = -Δφ_exact + ∇·M_exact
//
// Source for Magnetization WITH H coupling:
//   f_M = ∂M_exact/∂t + M_exact/τ_M - χ·H_exact/τ_M
// ============================================================================

/**
 * @brief Poisson MMS source WITH magnetization coupling
 *
 * f_φ = -Δφ_exact + ∇·M_exact
 */
template <int dim>
double compute_poisson_mms_source_with_M(
    const dealii::Point<dim>& pt,
    double time,
    double L_y = 1.0)
{
    const double x = pt[0];
    const double y = pt[1];
    const double pi = M_PI;
    const double t = time;

    // -Δφ_exact where φ = t·cos(πx)·cos(πy/L_y)
    // Δφ = -t·π²·cos(πx)·cos(πy/L_y) - t·(π/L_y)²·cos(πx)·cos(πy/L_y)
    //    = -t·π²(1 + 1/L_y²)·cos(πx)·cos(πy/L_y)
    const double neg_laplacian_phi = t * pi * pi * (1.0 + 1.0/(L_y*L_y))
                                     * std::cos(pi * x) * std::cos(pi * y / L_y);

    // ∇·M_exact where Mx = t·sin(πx)·sin(πy/L_y), My = t·cos(πx)·cos(πy/L_y)
    // ∂Mx/∂x = t·π·cos(πx)·sin(πy/L_y)
    // ∂My/∂y = -t·(π/L_y)·cos(πx)·sin(πy/L_y)
    // ∇·M = t·π·cos(πx)·sin(πy/L_y) - t·(π/L_y)·cos(πx)·sin(πy/L_y)
    //     = t·π·(1 - 1/L_y)·cos(πx)·sin(πy/L_y)
    const double div_M = t * pi * (1.0 - 1.0/L_y) * std::cos(pi * x) * std::sin(pi * y / L_y);

    // f_φ = -Δφ - ∇·M (note: Poisson is -Δφ = -∇·M, so source is -Δφ_exact + ∇·M_exact)
    return neg_laplacian_phi - div_M;
}

/**
 * @brief Magnetization MMS source WITH H = -∇φ coupling
 *
 * Equation: ∂M/∂t + M/τ_M = χ·H/τ_M
 * Source: f_M = ∂M_exact/∂t + M_exact/τ_M - χ·H_exact/τ_M
 */
template <int dim>
dealii::Tensor<1, dim> compute_mag_mms_source_with_H(
    const dealii::Point<dim>& pt,
    double t_new,
    double t_old,
    double tau_M,
    double chi,
    double L_y = 1.0)
{
    const double x = pt[0];
    const double y = pt[1];
    const double pi = M_PI;
    const double dt = t_new - t_old;

    // Exact M at new and old time
    const double Mx_new = t_new * std::sin(pi * x) * std::sin(pi * y / L_y);
    const double My_new = t_new * std::cos(pi * x) * std::cos(pi * y / L_y);
    const double Mx_old = t_old * std::sin(pi * x) * std::sin(pi * y / L_y);
    const double My_old = t_old * std::cos(pi * x) * std::cos(pi * y / L_y);

    // Exact H = -∇φ at new time
    // φ = t·cos(πx)·cos(πy/L_y)
    // ∇φ = [-t·π·sin(πx)·cos(πy/L_y), -t·(π/L_y)·cos(πx)·sin(πy/L_y)]
    // H = -∇φ = [t·π·sin(πx)·cos(πy/L_y), t·(π/L_y)·cos(πx)·sin(πy/L_y)]
    const double Hx = t_new * pi * std::sin(pi * x) * std::cos(pi * y / L_y);
    const double Hy = t_new * (pi / L_y) * std::cos(pi * x) * std::sin(pi * y / L_y);

    // Source: f_M = (M_new - M_old)/dt + M_new/τ_M - χ·H/τ_M
    dealii::Tensor<1, dim> f;
    f[0] = (Mx_new - Mx_old) / dt + Mx_new / tau_M - chi * Hx / tau_M;
    f[1] = (My_new - My_old) / dt + My_new / tau_M - chi * Hy / tau_M;

    return f;
}

// ============================================================================
// CH + NS COUPLING (for CH_NS test)
//
// CH equation: ∂θ/∂t + U·∇θ = γΔψ + f_θ
//
// Source for CH WITH advection:
//   f_θ = ∂θ_exact/∂t + U_exact·∇θ_exact + γΔψ_exact
// ============================================================================

/**
 * @brief CH theta MMS source WITH velocity advection
 *
 * f_θ = (θ^n - θ^{n-1})/dt + U^{n-1}·∇θ^{n-1} + γΔψ^n
 *
 * Note: Uses semi-implicit lagged advection (U^{n-1}·∇θ^{n-1})
 */
template <int dim>
double compute_ch_theta_mms_source_with_advection(
    const dealii::Point<dim>& pt,
    double t_new,
    double t_old,
    double gamma,
    double L_y = 1.0)
{
    const double x = pt[0];
    const double y = pt[1];
    const double pi = M_PI;
    const double dt = t_new - t_old;

    const double t4_new = t_new * t_new * t_new * t_new;
    const double t4_old = t_old * t_old * t_old * t_old;

    const double cos_px = std::cos(pi * x);
    const double cos_py = std::cos(pi * y);
    const double sin_px = std::sin(pi * x);
    const double sin_py = std::sin(pi * y);

    // Time derivative: (θ^n - θ^{n-1})/dt
    const double theta_new = t4_new * cos_px * cos_py;
    const double theta_old = t4_old * cos_px * cos_py;
    const double dtheta_dt = (theta_new - theta_old) / dt;

    // Velocity at t_old (lagged)
    const double ux_old = t_old * (pi / L_y) * sin_px * sin_px * std::sin(2.0 * pi * y / L_y);
    const double uy_old = -t_old * pi * std::sin(2.0 * pi * x) * std::sin(pi * y / L_y) * std::sin(pi * y / L_y);

    // ∇θ at t_old
    const double dtheta_dx_old = -t4_old * pi * sin_px * cos_py;
    const double dtheta_dy_old = -t4_old * pi * cos_px * sin_py;

    // Advection: U_old · ∇θ_old
    const double advection = ux_old * dtheta_dx_old + uy_old * dtheta_dy_old;

    // Δψ at t_new (ψ = t⁴·sin(πx)·sin(πy))
    // Δψ = -2π²·t⁴·sin(πx)·sin(πy)
    const double laplacian_psi = -2.0 * pi * pi * t4_new * sin_px * sin_py;

    // f_θ = (θ^n - θ^{n-1})/dt + U·∇θ + γΔψ
    return dtheta_dt + advection + gamma * laplacian_psi;
}

// ============================================================================
// FULL SYSTEM COUPLING (for FULL_SYSTEM test)
//
// All couplings active:
//   - CH with U·∇θ advection
//   - Poisson with ∇·M source
//   - Magnetization with χH relaxation AND U·∇M advection
//   - NS with Kelvin force μ₀(M·∇)H + capillary force + variable viscosity
//
// NOTE: The FULL_SYSTEM test uses PhaseFieldProblem directly with enable_mms=true.
// The source terms below are for reference and debugging.
// ============================================================================

/**
 * @brief Compute exact Kelvin force from manufactured M and H = -∇φ
 *
 * F_K = μ₀(M·∇)H where H = -∇φ
 *
 * Components:
 *   F_Kx = μ₀(Mx·∂Hx/∂x + My·∂Hx/∂y)
 *   F_Ky = μ₀(Mx·∂Hy/∂x + My·∂Hy/∂y)
 */
template <int dim>
dealii::Tensor<1, dim> compute_kelvin_force_exact(
    const dealii::Point<dim>& pt,
    double time,
    double mu_0,
    double L_y = 1.0)
{
    const double x = pt[0];
    const double y = pt[1];
    const double pi = M_PI;
    const double t = time;

    // Exact M
    const double Mx = t * std::sin(pi * x) * std::sin(pi * y / L_y);
    const double My = t * std::cos(pi * x) * std::cos(pi * y / L_y);

    // H = -∇φ where φ = t·cos(πx)·cos(πy/L_y)
    // Hx = t·π·sin(πx)·cos(πy/L_y)
    // Hy = t·(π/L_y)·cos(πx)·sin(πy/L_y)

    // ∂Hx/∂x = t·π²·cos(πx)·cos(πy/L_y)
    const double dHx_dx = t * pi * pi * std::cos(pi * x) * std::cos(pi * y / L_y);

    // ∂Hx/∂y = -t·(π²/L_y)·sin(πx)·sin(πy/L_y)
    const double dHx_dy = -t * (pi * pi / L_y) * std::sin(pi * x) * std::sin(pi * y / L_y);

    // ∂Hy/∂x = -t·(π²/L_y)·sin(πx)·sin(πy/L_y)
    const double dHy_dx = -t * (pi * pi / L_y) * std::sin(pi * x) * std::sin(pi * y / L_y);

    // ∂Hy/∂y = t·(π²/L_y²)·cos(πx)·cos(πy/L_y)
    const double dHy_dy = t * (pi * pi / (L_y * L_y)) * std::cos(pi * x) * std::cos(pi * y / L_y);

    // Kelvin force components
    dealii::Tensor<1, dim> F_K;
    F_K[0] = mu_0 * (Mx * dHx_dx + My * dHx_dy);
    F_K[1] = mu_0 * (Mx * dHy_dx + My * dHy_dy);

    return F_K;
}

/**
 * @brief Compute exact capillary force from manufactured θ and ψ
 *
 * F_cap = -λ·ψ·∇θ (Paper Eq. 42b)
 */
template <int dim>
dealii::Tensor<1, dim> compute_capillary_force_exact(
    const dealii::Point<dim>& pt,
    double time,
    double lambda)
{
    const double x = pt[0];
    const double y = pt[1];
    const double pi = M_PI;
    const double t4 = time * time * time * time;

    // ψ = t⁴·sin(πx)·sin(πy)
    const double psi = t4 * std::sin(pi * x) * std::sin(pi * y);

    // ∇θ where θ = t⁴·cos(πx)·cos(πy)
    const double dtheta_dx = -t4 * pi * std::sin(pi * x) * std::cos(pi * y);
    const double dtheta_dy = -t4 * pi * std::cos(pi * x) * std::sin(pi * y);

    dealii::Tensor<1, dim> F_cap;
    F_cap[0] = -lambda * psi * dtheta_dx;
    F_cap[1] = -lambda * psi * dtheta_dy;

    return F_cap;
}

/**
 * @brief NS source for FULL SYSTEM with all forces
 *
 * Forces included:
 *   - Kelvin: μ₀(M·∇)H
 *   - Capillary: -λψ∇θ
 *   - Variable viscosity: 2(∇ν)·T(U)
 *
 * f_U = f_NS_standalone - F_Kelvin - F_capillary + viscosity_correction
 */
template <int dim>
dealii::Tensor<1, dim> compute_ns_mms_source_full_system(
    const dealii::Point<dim>& pt,
    double t_new,
    double t_old,
    double nu_f,
    double nu_s,
    double mu_0,
    double lambda,
    double L_y = 1.0)
{
    const double x = pt[0];
    const double y = pt[1];
    const double pi = M_PI;
    const double pi2 = pi * pi;
    const double pi3 = pi * pi * pi;
    const double dt = t_new - t_old;

    const double sin_px = std::sin(pi * x);
    const double cos_px = std::cos(pi * x);
    const double sin_py = std::sin(pi * y / L_y);
    const double cos_py = std::cos(pi * y / L_y);
    const double sin_2px = std::sin(2.0 * pi * x);
    const double cos_2px = std::cos(2.0 * pi * x);
    const double sin_2py = std::sin(2.0 * pi * y / L_y);
    const double cos_2py = std::cos(2.0 * pi * y / L_y);

    // Velocities
    const double ux_new = t_new * (pi / L_y) * sin_px * sin_px * sin_2py;
    const double uy_new = -t_new * pi * sin_2px * sin_py * sin_py;
    const double ux_old = t_old * (pi / L_y) * sin_px * sin_px * sin_2py;
    const double uy_old = -t_old * pi * sin_2px * sin_py * sin_py;

    // Time derivative
    const double dux_dt = (ux_new - ux_old) / dt;
    const double duy_dt = (uy_new - uy_old) / dt;

    // Gradients at new time
    const double dux_dx = t_new * (pi2 / L_y) * sin_2px * sin_2py;
    const double dux_dy = t_new * (2.0 * pi2 / (L_y * L_y)) * sin_px * sin_px * cos_2py;
    const double duy_dx = -t_new * 2.0 * pi2 * cos_2px * sin_py * sin_py;
    const double duy_dy = -t_new * (pi2 / L_y) * sin_2px * sin_2py;

    // Convection (semi-implicit: U_old · ∇U_new)
    const double convect_x = ux_old * dux_dx + uy_old * dux_dy;
    const double convect_y = ux_old * duy_dx + uy_old * duy_dy;

    // Laplacians
    const double d2ux_dx2 = t_new * (2.0 * pi3 / L_y) * cos_2px * sin_2py;
    const double d2ux_dy2 = -t_new * (4.0 * pi3 / (L_y * L_y * L_y)) * sin_px * sin_px * sin_2py;
    const double lap_ux = d2ux_dx2 + d2ux_dy2;

    const double d2uy_dx2 = t_new * 4.0 * pi3 * sin_2px * sin_py * sin_py;
    const double d2uy_dy2 = -t_new * (2.0 * pi3 / (L_y * L_y)) * sin_2px * (cos_py * cos_py - sin_py * sin_py);
    const double lap_uy = d2uy_dx2 + d2uy_dy2;

    // Pressure gradient
    const double dp_dx = -t_new * pi * sin_px * cos_py;
    const double dp_dy = -t_new * (pi / L_y) * cos_px * sin_py;

    // Use average viscosity for base NS source
    const double nu_avg = 0.5 * (nu_f + nu_s);

    // Standalone NS source
    dealii::Tensor<1, dim> f_ns;
    f_ns[0] = dux_dt + convect_x - 2.0 * nu_avg * lap_ux + dp_dx;
    f_ns[1] = duy_dt + convect_y - 2.0 * nu_avg * lap_uy + dp_dy;

    // Subtract Kelvin force (it's on RHS of momentum equation)
    dealii::Tensor<1, dim> F_K = compute_kelvin_force_exact<dim>(pt, t_new, mu_0, L_y);
    f_ns[0] -= F_K[0];
    f_ns[1] -= F_K[1];

    // Subtract capillary force
    dealii::Tensor<1, dim> F_cap = compute_capillary_force_exact<dim>(pt, t_new, lambda);
    f_ns[0] -= F_cap[0];
    f_ns[1] -= F_cap[1];

    // Add variable viscosity correction: 2(∇ν)·T(U)
    // θ = t⁴·cos(πx)·cos(πy) at t_new
    const double t4 = t_new * t_new * t_new * t_new;
    const double cos_px_full = std::cos(pi * x);
    const double cos_py_full = std::cos(pi * y);
    const double sin_px_full = std::sin(pi * x);
    const double sin_py_full = std::sin(pi * y);

    // ∇ν = (ν_s - ν_f)/2 · ∇θ
    const double dnu_dx = 0.5 * (nu_s - nu_f) * (-t4 * pi * sin_px_full * cos_py_full);
    const double dnu_dy = 0.5 * (nu_s - nu_f) * (-t4 * pi * cos_px_full * sin_py_full);

    // Strain rate T(U)
    const double T_xx = 2.0 * dux_dx;
    const double T_xy = dux_dy + duy_dx;
    const double T_yy = 2.0 * duy_dy;

    // Variable viscosity term
    f_ns[0] += dnu_dx * T_xx + dnu_dy * T_xy;
    f_ns[1] += dnu_dx * T_xy + dnu_dy * T_yy;

    return f_ns;
}

/**
 * @brief Magnetization source for FULL SYSTEM with advection
 *
 * Equation: ∂M/∂t + (U·∇)M + M/τ_M = χ·H/τ_M
 * Source: f_M = ∂M/∂t + (U·∇)M + M/τ_M - χ·H/τ_M
 */
template <int dim>
dealii::Tensor<1, dim> compute_mag_mms_source_full_system(
    const dealii::Point<dim>& pt,
    double t_new,
    double t_old,
    double tau_M,
    double chi,
    double L_y = 1.0)
{
    const double x = pt[0];
    const double y = pt[1];
    const double pi = M_PI;
    const double dt = t_new - t_old;

    // Start with basic source (time derivative + relaxation - χH)
    dealii::Tensor<1, dim> f = compute_mag_mms_source_with_H<dim>(pt, t_new, t_old, tau_M, chi, L_y);

    // Add advection term (U·∇)M at t_old (lagged)
    // U at t_old
    const double sin_px = std::sin(pi * x);
    const double sin_py = std::sin(pi * y / L_y);
    const double cos_px = std::cos(pi * x);
    const double cos_py = std::cos(pi * y / L_y);

    const double ux_old = t_old * (pi / L_y) * sin_px * sin_px * std::sin(2.0 * pi * y / L_y);
    const double uy_old = -t_old * pi * std::sin(2.0 * pi * x) * sin_py * sin_py;

    // M at t_old
    const double Mx_old = t_old * sin_px * sin_py;
    const double My_old = t_old * cos_px * cos_py;

    // ∇M at t_old
    const double dMx_dx = t_old * pi * cos_px * sin_py;
    const double dMx_dy = t_old * (pi / L_y) * sin_px * cos_py;
    const double dMy_dx = -t_old * pi * sin_px * cos_py;
    const double dMy_dy = -t_old * (pi / L_y) * cos_px * sin_py;

    // (U·∇)M
    f[0] += ux_old * dMx_dx + uy_old * dMx_dy;
    f[1] += ux_old * dMy_dx + uy_old * dMy_dy;

    return f;
}

#endif // COUPLED_MMS_SOURCES_H