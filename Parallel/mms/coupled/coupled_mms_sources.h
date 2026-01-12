// ============================================================================
// mms/coupled/coupled_mms_sources.h - Coupled MMS Source Terms (PARALLEL)
//
// CRITICAL: These source terms account for coupling between subsystems.
//
// The key insight is that when subsystems are coupled, the manufactured
// solution from one subsystem appears in the equations of another.
// The MMS source must account for these additional terms.
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
// POISSON + MAGNETIZATION COUPLING
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
// CH + NS COUPLING
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
// NS + KELVIN FORCE COUPLING
//
// NS equation: ∂U/∂t + (U·∇)U - 2νΔU + ∇p = μ₀(M·∇)H + f_U
//
// Kelvin force: F_K = μ₀(M·∇)H = μ₀[Mx·∂H/∂x + My·∂H/∂y]
//
// Source for NS WITH Kelvin force:
//   f_U = NS_source_standalone - Kelvin_force_exact
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
 * @brief NS MMS source WITH Kelvin force coupling
 *
 * The Kelvin force is computed from the exact M and H fields and SUBTRACTED
 * from the standalone NS source (since it appears on the RHS of the equation).
 */
template <int dim>
dealii::Tensor<1, dim> compute_ns_mms_source_with_kelvin(
    const dealii::Point<dim>& pt,
    double t_new,
    double t_old,
    double nu,
    double mu_0,
    double L_y = 1.0)
{
    // Import the standalone NS source computation
    // (from ns_mms.h - compute_unsteady_ns_mms_source)

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

    // Pressure gradient (use pre-computed trig values)
    const double dp_dx = -t_new * pi * sin_px * cos_py;
    const double dp_dy = -t_new * (pi / L_y) * cos_px * sin_py;

    // Standalone NS source
    dealii::Tensor<1, dim> f_ns;
    f_ns[0] = dux_dt + convect_x - 2.0 * nu * lap_ux + dp_dx;
    f_ns[1] = duy_dt + convect_y - 2.0 * nu * lap_uy + dp_dy;

    // Kelvin force (computed from exact M and H)
    dealii::Tensor<1, dim> F_K = compute_kelvin_force_exact<dim>(pt, t_new, mu_0, L_y);

    // f = f_ns_standalone - F_K (since F_K is on RHS)
    f_ns[0] -= F_K[0];
    f_ns[1] -= F_K[1];

    return f_ns;
}

// ============================================================================
// FULL SYSTEM COUPLING
//
// All couplings active:
//   - CH with U·∇θ advection
//   - Poisson with ∇·M source
//   - Magnetization with χH relaxation
//   - NS with Kelvin force + phase-dependent viscosity
// ============================================================================

/**
 * @brief NS source with BOTH Kelvin force AND variable viscosity
 *
 * ν(θ) = ν_f + (ν_s - ν_f)·(1 + θ)/2
 *
 * This adds the term: 2∇ν·T(U) to the momentum equation
 */
template <int dim>
dealii::Tensor<1, dim> compute_ns_mms_source_full_coupling(
    const dealii::Point<dim>& pt,
    double t_new,
    double t_old,
    double nu_f,
    double nu_s,
    double mu_0,
    double L_y = 1.0)
{
    // Start with NS + Kelvin source
    // Use average viscosity for now (full variable viscosity requires strain rate)
    const double nu_avg = 0.5 * (nu_f + nu_s);
    dealii::Tensor<1, dim> f = compute_ns_mms_source_with_kelvin<dim>(
        pt, t_new, t_old, nu_avg, mu_0, L_y);

    // Add variable viscosity correction: 2(∇ν)·T(U)
    // θ = t⁴·cos(πx)·cos(πy) at t_new
    // ∇θ = [-t⁴·π·sin(πx)·cos(πy), -t⁴·π·cos(πx)·sin(πy)]
    // ∇ν = (ν_s - ν_f)/2 · ∇θ

    const double pi = M_PI;
    const double x = pt[0];
    const double y = pt[1];
    const double t4 = t_new * t_new * t_new * t_new;

    const double cos_px = std::cos(pi * x);
    const double cos_py = std::cos(pi * y);
    const double sin_px = std::sin(pi * x);
    const double sin_py = std::sin(pi * y);

    // ∇ν
    const double dnu_dx = 0.5 * (nu_s - nu_f) * (-t4 * pi * sin_px * cos_py);
    const double dnu_dy = 0.5 * (nu_s - nu_f) * (-t4 * pi * cos_px * sin_py);

    // Strain rate T(U) = ∇U + (∇U)^T components
    // T_xx = 2·∂ux/∂x
    // T_xy = T_yx = ∂ux/∂y + ∂uy/∂x
    // T_yy = 2·∂uy/∂y
    const double sin_2px = std::sin(2.0 * pi * x);
    const double sin_2py = std::sin(2.0 * pi * y / L_y);
    const double cos_2py = std::cos(2.0 * pi * y / L_y);

    const double dux_dx = t_new * (pi * pi / L_y) * sin_2px * sin_2py;
    const double dux_dy = t_new * (2.0 * pi * pi / (L_y * L_y)) * sin_px * sin_px * cos_2py;
    const double duy_dx = -t_new * 2.0 * pi * pi * std::cos(2.0 * pi * x) * std::sin(pi * y / L_y) * std::sin(pi * y / L_y);
    const double duy_dy = -t_new * (pi * pi / L_y) * sin_2px * sin_2py;

    const double T_xx = 2.0 * dux_dx;
    const double T_xy = dux_dy + duy_dx;
    const double T_yy = 2.0 * duy_dy;

    // Variable viscosity term: (∇ν)·T = [∇ν · T row 1, ∇ν · T row 2]
    // = [dnu_dx·T_xx + dnu_dy·T_xy, dnu_dx·T_xy + dnu_dy·T_yy]
    f[0] += dnu_dx * T_xx + dnu_dy * T_xy;
    f[1] += dnu_dx * T_xy + dnu_dy * T_yy;

    return f;
}

#endif // COUPLED_MMS_SOURCES_H