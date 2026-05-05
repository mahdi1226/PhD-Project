// ============================================================================
// ns_mms.h - Navier-Stokes MMS Exact Solutions (Parallel Version)
//
// PAPER EQUATION 42e (standalone, constant ν):
//   (U^n - U^{n-1})/dt + B_h(U^{n-1}, U^n, V) + ν(T(U^n), T(V))
//     - (p^n, ∇·V) + (∇·U^n, Q) = (f, V)
//
// VISCOUS TERM CONVENTION:
//   Paper: ν(T(U),T(V)) where T = ½(∇u + (∇u)^T)
//   Code helper returns D = ∇u + (∇u)^T = 2T, so bilinear form is (ν/4)(D,D)
//   Strong form: -(ν/2)∆U
//
// EXACT SOLUTIONS (derived from stream function ψ = t·sin²(πx)·sin²(πy/L_y)):
//   ux = t·(π/L_y)·sin²(πx)·sin(2πy/L_y)
//   uy = -t·π·sin(2πx)·sin²(πy/L_y)
//   p  = t·cos(πx)·cos(πy/L_y)
//
// Properties:
//   - ∇·U = 0 exactly (incompressible)
//   - U = 0 on all boundaries (no-slip)
//
// COUPLED MMS SOURCES:
//   - compute_kelvin_force_mms_source(): F_K(M*, H*) for NS+Magnetization
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef NS_MMS_H
#define NS_MMS_H

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/function.h>

#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Exact velocity x-component
// ux = t²·(π/L_y)·sin²(πx)·sin(2πy/L_y)
//
// Time profile changed from t to t² (2026-05-05): backward Euler is exact
// for linear-in-t functions, so the temporal-convergence test is silent
// when U is linear. With U ∝ t², BE truncation O(τ · d²U/dt²) ≠ 0 and the
// formal rate ≈ 1.0 becomes measurable. All downstream MMS source terms
// (Stokes, NS, unsteady NS) updated consistently.
// ============================================================================
template <int dim>
class NSExactVelocityX : public dealii::Function<dim>
{
public:
    NSExactVelocityX(double time = 1.0, double L_y = 1.0)
        : dealii::Function<dim>(1), time_(time), L_y_(L_y)
    {
    }

    virtual double value(const dealii::Point<dim>& p,
                         const unsigned int component = 0) const override
    {
        (void)component;
        const double x = p[0];
        const double y = p[1];
        const double sin_px = std::sin(M_PI * x);
        return time_ * time_ * (M_PI / L_y_) * sin_px * sin_px * std::sin(2.0 * M_PI * y / L_y_);
    }

    virtual dealii::Tensor<1, dim> gradient(const dealii::Point<dim>& p,
                                            const unsigned int component = 0) const override
    {
        (void)component;
        const double x = p[0];
        const double y = p[1];
        const double t2 = time_ * time_;

        dealii::Tensor<1, dim> grad;
        // ∂ux/∂x = t²·(π²/L_y)·sin(2πx)·sin(2πy/L_y)
        grad[0] = t2 * (M_PI * M_PI / L_y_) * std::sin(2.0 * M_PI * x) * std::sin(2.0 * M_PI * y / L_y_);
        // ∂ux/∂y = t²·(2π²/L_y²)·sin²(πx)·cos(2πy/L_y)
        const double sin_px = std::sin(M_PI * x);
        grad[1] = t2 * (2.0 * M_PI * M_PI / (L_y_ * L_y_)) * sin_px * sin_px * std::cos(2.0 * M_PI * y / L_y_);

        return grad;
    }

    void set_time(const double t) override { time_ = t; }
    double get_time() const { return time_; }

private:
    double time_;
    double L_y_;
};

// ============================================================================
// Exact velocity y-component
// uy = -t²·π·sin(2πx)·sin²(πy/L_y)
// ============================================================================
template <int dim>
class NSExactVelocityY : public dealii::Function<dim>
{
public:
    NSExactVelocityY(double time = 1.0, double L_y = 1.0)
        : dealii::Function<dim>(1), time_(time), L_y_(L_y)
    {
    }

    virtual double value(const dealii::Point<dim>& p,
                         const unsigned int component = 0) const override
    {
        (void)component;
        const double x = p[0];
        const double y = p[1];
        const double sin_py = std::sin(M_PI * y / L_y_);
        return -time_ * time_ * M_PI * std::sin(2.0 * M_PI * x) * sin_py * sin_py;
    }

    virtual dealii::Tensor<1, dim> gradient(const dealii::Point<dim>& p,
                                            const unsigned int component = 0) const override
    {
        (void)component;
        const double x = p[0];
        const double y = p[1];
        const double t2 = time_ * time_;

        dealii::Tensor<1, dim> grad;
        // ∂uy/∂x = -t²·2π²·cos(2πx)·sin²(πy/L_y)
        const double sin_py = std::sin(M_PI * y / L_y_);
        grad[0] = -t2 * 2.0 * M_PI * M_PI * std::cos(2.0 * M_PI * x) * sin_py * sin_py;
        // ∂uy/∂y = -t²·(π²/L_y)·sin(2πx)·sin(2πy/L_y)
        grad[1] = -t2 * (M_PI * M_PI / L_y_) * std::sin(2.0 * M_PI * x) * std::sin(2.0 * M_PI * y / L_y_);

        return grad;
    }

    void set_time(const double t) override { time_ = t; }

private:
    double time_;
    double L_y_;
};

// ============================================================================
// Exact pressure
// p = t²·cos(πx)·cos(πy/L_y)
//
// Pressure scales like t² to match the scaling of ν Δ U (which is t² · spatial
// after the U → t² update). Otherwise the steady Stokes balance -ν ΔU + ∇p = f
// would have time-mismatched terms in f.
// ============================================================================
template <int dim>
class NSExactPressure : public dealii::Function<dim>
{
public:
    NSExactPressure(double time = 1.0, double L_y = 1.0)
        : dealii::Function<dim>(1), time_(time), L_y_(L_y)
    {
    }

    virtual double value(const dealii::Point<dim>& p,
                         const unsigned int component = 0) const override
    {
        (void)component;
        const double x = p[0];
        const double y = p[1];
        return time_ * time_ * std::cos(M_PI * x) * std::cos(M_PI * y / L_y_);
    }

    virtual dealii::Tensor<1, dim> gradient(const dealii::Point<dim>& p,
                                            const unsigned int component = 0) const override
    {
        (void)component;
        const double x = p[0];
        const double y = p[1];
        const double t2 = time_ * time_;

        dealii::Tensor<1, dim> grad;
        grad[0] = -t2 * M_PI * std::sin(M_PI * x) * std::cos(M_PI * y / L_y_);
        grad[1] = -t2 * (M_PI / L_y_) * std::cos(M_PI * x) * std::sin(M_PI * y / L_y_);

        return grad;
    }

    void set_time(const double t) override { time_ = t; }

private:
    double time_;
    double L_y_;
};

// ============================================================================
// Convenience functions
// ============================================================================
template <int dim>
dealii::Tensor<1, dim> ns_mms_exact_velocity(
    const dealii::Point<dim>& p,
    double time,
    double L_y = 1.0)
{
    const double x = p[0];
    const double y = p[1];
    const double t2 = time * time;

    dealii::Tensor<1, dim> U;
    const double sin_px = std::sin(M_PI * x);
    U[0] = t2 * (M_PI / L_y) * sin_px * sin_px * std::sin(2.0 * M_PI * y / L_y);

    const double sin_py = std::sin(M_PI * y / L_y);
    U[1] = -t2 * M_PI * std::sin(2.0 * M_PI * x) * sin_py * sin_py;

    return U;
}

template <int dim>
double ns_mms_exact_pressure(
    const dealii::Point<dim>& p,
    double time,
    double L_y = 1.0)
{
    const double x = p[0];
    const double y = p[1];
    return time * time * std::cos(M_PI * x) * std::cos(M_PI * y / L_y);
}

// ============================================================================
// MMS Source Terms for Different Phases
// ============================================================================

/**
 * @brief Compute MMS source for STEADY STOKES (Phase A)
 *
 * Strong form: -(ν/2)∇²U + ∇p = f
 * Paper Eq. 42e: bilinear form is (ν T(U), T(V)) with T = ½(∇U + (∇U)^T).
 * Code helper returns D = ∇U + (∇U)^T = 2T, bilinear form is (ν/4)(D,D).
 * For div-free U*: (ν/4)(D,D) = (ν/2)(∇U,∇V), giving -(ν/2)∇²U after IBP.
 */
template <int dim>
dealii::Tensor<1, dim> compute_steady_stokes_mms_source(
    const dealii::Point<dim>& pt,
    double time,
    double nu,
    double L_y = 1.0)
{
    const double x = pt[0];
    const double y = pt[1];
    const double t = time;

    const double sin_px = std::sin(M_PI * x);
    const double cos_px = std::cos(M_PI * x);
    const double sin_py = std::sin(M_PI * y / L_y);
    const double cos_py = std::cos(M_PI * y / L_y);
    const double sin_2px = std::sin(2.0 * M_PI * x);
    const double cos_2px = std::cos(2.0 * M_PI * x);
    const double sin_2py = std::sin(2.0 * M_PI * y / L_y);

    const double pi = M_PI;
    (void)(M_PI * M_PI); // pi2 unused in this function
    const double pi3 = M_PI * M_PI * M_PI;
    const double t2 = t * t;

    // ∇²ux = ∂²ux/∂x² + ∂²ux/∂y²  (U ∝ t²)
    const double d2ux_dx2 = t2 * (2.0 * pi3 / L_y) * cos_2px * sin_2py;
    const double d2ux_dy2 = -t2 * (4.0 * pi3 / (L_y * L_y * L_y)) * sin_px * sin_px * sin_2py;
    const double laplacian_ux = d2ux_dx2 + d2ux_dy2;

    // ∇²uy = ∂²uy/∂x² + ∂²uy/∂y²
    const double d2uy_dx2 = t2 * 4.0 * pi3 * sin_2px * sin_py * sin_py;
    const double d2uy_dy2 = -t2 * (2.0 * pi3 / (L_y * L_y)) * sin_2px * (cos_py * cos_py - sin_py * sin_py);
    const double laplacian_uy = d2uy_dx2 + d2uy_dy2;

    // Viscous term: -(ν/2)∇²U  (paper Eq. 42e: (ν T, T) → strong form -(ν/2)∆U)
    const double viscous_x = -(nu / 2.0) * laplacian_ux;
    const double viscous_y = -(nu / 2.0) * laplacian_uy;

    // Pressure gradient  (p ∝ t²)
    const double dp_dx = -t2 * pi * sin_px * cos_py;
    const double dp_dy = -t2 * (pi / L_y) * cos_px * sin_py;

    // f = -(ν/2)∇²U + ∇p
    dealii::Tensor<1, dim> f;
    f[0] = viscous_x + dp_dx;
    f[1] = viscous_y + dp_dy;

    return f;
}

/**
 * @brief Compute MMS source for UNSTEADY STOKES (Phase B)
 *
 * Strong form: ∂U/∂t - (ν/2)∇²U + ∇p = f
 * Uses continuous time derivative.
 */
template <int dim>
dealii::Tensor<1, dim> compute_unsteady_stokes_mms_source(
    const dealii::Point<dim>& pt,
    double time,
    double nu,
    double L_y = 1.0)
{
    const double x = pt[0];
    const double y = pt[1];

    const double sin_px = std::sin(M_PI * x);
    const double sin_py = std::sin(M_PI * y / L_y);
    const double sin_2px = std::sin(2.0 * M_PI * x);
    const double sin_2py = std::sin(2.0 * M_PI * y / L_y);

    // Time derivative (continuous, U ∝ t²): ∂U/∂t = 2t · spatial
    //   ux = t²·(π/L_y)·sin²(πx)·sin(2πy/L_y) → ∂ux/∂t = 2t·(π/L_y)·sin²(πx)·sin(2πy/L_y)
    const double two_t = 2.0 * time;
    const double dux_dt = two_t * (M_PI / L_y) * sin_px * sin_px * sin_2py;
    const double duy_dt = -two_t * M_PI * sin_2px * sin_py * sin_py;

    // Steady Stokes source
    dealii::Tensor<1, dim> f_steady = compute_steady_stokes_mms_source<dim>(pt, time, nu, L_y);

    // f = ∂U/∂t + f_steady
    dealii::Tensor<1, dim> f;
    f[0] = dux_dt + f_steady[0];
    f[1] = duy_dt + f_steady[1];

    return f;
}

/**
 * @brief Compute MMS source for STEADY NS (Phase C)
 *
 * Strong form: (U·∇)U - (ν/2)∇²U + ∇p = f
 * Skew term ½(∇·U)U = 0 for divergence-free exact solution.
 */
template <int dim>
dealii::Tensor<1, dim> compute_steady_ns_mms_source(
    const dealii::Point<dim>& pt,
    double time,
    double nu,
    double L_y = 1.0)
{
    const double x = pt[0];
    const double y = pt[1];
    const double t = time;

    const double sin_px = std::sin(M_PI * x);
    const double sin_py = std::sin(M_PI * y / L_y);
    const double cos_2px = std::cos(2.0 * M_PI * x);
    const double sin_2px = std::sin(2.0 * M_PI * x);
    const double sin_2py = std::sin(2.0 * M_PI * y / L_y);
    const double cos_2py = std::cos(2.0 * M_PI * y / L_y);

    const double pi = M_PI;
    const double pi2 = M_PI * M_PI;
    const double t2 = t * t;

    // Exact velocity (U ∝ t²)
    const double ux = t2 * (pi / L_y) * sin_px * sin_px * sin_2py;
    const double uy = -t2 * pi * sin_2px * sin_py * sin_py;

    // Gradients (∇U ∝ t²)
    const double dux_dx = t2 * (pi2 / L_y) * sin_2px * sin_2py;
    const double dux_dy = t2 * (2.0 * pi2 / (L_y * L_y)) * sin_px * sin_px * cos_2py;
    const double duy_dx = -t2 * 2.0 * pi2 * cos_2px * sin_py * sin_py;
    const double duy_dy = -t2 * (pi2 / L_y) * sin_2px * sin_2py;

    // Convection: (U·∇)U  (U ∝ t², ∇U ∝ t² → convection ∝ t⁴)
    const double convect_x = ux * dux_dx + uy * dux_dy;
    const double convect_y = ux * duy_dx + uy * duy_dy;

    // Steady Stokes source
    dealii::Tensor<1, dim> f_steady = compute_steady_stokes_mms_source<dim>(pt, time, nu, L_y);

    // f = (U·∇)U + f_steady
    dealii::Tensor<1, dim> f;
    f[0] = convect_x + f_steady[0];
    f[1] = convect_y + f_steady[1];

    return f;
}

/**
 * @brief Compute MMS source for UNSTEADY NS (Phase D) - Semi-implicit
 *
 * Discrete form: (U^n - U^{n-1})/τ + (U^{n-1}·∇)U^n - (ν/2)∇²U^n + ∇p^n = f
 * This matches the production ns_assembler.cc discretization.
 */
template <int dim>
dealii::Tensor<1, dim> compute_unsteady_ns_mms_source(
    const dealii::Point<dim>& pt,
    double t_new,
    double t_old,
    double nu,
    double L_y = 1.0,
    bool analytical_dt = false)
{
    const double x = pt[0];
    const double y = pt[1];

    const double sin_px = std::sin(M_PI * x);
    const double sin_py = std::sin(M_PI * y / L_y);
    const double cos_2px = std::cos(2.0 * M_PI * x);
    const double sin_2px = std::sin(2.0 * M_PI * x);
    const double sin_2py = std::sin(2.0 * M_PI * y / L_y);
    const double cos_2py = std::cos(2.0 * M_PI * y / L_y);

    const double pi = M_PI;
    const double pi2 = M_PI * M_PI;
    const double t2_old = t_old * t_old;
    const double t2_new = t_new * t_new;

    // Exact velocities at OLD time (for convection) — U ∝ t²
    const double ux_old = t2_old * (pi / L_y) * sin_px * sin_px * sin_2py;
    const double uy_old = -t2_old * pi * sin_2px * sin_py * sin_py;

    // Exact velocities at NEW time
    const double ux_new = t2_new * (pi / L_y) * sin_px * sin_px * sin_2py;
    const double uy_new = -t2_new * pi * sin_2px * sin_py * sin_py;

    // Gradients at NEW time (∇U ∝ t²)
    const double dux_dx_new = t2_new * (pi2 / L_y) * sin_2px * sin_2py;
    const double dux_dy_new = t2_new * (2.0 * pi2 / (L_y * L_y)) * sin_px * sin_px * cos_2py;
    const double duy_dx_new = -t2_new * 2.0 * pi2 * cos_2px * sin_py * sin_py;
    const double duy_dy_new = -t2_new * (pi2 / L_y) * sin_2px * sin_2py;

    // dU/dt: discrete (cancels BE truncation by construction; default)
    //    vs analytical (exposes formal BE temporal rate ~1.0).
    // U ∝ t² ⇒  analytical dU/dt = 2t · spatial,
    //          discrete (U_new - U_old)/dt = ((t²_new - t²_old)/dt) · spatial
    //                                       = (t_new + t_old) · spatial.
    // The two differ by O(τ d²U/dt²) — the BE truncation error.
    double dux_dt, duy_dt;
    if (analytical_dt)
    {
        const double two_t = 2.0 * t_new;
        dux_dt =  two_t * (pi / L_y) * sin_px * sin_px * sin_2py;
        duy_dt = -two_t * pi * sin_2px * sin_py * sin_py;
    }
    else
    {
        const double dt = t_new - t_old;
        dux_dt = (ux_new - ux_old) / dt;
        duy_dt = (uy_new - uy_old) / dt;
    }

    // Semi-implicit convection: (U^{n-1}·∇)U^n
    const double convect_x = ux_old * dux_dx_new + uy_old * dux_dy_new;
    const double convect_y = ux_old * duy_dx_new + uy_old * duy_dy_new;

    // Steady Stokes source at t_new
    dealii::Tensor<1, dim> f_steady = compute_steady_stokes_mms_source<dim>(pt, t_new, nu, L_y);

    // f = (U^n - U^{n-1})/dt + (U^{n-1}·∇)U^n + f_steady(t_new)
    dealii::Tensor<1, dim> f;
    f[0] = dux_dt + convect_x + f_steady[0];
    f[1] = duy_dt + convect_y + f_steady[1];

    return f;
}

// ============================================================================
// KELVIN FORCE MMS SOURCE - For NS + Magnetization Coupled Test
//
// Paper Eq. 57: B_h^m(V, H, M) cell term = (V·∇)H·M + ½div(V)(H·M)
// By IBP (V=0 on ∂Ω): ½∫div(V)(H·M) = -½∫V·∇(H·M)
// Effective strong-form body force: F_K = μ₀[(M·∇)H - ½∇(H·M)]
//
// For MMS, we use:
//   M* = (t·sin(πx)·sin(πy/L_y), t·cos(πx)·sin(πy/L_y))  [from magnetization_mms.h]
//   φ* = t·cos(πx)·cos(πy/L_y)                           [from poisson_mms.h]
//   H* = ∇φ*
//
// The MMS test solves:
//   NS(U_h) = f_NS + μ₀ B_h^m(V, H_h, M_h)
//
// To converge to U*, we need to SUBTRACT F_K(M*, H*) from f_NS:
//   f_NS_coupled = f_NS - F_K(M*, H*)
//
// Then: f_NS_coupled + μ₀ B_h^m(V, H_h, M_h) → f_NS as h→0
// ============================================================================

/**
 * @brief Compute effective Kelvin body force F_K(M*, H*) for MMS
 *
 * F_K = μ₀[(M·∇)H - ½∇(H·M)]
 *
 * This is the strong-form equivalent of the assembler's B_h^m(V, H, M),
 * obtained by integration by parts of the ½div(V)(H·M) stabilization.
 * Should be SUBTRACTED from f_NS to get the coupled MMS source.
 *
 * @param pt Quadrature point
 * @param time Current time
 * @param mu_0 Magnetic permeability (typically 1.0 for nondimensional)
 * @param L_y Domain height
 * @return Effective Kelvin body force vector
 */
template <int dim>
dealii::Tensor<1, dim> compute_kelvin_force_mms_source(
    const dealii::Point<dim>& pt,
    double time,
    double mu_0,
    double L_y = 1.0)
{
    const double x = pt[0];
    const double y = pt[1];
    const double t = time;

    const double pi = M_PI;
    const double pi2 = pi * pi;

    // Trig functions
    const double sin_px = std::sin(pi * x);
    const double cos_px = std::cos(pi * x);
    const double sin_py = std::sin(pi * y / L_y);
    const double cos_py = std::cos(pi * y / L_y);

    // ---------------------------------------------
    // Exact M* from magnetization_mms.h (M ∝ t² as of 2026-05-05):
    //   Mx* = t²·sin(πx)·sin(πy/L_y)
    //   My* = t²·cos(πx)·sin(πy/L_y)
    // ---------------------------------------------
    const double t2 = t * t;
    const double Mx = t2 * sin_px * sin_py;
    const double My = t2 * cos_px * sin_py;

    // Gradients of M* (∇M ∝ t²):
    const double dMx_dx = t2 * pi * cos_px * sin_py;
    const double dMx_dy = t2 * (pi / L_y) * sin_px * cos_py;
    const double dMy_dx = -t2 * pi * sin_px * sin_py;
    const double dMy_dy = t2 * (pi / L_y) * cos_px * cos_py;

    // ---------------------------------------------
    // Exact φ* from poisson_mms.h:
    //   φ* = t·cos(πx)·cos(πy/L_y)
    //
    // Exact H* = ∇φ* (Nochetto CMAME 2016: H = ∇Φ):
    //   Hx* = ∂φ*/∂x = -t·π·sin(πx)·cos(πy/L_y)
    //   Hy* = ∂φ*/∂y = -t·(π/L_y)·cos(πx)·sin(πy/L_y)
    // ---------------------------------------------
    const double Hx = -t * pi * sin_px * cos_py;
    const double Hy = -t * (pi / L_y) * cos_px * sin_py;

    // Gradients of H* = Hess(φ*):
    //   ∂Hx/∂x = ∂²φ*/∂x² = -t·π²·cos(πx)·cos(πy/L_y)
    //   ∂Hx/∂y = ∂²φ*/∂x∂y = t·(π²/L_y)·sin(πx)·sin(πy/L_y)
    //   ∂Hy/∂x = ∂²φ*/∂y∂x = t·(π²/L_y)·sin(πx)·sin(πy/L_y)
    //   ∂Hy/∂y = ∂²φ*/∂y² = -t·(π²/L_y²)·cos(πx)·cos(πy/L_y)
    const double dHx_dx = -t * pi2 * cos_px * cos_py;
    const double dHx_dy = t * (pi2 / L_y) * sin_px * sin_py;
    const double dHy_dx = t * (pi2 / L_y) * sin_px * sin_py;
    const double dHy_dy = -t * (pi2 / (L_y * L_y)) * cos_px * cos_py;

    // ---------------------------------------------
    // Kelvin force (effective body force from B_h^m(V, H, M)):
    //
    //   The assembler computes μ₀[(M·∇)H·V + ½ div(V)(H·M)].
    //   By IBP: ½∫ div(V)(H·M) dx = -½∫ V·∇(H·M) dx  (V=0 on ∂Ω)
    //   So the effective strong-form body force is:
    //     F_K = μ₀[(M·∇)H - ½∇(H·M)]
    // ---------------------------------------------

    // (M·∇)H:
    const double M_grad_H_x = Mx * dHx_dx + My * dHx_dy;
    const double M_grad_H_y = Mx * dHy_dx + My * dHy_dy;

    // ∇(H·M):
    //   ∂(H·M)/∂x = ∂Hx/∂x·Mx + Hx·∂Mx/∂x + ∂Hy/∂x·My + Hy·∂My/∂x
    //   ∂(H·M)/∂y = ∂Hx/∂y·Mx + Hx·∂Mx/∂y + ∂Hy/∂y·My + Hy·∂My/∂y
    const double d_HdotM_dx = dHx_dx*Mx + Hx*dMx_dx + dHy_dx*My + Hy*dMy_dx;
    const double d_HdotM_dy = dHx_dy*Mx + Hx*dMx_dy + dHy_dy*My + Hy*dMy_dy;

    // Total Kelvin force: μ₀[(M·∇)H - ½∇(H·M)]
    dealii::Tensor<1, dim> F_K;
    F_K[0] = mu_0 * (M_grad_H_x - 0.5 * d_HdotM_dx);
    F_K[1] = mu_0 * (M_grad_H_y - 0.5 * d_HdotM_dy);

    return F_K;
}

/**
 * @brief Compute MMS source for NS + Magnetization coupled test
 *
 * f_NS_coupled = f_NS - F_K(M*, H*)
 *
 * This ensures that when the assembler adds the discrete Kelvin force,
 * the total RHS converges to the standalone NS source as h→0.
 *
 * @param pt Quadrature point
 * @param t_new New time level
 * @param t_old Old time level
 * @param nu Viscosity
 * @param mu_0 Magnetic permeability
 * @param L_y Domain height
 * @return Coupled MMS source for NS with Kelvin force
 */
template <int dim>
dealii::Tensor<1, dim> compute_ns_kelvin_coupled_mms_source(
    const dealii::Point<dim>& pt,
    double t_new,
    double t_old,
    double nu,
    double mu_0,
    double L_y = 1.0)
{
    // Standard NS MMS source (manufactures U* without Kelvin force)
    dealii::Tensor<1, dim> f_NS = compute_unsteady_ns_mms_source<dim>(pt, t_new, t_old, nu, L_y);

    // Exact Kelvin force at t_new
    dealii::Tensor<1, dim> F_K = compute_kelvin_force_mms_source<dim>(pt, t_new, mu_0, L_y);

    // Subtract Kelvin force so that total RHS converges to f_NS
    dealii::Tensor<1, dim> f_coupled;
    f_coupled[0] = f_NS[0] - F_K[0];
    f_coupled[1] = f_NS[1] - F_K[1];

    return f_coupled;
}

// ============================================================================
// Error structure
// ============================================================================
struct NSMMSError
{
    double ux_L2 = 0.0;
    double ux_H1 = 0.0;
    double uy_L2 = 0.0;
    double uy_H1 = 0.0;
    double p_L2 = 0.0;
    double div_U_L2 = 0.0;
};

template <int dim>
inline dealii::Tensor<1, dim> compute_ns_mms_source_semi_implicit(
    const dealii::Point<dim>& p,
    double t_new,
    double t_old,
    double nu,
    double L_y = 1.0)
{
    return compute_unsteady_ns_mms_source<dim>(p, t_new, t_old, nu, L_y);
}

#endif // NS_MMS_H