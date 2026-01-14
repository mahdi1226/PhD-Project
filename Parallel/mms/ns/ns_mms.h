// ============================================================================
// ns_mms.h - Navier-Stokes MMS Exact Solutions (Parallel Version)
//
// PAPER EQUATION 42e (standalone, constant ν):
//   (U^n - U^{n-1})/dt + B_h(U^{n-1}, U^n, V) + ν(T(U^n), T(V))/2
//     - (p^n, ∇·V) + (∇·U^n, Q) = (f, V)
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
// ux = t·(π/L_y)·sin²(πx)·sin(2πy/L_y)
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
        return time_ * (M_PI / L_y_) * sin_px * sin_px * std::sin(2.0 * M_PI * y / L_y_);
    }

    virtual dealii::Tensor<1, dim> gradient(const dealii::Point<dim>& p,
                                            const unsigned int component = 0) const override
    {
        (void)component;
        const double x = p[0];
        const double y = p[1];

        dealii::Tensor<1, dim> grad;
        // ∂ux/∂x = t·(π²/L_y)·sin(2πx)·sin(2πy/L_y)
        grad[0] = time_ * (M_PI * M_PI / L_y_) * std::sin(2.0 * M_PI * x) * std::sin(2.0 * M_PI * y / L_y_);
        // ∂ux/∂y = t·(2π²/L_y²)·sin²(πx)·cos(2πy/L_y)
        const double sin_px = std::sin(M_PI * x);
        grad[1] = time_ * (2.0 * M_PI * M_PI / (L_y_ * L_y_)) * sin_px * sin_px * std::cos(2.0 * M_PI * y / L_y_);

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
// uy = -t·π·sin(2πx)·sin²(πy/L_y)
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
        return -time_ * M_PI * std::sin(2.0 * M_PI * x) * sin_py * sin_py;
    }

    virtual dealii::Tensor<1, dim> gradient(const dealii::Point<dim>& p,
                                            const unsigned int component = 0) const override
    {
        (void)component;
        const double x = p[0];
        const double y = p[1];

        dealii::Tensor<1, dim> grad;
        // ∂uy/∂x = -t·2π²·cos(2πx)·sin²(πy/L_y)
        const double sin_py = std::sin(M_PI * y / L_y_);
        grad[0] = -time_ * 2.0 * M_PI * M_PI * std::cos(2.0 * M_PI * x) * sin_py * sin_py;
        // ∂uy/∂y = -t·(π²/L_y)·sin(2πx)·sin(2πy/L_y)
        grad[1] = -time_ * (M_PI * M_PI / L_y_) * std::sin(2.0 * M_PI * x) * std::sin(2.0 * M_PI * y / L_y_);

        return grad;
    }

    void set_time(const double t) override { time_ = t; }

private:
    double time_;
    double L_y_;
};

// ============================================================================
// Exact pressure
// p = t·cos(πx)·cos(πy/L_y)
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
        return time_ * std::cos(M_PI * x) * std::cos(M_PI * y / L_y_);
    }

    virtual dealii::Tensor<1, dim> gradient(const dealii::Point<dim>& p,
                                            const unsigned int component = 0) const override
    {
        (void)component;
        const double x = p[0];
        const double y = p[1];

        dealii::Tensor<1, dim> grad;
        grad[0] = -time_ * M_PI * std::sin(M_PI * x) * std::cos(M_PI * y / L_y_);
        grad[1] = -time_ * (M_PI / L_y_) * std::cos(M_PI * x) * std::sin(M_PI * y / L_y_);

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

    dealii::Tensor<1, dim> U;
    const double sin_px = std::sin(M_PI * x);
    U[0] = time * (M_PI / L_y) * sin_px * sin_px * std::sin(2.0 * M_PI * y / L_y);

    const double sin_py = std::sin(M_PI * y / L_y);
    U[1] = -time * M_PI * std::sin(2.0 * M_PI * x) * sin_py * sin_py;

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
    return time * std::cos(M_PI * x) * std::cos(M_PI * y / L_y);
}

// ============================================================================
// MMS Source Terms for Different Phases
// ============================================================================

/**
 * @brief Compute MMS source for STEADY STOKES (Phase A)
 *
 * Strong form: -2ν∇²U + ∇p = f
 * (Factor of 2 from symmetric gradient: T(U) = ∇U + (∇U)^T)
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

    // ∇²ux = ∂²ux/∂x² + ∂²ux/∂y²
    const double d2ux_dx2 = t * (2.0 * pi3 / L_y) * cos_2px * sin_2py;
    const double d2ux_dy2 = -t * (4.0 * pi3 / (L_y * L_y * L_y)) * sin_px * sin_px * sin_2py;
    const double laplacian_ux = d2ux_dx2 + d2ux_dy2;

    // ∇²uy = ∂²uy/∂x² + ∂²uy/∂y²
    const double d2uy_dx2 = t * 4.0 * pi3 * sin_2px * sin_py * sin_py;
    const double d2uy_dy2 = -t * (2.0 * pi3 / (L_y * L_y)) * sin_2px * (cos_py * cos_py - sin_py * sin_py);
    const double laplacian_uy = d2uy_dx2 + d2uy_dy2;

    // Viscous term: -ν∇²U
    const double viscous_x = -nu * laplacian_ux;
    const double viscous_y = -nu * laplacian_uy;

    // Pressure gradient
    const double dp_dx = -t * pi * sin_px * cos_py;
    const double dp_dy = -t * (pi / L_y) * cos_px * sin_py;

    // f = -2ν∇²U + ∇p
    dealii::Tensor<1, dim> f;
    f[0] = viscous_x + dp_dx;
    f[1] = viscous_y + dp_dy;

    return f;
}

/**
 * @brief Compute MMS source for UNSTEADY STOKES (Phase B)
 *
 * Strong form: ∂U/∂t - 2ν∇²U + ∇p = f
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

    // Time derivative (continuous): ∂U/∂t
    // ux = t·(π/L_y)·sin²(πx)·sin(2πy/L_y) → ∂ux/∂t = (π/L_y)·sin²(πx)·sin(2πy/L_y)
    const double dux_dt = (M_PI / L_y) * sin_px * sin_px * sin_2py;
    const double duy_dt = -M_PI * sin_2px * sin_py * sin_py;

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
 * Strong form: (U·∇)U - 2ν∇²U + ∇p = f
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

    // Exact velocity
    const double ux = t * (pi / L_y) * sin_px * sin_px * sin_2py;
    const double uy = -t * pi * sin_2px * sin_py * sin_py;

    // Gradients
    const double dux_dx = t * (pi2 / L_y) * sin_2px * sin_2py;
    const double dux_dy = t * (2.0 * pi2 / (L_y * L_y)) * sin_px * sin_px * cos_2py;
    const double duy_dx = -t * 2.0 * pi2 * cos_2px * sin_py * sin_py;
    const double duy_dy = -t * (pi2 / L_y) * sin_2px * sin_2py;

    // Convection: (U·∇)U
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
 * Discrete form: (U^n - U^{n-1})/τ + (U^{n-1}·∇)U^n - 2ν∇²U^n + ∇p^n = f
 * This matches the production ns_assembler.cc discretization.
 */
template <int dim>
dealii::Tensor<1, dim> compute_unsteady_ns_mms_source(
    const dealii::Point<dim>& pt,
    double t_new,
    double t_old,
    double nu,
    double L_y = 1.0)
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

    // Exact velocities at OLD time (for convection)
    const double ux_old = t_old * (pi / L_y) * sin_px * sin_px * sin_2py;
    const double uy_old = -t_old * pi * sin_2px * sin_py * sin_py;

    // Exact velocities at NEW time
    const double ux_new = t_new * (pi / L_y) * sin_px * sin_px * sin_2py;
    const double uy_new = -t_new * pi * sin_2px * sin_py * sin_py;

    // Gradients at NEW time (what's being convected)
    const double dux_dx_new = t_new * (pi2 / L_y) * sin_2px * sin_2py;
    const double dux_dy_new = t_new * (2.0 * pi2 / (L_y * L_y)) * sin_px * sin_px * cos_2py;
    const double duy_dx_new = -t_new * 2.0 * pi2 * cos_2px * sin_py * sin_py;
    const double duy_dy_new = -t_new * (pi2 / L_y) * sin_2px * sin_2py;

    // DISCRETE time derivative: (U^n - U^{n-1})/dt
    const double dt = t_new - t_old;
    const double dux_dt = (ux_new - ux_old) / dt;
    const double duy_dt = (uy_new - uy_old) / dt;

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