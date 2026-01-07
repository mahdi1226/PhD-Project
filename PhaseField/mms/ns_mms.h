// ============================================================================
// mms/ns_mms.h - Navier-Stokes Method of Manufactured Solutions (Header-Only)
//
// PAPER EQUATION 42e (standalone, constant ν):
//   (U^n - U^{n-1})/dt + B_h(U^{n-1}, U^n, V) + ν(T(U^n), T(V))/2
//     - (p^n, ∇·V) + (∇·U^n, Q) = (f, V)
//
// where:
//   T(U) = ∇U + (∇U)^T (symmetric gradient)
//   B_h(w,u,v) = (w·∇u, v) + ½(∇·w)(u, v) (skew convection)
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
// BUG FIX (2024): ∂²uy/∂y² was missing factor of 2.
//   Correct: d2uy_dy2 = -t·(2π³/L_y²)·sin(2πx)·cos(2πy/L_y)
//   Was:     d2uy_dy2 = -t·(π³/L_y²)·sin(2πx)·cos(2πy/L_y)  [WRONG]
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef NS_MMS_H
#define NS_MMS_H

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/numerics/vector_tools.h>

#include <cmath>
#include <string>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <map>

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
        : dealii::Function<dim>(1), time_(time), L_y_(L_y) {}

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

    void set_time(double t) override { time_ = t; }
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
        : dealii::Function<dim>(1), time_(time), L_y_(L_y) {}

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

    void set_time(double t) override { time_ = t; }

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
        : dealii::Function<dim>(1), time_(time), L_y_(L_y) {}

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

    void set_time(double t) override { time_ = t; }

private:
    double time_;
    double L_y_;
};

// ============================================================================
// Get exact velocity at a point (convenience function)
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

// ============================================================================
// Get exact pressure at a point (convenience function)
// ============================================================================
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
// Compute MMS source term for momentum equation
//
// f = ∂U/∂t + (U·∇)U - 2ν∇²U + ∇p
//
// Note: The actual ns_assembler.cc uses (ν)(T(U),T(V)), which corresponds to
// strong form -2ν∇²U (NOT -ν∇²U). This matches the paper's formulation.
// ============================================================================
template <int dim>
dealii::Tensor<1, dim> compute_ns_mms_source(
    const dealii::Point<dim>& pt,
    double time,
    double nu,
    double L_y = 1.0)
{
    const double x = pt[0];
    const double y = pt[1];
    const double t = time;

    // Precompute trig functions
    const double sin_px = std::sin(M_PI * x);
    const double cos_px = std::cos(M_PI * x);
    const double sin_2px = std::sin(2.0 * M_PI * x);
    const double cos_2px = std::cos(2.0 * M_PI * x);

    const double sin_py = std::sin(M_PI * y / L_y);
    const double cos_py = std::cos(M_PI * y / L_y);
    const double sin_2py = std::sin(2.0 * M_PI * y / L_y);
    const double cos_2py = std::cos(2.0 * M_PI * y / L_y);

    const double pi = M_PI;
    const double pi2 = M_PI * M_PI;

    // Exact solution values
    const double ux = t * (pi / L_y) * sin_px * sin_px * sin_2py;
    const double uy = -t * pi * sin_2px * sin_py * sin_py;

    // ========================================================================
    // Term 1: ∂U/∂t
    // ========================================================================
    const double dux_dt = (pi / L_y) * sin_px * sin_px * sin_2py;
    const double duy_dt = -pi * sin_2px * sin_py * sin_py;

    // ========================================================================
    // Term 2: (U·∇)U = ux·∂U/∂x + uy·∂U/∂y
    // ========================================================================
    // Gradients of ux
    const double dux_dx = t * (pi2 / L_y) * sin_2px * sin_2py;
    const double dux_dy = t * (2.0 * pi2 / (L_y * L_y)) * sin_px * sin_px * cos_2py;

    // Gradients of uy
    const double duy_dx = -t * 2.0 * pi2 * cos_2px * sin_py * sin_py;
    const double duy_dy = -t * (pi2 / L_y) * sin_2px * sin_2py;

    const double convect_x = ux * dux_dx + uy * dux_dy;
    const double convect_y = ux * duy_dx + uy * duy_dy;

    // ========================================================================
    // Term 3: -ν∇²U
    // For incompressible flow: -ν∇·T(U) = -ν∇²U (since ∇·U = 0)
    // ========================================================================
    // Second derivatives of ux
    const double d2ux_dx2 = t * (2.0 * pi2 * pi / L_y) * cos_2px * sin_2py;
    const double d2ux_dy2 = -t * (4.0 * pi2 * pi / (L_y * L_y * L_y)) * sin_px * sin_px * sin_2py;

    // Second derivatives of uy
    const double d2uy_dx2 = t * 4.0 * pi2 * pi * sin_2px * sin_py * sin_py;
    // FIX: Was missing factor of 2 in front of pi2
    const double d2uy_dy2 = -t * (2.0 * pi2 * pi / (L_y * L_y)) * sin_2px * (cos_py * cos_py - sin_py * sin_py);

    const double laplacian_ux = d2ux_dx2 + d2ux_dy2;
    const double laplacian_uy = d2uy_dx2 + d2uy_dy2;

    // Test: using -nu (standard Laplacian weak form)
    const double viscous_x = - 2 * nu * laplacian_ux;
    const double viscous_y = - 2 * nu * laplacian_uy;

    // ========================================================================
    // Term 4: ∇p
    // ========================================================================
    const double dp_dx = -t * pi * sin_px * cos_py;
    const double dp_dy = -t * (pi / L_y) * cos_px * sin_py;

    // ========================================================================
    // Total source: f = ∂U/∂t + (U·∇)U - ν∇²U + ∇p
    // ========================================================================
    dealii::Tensor<1, dim> f;
    f[0] = dux_dt + convect_x + viscous_x + dp_dx;
    f[1] = duy_dt + convect_y + viscous_y + dp_dy;

    return f;
}

// ============================================================================
// NS MMS Source for SEMI-IMPLICIT scheme
//
// For semi-implicit convection: (U^{n-1}·∇)U^n
// Source: f = ∂U/∂t|_{t_n} + (U(t_{n-1})·∇)U(t_n) - 2ν∇²U(t_n) + ∇p(t_n)
// Note: The actual ns_assembler.cc uses (ν)(T(U),T(V)), giving -2ν∇²U
// ============================================================================
template <int dim>
dealii::Tensor<1, dim> compute_ns_mms_source_semi_implicit(
    const dealii::Point<dim>& pt,
    double t_new,    // t^n - time for U^n
    double t_old,    // t^{n-1} - time for convecting velocity
    double nu,
    double L_y = 1.0)
{
    const double x = pt[0];
    const double y = pt[1];

    const double sin_px = std::sin(M_PI * x);
    const double cos_px = std::cos(M_PI * x);
    const double sin_2px = std::sin(2.0 * M_PI * x);
    const double cos_2px = std::cos(2.0 * M_PI * x);

    const double sin_py = std::sin(M_PI * y / L_y);
    const double cos_py = std::cos(M_PI * y / L_y);
    const double sin_2py = std::sin(2.0 * M_PI * y / L_y);
    const double cos_2py = std::cos(2.0 * M_PI * y / L_y);

    const double pi = M_PI;
    const double pi2 = M_PI * M_PI;


    // Exact velocities at OLD time (for convection)
    const double ux_old = t_old * (pi / L_y) * sin_px * sin_px * sin_2py;
    const double uy_old = -t_old * pi * sin_2px * sin_py * sin_py;

    // Exact velocities at NEW time (for discrete time derivative)
    const double ux_new = t_new * (pi / L_y) * sin_px * sin_px * sin_2py;
    const double uy_new = -t_new * pi * sin_2px * sin_py * sin_py;

    // Gradients of exact velocity at NEW time (what's being convected)
    const double dux_dx_new = t_new * (pi2 / L_y) * sin_2px * sin_2py;
    const double dux_dy_new = t_new * (2.0 * pi2 / (L_y * L_y)) * sin_px * sin_px * cos_2py;
    const double duy_dx_new = -t_new * 2.0 * pi2 * cos_2px * sin_py * sin_py;
    const double duy_dy_new = -t_new * (pi2 / L_y) * sin_2px * sin_2py;

    // ========================================================================
    // Term 1: (U^n - U^{n-1})/dt - DISCRETE time derivative
    // ========================================================================
    const double dt = t_new - t_old;
    const double dux_dt = (ux_new - ux_old) / dt;
    const double duy_dt = (uy_new - uy_old) / dt;

    // ========================================================================
    // Term 2: (U^{n-1}·∇)U^n - SEMI-IMPLICIT convection
    // ========================================================================
    const double convect_x = ux_old * dux_dx_new + uy_old * dux_dy_new;
    const double convect_y = ux_old * duy_dx_new + uy_old * duy_dy_new;

    // ========================================================================
    // Term 3: -ν∇²U at t_new
    // ========================================================================
    const double d2ux_dx2 = t_new * (2.0 * pi2 * pi / L_y) * cos_2px * sin_2py;
    const double d2ux_dy2 = -t_new * (4.0 * pi2 * pi / (L_y * L_y * L_y)) * sin_px * sin_px * sin_2py;
    const double d2uy_dx2 = t_new * 4.0 * pi2 * pi * sin_2px * sin_py * sin_py;
    // FIX: Was missing factor of 2 in front of pi2
    const double d2uy_dy2 = -t_new * (2.0 * pi2 * pi / (L_y * L_y)) * sin_2px * (cos_py * cos_py - sin_py * sin_py);

    const double laplacian_ux = d2ux_dx2 + d2ux_dy2;
    const double laplacian_uy = d2uy_dx2 + d2uy_dy2;

    // Test: using -nu (standard Laplacian weak form)
    const double viscous_x = - 2 * nu * laplacian_ux;
    const double viscous_y = - 2 * nu * laplacian_uy;

    // ========================================================================
    // Term 4: ∇p at t_new
    // ========================================================================
    const double dp_dx = - t_new * pi * sin_px * cos_py;
    const double dp_dy = - t_new * (pi / L_y) * cos_px * sin_py;

    // ========================================================================
    // Total: f = ∂U/∂t + (U^{n-1}·∇)U^n - ν∇²U + ∇p
    // ========================================================================
    dealii::Tensor<1, dim> f;
    f[0] = dux_dt + convect_x + viscous_x + dp_dx;
    f[1] = duy_dt + convect_y + viscous_y + dp_dy;

    return f;
}

// ============================================================================
// NS MMS Source WITHOUT convection (for debugging)
//
// f = ∂U/∂t - 2ν∇²U + ∇p  (no convection term)
// Note: The actual ns_assembler.cc uses (ν)(T(U),T(V)), giving -2ν∇²U
// ============================================================================
template <int dim>
dealii::Tensor<1, dim> compute_ns_mms_source_no_convection(
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
    const double sin_2px = std::sin(2.0 * M_PI * x);
    const double cos_2px = std::cos(2.0 * M_PI * x);

    const double sin_py = std::sin(M_PI * y / L_y);
    const double cos_py = std::cos(M_PI * y / L_y);
    const double sin_2py = std::sin(2.0 * M_PI * y / L_y);

    const double pi = M_PI;
    const double pi2 = M_PI * M_PI;

    // Term 1: ∂U/∂t
    const double dux_dt = (pi / L_y) * sin_px * sin_px * sin_2py;
    const double duy_dt = -pi * sin_2px * sin_py * sin_py;

    // Term 2: -ν∇²U
    const double d2ux_dx2 = t * (2.0 * pi2 * pi / L_y) * cos_2px * sin_2py;
    const double d2ux_dy2 = -t * (4.0 * pi2 * pi / (L_y * L_y * L_y)) * sin_px * sin_px * sin_2py;
    const double d2uy_dx2 = t * 4.0 * pi2 * pi * sin_2px * sin_py * sin_py;
    // FIX: Was missing factor of 2 in front of pi2
    const double d2uy_dy2 = -t * (2.0 * pi2 * pi / (L_y * L_y)) * sin_2px * (cos_py * cos_py - sin_py * sin_py);

    const double laplacian_ux = d2ux_dx2 + d2ux_dy2;
    const double laplacian_uy = d2uy_dx2 + d2uy_dy2;

    // Use -nu (same as semi_implicit source which works!)
    const double viscous_x = - 2 * nu * laplacian_ux;
    const double viscous_y = - 2 * nu * laplacian_uy;

    // Term 3: ∇p
    const double dp_dx = -t * pi * sin_px * cos_py;
    const double dp_dy = -t * (pi / L_y) * cos_px * sin_py;

    // Total source (NO convection): f = ∂U/∂t - ν∇²U + ∇p
    dealii::Tensor<1, dim> f;
    f[0] = dux_dt + viscous_x + dp_dx;
    f[1] = duy_dt + viscous_y + dp_dy;

    return f;
}

// ============================================================================
// NS MMS Source with ONLY mass and pressure (for debugging)
//
// f = ∂U/∂t + ∇p  (no convection, no viscosity)
// ============================================================================
template <int dim>
dealii::Tensor<1, dim> compute_ns_mms_source_mass_pressure_only(
    const dealii::Point<dim>& pt,
    double time,
    double L_y = 1.0)
{
    const double x = pt[0];
    const double y = pt[1];
    const double t = time;

    const double sin_px = std::sin(M_PI * x);
    const double cos_px = std::cos(M_PI * x);
    const double sin_2px = std::sin(2.0 * M_PI * x);

    const double sin_py = std::sin(M_PI * y / L_y);
    const double cos_py = std::cos(M_PI * y / L_y);
    const double sin_2py = std::sin(2.0 * M_PI * y / L_y);

    const double pi = M_PI;

    // Term 1: ∂U/∂t
    const double dux_dt = (pi / L_y) * sin_px * sin_px * sin_2py;
    const double duy_dt = -pi * sin_2px * sin_py * sin_py;

    // Term 2: ∇p
    const double dp_dx = -t * pi * sin_px * cos_py;
    const double dp_dy = -t * (pi / L_y) * cos_px * sin_py;

    // Total source: f = ∂U/∂t + ∇p
    dealii::Tensor<1, dim> f;
    f[0] = dux_dt + dp_dx;
    f[1] = duy_dt + dp_dy;

    return f;
}

// ============================================================================
// NS MMS Error Results
// ============================================================================
struct NSMMSError
{
    double ux_L2 = 0.0;
    double ux_H1 = 0.0;
    double uy_L2 = 0.0;
    double uy_H1 = 0.0;
    double p_L2 = 0.0;
    double div_U_L2 = 0.0;  // Should be ~0 for incompressible

    void print(unsigned int refinement, double h) const
    {
        std::cout << "[NS MMS] ref=" << refinement
                  << " h=" << std::scientific << std::setprecision(2) << h
                  << " ux_L2=" << std::setprecision(4) << ux_L2
                  << " ux_H1=" << ux_H1
                  << " uy_L2=" << uy_L2
                  << " p_L2=" << p_L2
                  << " divU=" << div_U_L2
                  << std::defaultfloat << "\n";
    }

    void print_for_convergence(double h) const
    {
        std::cout << std::scientific << std::setprecision(6)
                  << h << "\t"
                  << ux_L2 << "\t" << ux_H1 << "\t"
                  << uy_L2 << "\t" << uy_H1 << "\t"
                  << p_L2 << "\n";
    }
};

// ============================================================================
// Compute NS MMS errors
//
// For pressure: computed with zero-mean adjustment (pressure unique up to const)
// ============================================================================
template <int dim>
NSMMSError compute_ns_mms_error(
    const dealii::DoFHandler<dim>& ux_dof_handler,
    const dealii::DoFHandler<dim>& uy_dof_handler,
    const dealii::DoFHandler<dim>& p_dof_handler,
    const dealii::Vector<double>& ux_solution,
    const dealii::Vector<double>& uy_solution,
    const dealii::Vector<double>& p_solution,
    double time,
    double L_y = 1.0)
{
    NSMMSError error;

    // Setup FE evaluation for velocity (Q2)
    const auto& ux_fe = ux_dof_handler.get_fe();
    const auto& p_fe = p_dof_handler.get_fe();

    const unsigned int quad_degree = ux_fe.degree + 2;
    dealii::QGauss<dim> quadrature(quad_degree);
    const unsigned int n_q_points = quadrature.size();

    dealii::FEValues<dim> ux_fe_values(ux_fe, quadrature,
        dealii::update_values | dealii::update_gradients |
        dealii::update_quadrature_points | dealii::update_JxW_values);

    dealii::FEValues<dim> uy_fe_values(ux_fe, quadrature,
        dealii::update_values | dealii::update_gradients);

    dealii::FEValues<dim> p_fe_values(p_fe, quadrature,
        dealii::update_values);

    std::vector<double> ux_values(n_q_points);
    std::vector<double> uy_values(n_q_points);
    std::vector<double> p_values(n_q_points);
    std::vector<dealii::Tensor<1, dim>> ux_gradients(n_q_points);
    std::vector<dealii::Tensor<1, dim>> uy_gradients(n_q_points);

    NSExactVelocityX<dim> exact_ux(time, L_y);
    NSExactVelocityY<dim> exact_uy(time, L_y);
    NSExactPressure<dim> exact_p(time, L_y);

    double ux_L2_sq = 0.0, ux_H1_sq = 0.0;
    double uy_L2_sq = 0.0, uy_H1_sq = 0.0;
    double p_L2_sq = 0.0;
    double div_U_L2_sq = 0.0;

    // Compute mean pressure for both numerical and exact
    double p_mean_num = 0.0, p_mean_exact = 0.0, domain_area = 0.0;

    auto ux_cell = ux_dof_handler.begin_active();
    auto uy_cell = uy_dof_handler.begin_active();
    auto p_cell = p_dof_handler.begin_active();

    // First pass: compute mean pressures
    for (; ux_cell != ux_dof_handler.end(); ++ux_cell, ++uy_cell, ++p_cell)
    {
        ux_fe_values.reinit(ux_cell);
        p_fe_values.reinit(p_cell);

        p_fe_values.get_function_values(p_solution, p_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = ux_fe_values.JxW(q);
            const dealii::Point<dim>& x_q = ux_fe_values.quadrature_point(q);

            p_mean_num += p_values[q] * JxW;
            p_mean_exact += exact_p.value(x_q) * JxW;
            domain_area += JxW;
        }
    }

    p_mean_num /= domain_area;
    p_mean_exact /= domain_area;

    // Second pass: compute errors
    ux_cell = ux_dof_handler.begin_active();
    uy_cell = uy_dof_handler.begin_active();
    p_cell = p_dof_handler.begin_active();

    for (; ux_cell != ux_dof_handler.end(); ++ux_cell, ++uy_cell, ++p_cell)
    {
        ux_fe_values.reinit(ux_cell);
        uy_fe_values.reinit(uy_cell);
        p_fe_values.reinit(p_cell);

        ux_fe_values.get_function_values(ux_solution, ux_values);
        uy_fe_values.get_function_values(uy_solution, uy_values);
        p_fe_values.get_function_values(p_solution, p_values);
        ux_fe_values.get_function_gradients(ux_solution, ux_gradients);
        uy_fe_values.get_function_gradients(uy_solution, uy_gradients);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = ux_fe_values.JxW(q);
            const dealii::Point<dim>& x_q = ux_fe_values.quadrature_point(q);

            // Velocity errors
            const double ux_exact = exact_ux.value(x_q);
            const double uy_exact = exact_uy.value(x_q);
            const dealii::Tensor<1, dim> grad_ux_exact = exact_ux.gradient(x_q);
            const dealii::Tensor<1, dim> grad_uy_exact = exact_uy.gradient(x_q);

            const double ux_err = ux_values[q] - ux_exact;
            const double uy_err = uy_values[q] - uy_exact;
            const dealii::Tensor<1, dim> grad_ux_err = ux_gradients[q] - grad_ux_exact;
            const dealii::Tensor<1, dim> grad_uy_err = uy_gradients[q] - grad_uy_exact;

            ux_L2_sq += ux_err * ux_err * JxW;
            uy_L2_sq += uy_err * uy_err * JxW;
            ux_H1_sq += (grad_ux_err * grad_ux_err) * JxW;
            uy_H1_sq += (grad_uy_err * grad_uy_err) * JxW;

            // Pressure error (zero-mean adjusted)
            const double p_exact = exact_p.value(x_q);
            const double p_err = (p_values[q] - p_mean_num) - (p_exact - p_mean_exact);
            p_L2_sq += p_err * p_err * JxW;

            // Divergence (should be ~0 for incompressible)
            const double div_U = ux_gradients[q][0] + uy_gradients[q][1];
            div_U_L2_sq += div_U * div_U * JxW;
        }
    }

    error.ux_L2 = std::sqrt(ux_L2_sq);
    error.ux_H1 = std::sqrt(ux_H1_sq);
    error.uy_L2 = std::sqrt(uy_L2_sq);
    error.uy_H1 = std::sqrt(uy_H1_sq);
    error.p_L2 = std::sqrt(p_L2_sq);
    error.div_U_L2 = std::sqrt(div_U_L2_sq);

    return error;
}

// ============================================================================
// Apply NS MMS initial conditions
// ============================================================================
template <int dim>
void apply_ns_mms_initial_conditions(
    const dealii::DoFHandler<dim>& ux_dof_handler,
    const dealii::DoFHandler<dim>& uy_dof_handler,
    const dealii::DoFHandler<dim>& p_dof_handler,
    dealii::Vector<double>& ux_solution,
    dealii::Vector<double>& uy_solution,
    dealii::Vector<double>& p_solution,
    double time,
    double L_y = 1.0)
{
    NSExactVelocityX<dim> exact_ux(time, L_y);
    NSExactVelocityY<dim> exact_uy(time, L_y);
    NSExactPressure<dim> exact_p(time, L_y);

    dealii::VectorTools::interpolate(ux_dof_handler, exact_ux, ux_solution);
    dealii::VectorTools::interpolate(uy_dof_handler, exact_uy, uy_solution);
    dealii::VectorTools::interpolate(p_dof_handler, exact_p, p_solution);
}

// ============================================================================
// Apply NS MMS Dirichlet boundary conditions
//
// The exact solution has U = 0 on all boundaries (no-slip), so we apply
// homogeneous Dirichlet BCs for velocity.
//
// For pressure: pin one DoF to fix the constant (pure Neumann problem)
// ============================================================================
template <int dim>
void setup_ns_mms_velocity_constraints(
    const dealii::DoFHandler<dim>& ux_dof_handler,
    const dealii::DoFHandler<dim>& uy_dof_handler,
    dealii::AffineConstraints<double>& ux_constraints,
    dealii::AffineConstraints<double>& uy_constraints)
{
    // Velocity: homogeneous Dirichlet on all boundaries (exact solution is zero there)
    ux_constraints.clear();
    uy_constraints.clear();

    dealii::DoFTools::make_hanging_node_constraints(ux_dof_handler, ux_constraints);
    dealii::DoFTools::make_hanging_node_constraints(uy_dof_handler, uy_constraints);

    // Apply zero Dirichlet on all boundaries
    // Note: hyper_rectangle assigns boundary_id=0 to left/bottom, boundary_id=1 to right/top
    std::map<dealii::types::global_dof_index, double> ux_boundary_values;
    std::map<dealii::types::global_dof_index, double> uy_boundary_values;

    // Apply to both boundary IDs (0 and 1)
    for (dealii::types::boundary_id bid = 0; bid <= 1; ++bid)
    {
        dealii::VectorTools::interpolate_boundary_values(
            ux_dof_handler,
            bid,
            dealii::Functions::ZeroFunction<dim>(),
            ux_boundary_values);

        dealii::VectorTools::interpolate_boundary_values(
            uy_dof_handler,
            bid,
            dealii::Functions::ZeroFunction<dim>(),
            uy_boundary_values);
    }

    for (const auto& [dof, value] : ux_boundary_values)
    {
        if (!ux_constraints.is_constrained(dof))
        {
            ux_constraints.add_line(dof);
            ux_constraints.set_inhomogeneity(dof, value);
        }
    }

    for (const auto& [dof, value] : uy_boundary_values)
    {
        if (!uy_constraints.is_constrained(dof))
        {
            uy_constraints.add_line(dof);
            uy_constraints.set_inhomogeneity(dof, value);
        }
    }

    ux_constraints.close();
    uy_constraints.close();
}

template <int dim>
void setup_ns_mms_pressure_constraints(
    const dealii::DoFHandler<dim>& p_dof_handler,
    dealii::AffineConstraints<double>& p_constraints,
    double time = 1.0,    // Can remove this parameter now
    double L_y = 1.0)     // Can remove this parameter now
{
    p_constraints.clear();

    dealii::DoFTools::make_hanging_node_constraints(p_dof_handler, p_constraints);

    // Pin DoF 0 to zero to fix the constant (mean is subtracted in error computation)
    if (p_dof_handler.n_dofs() > 0)
    {
        if (!p_constraints.is_constrained(0))
        {
            p_constraints.add_line(0);
            p_constraints.set_inhomogeneity(0, 0.0);  // Pin to zero
        }
    }

    p_constraints.close();
}
// ============================================================================
// Compute NS MMS source WITH capillary force (for coupled CH-NS test)
//
// F_cap = ψ·∇θ (from ch_mms.h exact solutions)
// ============================================================================
template <int dim>
dealii::Tensor<1, dim> compute_ns_capillary_force(
    const dealii::Point<dim>& pt,
    double time,
    double lambda,
    double epsilon)
{
    const double x = pt[0], y = pt[1];
    const double t4 = time * time * time * time;

    // θ_exact = t⁴ cos(πx) cos(πy)
    const double theta = t4 * std::cos(M_PI * x) * std::cos(M_PI * y);

    // ∇ψ_exact where ψ = t⁴ sin(πx) sin(πy)
    dealii::Tensor<1, dim> grad_psi;
    grad_psi[0] = t4 * M_PI * std::cos(M_PI * x) * std::sin(M_PI * y);
    grad_psi[1] = t4 * M_PI * std::sin(M_PI * x) * std::cos(M_PI * y);

    // F_cap = (λ/ε) θ ∇ψ  (matching assembler)
    const double coeff = lambda / epsilon;
    return coeff * theta * grad_psi;
}

#endif // NS_MMS_H