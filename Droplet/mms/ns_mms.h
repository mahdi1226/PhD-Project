// ============================================================================
// mms/ns_mms.h - Navier-Stokes Method of Manufactured Solutions
//
// Exact solutions satisfying no-slip BCs on [0,1]×[0,L_y]:
//
//   Stream function: ψ = t · sin²(πx) · sin²(πy/L_y)
//
//   ux_exact = ∂ψ/∂y = t · (π/L_y) · sin²(πx) · sin(2πy/L_y)
//   uy_exact = -∂ψ/∂x = -t · π · sin(2πx) · sin²(πy/L_y)
//   p_exact  = t · cos(πx) · cos(πy/L_y)
//
// This ensures:
//   - div(U) = 0 exactly (incompressible)
//   - U = 0 on all boundaries (no-slip)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef NS_MMS_H
#define NS_MMS_H

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/function.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>

#include "utilities/parameters.h"

#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Exact velocity x-component
// ux = t · (π/L_y) · sin²(πx) · sin(2πy/L_y)
// ============================================================================
template <int dim>
class NSExactVelocityX : public dealii::Function<dim>
{
public:
    NSExactVelocityX(double time, double L_y = 0.6)
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
        // ∂ux/∂x = t·(π/L_y)·2·sin(πx)·cos(πx)·π·sin(2πy/L_y) = t·(π²/L_y)·sin(2πx)·sin(2πy/L_y)
        grad[0] = time_ * (M_PI * M_PI / L_y_) * std::sin(2.0 * M_PI * x) * std::sin(2.0 * M_PI * y / L_y_);
        // ∂ux/∂y = t·(π/L_y)·sin²(πx)·(2π/L_y)·cos(2πy/L_y) = t·(2π²/L_y²)·sin²(πx)·cos(2πy/L_y)
        const double sin_px = std::sin(M_PI * x);
        grad[1] = time_ * (2.0 * M_PI * M_PI / (L_y_ * L_y_)) * sin_px * sin_px * std::cos(2.0 * M_PI * y / L_y_);

        return grad;
    }

    void set_time(double t) override { time_ = t; }

private:
    double time_;
    double L_y_;
};

// ============================================================================
// Exact velocity y-component
// uy = -t · π · sin(2πx) · sin²(πy/L_y)
// ============================================================================
template <int dim>
class NSExactVelocityY : public dealii::Function<dim>
{
public:
    NSExactVelocityY(double time, double L_y = 0.6)
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
        // ∂uy/∂x = -t·π·2π·cos(2πx)·sin²(πy/L_y) = -t·2π²·cos(2πx)·sin²(πy/L_y)
        const double sin_py = std::sin(M_PI * y / L_y_);
        grad[0] = -time_ * 2.0 * M_PI * M_PI * std::cos(2.0 * M_PI * x) * sin_py * sin_py;
        // ∂uy/∂y = -t·π·sin(2πx)·2·sin(πy/L_y)·cos(πy/L_y)·(π/L_y) = -t·(π²/L_y)·sin(2πx)·sin(2πy/L_y)
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
// p = t · cos(πx) · cos(πy/L_y)
// ============================================================================
template <int dim>
class NSExactPressure : public dealii::Function<dim>
{
public:
    NSExactPressure(double time, double L_y = 0.6)
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
// NS MMS Error Results
// ============================================================================
struct NSMMSError
{
    double ux_L2 = 0.0;
    double ux_H1 = 0.0;
    double uy_L2 = 0.0;
    double uy_H1 = 0.0;
    double p_L2 = 0.0;
    double div_U_L2 = 0.0;

    void print(unsigned int refinement, double h) const;
    void print_for_convergence(double h) const;
};

// ============================================================================
// Compute NS MMS errors
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
    double L_y = 0.6);

// ============================================================================
// Compute MMS source term for momentum equation
// f = ∂U/∂t + (U·∇)U - ν∇·T(U) + ∇p
// where T(U) = ∇U + (∇U)^T
// ============================================================================
template <int dim>
dealii::Tensor<1, dim> compute_ns_mms_source(
    const dealii::Point<dim>& p,
    double time,
    double nu,
    double L_y = 0.6);

// ============================================================================
// Get exact velocity at a point
// ============================================================================
template <int dim>
dealii::Tensor<1, dim> ns_mms_exact_velocity(
    const dealii::Point<dim>& p,
    double time,
    double L_y = 0.6);

// ============================================================================
// Get exact pressure at a point
// ============================================================================
template <int dim>
double ns_mms_exact_pressure(
    const dealii::Point<dim>& p,
    double time,
    double L_y = 0.6);

#endif // NS_MMS_H