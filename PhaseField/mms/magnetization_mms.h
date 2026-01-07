// ============================================================================
// mms/magnetization_mms.h - Magnetization MMS Verification (Header-Only)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Equation 42c (magnetization transport with relaxation)
//
// PAPER EQUATION 42c:
//   (1/τ + 1/τ_M)(M^k, Z) - B_h^m(U^{k-1}, Z, M^k) = (1/τ_M)(χ_θ H^k, Z) + (1/τ)(M^{k-1}, Z)
//
// where:
//   - τ = dt (time step)
//   - τ_M = relaxation time
//   - χ_θ = χ(θ) = χ₀(1+θ)/2 (susceptibility)
//   - H = ∇φ (magnetic field)
//   - B_h^m = DG skew-symmetric transport (Eq. 57)
//
// EXACT SOLUTIONS:
//   Mx = t·sin(πx)·sin(πy/L_y)
//   My = t·cos(πx)·cos(πy/L_y)
//
// For STANDALONE test: U = 0 (no transport), θ = 1 (constant), φ prescribed
//
// ============================================================================
#ifndef MAGNETIZATION_MMS_H
#define MAGNETIZATION_MMS_H

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>

#include <cmath>
#include <iostream>
#include <iomanip>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Exact Mx component
// Mx = t·sin(πx)·sin(πy/L_y)
// ============================================================================
template <int dim>
class MagExactMx : public dealii::Function<dim>
{
public:
    MagExactMx(double time = 1.0, double L_y = 1.0)
        : dealii::Function<dim>(1), time_(time), L_y_(L_y) {}

    virtual double value(const dealii::Point<dim>& p,
                         const unsigned int = 0) const override
    {
        const double x = p[0];
        const double y = p[1];
        return time_ * std::sin(M_PI * x) * std::sin(M_PI * y / L_y_);
    }

    virtual dealii::Tensor<1, dim> gradient(const dealii::Point<dim>& p,
                                            const unsigned int = 0) const override
    {
        const double x = p[0];
        const double y = p[1];
        dealii::Tensor<1, dim> grad;
        grad[0] = time_ * M_PI * std::cos(M_PI * x) * std::sin(M_PI * y / L_y_);
        grad[1] = time_ * (M_PI / L_y_) * std::sin(M_PI * x) * std::cos(M_PI * y / L_y_);
        return grad;
    }

    void set_time(double t) { time_ = t; }
    double get_time() const { return time_; }

private:
    double time_;
    double L_y_;
};

// ============================================================================
// Exact My component
// My = t·cos(πx)·cos(πy/L_y)
// ============================================================================
template <int dim>
class MagExactMy : public dealii::Function<dim>
{
public:
    MagExactMy(double time = 1.0, double L_y = 1.0)
        : dealii::Function<dim>(1), time_(time), L_y_(L_y) {}

    virtual double value(const dealii::Point<dim>& p,
                         const unsigned int = 0) const override
    {
        const double x = p[0];
        const double y = p[1];
        return time_ * std::cos(M_PI * x) * std::cos(M_PI * y / L_y_);
    }

    virtual dealii::Tensor<1, dim> gradient(const dealii::Point<dim>& p,
                                            const unsigned int = 0) const override
    {
        const double x = p[0];
        const double y = p[1];
        dealii::Tensor<1, dim> grad;
        grad[0] = -time_ * M_PI * std::sin(M_PI * x) * std::cos(M_PI * y / L_y_);
        grad[1] = -time_ * (M_PI / L_y_) * std::cos(M_PI * x) * std::sin(M_PI * y / L_y_);
        return grad;
    }

    void set_time(double t) { time_ = t; }

private:
    double time_;
    double L_y_;
};

// ============================================================================
// Get exact magnetization at a point
// ============================================================================
template <int dim>
dealii::Tensor<1, dim> mag_mms_exact_M(
    const dealii::Point<dim>& p,
    double time,
    double L_y = 1.0)
{
    const double x = p[0];
    const double y = p[1];

    dealii::Tensor<1, dim> M;
    M[0] = time * std::sin(M_PI * x) * std::sin(M_PI * y / L_y);
    M[1] = time * std::cos(M_PI * x) * std::cos(M_PI * y / L_y);

    return M;
}

// ============================================================================
// Compute MMS source for magnetization equation (STANDALONE, U=0)
//
// With U = 0, the equation becomes:
//   (1/τ + 1/τ_M)(M^n, Z) = (1/τ_M)(χ·H, Z) + (1/τ)(M^{n-1}, Z) + (f_M, Z)
//
// So:
//   f_M = (1/τ)(M^n - M^{n-1}) + (1/τ_M)(M^n - χ·H)
//
// For simplicity, we set χ·H = 0 (no equilibrium magnetization), giving:
//   f_M = (M^n - M^{n-1})/τ + M^n/τ_M
//
// ============================================================================
template <int dim>
dealii::Tensor<1, dim> compute_mag_mms_source_standalone(
    const dealii::Point<dim>& pt,
    double t_new,
    double t_old,
    double tau_M,
    double L_y = 1.0)
{
    const double dt = t_new - t_old;

    // Get exact M at new and old times
    dealii::Tensor<1, dim> M_new = mag_mms_exact_M<dim>(pt, t_new, L_y);
    dealii::Tensor<1, dim> M_old = mag_mms_exact_M<dim>(pt, t_old, L_y);

    // Source: f = (M^n - M^{n-1})/dt + M^n/τ_M
    // (with χ·H = 0 assumption for standalone test)
    dealii::Tensor<1, dim> f;

    if (tau_M > 0.0)
    {
        f[0] = (M_new[0] - M_old[0]) / dt + M_new[0] / tau_M;
        f[1] = (M_new[1] - M_old[1]) / dt + M_new[1] / tau_M;
    }
    else
    {
        // No relaxation (τ_M → ∞): just time derivative
        f[0] = (M_new[0] - M_old[0]) / dt;
        f[1] = (M_new[1] - M_old[1]) / dt;
    }

    return f;
}

// ============================================================================
// Compute MMS source for magnetization WITH transport (U ≠ 0)
//
// Full equation:
//   (M^n - M^{n-1})/τ + (U·∇)M^n + (1/2)(∇·U)M^n + (M^n - χ·H)/τ_M = f_M
//
// For semi-implicit scheme using U^{n-1}:
//   f_M = (M^n - M^{n-1})/τ + (U^{n-1}·∇)M^n + (1/2)(∇·U^{n-1})M^n + (M^n - χ·H)/τ_M
//
// ============================================================================
template <int dim>
dealii::Tensor<1, dim> compute_mag_mms_source_with_transport(
    const dealii::Point<dim>& pt,
    double t_new,
    double t_old,
    double tau_M,
    double chi_val,           // χ(θ) value
    const dealii::Tensor<1, dim>& H,  // Magnetic field H = ∇φ
    const dealii::Tensor<1, dim>& U,  // Velocity
    double div_U,                      // ∇·U
    double L_y = 1.0)
{
    const double dt = t_new - t_old;

    // Get exact M at new and old times
    dealii::Tensor<1, dim> M_new = mag_mms_exact_M<dim>(pt, t_new, L_y);
    dealii::Tensor<1, dim> M_old = mag_mms_exact_M<dim>(pt, t_old, L_y);

    // Get gradient of M at new time
    const double x = pt[0];
    const double y = pt[1];

    // ∇Mx = (π·cos(πx)·sin(πy/L_y), (π/L_y)·sin(πx)·cos(πy/L_y)) * t
    dealii::Tensor<1, dim> grad_Mx;
    grad_Mx[0] = t_new * M_PI * std::cos(M_PI * x) * std::sin(M_PI * y / L_y);
    grad_Mx[1] = t_new * (M_PI / L_y) * std::sin(M_PI * x) * std::cos(M_PI * y / L_y);

    // ∇My = (-π·sin(πx)·cos(πy/L_y), -(π/L_y)·cos(πx)·sin(πy/L_y)) * t
    dealii::Tensor<1, dim> grad_My;
    grad_My[0] = -t_new * M_PI * std::sin(M_PI * x) * std::cos(M_PI * y / L_y);
    grad_My[1] = -t_new * (M_PI / L_y) * std::cos(M_PI * x) * std::sin(M_PI * y / L_y);

    // Convection: (U·∇)M
    const double convect_Mx = U[0] * grad_Mx[0] + U[1] * grad_Mx[1];
    const double convect_My = U[0] * grad_My[0] + U[1] * grad_My[1];

    // Skew term: (1/2)(∇·U)M
    const double skew_Mx = 0.5 * div_U * M_new[0];
    const double skew_My = 0.5 * div_U * M_new[1];

    // Equilibrium magnetization: χ·H
    const double chi_H_x = chi_val * H[0];
    const double chi_H_y = chi_val * H[1];

    // Source term
    dealii::Tensor<1, dim> f;

    // f = (M^n - M^{n-1})/dt + (U·∇)M + (1/2)(∇·U)M + (M - χ·H)/τ_M
    f[0] = (M_new[0] - M_old[0]) / dt + convect_Mx + skew_Mx;
    f[1] = (M_new[1] - M_old[1]) / dt + convect_My + skew_My;

    if (tau_M > 0.0)
    {
        f[0] += (M_new[0] - chi_H_x) / tau_M;
        f[1] += (M_new[1] - chi_H_y) / tau_M;
    }

    return f;
}

// ============================================================================
// Magnetization MMS Error Results
// ============================================================================
struct MagMMSError
{
    double Mx_L2 = 0.0;
    double My_L2 = 0.0;
    double M_L2 = 0.0;  // Combined ||M||_L2

    void print(unsigned int refinement, double h) const
    {
        std::cout << "[MAG MMS] ref=" << refinement
                  << " h=" << std::scientific << std::setprecision(2) << h
                  << " Mx_L2=" << std::setprecision(4) << Mx_L2
                  << " My_L2=" << My_L2
                  << " M_L2=" << M_L2
                  << std::defaultfloat << "\n";
    }
};

// ============================================================================
// Compute magnetization MMS errors
// ============================================================================
template <int dim>
MagMMSError compute_mag_mms_error(
    const dealii::DoFHandler<dim>& M_dof_handler,
    const dealii::Vector<double>& Mx_solution,
    const dealii::Vector<double>& My_solution,
    double time,
    double L_y = 1.0)
{
    MagMMSError error;

    const auto& fe = M_dof_handler.get_fe();
    dealii::QGauss<dim> quadrature(fe.degree + 2);
    const unsigned int n_q_points = quadrature.size();

    dealii::FEValues<dim> fe_values(fe, quadrature,
        dealii::update_values | dealii::update_quadrature_points | dealii::update_JxW_values);

    std::vector<double> Mx_values(n_q_points);
    std::vector<double> My_values(n_q_points);

    MagExactMx<dim> exact_Mx(time, L_y);
    MagExactMy<dim> exact_My(time, L_y);

    double Mx_L2_sq = 0.0;
    double My_L2_sq = 0.0;

    for (const auto& cell : M_dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        fe_values.get_function_values(Mx_solution, Mx_values);
        fe_values.get_function_values(My_solution, My_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = fe_values.JxW(q);
            const dealii::Point<dim>& x_q = fe_values.quadrature_point(q);

            const double Mx_exact = exact_Mx.value(x_q);
            const double My_exact = exact_My.value(x_q);

            const double Mx_err = Mx_values[q] - Mx_exact;
            const double My_err = My_values[q] - My_exact;

            Mx_L2_sq += Mx_err * Mx_err * JxW;
            My_L2_sq += My_err * My_err * JxW;
        }
    }

    error.Mx_L2 = std::sqrt(Mx_L2_sq);
    error.My_L2 = std::sqrt(My_L2_sq);
    error.M_L2 = std::sqrt(Mx_L2_sq + My_L2_sq);

    return error;
}

// ============================================================================
// Apply magnetization MMS initial conditions
// ============================================================================
template <int dim>
void apply_mag_mms_initial_conditions(
    const dealii::DoFHandler<dim>& M_dof_handler,
    dealii::Vector<double>& Mx_solution,
    dealii::Vector<double>& My_solution,
    double time,
    double L_y = 1.0)
{
    MagExactMx<dim> exact_Mx(time, L_y);
    MagExactMy<dim> exact_My(time, L_y);

    dealii::VectorTools::interpolate(M_dof_handler, exact_Mx, Mx_solution);
    dealii::VectorTools::interpolate(M_dof_handler, exact_My, My_solution);
}

#endif // MAGNETIZATION_MMS_H