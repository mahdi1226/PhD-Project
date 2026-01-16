// ============================================================================
// mms/magnetization/magnetization_mms.h - Magnetization MMS (PARALLEL)
//
// EXACT SOLUTIONS (chosen so M*·n = 0 on all boundaries):
//   Mx = t·sin(πx)·sin(πy/L_y)
//   My = t·cos(πx)·sin(πy/L_y)   ← sin(πy/L_y) ensures My=0 at y=0,L_y
//
// This ensures the boundary integral ∫_{∂Ω} (M*·n) χ ds = 0
// which is required for the coupled Poisson-Magnetization MMS test.
//
// PARALLEL VERSION:
//   - Adds compute_mag_mms_errors_parallel() with MPI reductions
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef MAGNETIZATION_MMS_H
#define MAGNETIZATION_MMS_H

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/trilinos_vector.h>

#include <mpi.h>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Exact Mx = t·sin(πx)·sin(πy/L_y)
//
// Boundary values:
//   x=0: Mx = 0
//   x=1: Mx = 0
//   y=0: Mx = 0
//   y=L_y: Mx = 0
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
        return time_ * std::sin(M_PI * p[0]) * std::sin(M_PI * p[1] / L_y_);
    }

    virtual dealii::Tensor<1, dim> gradient(const dealii::Point<dim>& p,
                                            const unsigned int = 0) const override
    {
        dealii::Tensor<1, dim> grad;
        grad[0] = time_ * M_PI * std::cos(M_PI * p[0]) * std::sin(M_PI * p[1] / L_y_);
        grad[1] = time_ * (M_PI / L_y_) * std::sin(M_PI * p[0]) * std::cos(M_PI * p[1] / L_y_);
        return grad;
    }

    void set_time(const double t) override { time_ = t; }

private:
    double time_;
    double L_y_;
};

// ============================================================================
// Exact My = t·cos(πx)·sin(πy/L_y)
//
// CRITICAL: Using sin(πy/L_y) ensures My = 0 at y=0 and y=L_y
// This makes M*·n = 0 on horizontal boundaries (required for coupled MMS)
//
// Boundary values:
//   x=0: My = t·sin(πy/L_y)  (but n = (-1,0), so M·n = -Mx = 0)
//   x=1: My = -t·sin(πy/L_y) (but n = (1,0), so M·n = Mx = 0)
//   y=0: My = 0
//   y=L_y: My = 0
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
        // Changed from cos(πy/L_y) to sin(πy/L_y) for zero BC at y=0,L_y
        return time_ * std::cos(M_PI * p[0]) * std::sin(M_PI * p[1] / L_y_);
    }

    virtual dealii::Tensor<1, dim> gradient(const dealii::Point<dim>& p,
                                            const unsigned int = 0) const override
    {
        dealii::Tensor<1, dim> grad;
        // ∂My/∂x = -t·π·sin(πx)·sin(πy/L_y)
        grad[0] = -time_ * M_PI * std::sin(M_PI * p[0]) * std::sin(M_PI * p[1] / L_y_);
        // ∂My/∂y = t·(π/L_y)·cos(πx)·cos(πy/L_y)
        grad[1] = time_ * (M_PI / L_y_) * std::cos(M_PI * p[0]) * std::cos(M_PI * p[1] / L_y_);
        return grad;
    }

    void set_time(const double t) override { time_ = t; }

private:
    double time_;
    double L_y_;
};

// ============================================================================
// Get exact magnetization vector at a point
// ============================================================================
template <int dim>
dealii::Tensor<1, dim> mag_mms_exact_M(
    const dealii::Point<dim>& p,
    double time,
    double L_y = 1.0)
{
    dealii::Tensor<1, dim> M;
    M[0] = time * std::sin(M_PI * p[0]) * std::sin(M_PI * p[1] / L_y);
    M[1] = time * std::cos(M_PI * p[0]) * std::sin(M_PI * p[1] / L_y);  // sin, not cos
    return M;
}

// ============================================================================
// MMS source for STANDALONE test (U=0, H=0)
//
// Equation: ∂M/∂t + M/τ_M = f_M
// Discretized: (M^n - M^{n-1})/τ + M^n/τ_M = f_M
//
// f_M = (M*^n - M*^{n-1})/τ + M*^n/τ_M
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

    dealii::Tensor<1, dim> M_new = mag_mms_exact_M<dim>(pt, t_new, L_y);
    dealii::Tensor<1, dim> M_old = mag_mms_exact_M<dim>(pt, t_old, L_y);

    dealii::Tensor<1, dim> f;
    if (tau_M > 0.0)
    {
        f[0] = (M_new[0] - M_old[0]) / dt + M_new[0] / tau_M;
        f[1] = (M_new[1] - M_old[1]) / dt + M_new[1] / tau_M;
    }
    else
    {
        f[0] = (M_new[0] - M_old[0]) / dt;
        f[1] = (M_new[1] - M_old[1]) / dt;
    }
    return f;
}

// ============================================================================
// MMS source WITH transport (U ≠ 0) or coupling (H ≠ 0)
//
// Full equation: ∂M/∂t + (U·∇)M + ½(∇·U)M + (M - χH)/τ_M = f_M
//
// Uses DISCRETE H (passed in), not analytical, so that terms cancel properly
// in the weak form.
// ============================================================================
template <int dim>
dealii::Tensor<1, dim> compute_mag_mms_source_with_transport(
    const dealii::Point<dim>& pt,
    double t_new,
    double t_old,
    double tau_M,
    double chi_val,
    const dealii::Tensor<1, dim>& H,  // Discrete H from Poisson solve
    const dealii::Tensor<1, dim>& U,
    double div_U,
    double L_y = 1.0)
{
    const double dt = t_new - t_old;

    dealii::Tensor<1, dim> M_new = mag_mms_exact_M<dim>(pt, t_new, L_y);
    dealii::Tensor<1, dim> M_old = mag_mms_exact_M<dim>(pt, t_old, L_y);

    // Gradient of exact M at new time
    const double x = pt[0];
    const double y = pt[1];

    // ∇Mx where Mx = t·sin(πx)·sin(πy/L_y)
    dealii::Tensor<1, dim> grad_Mx;
    grad_Mx[0] = t_new * M_PI * std::cos(M_PI * x) * std::sin(M_PI * y / L_y);
    grad_Mx[1] = t_new * (M_PI / L_y) * std::sin(M_PI * x) * std::cos(M_PI * y / L_y);

    // ∇My where My = t·cos(πx)·sin(πy/L_y)
    dealii::Tensor<1, dim> grad_My;
    grad_My[0] = -t_new * M_PI * std::sin(M_PI * x) * std::sin(M_PI * y / L_y);
    grad_My[1] = t_new * (M_PI / L_y) * std::cos(M_PI * x) * std::cos(M_PI * y / L_y);

    // Convection: (U·∇)M
    const double convect_Mx = U[0] * grad_Mx[0] + U[1] * grad_Mx[1];
    const double convect_My = U[0] * grad_My[0] + U[1] * grad_My[1];

    // Skew term: (1/2)(∇·U)M
    const double skew_Mx = 0.5 * div_U * M_new[0];
    const double skew_My = 0.5 * div_U * M_new[1];

    // Equilibrium: χ·H (using DISCRETE H so it cancels in weak form)
    const double chi_H_x = chi_val * H[0];
    const double chi_H_y = chi_val * H[1];

    dealii::Tensor<1, dim> f;
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
};

// ============================================================================
// Compute magnetization MMS errors (PARALLEL)
// ============================================================================
template <int dim>
MagMMSError compute_mag_mms_errors_parallel(
    const dealii::DoFHandler<dim>& M_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& Mx_solution,
    const dealii::TrilinosWrappers::MPI::Vector& My_solution,
    double time,
    double L_y,
    MPI_Comm mpi_communicator)
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

    double local_Mx_L2_sq = 0.0;
    double local_My_L2_sq = 0.0;

    for (const auto& cell : M_dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);
        fe_values.get_function_values(Mx_solution, Mx_values);
        fe_values.get_function_values(My_solution, My_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = fe_values.JxW(q);
            const auto& x_q = fe_values.quadrature_point(q);

            const double Mx_exact = exact_Mx.value(x_q);
            const double My_exact = exact_My.value(x_q);

            const double Mx_err = Mx_values[q] - Mx_exact;
            const double My_err = My_values[q] - My_exact;

            local_Mx_L2_sq += Mx_err * Mx_err * JxW;
            local_My_L2_sq += My_err * My_err * JxW;
        }
    }

    // Global reductions
    double global_Mx_L2_sq = 0.0, global_My_L2_sq = 0.0;
    MPI_Allreduce(&local_Mx_L2_sq, &global_Mx_L2_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
    MPI_Allreduce(&local_My_L2_sq, &global_My_L2_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);

    error.Mx_L2 = std::sqrt(global_Mx_L2_sq);
    error.My_L2 = std::sqrt(global_My_L2_sq);
    error.M_L2 = std::sqrt(global_Mx_L2_sq + global_My_L2_sq);

    return error;
}

#endif // MAGNETIZATION_MMS_H