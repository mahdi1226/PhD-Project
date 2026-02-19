// ============================================================================
// poisson/tests/poisson_mms.h - MMS Exact Solutions and Source Terms
//
// PAPER EQUATION 42d (Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531):
//   (∇φ, ∇X) = (h_a − M, ∇X)    ∀X ∈ X_h
//
// STANDALONE MMS (M = 0, h_a = 0):
//   Strong form: −Δφ = f_mms
//   Weak form:   (∇φ, ∇X) = (f_mms, X)
//
// EXACT SOLUTION:
//   φ_exact(x, y, t) = t · cos(πx) · cos(πy/L_y)
//
// Properties:
//   - Satisfies homogeneous Neumann BCs: ∂φ/∂n = 0 on ∂Ω
//   - Smooth (C^∞) — convergence rates limited only by FE degree
//   - Non-trivial gradient in both x and y directions
//   - Time factor t allows temporal testing (t = 1 for pure spatial)
//
// MMS SOURCE:
//   f_mms = −Δφ_exact = t·π²(1 + 1/L_y²)·cos(πx)·cos(πy/L_y)
//
// EXPECTED CONVERGENCE (Q1):
//   L2 error: O(h²) — rate ≈ 2.0
//   H1 error: O(h¹) — rate ≈ 1.0
//
// ERROR COMPUTATION:
//   For pure Neumann, solution is unique up to a constant.
//   L2 error computed with mean-shift correction (two-pass).
//   H1 error (gradient) is unaffected by constant shift.
// ============================================================================
#ifndef POISSON_MMS_H
#define POISSON_MMS_H

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
// Exact solution: φ_exact = t · cos(πx) · cos(πy/L_y) [· cos(πz/L_z) in 3D]
// ============================================================================
template <int dim>
class PoissonExactSolution : public dealii::Function<dim>
{
public:
    PoissonExactSolution(double time = 1.0, double L_y = 1.0, double L_z = 1.0)
        : dealii::Function<dim>(1), time_(time), L_y_(L_y), L_z_(L_z) {}

    virtual double value(const dealii::Point<dim>& p,
                         const unsigned int /*component*/ = 0) const override
    {
        const double x = p[0];
        const double y = (dim >= 2) ? p[1] : 0.0;
        double val = time_ * std::cos(M_PI * x)
                           * std::cos(M_PI * y / L_y_);
        if constexpr (dim >= 3)
            val *= std::cos(M_PI * p[2] / L_z_);
        return val;
    }

    virtual dealii::Tensor<1, dim> gradient(
        const dealii::Point<dim>& p,
        const unsigned int /*component*/ = 0) const override
    {
        const double x = p[0];
        const double y = (dim >= 2) ? p[1] : 0.0;

        dealii::Tensor<1, dim> grad;
        grad[0] = -time_ * M_PI
                   * std::sin(M_PI * x)
                   * std::cos(M_PI * y / L_y_);
        if constexpr (dim >= 3)
            grad[0] *= std::cos(M_PI * p[2] / L_z_);

        if constexpr (dim >= 2)
        {
            grad[1] = -time_ * (M_PI / L_y_)
                       * std::cos(M_PI * x)
                       * std::sin(M_PI * y / L_y_);
            if constexpr (dim >= 3)
                grad[1] *= std::cos(M_PI * p[2] / L_z_);
        }

        if constexpr (dim >= 3)
        {
            grad[2] = -time_ * (M_PI / L_z_)
                       * std::cos(M_PI * x)
                       * std::cos(M_PI * y / L_y_)
                       * std::sin(M_PI * p[2] / L_z_);
        }
        return grad;
    }

    void set_time(double t) override { time_ = t; }
    double get_time() const { return time_; }

private:
    double time_;
    double L_y_;
    double L_z_;
};

// ============================================================================
// MMS source for standalone test (M = 0, h_a = 0)
//
// 2D: f_mms = −Δφ_exact = t·π²(1 + 1/L_y²)·cos(πx)·cos(πy/L_y)
// 3D: f_mms = −Δφ_exact = t·π²(1 + 1/L_y² + 1/L_z²)·cos(πx)·cos(πy/L_y)·cos(πz/L_z)
// ============================================================================
template <int dim>
double compute_poisson_mms_source_standalone(
    const dealii::Point<dim>& pt,
    double time,
    double L_y = 1.0,
    double L_z = 1.0)
{
    const double x = pt[0];
    const double y = (dim >= 2) ? pt[1] : 0.0;

    double coeff = 1.0 + 1.0 / (L_y * L_y);
    if constexpr (dim >= 3)
        coeff += 1.0 / (L_z * L_z);

    double val = time * M_PI * M_PI * coeff
           * std::cos(M_PI * x)
           * std::cos(M_PI * y / L_y);
    if constexpr (dim >= 3)
        val *= std::cos(M_PI * pt[2] / L_z);

    return val;
}

// ============================================================================
// Error computation (parallel, two-pass for mean-shift correction)
// ============================================================================
struct PoissonMMSErrors
{
    double L2 = 0.0;
    double H1 = 0.0;
    double Linf = 0.0;
};

template <int dim>
PoissonMMSErrors compute_poisson_mms_errors(
    const dealii::DoFHandler<dim>& dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& solution_relevant,
    double time,
    double L_y,
    MPI_Comm mpi_comm,
    double L_z = 1.0)
{
    PoissonMMSErrors errors;

    const auto& fe = dof_handler.get_fe();
    const unsigned int quad_degree = fe.degree + 2;
    dealii::QGauss<dim> quadrature(quad_degree);
    const unsigned int n_q_points = quadrature.size();

    dealii::FEValues<dim> fe_values(fe, quadrature,
        dealii::update_values | dealii::update_gradients |
        dealii::update_quadrature_points | dealii::update_JxW_values);

    std::vector<double> phi_values(n_q_points);
    std::vector<dealii::Tensor<1, dim>> phi_gradients(n_q_points);

    PoissonExactSolution<dim> exact(time, L_y, L_z);

    // ---- Pass 1: mean difference + H1 seminorm ----
    double local_volume = 0.0;
    double local_mean_diff = 0.0;
    double local_H1_sq = 0.0;

    for (const auto& cell : dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);
        fe_values.get_function_values(solution_relevant, phi_values);
        fe_values.get_function_gradients(solution_relevant, phi_gradients);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = fe_values.JxW(q);
            const auto& x_q = fe_values.quadrature_point(q);

            const double phi_ex = exact.value(x_q);
            const double diff = phi_values[q] - phi_ex;
            local_mean_diff += diff * JxW;
            local_volume += JxW;

            const auto grad_ex = exact.gradient(x_q);
            const auto grad_err = phi_gradients[q] - grad_ex;
            local_H1_sq += (grad_err * grad_err) * JxW;
        }
    }

    double global_volume = 0.0, global_mean_diff = 0.0;
    double global_H1_sq = 0.0;
    MPI_Allreduce(&local_volume, &global_volume, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
    MPI_Allreduce(&local_mean_diff, &global_mean_diff, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
    MPI_Allreduce(&local_H1_sq, &global_H1_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);

    const double c_shift = global_mean_diff / global_volume;

    // ---- Pass 2: L2 and Linf with mean-shifted solution ----
    double local_L2_sq = 0.0;
    double local_Linf = 0.0;

    for (const auto& cell : dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);
        fe_values.get_function_values(solution_relevant, phi_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = fe_values.JxW(q);
            const auto& x_q = fe_values.quadrature_point(q);

            const double phi_ex = exact.value(x_q);
            const double err = (phi_values[q] - c_shift) - phi_ex;
            local_L2_sq += err * err * JxW;
            local_Linf = std::max(local_Linf, std::abs(err));
        }
    }

    double global_L2_sq = 0.0;
    double global_Linf = 0.0;
    MPI_Allreduce(&local_L2_sq, &global_L2_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);
    MPI_Allreduce(&local_Linf, &global_Linf, 1, MPI_DOUBLE, MPI_MAX, mpi_comm);

    errors.L2 = std::sqrt(global_L2_sq);
    errors.H1 = std::sqrt(global_H1_sq);
    errors.Linf = global_Linf;

    return errors;
}

#endif // POISSON_MMS_H