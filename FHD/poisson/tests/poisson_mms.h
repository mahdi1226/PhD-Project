// ============================================================================
// poisson/tests/poisson_mms.h - MMS Exact Solutions and Source Terms
//
// PAPER EQUATION 42d (Nochetto, Salgado & Tomas, arXiv:1511.04381):
//   (∇φ, ∇X) = (h_a − M, ∇X)    ∀X ∈ X_h
//
// STANDALONE MMS (M = 0, h_a = 0):
//   Strong form: −Δφ = f_mms
//   Weak form:   (∇φ, ∇X) = (f_mms, X)
//
// EXACT SOLUTION:
//   φ*(x, y, t) = t · cos(πx) · cos(πy)
//
// Properties:
//   - Satisfies homogeneous Neumann BCs: ∂φ/∂n = 0 on ∂[0,1]²
//   - Smooth (C^∞) — convergence limited only by FE degree
//   - Non-trivial gradient in both directions
//   - Time factor t for temporal testing (t = 1 for pure spatial)
//
// MMS SOURCE:
//   f_mms = −Δφ* = 2π²t · cos(πx) · cos(πy)
//
// EXPECTED CONVERGENCE (CG Q_ℓ):
//   L2 error: O(h^{ℓ+1})    (ℓ=2 → rate ≈ 3.0)
//   H1 error: O(h^{ℓ})      (ℓ=2 → rate ≈ 2.0)
//
// ERROR COMPUTATION:
//   Pure Neumann → solution unique up to constant.
//   L2 uses mean-shift correction (two-pass).
//   H1 (gradient) is unaffected by constant shift.
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015), Section 6
// ============================================================================
#ifndef FHD_POISSON_MMS_H
#define FHD_POISSON_MMS_H

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
// Exact solution: φ* = t · cos(πx) · cos(πy)
// ============================================================================
template <int dim>
class PoissonExactSolution : public dealii::Function<dim>
{
public:
    PoissonExactSolution(double time = 1.0)
        : dealii::Function<dim>(1), time_(time) {}

    virtual double value(const dealii::Point<dim>& p,
                         const unsigned int /*component*/ = 0) const override
    {
        double val = time_ * std::cos(M_PI * p[0]);
        if constexpr (dim >= 2)
            val *= std::cos(M_PI * p[1]);
        if constexpr (dim >= 3)
            val *= std::cos(M_PI * p[2]);
        return val;
    }

    virtual dealii::Tensor<1, dim> gradient(
        const dealii::Point<dim>& p,
        const unsigned int /*component*/ = 0) const override
    {
        dealii::Tensor<1, dim> grad;

        // Common factor
        double base = time_;
        for (unsigned int d = 0; d < dim; ++d)
            base *= std::cos(M_PI * p[d]);

        // ∂φ*/∂x_i = -πt Π_{j≠i} cos(πx_j) · sin(πx_i)
        for (unsigned int i = 0; i < dim; ++i)
        {
            // base / cos(πx_i) * (-sin(πx_i)) * π
            const double c = std::cos(M_PI * p[i]);
            if (std::abs(c) > 1e-14)
                grad[i] = -M_PI * base * std::sin(M_PI * p[i]) / c;
            else
            {
                // Direct computation when cos is near zero
                double val = -M_PI * time_ * std::sin(M_PI * p[i]);
                for (unsigned int d = 0; d < dim; ++d)
                    if (d != i)
                        val *= std::cos(M_PI * p[d]);
                grad[i] = val;
            }
        }

        return grad;
    }

    void set_time(double t) override { time_ = t; }
    double get_time() const { return time_; }

private:
    double time_;
};

// ============================================================================
// MMS source for standalone test (M = 0, h_a = 0)
//
// f_mms = −Δφ* = dim·π²·t · Π cos(πx_d)
//
// 2D: f_mms = 2π²t · cos(πx) · cos(πy)
// 3D: f_mms = 3π²t · cos(πx) · cos(πy) · cos(πz)
// ============================================================================
template <int dim>
double compute_poisson_mms_source(
    const dealii::Point<dim>& pt,
    double time)
{
    double val = dim * M_PI * M_PI * time;
    for (unsigned int d = 0; d < dim; ++d)
        val *= std::cos(M_PI * pt[d]);
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
    MPI_Comm mpi_comm)
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

    PoissonExactSolution<dim> exact(time);

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

#endif // FHD_POISSON_MMS_H
