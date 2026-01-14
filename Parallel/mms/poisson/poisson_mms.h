// ============================================================================
// mms/poisson/poisson_mms.h - Poisson MMS Exact Solutions (PARALLEL)
//
// EXACT SOLUTION:
//   φ_exact = t · cos(πx) · cos(πy/L_y)
//
// This satisfies:
//   - Homogeneous Neumann BC (∂φ/∂n = 0 on all boundaries)
//   - Zero mean (compatible with pinning any DoF)
//
// PARALLEL VERSION:
//   - Adds compute_poisson_mms_errors_parallel() with MPI reduction
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
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
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Exact solution: φ_exact = t · cos(πx) · cos(πy/L_y)
// ============================================================================
template <int dim>
class PoissonExactSolution : public dealii::Function<dim>
{
public:
    PoissonExactSolution(double time = 1.0, double L_y = 1.0)
        : dealii::Function<dim>(1), time_(time), L_y_(L_y) {}

    virtual double value(const dealii::Point<dim>& p,
                         const unsigned int component = 0) const override
    {
        (void)component;
        const double x = p[0];
        const double y = (dim >= 2) ? p[1] : 0.0;
        return time_ * std::cos(M_PI * x) * std::cos(M_PI * y / L_y_);
    }

    virtual dealii::Tensor<1, dim> gradient(const dealii::Point<dim>& p,
                                            const unsigned int component = 0) const override
    {
        (void)component;
        const double x = p[0];
        const double y = (dim >= 2) ? p[1] : 0.0;

        dealii::Tensor<1, dim> grad;
        grad[0] = -time_ * M_PI * std::sin(M_PI * x) * std::cos(M_PI * y / L_y_);
        if constexpr (dim >= 2)
            grad[1] = -time_ * (M_PI / L_y_) * std::cos(M_PI * x) * std::sin(M_PI * y / L_y_);
        return grad;
    }

    void set_time(double t) override { time_ = t; }
    double get_time() const { return time_; }

private:
    double time_;
    double L_y_;
};

// ============================================================================
// MMS source for STANDALONE test (M=0)
//
// Strong form: -Δφ = f_MMS
// f_MMS = -Δφ_exact = t·π²(1 + 1/L_y²)·cos(πx)·cos(πy/L_y)
// ============================================================================
template <int dim>
double compute_poisson_mms_source_standalone(
    const dealii::Point<dim>& pt,
    double time,
    double L_y = 1.0)
{
    const double x = pt[0];
    const double y = pt[1];

    return time * M_PI * M_PI * (1.0 + 1.0/(L_y*L_y))
           * std::cos(M_PI * x) * std::cos(M_PI * y / L_y);
}

// ============================================================================
// MMS source for COUPLED Poisson-Magnetization test
//
// f_MMS = -Δφ_exact - ∇·M_exact
// ============================================================================
template <int dim>
double compute_poisson_mms_source_coupled(
    const dealii::Point<dim>& pt,
    double time,
    double L_y = 1.0)
{
    const double x = pt[0];
    const double y = pt[1];

    // -Δφ_exact
    const double neg_laplacian_phi = time * M_PI * M_PI * (1.0 + 1.0/(L_y*L_y))
                                     * std::cos(M_PI * x) * std::cos(M_PI * y / L_y);

    // ∇·M_exact (assuming M from magnetization_mms.h)
    const double div_M = time * M_PI * std::cos(M_PI * x) * std::sin(M_PI * y / L_y)
                       - time * (M_PI / L_y) * std::cos(M_PI * x) * std::sin(M_PI * y / L_y);

    return neg_laplacian_phi - div_M;
}

// ============================================================================
// Poisson MMS Error Results
// ============================================================================
struct PoissonMMSError
{
    double L2_error = 0.0;
    double H1_error = 0.0;
    double Linf_error = 0.0;
};

// ============================================================================
// Compute Poisson MMS errors (PARALLEL)
//
// For pure Neumann, the solution is unique up to a constant. We:
//   1. Compute H1 seminorm (gradient error) - unaffected by constant
//   2. For L2, compute the mean difference and subtract it
// ============================================================================
template <int dim>
PoissonMMSError compute_poisson_mms_errors_parallel(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& phi_solution,
    double time,
    double L_y,
    MPI_Comm mpi_communicator)
{
    PoissonMMSError errors;

    const auto& fe = phi_dof_handler.get_fe();
    const unsigned int quad_degree = fe.degree + 2;
    dealii::QGauss<dim> quadrature(quad_degree);
    const unsigned int n_q_points = quadrature.size();

    dealii::FEValues<dim> fe_values(fe, quadrature,
        dealii::update_values | dealii::update_gradients |
        dealii::update_quadrature_points | dealii::update_JxW_values);

    std::vector<double> phi_values(n_q_points);
    std::vector<dealii::Tensor<1, dim>> phi_gradients(n_q_points);

    PoissonExactSolution<dim> exact_solution(time, L_y);

    // First pass: compute mean difference, H1 seminorm (LOCAL contributions)
    double local_volume = 0.0;
    double local_mean_diff = 0.0;
    double local_H1_sq = 0.0;
    double local_Linf = 0.0;

    for (const auto& cell : phi_dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);
        fe_values.get_function_values(phi_solution, phi_values);
        fe_values.get_function_gradients(phi_solution, phi_gradients);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = fe_values.JxW(q);
            const auto& x_q = fe_values.quadrature_point(q);

            const double phi_exact = exact_solution.value(x_q);
            const double diff = phi_values[q] - phi_exact;
            local_mean_diff += diff * JxW;
            local_volume += JxW;
            local_Linf = std::max(local_Linf, std::abs(diff));

            // H1 seminorm (gradient error)
            const auto grad_exact = exact_solution.gradient(x_q);
            const auto grad_error = phi_gradients[q] - grad_exact;
            local_H1_sq += (grad_error * grad_error) * JxW;
        }
    }

    // Global reductions
    double global_volume = 0.0, global_mean_diff = 0.0, global_H1_sq = 0.0, global_Linf = 0.0;
    MPI_Allreduce(&local_volume, &global_volume, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
    MPI_Allreduce(&local_mean_diff, &global_mean_diff, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
    MPI_Allreduce(&local_H1_sq, &global_H1_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
    MPI_Allreduce(&local_Linf, &global_Linf, 1, MPI_DOUBLE, MPI_MAX, mpi_communicator);

    // Compute mean shift
    const double c_shift = global_mean_diff / global_volume;

    // Second pass: compute L2 error with mean-shifted solution (LOCAL)
    double local_L2_sq = 0.0;

    for (const auto& cell : phi_dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);
        fe_values.get_function_values(phi_solution, phi_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = fe_values.JxW(q);
            const auto& x_q = fe_values.quadrature_point(q);

            const double phi_exact = exact_solution.value(x_q);
            const double value_error = (phi_values[q] - c_shift) - phi_exact;
            local_L2_sq += value_error * value_error * JxW;
        }
    }

    // Global reduction for L2
    double global_L2_sq = 0.0;
    MPI_Allreduce(&local_L2_sq, &global_L2_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);

    errors.L2_error = std::sqrt(global_L2_sq);
    errors.H1_error = std::sqrt(global_H1_sq);
    errors.Linf_error = global_Linf;

    return errors;
}

#endif // POISSON_MMS_H