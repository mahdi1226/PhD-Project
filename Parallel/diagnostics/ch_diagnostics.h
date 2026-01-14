// ============================================================================
// diagnostics/ch_diagnostics.h - Cahn-Hilliard Diagnostics (Parallel)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
//
// Computes:
//   - θ bounds: should stay in [-1, 1]
//   - Mass: ∫θ dΩ (conserved for Neumann BCs)
//   - Energy: E_CH = ∫[ε/2|∇θ|² + (1/ε)W(θ)] dΩ
//
// All quantities are MPI-reduced for parallel correctness.
// ============================================================================
#ifndef CH_DIAGNOSTICS_H
#define CH_DIAGNOSTICS_H

#include "utilities/parameters.h"
#include "utilities/mpi_tools.h"

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <algorithm>
#include <cmath>

// ============================================================================
// CH Diagnostic Data
// ============================================================================
struct CHDiagnostics
{
    double theta_min = 0.0;
    double theta_max = 0.0;
    double mass = 0.0;           // ∫θ dΩ
    double energy = 0.0;         // E_CH = ∫[ε/2|∇θ|² + (1/ε)W(θ)] dΩ
    bool bounds_violated = false;
};

// ============================================================================
// Compute CH diagnostics (parallel version with Trilinos vectors)
// ============================================================================
template <int dim>
CHDiagnostics compute_ch_diagnostics(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& theta_solution,
    const Parameters& params,
    MPI_Comm comm = MPI_COMM_WORLD)
{
    const double epsilon = params.physics.epsilon;

    const auto& fe = theta_dof_handler.get_fe();
    dealii::QGauss<dim> quadrature(fe.degree + 1);
    dealii::FEValues<dim> fe_values(fe, quadrature,
        dealii::update_values |
        dealii::update_gradients |
        dealii::update_JxW_values);

    const unsigned int n_q_points = quadrature.size();
    std::vector<double> theta_values(n_q_points);
    std::vector<dealii::Tensor<1, dim>> theta_gradients(n_q_points);

    // Local accumulators
    double local_mass = 0.0;
    double local_energy = 0.0;
    double local_theta_min = std::numeric_limits<double>::max();
    double local_theta_max = std::numeric_limits<double>::lowest();

    for (const auto& cell : theta_dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);
        fe_values.get_function_values(theta_solution, theta_values);
        fe_values.get_function_gradients(theta_solution, theta_gradients);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double theta = theta_values[q];
            const double grad_theta_sq = theta_gradients[q].norm_square();
            const double JxW = fe_values.JxW(q);

            // Mass: ∫θ dΩ
            local_mass += theta * JxW;

            // W(θ) = (1/4)(θ² - 1)² (double-well potential)
            const double theta_sq = theta * theta;
            const double W_theta = 0.25 * (theta_sq - 1.0) * (theta_sq - 1.0);

            // E_CH = ε/2 |∇θ|² + (1/ε) W(θ)
            const double E_grad = 0.5 * epsilon * grad_theta_sq;
            const double E_bulk = (1.0 / epsilon) * W_theta;
            local_energy += (E_grad + E_bulk) * JxW;

            // Track bounds
            local_theta_min = std::min(local_theta_min, theta);
            local_theta_max = std::max(local_theta_max, theta);
        }
    }

    // MPI reductions
    CHDiagnostics result;
    result.mass = MPIUtils::reduce_sum(local_mass, comm);
    result.energy = MPIUtils::reduce_sum(local_energy, comm);
    result.theta_min = MPIUtils::reduce_min(local_theta_min, comm);
    result.theta_max = MPIUtils::reduce_max(local_theta_max, comm);
    result.bounds_violated = (result.theta_min < -1.01 || result.theta_max > 1.01);

    return result;
}

// ============================================================================
// Compute CH diagnostics (serial version with deal.II vectors)
// ============================================================================
template <int dim>
CHDiagnostics compute_ch_diagnostics(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::Vector<double>& theta_solution,
    const Parameters& params)
{
    const double epsilon = params.physics.epsilon;

    const auto& fe = theta_dof_handler.get_fe();
    dealii::QGauss<dim> quadrature(fe.degree + 1);
    dealii::FEValues<dim> fe_values(fe, quadrature,
        dealii::update_values |
        dealii::update_gradients |
        dealii::update_JxW_values);

    const unsigned int n_q_points = quadrature.size();
    std::vector<double> theta_values(n_q_points);
    std::vector<dealii::Tensor<1, dim>> theta_gradients(n_q_points);

    CHDiagnostics result;
    result.theta_min = std::numeric_limits<double>::max();
    result.theta_max = std::numeric_limits<double>::lowest();

    for (const auto& cell : theta_dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        fe_values.get_function_values(theta_solution, theta_values);
        fe_values.get_function_gradients(theta_solution, theta_gradients);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double theta = theta_values[q];
            const double grad_theta_sq = theta_gradients[q].norm_square();
            const double JxW = fe_values.JxW(q);

            result.mass += theta * JxW;

            const double theta_sq = theta * theta;
            const double W_theta = 0.25 * (theta_sq - 1.0) * (theta_sq - 1.0);
            const double E_grad = 0.5 * epsilon * grad_theta_sq;
            const double E_bulk = (1.0 / epsilon) * W_theta;
            result.energy += (E_grad + E_bulk) * JxW;

            result.theta_min = std::min(result.theta_min, theta);
            result.theta_max = std::max(result.theta_max, theta);
        }
    }

    result.bounds_violated = (result.theta_min < -1.01 || result.theta_max > 1.01);
    return result;
}

#endif // CH_DIAGNOSTICS_H