// ============================================================================
// diagnostics/poisson_diagnostics.h - Magnetostatic Poisson Diagnostics (Parallel)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
//
// Computes:
//   - φ bounds (min, max)
//   - H field magnitude |∇φ| (L², max)
//   - Magnetic energy: ½∫ μ(θ)|∇φ|² dx
//   - M magnitude (quasi-equilibrium): |χ(θ)∇φ|
//
// All quantities are MPI-reduced for parallel correctness.
// ============================================================================
#ifndef POISSON_DIAGNOSTICS_H
#define POISSON_DIAGNOSTICS_H

#include "utilities/parameters.h"
#include "utilities/mpi_tools.h"

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <algorithm>
#include <cmath>
#include <limits>

// ============================================================================
// Poisson Diagnostic Data
// ============================================================================
struct PoissonDiagnostics
{
    // Potential bounds
    double phi_min = 0.0;
    double phi_max = 0.0;

    // H field = ∇φ
    double H_L2_norm = 0.0;      // ||∇φ||_{L²}
    double H_max = 0.0;          // max |∇φ|

    // Magnetization (quasi-equilibrium: M = χ(θ)∇φ)
    double M_L2_norm = 0.0;      // ||M||_{L²}
    double M_max = 0.0;          // max |M|

    // Magnetic energy: ½∫ μ(θ)|∇φ|² dx
    double magnetic_energy = 0.0;

    // Permeability range
    double mu_min = 1.0;
    double mu_max = 1.0;
};

// ============================================================================
// Helper: Smooth Heaviside function for susceptibility
// ============================================================================
namespace detail
{
    inline double smooth_heaviside(double x)
    {
        if (x > 20.0) return 1.0;
        if (x < -20.0) return 0.0;
        return 1.0 / (1.0 + std::exp(-x));
    }

    inline double susceptibility(double theta, double epsilon, double chi_0)
    {
        return chi_0 * smooth_heaviside(theta / epsilon);
    }
}

// ============================================================================
// Compute Poisson diagnostics (parallel version with Trilinos vectors)
// ============================================================================
template <int dim>
PoissonDiagnostics compute_poisson_diagnostics(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& phi_solution,
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& theta_solution,
    const Parameters& params,
    MPI_Comm comm = MPI_COMM_WORLD)
{
    const auto& phi_fe = phi_dof_handler.get_fe();
    const auto& theta_fe = theta_dof_handler.get_fe();

    const unsigned int quad_degree = std::max(phi_fe.degree, theta_fe.degree) + 2;
    dealii::QGauss<dim> quadrature(quad_degree);
    const unsigned int n_q_points = quadrature.size();

    dealii::FEValues<dim> phi_fe_values(phi_fe, quadrature,
        dealii::update_gradients | dealii::update_JxW_values);

    dealii::FEValues<dim> theta_fe_values(theta_fe, quadrature,
        dealii::update_values);

    std::vector<dealii::Tensor<1, dim>> grad_phi_values(n_q_points);
    std::vector<double> theta_values(n_q_points);

    // Local accumulators
    double local_H_L2_sq = 0.0;
    double local_M_L2_sq = 0.0;
    double local_magnetic_energy = 0.0;
    double local_H_max = 0.0;
    double local_M_max = 0.0;
    double local_phi_min = std::numeric_limits<double>::max();
    double local_phi_max = std::numeric_limits<double>::lowest();
    double local_mu_min = std::numeric_limits<double>::max();
    double local_mu_max = std::numeric_limits<double>::lowest();

    auto phi_cell = phi_dof_handler.begin_active();
    auto theta_cell = theta_dof_handler.begin_active();

    for (; phi_cell != phi_dof_handler.end(); ++phi_cell, ++theta_cell)
    {
        if (!phi_cell->is_locally_owned())
            continue;

        phi_fe_values.reinit(phi_cell);
        theta_fe_values.reinit(theta_cell);

        phi_fe_values.get_function_gradients(phi_solution, grad_phi_values);
        theta_fe_values.get_function_values(theta_solution, theta_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = phi_fe_values.JxW(q);

            // H = ∇φ
            const dealii::Tensor<1, dim>& H = grad_phi_values[q];
            const double H_norm = H.norm();

            // θ and derived quantities
            const double theta_q = theta_values[q];
            const double chi_theta = detail::susceptibility(theta_q,
                params.physics.epsilon, params.physics.chi_0);
            const double mu_theta = 1.0 + chi_theta;

            // M = χ(θ)H (quasi-equilibrium)
            const double M_norm = chi_theta * H_norm;

            // Accumulate integrals
            local_H_L2_sq += H_norm * H_norm * JxW;
            local_M_L2_sq += M_norm * M_norm * JxW;
            local_magnetic_energy += 0.5 * mu_theta * H_norm * H_norm * JxW;

            // Track extrema
            local_H_max = std::max(local_H_max, H_norm);
            local_M_max = std::max(local_M_max, M_norm);
            local_mu_min = std::min(local_mu_min, mu_theta);
            local_mu_max = std::max(local_mu_max, mu_theta);
        }
    }

    // φ bounds from solution vector (locally owned entries)
    for (const auto& cell : phi_dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        std::vector<dealii::types::global_dof_index> local_dof_indices(
            phi_dof_handler.get_fe().dofs_per_cell);
        cell->get_dof_indices(local_dof_indices);

        for (const auto& idx : local_dof_indices)
        {
            if (phi_solution.locally_owned_elements().is_element(idx))
            {
                const double phi_val = phi_solution[idx];
                local_phi_min = std::min(local_phi_min, phi_val);
                local_phi_max = std::max(local_phi_max, phi_val);
            }
        }
    }

    // MPI reductions
    PoissonDiagnostics result;

    double global_H_L2_sq = MPIUtils::reduce_sum(local_H_L2_sq, comm);
    double global_M_L2_sq = MPIUtils::reduce_sum(local_M_L2_sq, comm);

    result.H_L2_norm = std::sqrt(global_H_L2_sq);
    result.M_L2_norm = std::sqrt(global_M_L2_sq);
    result.magnetic_energy = MPIUtils::reduce_sum(local_magnetic_energy, comm);

    result.H_max = MPIUtils::reduce_max(local_H_max, comm);
    result.M_max = MPIUtils::reduce_max(local_M_max, comm);

    result.phi_min = MPIUtils::reduce_min(local_phi_min, comm);
    result.phi_max = MPIUtils::reduce_max(local_phi_max, comm);

    result.mu_min = MPIUtils::reduce_min(local_mu_min, comm);
    result.mu_max = MPIUtils::reduce_max(local_mu_max, comm);

    return result;
}

// ============================================================================
// Compute Poisson diagnostics (serial version with deal.II vectors)
// ============================================================================
template <int dim>
PoissonDiagnostics compute_poisson_diagnostics(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::Vector<double>& phi_solution,
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::Vector<double>& theta_solution,
    const Parameters& params)
{
    PoissonDiagnostics result;

    // φ bounds
    result.phi_min = *std::min_element(phi_solution.begin(), phi_solution.end());
    result.phi_max = *std::max_element(phi_solution.begin(), phi_solution.end());

    result.mu_min = std::numeric_limits<double>::max();
    result.mu_max = std::numeric_limits<double>::lowest();

    const auto& phi_fe = phi_dof_handler.get_fe();
    const auto& theta_fe = theta_dof_handler.get_fe();

    const unsigned int quad_degree = std::max(phi_fe.degree, theta_fe.degree) + 2;
    dealii::QGauss<dim> quadrature(quad_degree);
    const unsigned int n_q_points = quadrature.size();

    dealii::FEValues<dim> phi_fe_values(phi_fe, quadrature,
        dealii::update_gradients | dealii::update_JxW_values);

    dealii::FEValues<dim> theta_fe_values(theta_fe, quadrature,
        dealii::update_values);

    std::vector<dealii::Tensor<1, dim>> grad_phi_values(n_q_points);
    std::vector<double> theta_values(n_q_points);

    double H_L2_sq = 0.0;
    double M_L2_sq = 0.0;

    auto phi_cell = phi_dof_handler.begin_active();
    auto theta_cell = theta_dof_handler.begin_active();

    for (; phi_cell != phi_dof_handler.end(); ++phi_cell, ++theta_cell)
    {
        phi_fe_values.reinit(phi_cell);
        theta_fe_values.reinit(theta_cell);

        phi_fe_values.get_function_gradients(phi_solution, grad_phi_values);
        theta_fe_values.get_function_values(theta_solution, theta_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = phi_fe_values.JxW(q);
            const dealii::Tensor<1, dim>& H = grad_phi_values[q];
            const double H_norm = H.norm();

            const double theta_q = theta_values[q];
            const double chi_theta = detail::susceptibility(theta_q,
                params.physics.epsilon, params.physics.chi_0);
            const double mu_theta = 1.0 + chi_theta;
            const double M_norm = chi_theta * H_norm;

            H_L2_sq += H_norm * H_norm * JxW;
            M_L2_sq += M_norm * M_norm * JxW;
            result.magnetic_energy += 0.5 * mu_theta * H_norm * H_norm * JxW;

            result.H_max = std::max(result.H_max, H_norm);
            result.M_max = std::max(result.M_max, M_norm);
            result.mu_min = std::min(result.mu_min, mu_theta);
            result.mu_max = std::max(result.mu_max, mu_theta);
        }
    }

    result.H_L2_norm = std::sqrt(H_L2_sq);
    result.M_L2_norm = std::sqrt(M_L2_sq);

    return result;
}

#endif // POISSON_DIAGNOSTICS_H