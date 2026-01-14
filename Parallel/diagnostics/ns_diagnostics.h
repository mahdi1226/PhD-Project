// ============================================================================
// diagnostics/ns_diagnostics.h - Navier-Stokes Diagnostics (Parallel)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
//
// Computes:
//   - Velocity bounds and norms
//   - Pressure bounds
//   - Kinetic energy: ½∫|U|² dx
//   - Divergence (incompressibility): ||div U||
//   - CFL number
//
// All quantities are MPI-reduced for parallel correctness.
// ============================================================================
#ifndef NS_DIAGNOSTICS_H
#define NS_DIAGNOSTICS_H

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
// NS Diagnostic Data
// ============================================================================
struct NSDiagnostics
{
    // Velocity bounds
    double ux_min = 0.0;
    double ux_max = 0.0;
    double uy_min = 0.0;
    double uy_max = 0.0;

    // Velocity norms
    double U_L2_norm = 0.0;     // ||U||_{L²}
    double U_max = 0.0;         // max |U|

    // Pressure bounds
    double p_min = 0.0;
    double p_max = 0.0;

    // Kinetic energy: ½∫|U|² dx
    double kinetic_energy = 0.0;

    // Incompressibility: div(U)
    double div_U_L2 = 0.0;      // ||div U||_{L²}
    double div_U_max = 0.0;     // max |div U|

    // CFL number: max|U| * dt / h_min
    double cfl = 0.0;
};

// ============================================================================
// Compute NS diagnostics (parallel version with Trilinos vectors)
// ============================================================================
template <int dim>
NSDiagnostics compute_ns_diagnostics(
    const dealii::DoFHandler<dim>& ux_dof_handler,
    const dealii::DoFHandler<dim>& uy_dof_handler,
    const dealii::DoFHandler<dim>& p_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& ux_solution,
    const dealii::TrilinosWrappers::MPI::Vector& uy_solution,
    const dealii::TrilinosWrappers::MPI::Vector& p_solution,
    double dt,
    double h_min,
    MPI_Comm comm = MPI_COMM_WORLD)
{
    const auto& ux_fe = ux_dof_handler.get_fe();
    const auto& uy_fe = uy_dof_handler.get_fe();

    dealii::QGauss<dim> quadrature(ux_fe.degree + 2);
    const unsigned int n_q_points = quadrature.size();

    dealii::FEValues<dim> ux_fe_values(ux_fe, quadrature,
        dealii::update_values | dealii::update_gradients | dealii::update_JxW_values);

    dealii::FEValues<dim> uy_fe_values(uy_fe, quadrature,
        dealii::update_values | dealii::update_gradients);

    std::vector<double> ux_values(n_q_points);
    std::vector<double> uy_values(n_q_points);
    std::vector<dealii::Tensor<1, dim>> ux_gradients(n_q_points);
    std::vector<dealii::Tensor<1, dim>> uy_gradients(n_q_points);

    // Local accumulators
    double local_U_L2_sq = 0.0;
    double local_kinetic_energy = 0.0;
    double local_div_U_L2_sq = 0.0;
    double local_U_max = 0.0;
    double local_div_U_max = 0.0;
    double local_ux_min = std::numeric_limits<double>::max();
    double local_ux_max = std::numeric_limits<double>::lowest();
    double local_uy_min = std::numeric_limits<double>::max();
    double local_uy_max = std::numeric_limits<double>::lowest();
    double local_p_min = std::numeric_limits<double>::max();
    double local_p_max = std::numeric_limits<double>::lowest();

    auto ux_cell = ux_dof_handler.begin_active();
    auto uy_cell = uy_dof_handler.begin_active();

    for (; ux_cell != ux_dof_handler.end(); ++ux_cell, ++uy_cell)
    {
        if (!ux_cell->is_locally_owned())
            continue;

        ux_fe_values.reinit(ux_cell);
        uy_fe_values.reinit(uy_cell);

        ux_fe_values.get_function_values(ux_solution, ux_values);
        uy_fe_values.get_function_values(uy_solution, uy_values);
        ux_fe_values.get_function_gradients(ux_solution, ux_gradients);
        uy_fe_values.get_function_gradients(uy_solution, uy_gradients);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = ux_fe_values.JxW(q);
            const double ux_q = ux_values[q];
            const double uy_q = uy_values[q];

            // |U|² = ux² + uy²
            const double U_sq = ux_q * ux_q + uy_q * uy_q;
            const double U_norm = std::sqrt(U_sq);

            // div(U) = ∂ux/∂x + ∂uy/∂y
            const double div_U = ux_gradients[q][0] + uy_gradients[q][1];

            // Accumulate integrals
            local_U_L2_sq += U_sq * JxW;
            local_kinetic_energy += 0.5 * U_sq * JxW;
            local_div_U_L2_sq += div_U * div_U * JxW;

            // Track maxima
            local_U_max = std::max(local_U_max, U_norm);
            local_div_U_max = std::max(local_div_U_max, std::abs(div_U));

            // Velocity bounds at quadrature points
            local_ux_min = std::min(local_ux_min, ux_q);
            local_ux_max = std::max(local_ux_max, ux_q);
            local_uy_min = std::min(local_uy_min, uy_q);
            local_uy_max = std::max(local_uy_max, uy_q);
        }
    }

    // Pressure bounds from solution vector (locally owned entries)
    for (const auto& cell : p_dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        std::vector<dealii::types::global_dof_index> local_dof_indices(
            p_dof_handler.get_fe().dofs_per_cell);
        cell->get_dof_indices(local_dof_indices);

        for (const auto& idx : local_dof_indices)
        {
            if (p_solution.locally_owned_elements().is_element(idx))
            {
                const double p_val = p_solution[idx];
                local_p_min = std::min(local_p_min, p_val);
                local_p_max = std::max(local_p_max, p_val);
            }
        }
    }

    // MPI reductions
    NSDiagnostics result;

    double global_U_L2_sq = MPIUtils::reduce_sum(local_U_L2_sq, comm);
    double global_div_U_L2_sq = MPIUtils::reduce_sum(local_div_U_L2_sq, comm);

    result.U_L2_norm = std::sqrt(global_U_L2_sq);
    result.kinetic_energy = MPIUtils::reduce_sum(local_kinetic_energy, comm);
    result.div_U_L2 = std::sqrt(global_div_U_L2_sq);
    result.div_U_max = MPIUtils::reduce_max(local_div_U_max, comm);
    result.U_max = MPIUtils::reduce_max(local_U_max, comm);

    result.ux_min = MPIUtils::reduce_min(local_ux_min, comm);
    result.ux_max = MPIUtils::reduce_max(local_ux_max, comm);
    result.uy_min = MPIUtils::reduce_min(local_uy_min, comm);
    result.uy_max = MPIUtils::reduce_max(local_uy_max, comm);
    result.p_min = MPIUtils::reduce_min(local_p_min, comm);
    result.p_max = MPIUtils::reduce_max(local_p_max, comm);

    // CFL number
    result.cfl = (h_min > 0.0) ? result.U_max * dt / h_min : 0.0;

    return result;
}

// ============================================================================
// Compute NS diagnostics (serial version with deal.II vectors)
// ============================================================================
template <int dim>
NSDiagnostics compute_ns_diagnostics(
    const dealii::DoFHandler<dim>& ux_dof_handler,
    const dealii::DoFHandler<dim>& uy_dof_handler,
    const dealii::DoFHandler<dim>& p_dof_handler,
    const dealii::Vector<double>& ux_solution,
    const dealii::Vector<double>& uy_solution,
    const dealii::Vector<double>& p_solution,
    double dt,
    double h_min)
{
    NSDiagnostics result;

    // Velocity bounds from solution vectors
    result.ux_min = *std::min_element(ux_solution.begin(), ux_solution.end());
    result.ux_max = *std::max_element(ux_solution.begin(), ux_solution.end());
    result.uy_min = *std::min_element(uy_solution.begin(), uy_solution.end());
    result.uy_max = *std::max_element(uy_solution.begin(), uy_solution.end());

    // Pressure bounds
    result.p_min = *std::min_element(p_solution.begin(), p_solution.end());
    result.p_max = *std::max_element(p_solution.begin(), p_solution.end());

    const auto& ux_fe = ux_dof_handler.get_fe();
    const auto& uy_fe = uy_dof_handler.get_fe();

    dealii::QGauss<dim> quadrature(ux_fe.degree + 2);
    const unsigned int n_q_points = quadrature.size();

    dealii::FEValues<dim> ux_fe_values(ux_fe, quadrature,
        dealii::update_values | dealii::update_gradients | dealii::update_JxW_values);

    dealii::FEValues<dim> uy_fe_values(uy_fe, quadrature,
        dealii::update_values | dealii::update_gradients);

    std::vector<double> ux_values(n_q_points);
    std::vector<double> uy_values(n_q_points);
    std::vector<dealii::Tensor<1, dim>> ux_gradients(n_q_points);
    std::vector<dealii::Tensor<1, dim>> uy_gradients(n_q_points);

    double U_L2_sq = 0.0;
    double kinetic_energy = 0.0;
    double div_U_L2_sq = 0.0;
    double U_max = 0.0;
    double div_U_max = 0.0;

    auto ux_cell = ux_dof_handler.begin_active();
    auto uy_cell = uy_dof_handler.begin_active();

    for (; ux_cell != ux_dof_handler.end(); ++ux_cell, ++uy_cell)
    {
        ux_fe_values.reinit(ux_cell);
        uy_fe_values.reinit(uy_cell);

        ux_fe_values.get_function_values(ux_solution, ux_values);
        uy_fe_values.get_function_values(uy_solution, uy_values);
        ux_fe_values.get_function_gradients(ux_solution, ux_gradients);
        uy_fe_values.get_function_gradients(uy_solution, uy_gradients);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = ux_fe_values.JxW(q);
            const double ux_q = ux_values[q];
            const double uy_q = uy_values[q];

            const double U_sq = ux_q * ux_q + uy_q * uy_q;
            const double U_norm = std::sqrt(U_sq);
            const double div_U = ux_gradients[q][0] + uy_gradients[q][1];

            U_L2_sq += U_sq * JxW;
            kinetic_energy += 0.5 * U_sq * JxW;
            div_U_L2_sq += div_U * div_U * JxW;

            U_max = std::max(U_max, U_norm);
            div_U_max = std::max(div_U_max, std::abs(div_U));
        }
    }

    result.U_L2_norm = std::sqrt(U_L2_sq);
    result.U_max = U_max;
    result.kinetic_energy = kinetic_energy;
    result.div_U_L2 = std::sqrt(div_U_L2_sq);
    result.div_U_max = div_U_max;
    result.cfl = (h_min > 0.0) ? U_max * dt / h_min : 0.0;

    return result;
}

#endif // NS_DIAGNOSTICS_H