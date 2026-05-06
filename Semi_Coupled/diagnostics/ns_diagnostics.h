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
#include <vector>

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
// Compute NS diagnostics from COUPLED solution (parallel Trilinos version)
//
// Works directly with the monolithic NS solution vector using index maps.
// ============================================================================
// ============================================================================
// Compute NS diagnostics (parallel version with separate component vectors)
//
// This matches the decoupled NS storage used in phase_field.cc:
//   ux_solution, uy_solution, p_solution stored as separate Trilinos vectors.
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

    // Two quadratures, sharing the same FEValues call sites:
    //   - QGauss for L2-norm-style integrals (correct quadrature accuracy).
    //   - QGaussLobatto for max-norm sampling. Lobatto includes the cell
    //     endpoints/vertices, so for Q2 velocities the sample points span
    //     the support points where the polynomial maximum is most likely.
    //     Plain QGauss systematically underestimated U_max → CFL underestimate
    //     (A6-10 finding).
    dealii::QGauss<dim>        q_integ(ux_fe.degree + 2);
    dealii::QGaussLobatto<dim> q_max  (ux_fe.degree + 2);

    dealii::FEValues<dim> ux_int(
        ux_fe, q_integ,
        dealii::update_values | dealii::update_gradients | dealii::update_JxW_values);
    dealii::FEValues<dim> uy_int(
        uy_fe, q_integ,
        dealii::update_values | dealii::update_gradients);

    dealii::FEValues<dim> ux_max(
        ux_fe, q_max,
        dealii::update_values | dealii::update_gradients);
    dealii::FEValues<dim> uy_max(
        uy_fe, q_max,
        dealii::update_values | dealii::update_gradients);

    // Function-value scratch (A4-2: replaces O(dofs_per_cell × n_q) manual
    // shape_value loops with deal.II's vectorized get_function_values).
    std::vector<double> ux_int_vals(q_integ.size()), uy_int_vals(q_integ.size());
    std::vector<dealii::Tensor<1,dim>> ux_int_grads(q_integ.size()), uy_int_grads(q_integ.size());
    std::vector<double> ux_max_vals(q_max.size()), uy_max_vals(q_max.size());
    std::vector<dealii::Tensor<1,dim>> ux_max_grads(q_max.size()), uy_max_grads(q_max.size());

    double local_U_L2_sq        = 0.0;
    double local_kinetic_energy = 0.0;
    double local_div_U_L2_sq    = 0.0;
    double local_U_max          = 0.0;
    double local_div_U_max      = 0.0;

    double local_ux_min = std::numeric_limits<double>::max();
    double local_ux_max = std::numeric_limits<double>::lowest();
    double local_uy_min = std::numeric_limits<double>::max();
    double local_uy_max = std::numeric_limits<double>::lowest();
    double local_p_min  = std::numeric_limits<double>::max();
    double local_p_max  = std::numeric_limits<double>::lowest();

    for (const auto& ux_cell : ux_dof_handler.active_cell_iterators())
    {
        if (!ux_cell->is_locally_owned())
            continue;

        const auto uy_cell =
            typename dealii::DoFHandler<dim>::active_cell_iterator(
                &uy_dof_handler.get_triangulation(),
                ux_cell->level(),
                ux_cell->index(),
                &uy_dof_handler);

        // ---- L2 / kinetic / div integrals at Gauss points ----
        ux_int.reinit(ux_cell);
        uy_int.reinit(uy_cell);
        ux_int.get_function_values   (ux_solution, ux_int_vals);
        ux_int.get_function_gradients(ux_solution, ux_int_grads);
        uy_int.get_function_values   (uy_solution, uy_int_vals);
        uy_int.get_function_gradients(uy_solution, uy_int_grads);

        for (unsigned int q = 0; q < q_integ.size(); ++q)
        {
            const double JxW = ux_int.JxW(q);
            const double ux_q = ux_int_vals[q];
            const double uy_q = uy_int_vals[q];
            const double U_sq = ux_q * ux_q + uy_q * uy_q;
            const double div_U = ux_int_grads[q][0] + uy_int_grads[q][1];

            local_U_L2_sq        += U_sq * JxW;
            local_kinetic_energy += 0.5 * U_sq * JxW;
            local_div_U_L2_sq    += div_U * div_U * JxW;
            local_div_U_max = std::max(local_div_U_max, std::abs(div_U));
        }

        // ---- max-norm sampling at Gauss-Lobatto (vertex-inclusive) points ----
        ux_max.reinit(ux_cell);
        uy_max.reinit(uy_cell);
        ux_max.get_function_values   (ux_solution, ux_max_vals);
        uy_max.get_function_values   (uy_solution, uy_max_vals);

        for (unsigned int q = 0; q < q_max.size(); ++q)
        {
            const double ux_q = ux_max_vals[q];
            const double uy_q = uy_max_vals[q];
            const double U_norm = std::sqrt(ux_q * ux_q + uy_q * uy_q);
            local_U_max  = std::max(local_U_max,  U_norm);
            local_ux_min = std::min(local_ux_min, ux_q);
            local_ux_max = std::max(local_ux_max, ux_q);
            local_uy_min = std::min(local_uy_min, uy_q);
            local_uy_max = std::max(local_uy_max, uy_q);
        }
    }

    // Pressure bounds from locally owned DoFs
    const dealii::IndexSet& p_owned = p_dof_handler.locally_owned_dofs();
    for (auto it = p_owned.begin(); it != p_owned.end(); ++it)
    {
        const double val = p_solution[*it];
        local_p_min = std::min(local_p_min, val);
        local_p_max = std::max(local_p_max, val);
    }

    NSDiagnostics result;
    result.U_L2_norm      = std::sqrt(MPIUtils::reduce_sum(local_U_L2_sq, comm));
    result.kinetic_energy = MPIUtils::reduce_sum(local_kinetic_energy, comm);
    result.div_U_L2       = std::sqrt(MPIUtils::reduce_sum(local_div_U_L2_sq, comm));
    result.div_U_max      = MPIUtils::reduce_max(local_div_U_max, comm);
    result.U_max          = MPIUtils::reduce_max(local_U_max, comm);

    result.ux_min = MPIUtils::reduce_min(local_ux_min, comm);
    result.ux_max = MPIUtils::reduce_max(local_ux_max, comm);
    result.uy_min = MPIUtils::reduce_min(local_uy_min, comm);
    result.uy_max = MPIUtils::reduce_max(local_uy_max, comm);
    result.p_min  = MPIUtils::reduce_min(local_p_min, comm);
    result.p_max  = MPIUtils::reduce_max(local_p_max, comm);

    result.cfl = (h_min > 0.0) ? result.U_max * dt / h_min : 0.0;
    return result;
}

#endif // NS_DIAGNOSTICS_H