// ============================================================================
// diagnostics/ch_mms.cc - Implementation of CH MMS functions
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#include "ch_mms.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <iomanip>
#include <cmath>

// ============================================================================
// CHMMSErrors print functions
// ============================================================================
void CHMMSErrors::print() const
{
    std::cout << "  CH MMS Errors:\n"
              << "    θ L2 error: " << std::scientific << std::setprecision(6) << theta_L2 << "\n"
              << "    θ H1 error: " << theta_H1 << "\n"
              << "    ψ L2 error: " << psi_L2 << "\n";
}

void CHMMSErrors::print_for_convergence() const
{
    // Format: h, theta_L2, theta_H1, psi_L2 (tab-separated for easy plotting)
    std::cout << std::scientific << std::setprecision(6)
              << h << "\t" << theta_L2 << "\t" << theta_H1 << "\t" << psi_L2 << "\n";
}

// ============================================================================
// Compute CH MMS errors
// ============================================================================
template <int dim>
CHMMSErrors compute_ch_mms_errors(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::DoFHandler<dim>& psi_dof_handler,
    const dealii::Vector<double>& theta_solution,
    const dealii::Vector<double>& psi_solution,
    double current_time)
{
    CHMMSErrors errors;

    // Get minimum cell diameter for h
    double min_h = std::numeric_limits<double>::max();
    for (const auto& cell : theta_dof_handler.active_cell_iterators())
        min_h = std::min(min_h, cell->diameter());
    errors.h = min_h;

    // Create exact solution objects and set time
    CHExactTheta<dim> theta_exact;
    CHExactPsi<dim> psi_exact;
    theta_exact.set_time(current_time);
    psi_exact.set_time(current_time);

    // Compute θ errors using deal.II VectorTools
    dealii::Vector<double> theta_diff(theta_dof_handler.n_dofs());
    dealii::VectorTools::interpolate(theta_dof_handler, theta_exact, theta_diff);
    theta_diff -= theta_solution;

    // L2 norm via integrate_difference
    dealii::Vector<float> theta_L2_per_cell(theta_dof_handler.get_triangulation().n_active_cells());
    dealii::VectorTools::integrate_difference(
        theta_dof_handler,
        theta_solution,
        theta_exact,
        theta_L2_per_cell,
        dealii::QGauss<dim>(theta_dof_handler.get_fe().degree + 2),
        dealii::VectorTools::L2_norm);
    errors.theta_L2 = dealii::VectorTools::compute_global_error(
        theta_dof_handler.get_triangulation(),
        theta_L2_per_cell,
        dealii::VectorTools::L2_norm);

    // H1 seminorm
    dealii::Vector<float> theta_H1_per_cell(theta_dof_handler.get_triangulation().n_active_cells());
    dealii::VectorTools::integrate_difference(
        theta_dof_handler,
        theta_solution,
        theta_exact,
        theta_H1_per_cell,
        dealii::QGauss<dim>(theta_dof_handler.get_fe().degree + 2),
        dealii::VectorTools::H1_seminorm);
    errors.theta_H1 = dealii::VectorTools::compute_global_error(
        theta_dof_handler.get_triangulation(),
        theta_H1_per_cell,
        dealii::VectorTools::H1_seminorm);

    // Compute ψ L2 error
    dealii::Vector<float> psi_L2_per_cell(psi_dof_handler.get_triangulation().n_active_cells());
    dealii::VectorTools::integrate_difference(
        psi_dof_handler,
        psi_solution,
        psi_exact,
        psi_L2_per_cell,
        dealii::QGauss<dim>(psi_dof_handler.get_fe().degree + 2),
        dealii::VectorTools::L2_norm);
    errors.psi_L2 = dealii::VectorTools::compute_global_error(
        psi_dof_handler.get_triangulation(),
        psi_L2_per_cell,
        dealii::VectorTools::L2_norm);

    return errors;
}

// ============================================================================
// Apply MMS Dirichlet boundary constraints
// ============================================================================
template <int dim>
void apply_ch_mms_boundary_constraints(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::DoFHandler<dim>& psi_dof_handler,
    dealii::AffineConstraints<double>& theta_constraints,
    dealii::AffineConstraints<double>& psi_constraints,
    double current_time)
{
    // Create boundary value functions
    CHMMSBoundaryTheta<dim> theta_bc;
    CHMMSBoundaryPsi<dim> psi_bc;
    theta_bc.set_time(current_time);
    psi_bc.set_time(current_time);

    // Apply Dirichlet BCs on all boundaries (boundary_id not specified = all)
    // For MMS we want θ = θ_exact and ψ = ψ_exact on entire boundary
    dealii::VectorTools::interpolate_boundary_values(
        theta_dof_handler,
        0,  // boundary_id 0 (bottom)
        theta_bc,
        theta_constraints);
    dealii::VectorTools::interpolate_boundary_values(
        theta_dof_handler,
        1,  // boundary_id 1 (right)
        theta_bc,
        theta_constraints);
    dealii::VectorTools::interpolate_boundary_values(
        theta_dof_handler,
        2,  // boundary_id 2 (top)
        theta_bc,
        theta_constraints);
    dealii::VectorTools::interpolate_boundary_values(
        theta_dof_handler,
        3,  // boundary_id 3 (left)
        theta_bc,
        theta_constraints);

    // Same for ψ
    dealii::VectorTools::interpolate_boundary_values(
        psi_dof_handler,
        0,
        psi_bc,
        psi_constraints);
    dealii::VectorTools::interpolate_boundary_values(
        psi_dof_handler,
        1,
        psi_bc,
        psi_constraints);
    dealii::VectorTools::interpolate_boundary_values(
        psi_dof_handler,
        2,
        psi_bc,
        psi_constraints);
    dealii::VectorTools::interpolate_boundary_values(
        psi_dof_handler,
        3,
        psi_bc,
        psi_constraints);
}

// ============================================================================
// Apply MMS initial conditions
// ============================================================================
template <int dim>
void apply_ch_mms_initial_conditions(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::DoFHandler<dim>& psi_dof_handler,
    dealii::Vector<double>& theta_solution,
    dealii::Vector<double>& psi_solution,
    double t_init)
{
    CHMMSInitialTheta<dim> theta_ic(t_init);
    CHMMSInitialPsi<dim> psi_ic(t_init);

    dealii::VectorTools::interpolate(theta_dof_handler, theta_ic, theta_solution);
    dealii::VectorTools::interpolate(psi_dof_handler, psi_ic, psi_solution);
}

// ============================================================================
// Explicit template instantiations
// ============================================================================
template CHMMSErrors compute_ch_mms_errors<2>(
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    const dealii::Vector<double>&,
    const dealii::Vector<double>&,
    double);

template void apply_ch_mms_boundary_constraints<2>(
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    dealii::AffineConstraints<double>&,
    dealii::AffineConstraints<double>&,
    double);

template void apply_ch_mms_initial_conditions<2>(
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    dealii::Vector<double>&,
    dealii::Vector<double>&,
    double);