// ============================================================================
// diagnostics/ns_diagnostics.cc - Navier-Stokes Diagnostics
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "diagnostics/ns_diagnostics.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <algorithm>

// ============================================================================
// NSDiagnostics member functions
// ============================================================================

void NSDiagnostics::print(unsigned int step, double time) const
{
    std::cout << "[NS] step=" << step
              << " t=" << std::scientific << std::setprecision(3) << time
              << " |U|_max=" << std::setprecision(3) << U_max
              << " E_kin=" << kinetic_energy
              << " |divU|=" << div_U_max
              << " CFL=" << std::setprecision(2) << cfl
              << " p∈[" << p_min << "," << p_max << "]"
              << std::defaultfloat << "\n";
}

std::string NSDiagnostics::header()
{
    return "step,time,ux_min,ux_max,uy_min,uy_max,U_L2,U_max,p_min,p_max,E_kin,div_U_L2,div_U_max,cfl";
}

std::string NSDiagnostics::to_csv(unsigned int step, double time) const
{
    std::ostringstream oss;
    oss << step << ","
        << std::scientific << std::setprecision(6)
        << time << ","
        << ux_min << ","
        << ux_max << ","
        << uy_min << ","
        << uy_max << ","
        << U_L2_norm << ","
        << U_max << ","
        << p_min << ","
        << p_max << ","
        << kinetic_energy << ","
        << div_U_L2 << ","
        << div_U_max << ","
        << cfl;
    return oss.str();
}

// ============================================================================
// Compute NS diagnostics
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
    NSDiagnostics diag;

    // Velocity bounds from solution vectors
    diag.ux_min = *std::min_element(ux_solution.begin(), ux_solution.end());
    diag.ux_max = *std::max_element(ux_solution.begin(), ux_solution.end());
    diag.uy_min = *std::min_element(uy_solution.begin(), uy_solution.end());
    diag.uy_max = *std::max_element(uy_solution.begin(), uy_solution.end());

    // Pressure bounds
    diag.p_min = *std::min_element(p_solution.begin(), p_solution.end());
    diag.p_max = *std::max_element(p_solution.begin(), p_solution.end());

    // Setup FE evaluation
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

    // Accumulated values
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

            // |U|² = ux² + uy²
            const double U_sq = ux_q * ux_q + uy_q * uy_q;
            const double U_norm = std::sqrt(U_sq);

            // div(U) = ∂ux/∂x + ∂uy/∂y
            const double div_U = ux_gradients[q][0] + uy_gradients[q][1];

            // Accumulate integrals
            U_L2_sq += U_sq * JxW;
            kinetic_energy += 0.5 * U_sq * JxW;
            div_U_L2_sq += div_U * div_U * JxW;

            // Track maxima
            U_max = std::max(U_max, U_norm);
            div_U_max = std::max(div_U_max, std::abs(div_U));
        }
    }

    // Store results
    diag.U_L2_norm = std::sqrt(U_L2_sq);
    diag.U_max = U_max;
    diag.kinetic_energy = kinetic_energy;
    diag.div_U_L2 = std::sqrt(div_U_L2_sq);
    diag.div_U_max = div_U_max;

    // CFL number
    if (h_min > 0.0)
        diag.cfl = U_max * dt / h_min;
    else
        diag.cfl = 0.0;

    return diag;
}

// ============================================================================
// Explicit instantiations
// ============================================================================
template NSDiagnostics compute_ns_diagnostics<2>(
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    const dealii::Vector<double>&,
    const dealii::Vector<double>&,
    const dealii::Vector<double>&,
    double,
    double);