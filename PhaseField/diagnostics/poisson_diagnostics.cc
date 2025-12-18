// ============================================================================
// diagnostics/poisson_diagnostics.cc - Magnetostatic Poisson Diagnostics
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "diagnostics/poisson_diagnostics.h"
#include "assembly/poisson_assembler.h"  // For compute_applied_field, compute_susceptibility

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <algorithm>

// ============================================================================
// PoissonDiagnostics member functions
// ============================================================================

void PoissonDiagnostics::print(unsigned int step, double time) const
{
    std::cout << "[Poisson] step=" << step
              << " t=" << std::scientific << std::setprecision(3) << time
              << " φ∈[" << std::setprecision(2) << phi_min << "," << phi_max << "]"
              << " |H|_max=" << std::setprecision(3) << H_max
              << " |M|_max=" << M_max
              << " E_mag=" << magnetic_energy
              << " μ∈[" << std::setprecision(2) << mu_min << "," << mu_max << "]"
              << std::defaultfloat << "\n";
}

std::string PoissonDiagnostics::header()
{
    return "step,time,phi_min,phi_max,H_L2,H_max,M_L2,M_max,h_a_max,h_a_dot_grad_phi,E_mag,mu_min,mu_max";
}

std::string PoissonDiagnostics::to_csv(unsigned int step, double time) const
{
    std::ostringstream oss;
    oss << step << ","
        << std::scientific << std::setprecision(6)
        << time << ","
        << phi_min << ","
        << phi_max << ","
        << H_L2_norm << ","
        << H_max << ","
        << M_L2_norm << ","
        << M_max << ","
        << h_a_max << ","
        << h_a_dot_grad_phi << ","
        << magnetic_energy << ","
        << mu_min << ","
        << mu_max;
    return oss.str();
}

// ============================================================================
// Compute diagnostics for quasi-equilibrium model
// ============================================================================
template <int dim>
PoissonDiagnostics compute_poisson_diagnostics(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::Vector<double>& phi_solution,
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::Vector<double>& theta_solution,
    const Parameters& params,
    double current_time)
{
    PoissonDiagnostics diag;

    // φ bounds from solution vector
    diag.phi_min = *std::min_element(phi_solution.begin(), phi_solution.end());
    diag.phi_max = *std::max_element(phi_solution.begin(), phi_solution.end());

    // Setup FE evaluation
    const auto& phi_fe = phi_dof_handler.get_fe();
    const auto& theta_fe = theta_dof_handler.get_fe();

    const unsigned int quad_degree = std::max(phi_fe.degree, theta_fe.degree) + 2;
    dealii::QGauss<dim> quadrature(quad_degree);
    const unsigned int n_q_points = quadrature.size();

    dealii::FEValues<dim> phi_fe_values(phi_fe, quadrature,
        dealii::update_gradients | dealii::update_quadrature_points | dealii::update_JxW_values);

    dealii::FEValues<dim> theta_fe_values(theta_fe, quadrature,
        dealii::update_values);

    std::vector<dealii::Tensor<1, dim>> grad_phi_values(n_q_points);
    std::vector<double> theta_values(n_q_points);

    const double epsilon = params.ch.epsilon;
    const double chi_0 = params.magnetization.chi_0;

    // Accumulated values
    double H_L2_sq = 0.0;
    double M_L2_sq = 0.0;
    double magnetic_energy = 0.0;
    double h_a_dot_grad_phi = 0.0;

    double H_max = 0.0;
    double M_max = 0.0;
    double h_a_max = 0.0;
    double mu_min = 1e10;
    double mu_max = 0.0;

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
            const dealii::Point<dim>& x_q = phi_fe_values.quadrature_point(q);

            // H = ∇φ
            const dealii::Tensor<1, dim>& H = grad_phi_values[q];
            const double H_norm = H.norm();

            // θ and derived quantities
            const double theta_q = theta_values[q];
            const double chi_theta = compute_susceptibility(theta_q, epsilon, chi_0);
            const double mu_theta = 1.0 + chi_theta;

            // M = χ(θ)H (quasi-equilibrium)
            const double M_norm = chi_theta * H_norm;

            // Applied field h_a
            dealii::Tensor<1, dim> h_a = compute_applied_field(x_q, params, current_time);
            const double h_a_norm = h_a.norm();

            // Accumulate integrals
            H_L2_sq += H_norm * H_norm * JxW;
            M_L2_sq += M_norm * M_norm * JxW;
            magnetic_energy += 0.5 * mu_theta * H_norm * H_norm * JxW;
            h_a_dot_grad_phi += (h_a * H) * JxW;

            // Track maxima
            H_max = std::max(H_max, H_norm);
            M_max = std::max(M_max, M_norm);
            h_a_max = std::max(h_a_max, h_a_norm);
            mu_min = std::min(mu_min, mu_theta);
            mu_max = std::max(mu_max, mu_theta);
        }
    }

    // Store results
    diag.H_L2_norm = std::sqrt(H_L2_sq);
    diag.H_max = H_max;
    diag.M_L2_norm = std::sqrt(M_L2_sq);
    diag.M_max = M_max;
    diag.h_a_max = h_a_max;
    diag.h_a_dot_grad_phi = h_a_dot_grad_phi;
    diag.magnetic_energy = magnetic_energy;
    diag.mu_min = mu_min;
    diag.mu_max = mu_max;

    return diag;
}

// ============================================================================
// Compute diagnostics for simplified model (M = 0, μ = 1)
// ============================================================================
template <int dim>
PoissonDiagnostics compute_poisson_diagnostics_simplified(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::Vector<double>& phi_solution,
    const Parameters& params,
    double current_time)
{
    PoissonDiagnostics diag;

    // φ bounds
    diag.phi_min = *std::min_element(phi_solution.begin(), phi_solution.end());
    diag.phi_max = *std::max_element(phi_solution.begin(), phi_solution.end());

    // Setup FE evaluation
    const auto& phi_fe = phi_dof_handler.get_fe();

    dealii::QGauss<dim> quadrature(phi_fe.degree + 2);
    const unsigned int n_q_points = quadrature.size();

    dealii::FEValues<dim> phi_fe_values(phi_fe, quadrature,
        dealii::update_gradients | dealii::update_quadrature_points | dealii::update_JxW_values);

    std::vector<dealii::Tensor<1, dim>> grad_phi_values(n_q_points);

    double H_L2_sq = 0.0;
    double magnetic_energy = 0.0;
    double h_a_dot_grad_phi = 0.0;

    double H_max = 0.0;
    double h_a_max = 0.0;

    for (const auto& cell : phi_dof_handler.active_cell_iterators())
    {
        phi_fe_values.reinit(cell);
        phi_fe_values.get_function_gradients(phi_solution, grad_phi_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = phi_fe_values.JxW(q);
            const dealii::Point<dim>& x_q = phi_fe_values.quadrature_point(q);

            const dealii::Tensor<1, dim>& H = grad_phi_values[q];
            const double H_norm = H.norm();

            dealii::Tensor<1, dim> h_a = compute_applied_field(x_q, params, current_time);
            const double h_a_norm = h_a.norm();

            // For simplified: μ = 1, M = 0
            H_L2_sq += H_norm * H_norm * JxW;
            magnetic_energy += 0.5 * H_norm * H_norm * JxW;  // μ = 1
            h_a_dot_grad_phi += (h_a * H) * JxW;

            H_max = std::max(H_max, H_norm);
            h_a_max = std::max(h_a_max, h_a_norm);
        }
    }

    diag.H_L2_norm = std::sqrt(H_L2_sq);
    diag.H_max = H_max;
    diag.M_L2_norm = 0.0;  // M = 0 for simplified
    diag.M_max = 0.0;
    diag.h_a_max = h_a_max;
    diag.h_a_dot_grad_phi = h_a_dot_grad_phi;
    diag.magnetic_energy = magnetic_energy;
    diag.mu_min = 1.0;
    diag.mu_max = 1.0;

    return diag;
}

// ============================================================================
// Explicit instantiations
// ============================================================================
template PoissonDiagnostics compute_poisson_diagnostics<2>(
    const dealii::DoFHandler<2>&,
    const dealii::Vector<double>&,
    const dealii::DoFHandler<2>&,
    const dealii::Vector<double>&,
    const Parameters&,
    double);

template PoissonDiagnostics compute_poisson_diagnostics_simplified<2>(
    const dealii::DoFHandler<2>&,
    const dealii::Vector<double>&,
    const Parameters&,
    double);