// ============================================================================
// mms/poisson_mms.cc - Poisson Method of Manufactured Solutions
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "mms/poisson_mms.h"
#include "physics/material_properties.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/full_matrix.h>

#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>

// ============================================================================
// Print MMS errors
// ============================================================================
void PoissonMMSError::print(unsigned int refinement, double h) const
{
    std::cout << "[Poisson MMS] ref=" << refinement
              << " h=" << std::scientific << std::setprecision(2) << h
              << " L2=" << std::setprecision(4) << L2_error
              << " H1=" << H1_error
              << " Linf=" << Linf_error
              << std::defaultfloat << "\n";
}

// ============================================================================
// Compute Poisson MMS errors
// ============================================================================
template <int dim>
PoissonMMSError compute_poisson_mms_error(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::Vector<double>& phi_solution,
    double time,
    double L_y)
{
    PoissonMMSError error;

    const auto& fe = phi_dof_handler.get_fe();
    dealii::QGauss<dim> quadrature(fe.degree + 2);
    const unsigned int n_q_points = quadrature.size();

    dealii::FEValues<dim> fe_values(fe, quadrature,
        dealii::update_values | dealii::update_gradients |
        dealii::update_quadrature_points | dealii::update_JxW_values);

    std::vector<double> phi_values(n_q_points);
    std::vector<dealii::Tensor<1, dim>> phi_gradients(n_q_points);

    PoissonExactSolution<dim> exact_solution(time, L_y);

    double L2_sq = 0.0;
    double H1_sq = 0.0;
    double Linf = 0.0;

    for (const auto& cell : phi_dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        fe_values.get_function_values(phi_solution, phi_values);
        fe_values.get_function_gradients(phi_solution, phi_gradients);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = fe_values.JxW(q);
            const dealii::Point<dim>& x_q = fe_values.quadrature_point(q);

            // Value error
            const double phi_exact = exact_solution.value(x_q);
            const double value_error = phi_values[q] - phi_exact;
            L2_sq += value_error * value_error * JxW;
            Linf = std::max(Linf, std::abs(value_error));

            // Gradient error
            const dealii::Tensor<1, dim> grad_exact = exact_solution.gradient(x_q);
            const dealii::Tensor<1, dim> grad_error = phi_gradients[q] - grad_exact;
            H1_sq += (grad_error * grad_error) * JxW;
        }
    }

    error.L2_error = std::sqrt(L2_sq);
    error.H1_error = std::sqrt(H1_sq);
    error.Linf_error = Linf;

    return error;
}

// ============================================================================
// Assemble Poisson system with MMS source (SIMPLIFIED, μ = 1)
//
// (∇φ, ∇χ) = (h_a_MMS, ∇χ)  where h_a_MMS = ∇φ_exact
// ============================================================================
template <int dim>
void assemble_poisson_system_mms_simplified(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    double current_time,
    double L_y,
    dealii::SparseMatrix<double>& phi_matrix,
    dealii::Vector<double>& phi_rhs,
    const dealii::AffineConstraints<double>& phi_constraints)
{
    phi_matrix = 0;
    phi_rhs = 0;

    const auto& fe = phi_dof_handler.get_fe();
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

    dealii::QGauss<dim> quadrature(fe.degree + 2);
    const unsigned int n_q_points = quadrature.size();

    dealii::FEValues<dim> fe_values(fe, quadrature,
        dealii::update_values | dealii::update_gradients |
        dealii::update_quadrature_points | dealii::update_JxW_values);

    dealii::FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    dealii::Vector<double> local_rhs(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto& cell : phi_dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        local_matrix = 0;
        local_rhs = 0;

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = fe_values.JxW(q);
            const dealii::Point<dim>& x_q = fe_values.quadrature_point(q);

            // MMS source: h_a = ∇φ_exact
            dealii::Tensor<1, dim> h_a = poisson_mms_source_simplified<dim>(x_q, current_time, L_y);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const auto& grad_chi_i = fe_values.shape_grad(i, q);

                // RHS: (h_a, ∇χ)
                local_rhs(i) += (h_a * grad_chi_i) * JxW;

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    const auto& grad_chi_j = fe_values.shape_grad(j, q);
                    // LHS: (∇φ, ∇χ)
                    local_matrix(i, j) += (grad_chi_i * grad_chi_j) * JxW;
                }
            }
        }

        cell->get_dof_indices(local_dof_indices);
        phi_constraints.distribute_local_to_global(
            local_matrix, local_rhs, local_dof_indices,
            phi_matrix, phi_rhs);
    }
}

// ============================================================================
// Assemble Poisson system with MMS source (QUASI-EQUILIBRIUM)
//
// (μ(θ)∇φ, ∇χ) = (h_a_MMS, ∇χ)  where h_a_MMS = μ(θ)∇φ_exact
// ============================================================================
template <int dim>
void assemble_poisson_system_mms_quasi_equilibrium(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::Vector<double>& theta_solution,
    const Parameters& params,
    double current_time,
    double L_y,
    dealii::SparseMatrix<double>& phi_matrix,
    dealii::Vector<double>& phi_rhs,
    const dealii::AffineConstraints<double>& phi_constraints)
{
    phi_matrix = 0;
    phi_rhs = 0;

    const auto& phi_fe = phi_dof_handler.get_fe();
    const auto& theta_fe = theta_dof_handler.get_fe();
    const unsigned int dofs_per_cell = phi_fe.n_dofs_per_cell();

    const unsigned int quad_degree = std::max(phi_fe.degree, theta_fe.degree) + 2;
    dealii::QGauss<dim> quadrature(quad_degree);
    const unsigned int n_q_points = quadrature.size();

    dealii::FEValues<dim> phi_fe_values(phi_fe, quadrature,
        dealii::update_values | dealii::update_gradients |
        dealii::update_quadrature_points | dealii::update_JxW_values);

    dealii::FEValues<dim> theta_fe_values(theta_fe, quadrature,
        dealii::update_values);

    dealii::FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    dealii::Vector<double> local_rhs(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<double> theta_values(n_q_points);

    auto phi_cell = phi_dof_handler.begin_active();
    auto theta_cell = theta_dof_handler.begin_active();

    for (; phi_cell != phi_dof_handler.end(); ++phi_cell, ++theta_cell)
    {
        phi_fe_values.reinit(phi_cell);
        theta_fe_values.reinit(theta_cell);

        local_matrix = 0;
        local_rhs = 0;

        theta_fe_values.get_function_values(theta_solution, theta_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = phi_fe_values.JxW(q);
            const dealii::Point<dim>& x_q = phi_fe_values.quadrature_point(q);

            // θ value and μ(θ)
            const double theta_q = theta_values[q];
            const double sigmoid_val = 1.0 / (1.0 + std::exp(-theta_q / epsilon));
            const double chi_theta = chi_0 * sigmoid_val;
            const double mu_theta = 1.0 + chi_theta;

            // MMS source: h_a = μ(θ)∇φ_exact
            dealii::Tensor<1, dim> h_a = poisson_mms_source_quasi_equilibrium<dim>(
                x_q, theta_q, current_time, epsilon, chi_0, L_y);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const auto& grad_chi_i = phi_fe_values.shape_grad(i, q);

                // RHS: (h_a, ∇χ) = (μ(θ)∇φ_exact, ∇χ)
                local_rhs(i) += (h_a * grad_chi_i) * JxW;

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    const auto& grad_chi_j = phi_fe_values.shape_grad(j, q);
                    // LHS: (μ(θ)∇φ, ∇χ)
                    local_matrix(i, j) += mu_theta * (grad_chi_i * grad_chi_j) * JxW;
                }
            }
        }

        phi_cell->get_dof_indices(local_dof_indices);
        phi_constraints.distribute_local_to_global(
            local_matrix, local_rhs, local_dof_indices,
            phi_matrix, phi_rhs);
    }
}

// ============================================================================
// Explicit instantiations
// ============================================================================
template PoissonMMSError compute_poisson_mms_error<2>(
    const dealii::DoFHandler<2>&,
    const dealii::Vector<double>&,
    double,
    double);

template void assemble_poisson_system_mms_simplified<2>(
    const dealii::DoFHandler<2>&,
    double,
    double,
    dealii::SparseMatrix<double>&,
    dealii::Vector<double>&,
    const dealii::AffineConstraints<double>&);

template void assemble_poisson_system_mms_quasi_equilibrium<2>(
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    const dealii::Vector<double>&,
    const Parameters&,
    double,
    double,
    dealii::SparseMatrix<double>&,
    dealii::Vector<double>&,
    const dealii::AffineConstraints<double>&);