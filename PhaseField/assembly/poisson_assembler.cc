// ============================================================================
// assembly/poisson_assembler.cc - Magnetostatic Poisson Assembly (MMS-AWARE)
//
// PAPER EQUATION 42d (corrected interpretation):
//   (∇φ, ∇χ) = (-M^k, ∇χ)  ∀χ ∈ X_h
//
// This solves for the demagnetizing potential φ where h_d = ∇φ.
// The applied field h_a is handled in the magnetization equation.
//
// MMS MODE:
//   When params.enable_mms=true, adds MMS source term:
//   (∇φ, ∇χ) = (-M^k, ∇χ) + (f_MMS, χ)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "assembly/poisson_assembler.h"
#include "mms/poisson_mms.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/full_matrix.h>

#include <iostream>
#include <memory>

// ============================================================================
// Assemble the magnetostatic Poisson system (Paper Eq. 42d)
//
//   (∇φ, ∇χ) = (h_a - M^k, ∇χ)  ∀χ ∈ X_h
//
// M is computed by solve_magnetization() before this is called.
//
// In MMS mode, adds source term (f_MMS, χ) to RHS where f_MMS is computed
// from exact solutions φ_exact and M_exact.
// ============================================================================
template <int dim>
void assemble_poisson_system(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::DoFHandler<dim>& M_dof_handler,
    const dealii::Vector<double>& mx_solution,
    const dealii::Vector<double>& my_solution,
    const Parameters& params,
    double current_time,
    dealii::SparseMatrix<double>& phi_matrix,
    dealii::Vector<double>& phi_rhs,
    const dealii::AffineConstraints<double>& phi_constraints)
{
    const bool mms_mode = params.enable_mms;
    const double L_y = params.domain.y_max - params.domain.y_min;

    // Check if M is provided (non-empty vectors)
    const bool has_M = (mx_solution.size() > 0);

    phi_matrix = 0;
    phi_rhs = 0;

    const auto& phi_fe = phi_dof_handler.get_fe();
    const unsigned int dofs_per_cell = phi_fe.n_dofs_per_cell();

    const unsigned int quad_degree = phi_fe.degree + 2;
    dealii::QGauss<dim> quadrature(quad_degree);
    const unsigned int n_q_points = quadrature.size();

    // FEValues for φ
    dealii::FEValues<dim> phi_fe_values(phi_fe, quadrature,
        dealii::update_values |
        dealii::update_gradients |
        dealii::update_quadrature_points |
        dealii::update_JxW_values);

    // FEValues for M (only if M is provided)
    std::unique_ptr<dealii::FEValues<dim>> M_fe_values_ptr;
    if (has_M)
    {
        M_fe_values_ptr = std::make_unique<dealii::FEValues<dim>>(
            M_dof_handler.get_fe(), quadrature, dealii::update_values);
    }

    dealii::FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    dealii::Vector<double> local_rhs(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<double> mx_values(n_q_points);
    std::vector<double> my_values(n_q_points);

    auto phi_cell = phi_dof_handler.begin_active();
    auto M_cell = has_M ? M_dof_handler.begin_active() : decltype(M_dof_handler.begin_active())();

    for (; phi_cell != phi_dof_handler.end(); ++phi_cell)
    {
        phi_fe_values.reinit(phi_cell);

        local_matrix = 0;
        local_rhs = 0;

        // Get M values if provided
        if (has_M)
        {
            M_fe_values_ptr->reinit(M_cell);
            M_fe_values_ptr->get_function_values(mx_solution, mx_values);
            M_fe_values_ptr->get_function_values(my_solution, my_values);
        }

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = phi_fe_values.JxW(q);
            const dealii::Point<dim>& x_q = phi_fe_values.quadrature_point(q);

            // Magnetization M (zero if not provided)
            dealii::Tensor<1, dim> M;
            if (has_M)
            {
                M[0] = mx_values[q];
                M[1] = my_values[q];
            }
            else
            {
                M[0] = 0.0;
                M[1] = 0.0;
            }

            // RHS source: -M (demagnetizing field only)
            dealii::Tensor<1, dim> source = -M;

            // MMS source term
            double f_mms = 0.0;
            if (mms_mode)
            {
                if (has_M)
                {
                    // Coupled: f_MMS = -Δφ_exact - ∇·M_exact
                    f_mms = compute_poisson_mms_source_coupled<dim>(x_q, current_time, L_y);
                }
                else
                {
                    // Standalone: f_MMS = -Δφ_exact
                    f_mms = compute_poisson_mms_source_standalone<dim>(x_q, current_time, L_y);
                }
            }

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const double chi_i = phi_fe_values.shape_value(i, q);
                const auto& grad_chi_i = phi_fe_values.shape_grad(i, q);

                // RHS: (-M, ∇χ)
                local_rhs(i) += (source * grad_chi_i) * JxW;

                // MMS: add (f_MMS, χ) term
                if (mms_mode)
                {
                    local_rhs(i) += f_mms * chi_i * JxW;
                }

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    const auto& grad_chi_j = phi_fe_values.shape_grad(j, q);
                    // LHS: (∇φ, ∇χ)
                    local_matrix(i, j) += (grad_chi_i * grad_chi_j) * JxW;
                }
            }
        }

        phi_cell->get_dof_indices(local_dof_indices);
        phi_constraints.distribute_local_to_global(
            local_matrix, local_rhs, local_dof_indices,
            phi_matrix, phi_rhs);

        // Advance M cell iterator if M is provided
        if (has_M)
            ++M_cell;
    }
}

// ============================================================================
// Explicit instantiation
// ============================================================================
template void assemble_poisson_system<2>(
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    const dealii::Vector<double>&,
    const dealii::Vector<double>&,
    const Parameters&,
    double,
    dealii::SparseMatrix<double>&,
    dealii::Vector<double>&,
    const dealii::AffineConstraints<double>&);
