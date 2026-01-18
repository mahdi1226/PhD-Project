// ============================================================================
// assembly/poisson_assembler.cc - Magnetostatic Poisson Assembly (PARALLEL)
//
// PAPER EQUATION 42d:
//   (∇φ, ∇χ) = (h_a - M^k, ∇χ)  ∀χ ∈ X_h
//
// OPTIMIZATION: Split into matrix (once) and RHS (every timestep)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "assembly/poisson_assembler.h"
#include "mms/poisson/poisson_mms.h"
#include "physics/applied_field.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/full_matrix.h>

#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// assemble_poisson_matrix - Laplacian only (ONCE at setup)
// ============================================================================
template <int dim>
void assemble_poisson_matrix(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::AffineConstraints<double>& phi_constraints,
    dealii::TrilinosWrappers::SparseMatrix& phi_matrix)
{
    phi_matrix = 0;

    const auto& phi_fe = phi_dof_handler.get_fe();
    const unsigned int dofs_per_cell = phi_fe.n_dofs_per_cell();

    const unsigned int quad_degree = phi_fe.degree + 1;  // Sufficient for Laplacian
    dealii::QGauss<dim> quadrature(quad_degree);
    const unsigned int n_q_points = quadrature.size();

    dealii::FEValues<dim> phi_fe_values(phi_fe, quadrature,
        dealii::update_gradients | dealii::update_JxW_values);

    dealii::FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto& cell : phi_dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        phi_fe_values.reinit(cell);
        local_matrix = 0;

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = phi_fe_values.JxW(q);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const auto& grad_chi_i = phi_fe_values.shape_grad(i, q);

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    const auto& grad_chi_j = phi_fe_values.shape_grad(j, q);
                    // LHS: (∇φ, ∇χ)
                    local_matrix(i, j) += (grad_chi_i * grad_chi_j) * JxW;
                }
            }
        }

        cell->get_dof_indices(local_dof_indices);
        phi_constraints.distribute_local_to_global(
            local_matrix, local_dof_indices, phi_matrix);
    }

    phi_matrix.compress(dealii::VectorOperation::add);
}

// ============================================================================
// assemble_poisson_rhs - RHS only (EVERY timestep)
// ============================================================================
template <int dim>
void assemble_poisson_rhs(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::DoFHandler<dim>& M_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& mx_solution,
    const dealii::TrilinosWrappers::MPI::Vector& my_solution,
    const Parameters& params,
    double current_time,
    const dealii::AffineConstraints<double>& phi_constraints,
    dealii::TrilinosWrappers::MPI::Vector& phi_rhs)
{
    const bool mms_mode = params.enable_mms;
    const double L_y = params.domain.y_max - params.domain.y_min;

    const bool has_M = (mx_solution.size() > 0);
    const bool has_applied_field = (!mms_mode && params.enable_magnetic &&
                                    !params.dipoles.positions.empty());

    phi_rhs = 0;

    const auto& phi_fe = phi_dof_handler.get_fe();
    const unsigned int dofs_per_cell = phi_fe.n_dofs_per_cell();

    const unsigned int quad_degree = phi_fe.degree + 2;
    dealii::QGauss<dim> quadrature(quad_degree);
    const unsigned int n_q_points = quadrature.size();

    dealii::FEValues<dim> phi_fe_values(phi_fe, quadrature,
        dealii::update_values |
        dealii::update_gradients |
        dealii::update_quadrature_points |
        dealii::update_JxW_values);

    std::unique_ptr<dealii::FEValues<dim>> M_fe_values_ptr;
    if (has_M)
    {
        M_fe_values_ptr = std::make_unique<dealii::FEValues<dim>>(
            M_dof_handler.get_fe(), quadrature, dealii::update_values);
    }

    dealii::Vector<double> local_rhs(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<double> mx_values(n_q_points);
    std::vector<double> my_values(n_q_points);

    auto phi_cell = phi_dof_handler.begin_active();
    auto M_cell = has_M ? M_dof_handler.begin_active() : decltype(M_dof_handler.begin_active())();

    for (; phi_cell != phi_dof_handler.end(); ++phi_cell)
    {
        if (!phi_cell->is_locally_owned())
        {
            if (has_M)
                ++M_cell;
            continue;
        }

        phi_fe_values.reinit(phi_cell);
        local_rhs = 0;

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

            // Magnetization M
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

            // Applied field h_a from dipoles
            dealii::Tensor<1, dim> h_a;
            h_a[0] = 0.0;
            h_a[1] = 0.0;

            if (has_applied_field)
            {
                h_a = compute_applied_field<dim>(x_q, params, current_time);
            }

            // RHS source: (h_a - M, ∇χ)
            // RHS source: (h_a - M, ∇χ)
// ALWAYS use the same operator for MMS and production
dealii::Tensor<1, dim> source;
source[0] = h_a[0] - M[0];
source[1] = h_a[1] - M[1];

            // MMS source term
            double f_mms = 0.0;
            if (mms_mode)
            {
                if (has_M)
                {
                    f_mms = compute_poisson_mms_source_coupled<dim>(x_q, current_time, L_y);
                }
                else
                {
                    f_mms = compute_poisson_mms_source_standalone<dim>(x_q, current_time, L_y);
                }
            }

            // Assemble local RHS
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const double chi_i = phi_fe_values.shape_value(i, q);
                const auto& grad_chi_i = phi_fe_values.shape_grad(i, q);

                local_rhs(i) += (source * grad_chi_i) * JxW;

                if (mms_mode)
                {
                    local_rhs(i) += f_mms * chi_i * JxW;
                }
            }
        }

        phi_cell->get_dof_indices(local_dof_indices);
        phi_constraints.distribute_local_to_global(local_rhs, local_dof_indices, phi_rhs);

        if (has_M)
            ++M_cell;
    }

    phi_rhs.compress(dealii::VectorOperation::add);
}

// ============================================================================
// assemble_poisson_system - LEGACY (both matrix and RHS)
// ============================================================================
template <int dim>
void assemble_poisson_system(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::DoFHandler<dim>& M_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& mx_solution,
    const dealii::TrilinosWrappers::MPI::Vector& my_solution,
    const Parameters& params,
    double current_time,
    const dealii::AffineConstraints<double>& phi_constraints,
    dealii::TrilinosWrappers::SparseMatrix& phi_matrix,
    dealii::TrilinosWrappers::MPI::Vector& phi_rhs)
{
    // Just call both functions
    assemble_poisson_matrix<dim>(phi_dof_handler, phi_constraints, phi_matrix);
    assemble_poisson_rhs<dim>(phi_dof_handler, M_dof_handler,
                              mx_solution, my_solution,
                              params, current_time,
                              phi_constraints, phi_rhs);
}

// ============================================================================
// Explicit instantiations
// ============================================================================
template void assemble_poisson_matrix<2>(
    const dealii::DoFHandler<2>&,
    const dealii::AffineConstraints<double>&,
    dealii::TrilinosWrappers::SparseMatrix&);

template void assemble_poisson_rhs<2>(
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const Parameters&,
    double,
    const dealii::AffineConstraints<double>&,
    dealii::TrilinosWrappers::MPI::Vector&);

template void assemble_poisson_system<2>(
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const Parameters&,
    double,
    const dealii::AffineConstraints<double>&,
    dealii::TrilinosWrappers::SparseMatrix&,
    dealii::TrilinosWrappers::MPI::Vector&);