// ============================================================================
// poisson/poisson_assemble.cc - Matrix and RHS Assembly
//
// PAPER EQUATION 42d (Nochetto, Salgado & Tomas, arXiv:1511.04381):
//
//   (∇φ^k, ∇X) = (h_a^k − M^k, ∇X)    ∀X ∈ X_h
//
// Matrix (LHS): (∇φ, ∇X) — constant-coefficient Laplacian, assembled ONCE
// RHS:          (h_a − M, ∇X) — reassembled each Picard iteration / timestep
//
// When enable_mms = true, adds volumetric MMS source: + (f_mms, X)
//
// Assembly uses FEValues on CG Q_ℓ elements with Gauss quadrature.
// M is evaluated via FEValues on the DG DoFHandler (cross-mesh evaluation).
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015)
// ============================================================================

#include "poisson/poisson.h"
#include "physics/applied_field.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_dgq.h>

template <int dim>
void PoissonSubsystem<dim>::assemble_matrix()
{
    const dealii::QGauss<dim> quadrature(fe_.degree + 1);
    dealii::FEValues<dim> fe_values(fe_, quadrature,
                                    dealii::update_gradients |
                                    dealii::update_JxW_values);

    const unsigned int dofs_per_cell = fe_.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature.size();

    dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    system_matrix_ = 0.0;

    for (const auto& cell : dof_handler_.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);
        cell_matrix = 0.0;

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = fe_values.JxW(q);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    // (∇φ, ∇X) = Σ_q (∇φ_j · ∇X_i) JxW
                    cell_matrix(i, j) +=
                        fe_values.shape_grad(j, q) *
                        fe_values.shape_grad(i, q) * JxW;
                }
        }

        cell->get_dof_indices(local_dof_indices);
        constraints_.distribute_local_to_global(
            cell_matrix, local_dof_indices, system_matrix_);
    }

    system_matrix_.compress(dealii::VectorOperation::add);
}

template <int dim>
void PoissonSubsystem<dim>::initialize_preconditioner()
{
    dealii::TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
    amg_data.elliptic = true;
    amg_data.higher_order_elements = (fe_.degree > 1);
    amg_data.smoother_sweeps = 2;
    amg_data.aggregation_threshold = 0.02;

    amg_preconditioner_.initialize(system_matrix_, amg_data);
    amg_initialized_ = true;
}

template <int dim>
void PoissonSubsystem<dim>::assemble_rhs(
    const dealii::TrilinosWrappers::MPI::Vector& M_x_relevant,
    const dealii::TrilinosWrappers::MPI::Vector& M_y_relevant,
    const dealii::DoFHandler<dim>& M_dof_handler,
    double current_time)
{
    dealii::Timer timer;
    timer.start();

    const dealii::QGauss<dim> quadrature(fe_.degree + 1);
    const unsigned int n_q_points = quadrature.size();
    const unsigned int dofs_per_cell = fe_.n_dofs_per_cell();

    // FEValues for potential (CG) test functions
    dealii::FEValues<dim> fe_values(fe_, quadrature,
                                    dealii::update_values |
                                    dealii::update_gradients |
                                    dealii::update_quadrature_points |
                                    dealii::update_JxW_values);

    // Check if M is available (size > 0 means coupled mode)
    const bool has_M = (M_x_relevant.size() > 0);

    // FEValues for magnetization (DG) — only if M is available
    std::unique_ptr<dealii::FEValues<dim>> fe_values_M;
    if (has_M)
    {
        fe_values_M = std::make_unique<dealii::FEValues<dim>>(
            M_dof_handler.get_fe(), quadrature,
            dealii::update_values);
    }

    dealii::Vector<double> cell_rhs(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    // Buffers for M values at quadrature points
    std::vector<double> M_x_values(n_q_points);
    std::vector<double> M_y_values(n_q_points);

    system_rhs_ = 0.0;

    for (const auto& cell : dof_handler_.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);
        cell_rhs = 0.0;

        // Get M values on this cell (if available)
        if (has_M)
        {
            // Find corresponding DG cell
            const auto M_cell = cell->as_dof_handler_iterator(M_dof_handler);
            fe_values_M->reinit(M_cell);
            fe_values_M->get_function_values(M_x_relevant, M_x_values);
            fe_values_M->get_function_values(M_y_relevant, M_y_values);
        }

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const auto& x_q = fe_values.quadrature_point(q);
            const double JxW = fe_values.JxW(q);

            // Compute h_a at this quadrature point
            const dealii::Tensor<1, dim> h_a =
                compute_applied_field<dim>(x_q, params_, current_time);

            // RHS integrand: (h_a − M, ∇X)
            dealii::Tensor<1, dim> h_a_minus_M;
            h_a_minus_M[0] = h_a[0] - (has_M ? M_x_values[q] : 0.0);
            h_a_minus_M[1] = h_a[1] - (has_M ? M_y_values[q] : 0.0);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                // (h_a − M, ∇X_i)
                cell_rhs(i) += (h_a_minus_M *
                                fe_values.shape_grad(i, q)) * JxW;

                // MMS source: (f_mms, X_i)
                if (params_.enable_mms && mms_source_)
                {
                    cell_rhs(i) += mms_source_(x_q, current_time) *
                                   fe_values.shape_value(i, q) * JxW;
                }
            }
        }

        cell->get_dof_indices(local_dof_indices);
        constraints_.distribute_local_to_global(
            cell_rhs, local_dof_indices, system_rhs_);
    }

    system_rhs_.compress(dealii::VectorOperation::add);

    timer.stop();
    last_assemble_time_ = timer.wall_time();
}

// Explicit instantiations
template void PoissonSubsystem<2>::assemble_matrix();
template void PoissonSubsystem<2>::initialize_preconditioner();
template void PoissonSubsystem<2>::assemble_rhs(
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::DoFHandler<2>&, double);

template void PoissonSubsystem<3>::assemble_matrix();
template void PoissonSubsystem<3>::initialize_preconditioner();
template void PoissonSubsystem<3>::assemble_rhs(
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::DoFHandler<3>&, double);
