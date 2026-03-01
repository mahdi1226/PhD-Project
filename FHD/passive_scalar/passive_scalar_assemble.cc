// ============================================================================
// passive_scalar/passive_scalar_assemble.cc - Cell Assembly
//
// PAPER EQUATION 104 (Nochetto, Salgado & Tomas, arXiv:1511.04381):
//
//   c_t + u · ∇c − α Δc = 0
//
// Backward Euler discretization:
//   LHS: (1/τ)(c^k, v) + α(∇c^k, ∇v) + b_h(u^k; c^k, v)
//   RHS: (1/τ)(c^{k-1}, v)
//
// b_h = skew-symmetric convection for energy stability
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015)
// ============================================================================

#include "passive_scalar/passive_scalar.h"
#include "physics/skew_forms.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/fe/fe_values.h>

template <int dim>
void PassiveScalarSubsystem<dim>::assemble(
    const dealii::TrilinosWrappers::MPI::Vector& c_old_relevant,
    double dt,
    const dealii::TrilinosWrappers::MPI::Vector& ux_relevant,
    const dealii::TrilinosWrappers::MPI::Vector& uy_relevant,
    const dealii::DoFHandler<dim>& vel_dof_handler)
{
    dealii::Timer timer;
    timer.start();

    const double alpha = params_.passive_scalar.alpha;
    const double mass_coeff = 1.0 / dt;

    const bool has_old = (c_old_relevant.size() > 0);
    const bool has_vel = (ux_relevant.size() > 0);

    const unsigned int degree = fe_.degree;
    const dealii::QGauss<dim> quadrature(degree + 1);
    const unsigned int n_q = quadrature.size();
    const unsigned int dpc = fe_.n_dofs_per_cell();

    dealii::FEValues<dim> fe_values(fe_, quadrature,
        dealii::update_values | dealii::update_gradients |
        dealii::update_JxW_values);

    // FEValues for velocity (for convection)
    std::unique_ptr<dealii::FEValues<dim>> fe_values_vel;
    if (has_vel)
    {
        fe_values_vel = std::make_unique<dealii::FEValues<dim>>(
            vel_dof_handler.get_fe(), quadrature,
            dealii::update_values | dealii::update_gradients);
    }

    dealii::FullMatrix<double> cell_matrix(dpc, dpc);
    dealii::Vector<double> cell_rhs(dpc);

    std::vector<dealii::types::global_dof_index> local_dofs(dpc);

    // Buffers
    std::vector<double> c_old_vals(n_q);
    std::vector<double> ux_vals(n_q), uy_vals(n_q);
    std::vector<dealii::Tensor<1, dim>> grad_ux_vals(n_q), grad_uy_vals(n_q);

    // Zero global
    system_matrix_ = 0.0;
    system_rhs_ = 0.0;

    // ========================================================================
    // Cell loop
    // ========================================================================
    for (const auto& cell : dof_handler_.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);
        cell->get_dof_indices(local_dofs);

        // Old concentration
        if (has_old)
            fe_values.get_function_values(c_old_relevant, c_old_vals);

        // Velocity (for convection)
        if (has_vel)
        {
            const auto vel_cell = cell->as_dof_handler_iterator(vel_dof_handler);
            fe_values_vel->reinit(vel_cell);
            fe_values_vel->get_function_values(ux_relevant, ux_vals);
            fe_values_vel->get_function_values(uy_relevant, uy_vals);
            fe_values_vel->get_function_gradients(ux_relevant, grad_ux_vals);
            fe_values_vel->get_function_gradients(uy_relevant, grad_uy_vals);
        }

        cell_matrix = 0.0;
        cell_rhs = 0.0;

        for (unsigned int q = 0; q < n_q; ++q)
        {
            const double JxW = fe_values.JxW(q);

            // Velocity at quadrature point
            dealii::Tensor<1, dim> U_q;
            double div_U_q = 0.0;
            if (has_vel)
            {
                U_q[0] = ux_vals[q];
                U_q[1] = uy_vals[q];
                div_U_q = grad_ux_vals[q][0] + grad_uy_vals[q][1];
            }

            for (unsigned int i = 0; i < dpc; ++i)
            {
                const double phi_i = fe_values.shape_value(i, q);
                const auto& grad_phi_i = fe_values.shape_grad(i, q);

                // RHS: (1/τ) c_old * φ_i
                if (has_old)
                    cell_rhs(i) += mass_coeff * c_old_vals[q] * phi_i * JxW;

                for (unsigned int j = 0; j < dpc; ++j)
                {
                    const double phi_j = fe_values.shape_value(j, q);
                    const auto& grad_phi_j = fe_values.shape_grad(j, q);

                    // Mass: (1/τ)(c, v)
                    double mass = mass_coeff * phi_j * phi_i;

                    // Diffusion: α(∇c, ∇v)
                    double diffusion = alpha * (grad_phi_j * grad_phi_i);

                    // Convection: b_h(u; c, v) = (u·∇c)v + 0.5(div u)(c·v)
                    double conv = 0.0;
                    if (has_vel)
                    {
                        conv = skew_angular_convection_scalar<dim>(
                            U_q, div_U_q, phi_j, grad_phi_j, phi_i);
                    }

                    cell_matrix(i, j) += (mass + diffusion + conv) * JxW;
                }
            }
        }

        constraints_.distribute_local_to_global(
            cell_matrix, cell_rhs, local_dofs,
            system_matrix_, system_rhs_);
    }

    system_matrix_.compress(dealii::VectorOperation::add);
    system_rhs_.compress(dealii::VectorOperation::add);

    timer.stop();
    last_assemble_time_ = timer.wall_time();
}

// Explicit instantiations
template void PassiveScalarSubsystem<2>::assemble(
    const dealii::TrilinosWrappers::MPI::Vector&,
    double,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::DoFHandler<2>&);
template void PassiveScalarSubsystem<3>::assemble(
    const dealii::TrilinosWrappers::MPI::Vector&,
    double,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::DoFHandler<3>&);
