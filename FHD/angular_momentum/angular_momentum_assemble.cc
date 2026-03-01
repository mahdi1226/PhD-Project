// ============================================================================
// angular_momentum/angular_momentum_assemble.cc - Cell Assembly
//
// PAPER EQUATION 42f (Nochetto, Salgado & Tomas, arXiv:1511.04381):
//
// In 2D, w is scalar (z-component of spin):
//
//   j(w^k/τ, z) + j·b_h(u^{k-1}; w^k, z) + c₁(∇w^k, ∇z) + 4ν_r(w^k, z)
//     = j(w^{k-1}/τ, z) + 2ν_r(curl u^k, z) + μ₀(m^k × h^k, z) + (f_mms, z)
//
// LHS per cell:
//   A_cell = (j/τ + 4ν_r) M + c₁ K + j·convection
// where M = mass, K = stiffness (Laplacian)
//
// RHS per cell:
//   f_cell = (j/τ) M w_old + 2ν_r(curl u, z) + μ₀(m×h, z) + (f_mms, z)
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015)
// ============================================================================

#include "angular_momentum/angular_momentum.h"
#include "physics/skew_forms.h"
#include "physics/kelvin_force.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/fe/fe_values.h>

template <int dim>
void AngularMomentumSubsystem<dim>::assemble(
    const dealii::TrilinosWrappers::MPI::Vector& w_old_relevant,
    double dt,
    double current_time,
    const dealii::TrilinosWrappers::MPI::Vector& ux_relevant,
    const dealii::TrilinosWrappers::MPI::Vector& uy_relevant,
    const dealii::DoFHandler<dim>& vel_dof_handler,
    bool include_convection,
    const dealii::TrilinosWrappers::MPI::Vector& Mx_relevant,
    const dealii::TrilinosWrappers::MPI::Vector& My_relevant,
    const dealii::DoFHandler<dim>* M_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& phi_relevant,
    const dealii::DoFHandler<dim>* phi_dof_handler)
{
    dealii::Timer timer;
    timer.start();

    const double j = params_.physics.j_micro;
    const double c1 = params_.physics.c_1;
    const double nu_r = params_.physics.nu_r;
    const double mu_0 = params_.physics.mu_0;

    const bool has_time = (dt > 0.0 && dt < 1e30);
    const double mass_coeff = has_time ? j / dt : 0.0;
    const double reaction_coeff = 4.0 * nu_r;

    const bool has_old = (w_old_relevant.size() > 0);
    const bool has_vel = (ux_relevant.size() > 0);
    const bool has_torque = (Mx_relevant.size() > 0
                             && M_dof_handler != nullptr
                             && phi_relevant.size() > 0
                             && phi_dof_handler != nullptr);

    const unsigned int degree = fe_.degree;
    const dealii::QGauss<dim> quadrature(degree + 1);
    const unsigned int n_q = quadrature.size();
    const unsigned int dpc = fe_.n_dofs_per_cell();

    dealii::FEValues<dim> fe_values(fe_, quadrature,
        dealii::update_values | dealii::update_gradients |
        dealii::update_quadrature_points | dealii::update_JxW_values);

    // FEValues for velocity (only needed for convection or curl)
    std::unique_ptr<dealii::FEValues<dim>> fe_values_vel;
    if (has_vel)
    {
        fe_values_vel = std::make_unique<dealii::FEValues<dim>>(
            vel_dof_handler.get_fe(), quadrature,
            dealii::update_values | dealii::update_gradients);
    }

    // FEValues for magnetization M (DG, values only for torque)
    std::unique_ptr<dealii::FEValues<dim>> fe_values_M;
    // FEValues for potential φ (CG, gradients for H = ∇φ)
    std::unique_ptr<dealii::FEValues<dim>> fe_values_phi;
    if (has_torque)
    {
        fe_values_M = std::make_unique<dealii::FEValues<dim>>(
            M_dof_handler->get_fe(), quadrature,
            dealii::update_values);
        fe_values_phi = std::make_unique<dealii::FEValues<dim>>(
            phi_dof_handler->get_fe(), quadrature,
            dealii::update_gradients);
    }

    dealii::FullMatrix<double> cell_matrix(dpc, dpc);
    dealii::Vector<double> cell_rhs(dpc);

    std::vector<dealii::types::global_dof_index> local_dofs(dpc);

    // Buffers
    std::vector<double> w_old_vals(n_q);
    std::vector<double> ux_vals(n_q), uy_vals(n_q);
    std::vector<dealii::Tensor<1, dim>> grad_ux_vals(n_q), grad_uy_vals(n_q);

    // Magnetic torque buffers
    std::vector<double> Mx_vals(n_q), My_vals(n_q);
    std::vector<dealii::Tensor<1, dim>> phi_grad_vals(n_q);

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

        // Old w
        if (has_old)
            fe_values.get_function_values(w_old_relevant, w_old_vals);

        // Velocity (for convection and/or curl source)
        if (has_vel)
        {
            const auto vel_cell = cell->as_dof_handler_iterator(vel_dof_handler);
            fe_values_vel->reinit(vel_cell);
            fe_values_vel->get_function_values(ux_relevant, ux_vals);
            fe_values_vel->get_function_values(uy_relevant, uy_vals);
            fe_values_vel->get_function_gradients(ux_relevant, grad_ux_vals);
            fe_values_vel->get_function_gradients(uy_relevant, grad_uy_vals);
        }

        // Magnetization M and potential φ for magnetic torque
        if (has_torque)
        {
            const auto M_cell = cell->as_dof_handler_iterator(*M_dof_handler);
            fe_values_M->reinit(M_cell);
            fe_values_M->get_function_values(Mx_relevant, Mx_vals);
            fe_values_M->get_function_values(My_relevant, My_vals);

            const auto phi_cell = cell->as_dof_handler_iterator(*phi_dof_handler);
            fe_values_phi->reinit(phi_cell);
            fe_values_phi->get_function_gradients(phi_relevant, phi_grad_vals);
        }

        cell_matrix = 0.0;
        cell_rhs = 0.0;

        for (unsigned int q = 0; q < n_q; ++q)
        {
            const auto& x_q = fe_values.quadrature_point(q);
            const double JxW = fe_values.JxW(q);

            // Old velocity at q
            dealii::Tensor<1, dim> U_old;
            double div_U_old = 0.0;
            if (has_vel && include_convection)
            {
                U_old[0] = ux_vals[q];
                U_old[1] = uy_vals[q];
                div_U_old = grad_ux_vals[q][0] + grad_uy_vals[q][1];
            }

            // MMS source
            double f_mms = 0.0;
            if (params_.enable_mms && mms_source_)
            {
                double w_old_disc = 0.0;
                if (has_old)
                    w_old_disc = w_old_vals[q];
                f_mms = mms_source_(x_q, current_time,
                                     current_time - dt,
                                     j, c1, nu_r, w_old_disc,
                                     U_old, div_U_old,
                                     include_convection);
            }

            for (unsigned int i = 0; i < dpc; ++i)
            {
                const double phi_i = fe_values.shape_value(i, q);
                const auto& grad_phi_i = fe_values.shape_grad(i, q);

                // RHS: old mass + MMS source
                if (has_time && has_old)
                    cell_rhs(i) += mass_coeff * w_old_vals[q] * phi_i * JxW;

                cell_rhs(i) += f_mms * phi_i * JxW;

                // Curl coupling: 2ν_r(∇×u^k, z)
                // In 2D: curl(u) = ∂uy/∂x - ∂ux/∂y (scalar)
                if (has_vel)
                {
                    const double curl_u = grad_uy_vals[q][0] - grad_ux_vals[q][1];
                    cell_rhs(i) += 2.0 * nu_r * curl_u * phi_i * JxW;
                }

                // Magnetic torque: μ₀(m × h, z)
                // In 2D: m × h = mx*hy - my*hx (scalar)
                // H = ∇φ (total field; h_a encoded via Poisson RHS)
                if (has_torque)
                {
                    const dealii::Tensor<1, dim>& H_q = phi_grad_vals[q];

                    const double torque = KelvinForce::magnetic_torque_2d(
                        Mx_vals[q], My_vals[q],
                        H_q[0], H_q[1]);
                    cell_rhs(i) += mu_0 * torque * phi_i * JxW;
                }

                for (unsigned int j_dof = 0; j_dof < dpc; ++j_dof)
                {
                    const double phi_j = fe_values.shape_value(j_dof, q);
                    const auto& grad_phi_j = fe_values.shape_grad(j_dof, q);

                    // Mass + reaction: (j/τ + 4ν_r)(w, z)
                    double mass_react = (mass_coeff + reaction_coeff)
                                        * phi_j * phi_i;

                    // Diffusion: c₁(∇w, ∇z)
                    double diffusion = c1 * (grad_phi_j * grad_phi_i);

                    // Convection: j·b_h(u; w, z)
                    double conv = 0.0;
                    if (has_vel && include_convection)
                    {
                        conv = j * skew_angular_convection_scalar<dim>(
                            U_old, div_U_old, phi_j, grad_phi_j, phi_i);
                    }

                    cell_matrix(i, j_dof) += (mass_react + diffusion + conv) * JxW;
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
template void AngularMomentumSubsystem<2>::assemble(
    const dealii::TrilinosWrappers::MPI::Vector&,
    double, double,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::DoFHandler<2>&, bool,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::DoFHandler<2>*,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::DoFHandler<2>*);
template void AngularMomentumSubsystem<3>::assemble(
    const dealii::TrilinosWrappers::MPI::Vector&,
    double, double,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::DoFHandler<3>&, bool,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::DoFHandler<3>*,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::DoFHandler<3>*);
