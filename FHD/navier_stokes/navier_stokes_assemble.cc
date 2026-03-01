// ============================================================================
// navier_stokes/navier_stokes_assemble.cc - Monolithic Saddle-Point Assembly
//
// PAPER EQUATION 42e (Nochetto, Salgado & Tomas, arXiv:1511.04381):
//
//   (u^k/τ, v) + (ν+ν_r)(D(u^k), D(v)) + b_h(u^{k-1}; u^k, v)
//     - (p^k, ∇·v) + (∇·u^k, q)
//     = (u^{k-1}/τ, v) + f_mms
//
// LHS blocks (per cell):
//   A_ux_ux:  mass(1/τ) + viscous(ν_eff) + convection(u_old)
//   A_ux_uy:  viscous cross-term (from D(u):D(v))
//   A_uy_ux:  viscous cross-term
//   A_uy_uy:  mass(1/τ) + viscous(ν_eff) + convection(u_old)
//   B_ux:     -(p, ∂vx/∂x) and (∂ux/∂x, q)
//   B_uy:     -(p, ∂vy/∂y) and (∂uy/∂y, q)
//
// Viscous term uses strain-rate: (ν+ν_r)(D(u), D(v)) = (ν_eff/4)(T(u):T(v))
// where T(u) = ∇u + ∇uᵀ = 2D(u).
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015)
// ============================================================================

#include "navier_stokes/navier_stokes.h"
#include "physics/skew_forms.h"
#include "physics/kelvin_force.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>

namespace
{
    // Strain-rate tensor T for test function in ux direction: v = (vx, 0)
    // T(v) = ∇v + ∇vᵀ = [[2∂vx/∂x, ∂vx/∂y], [∂vx/∂y, 0]]
    template <int dim>
    dealii::SymmetricTensor<2, dim>
    T_test_ux(const dealii::Tensor<1, dim>& grad_vx)
    {
        dealii::SymmetricTensor<2, dim> T;
        T[0][0] = 2.0 * grad_vx[0];
        T[0][1] = grad_vx[1];
        // T[1][1] = 0 (default)
        return T;
    }

    // Strain-rate tensor T for test function in uy direction: v = (0, vy)
    // T(v) = [[0, ∂vy/∂x], [∂vy/∂x, 2∂vy/∂y]]
    template <int dim>
    dealii::SymmetricTensor<2, dim>
    T_test_uy(const dealii::Tensor<1, dim>& grad_vy)
    {
        dealii::SymmetricTensor<2, dim> T;
        T[0][1] = grad_vy[0];
        T[1][1] = 2.0 * grad_vy[1];
        return T;
    }
}

template <int dim>
void NavierStokesSubsystem<dim>::assemble(
    const dealii::TrilinosWrappers::MPI::Vector& ux_old_relevant,
    const dealii::TrilinosWrappers::MPI::Vector& uy_old_relevant,
    double dt,
    double current_time,
    bool include_convection,
    const dealii::TrilinosWrappers::MPI::Vector& w_relevant,
    const dealii::DoFHandler<dim>& w_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& Mx_relevant,
    const dealii::TrilinosWrappers::MPI::Vector& My_relevant,
    const dealii::DoFHandler<dim>* M_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& phi_relevant,
    const dealii::DoFHandler<dim>* phi_dof_handler)
{
    dealii::Timer timer;
    timer.start();

    const double nu_eff = params_.physics.nu + params_.physics.nu_r;
    const double nu_r = params_.physics.nu_r;
    const double mu_0 = params_.physics.mu_0;
    const double mass_coeff = 1.0 / dt;

    const bool has_time = (dt > 0.0 && dt < 1e30);
    const bool has_old = (ux_old_relevant.size() > 0);
    const bool has_w = (w_relevant.size() > 0);
    const bool has_kelvin = (Mx_relevant.size() > 0
                             && M_dof_handler != nullptr
                             && phi_relevant.size() > 0
                             && phi_dof_handler != nullptr);

    const unsigned int vel_degree = fe_velocity_.degree;
    const dealii::QGauss<dim> quadrature(vel_degree + 1);
    const unsigned int n_q = quadrature.size();

    const unsigned int ux_dpc = fe_velocity_.n_dofs_per_cell();
    const unsigned int uy_dpc = fe_velocity_.n_dofs_per_cell();
    const unsigned int p_dpc  = fe_pressure_.n_dofs_per_cell();

    // FEValues for velocity (ux, uy share the same FE)
    dealii::FEValues<dim> fe_values_vel(fe_velocity_, quadrature,
        dealii::update_values | dealii::update_gradients |
        dealii::update_quadrature_points | dealii::update_JxW_values);

    dealii::FEValues<dim> fe_values_p(fe_pressure_, quadrature,
        dealii::update_values);

    // FEValues for angular velocity w (CG, for micropolar coupling)
    std::unique_ptr<dealii::FEValues<dim>> fe_values_w;
    if (has_w)
    {
        fe_values_w = std::make_unique<dealii::FEValues<dim>>(
            w_dof_handler.get_fe(), quadrature,
            dealii::update_values);
    }

    // FEValues for magnetization M (DG, for Kelvin force: values + gradients)
    std::unique_ptr<dealii::FEValues<dim>> fe_values_M;
    // FEValues for potential φ (CG, for Kelvin force: gradients + hessians)
    std::unique_ptr<dealii::FEValues<dim>> fe_values_phi;
    if (has_kelvin)
    {
        fe_values_M = std::make_unique<dealii::FEValues<dim>>(
            M_dof_handler->get_fe(), quadrature,
            dealii::update_values | dealii::update_gradients);
        fe_values_phi = std::make_unique<dealii::FEValues<dim>>(
            phi_dof_handler->get_fe(), quadrature,
            dealii::update_gradients | dealii::update_hessians);
    }

    // Local matrices (block structure)
    dealii::FullMatrix<double> local_ux_ux(ux_dpc, ux_dpc);
    dealii::FullMatrix<double> local_ux_uy(ux_dpc, uy_dpc);
    dealii::FullMatrix<double> local_uy_ux(uy_dpc, ux_dpc);
    dealii::FullMatrix<double> local_uy_uy(uy_dpc, uy_dpc);
    dealii::FullMatrix<double> local_ux_p(ux_dpc, p_dpc);
    dealii::FullMatrix<double> local_uy_p(uy_dpc, p_dpc);
    dealii::FullMatrix<double> local_p_ux(p_dpc, ux_dpc);
    dealii::FullMatrix<double> local_p_uy(p_dpc, uy_dpc);
    dealii::FullMatrix<double> local_p_p(p_dpc, p_dpc);  // zero — needed for constraint diagonal

    dealii::Vector<double> local_rhs_ux(ux_dpc);
    dealii::Vector<double> local_rhs_uy(uy_dpc);
    dealii::Vector<double> local_rhs_p(p_dpc);

    std::vector<dealii::types::global_dof_index> ux_dofs(ux_dpc);
    std::vector<dealii::types::global_dof_index> uy_dofs(uy_dpc);
    std::vector<dealii::types::global_dof_index> p_dofs(p_dpc);

    // Old velocity buffers
    std::vector<double> ux_old_vals(n_q), uy_old_vals(n_q);
    std::vector<dealii::Tensor<1, dim>> grad_ux_old(n_q), grad_uy_old(n_q);

    // Angular velocity buffer (for micropolar coupling)
    std::vector<double> w_vals(n_q);

    // Kelvin force buffers (magnetization + potential)
    std::vector<double> Mx_vals(n_q), My_vals(n_q);
    std::vector<dealii::Tensor<1, dim>> grad_Mx_vals(n_q), grad_My_vals(n_q);
    std::vector<dealii::Tensor<1, dim>> phi_grad_vals(n_q);
    std::vector<dealii::Tensor<2, dim>> phi_hess_vals(n_q);

    // Zero global system
    ns_matrix_ = 0.0;
    ns_rhs_ = 0.0;

    // ========================================================================
    // Cell loop
    // ========================================================================
    for (const auto& ux_cell : ux_dof_handler_.active_cell_iterators())
    {
        if (!ux_cell->is_locally_owned())
            continue;

        fe_values_vel.reinit(ux_cell);
        ux_cell->get_dof_indices(ux_dofs);

        // Synchronized cell iteration for uy and p
        const auto uy_cell = ux_cell->as_dof_handler_iterator(uy_dof_handler_);
        uy_cell->get_dof_indices(uy_dofs);

        const auto p_cell = ux_cell->as_dof_handler_iterator(p_dof_handler_);
        fe_values_p.reinit(p_cell);
        p_cell->get_dof_indices(p_dofs);

        // Get old velocity values (needed for mass term) and gradients (for convection)
        if (has_old)
        {
            fe_values_vel.get_function_values(ux_old_relevant, ux_old_vals);
            fe_values_vel.get_function_values(uy_old_relevant, uy_old_vals);
            if (include_convection)
            {
                fe_values_vel.get_function_gradients(ux_old_relevant, grad_ux_old);
                fe_values_vel.get_function_gradients(uy_old_relevant, grad_uy_old);
            }
        }

        // Get angular velocity w for micropolar coupling
        if (has_w)
        {
            const auto w_cell = ux_cell->as_dof_handler_iterator(w_dof_handler);
            fe_values_w->reinit(w_cell);
            fe_values_w->get_function_values(w_relevant, w_vals);
        }

        // Get magnetization M and potential φ for Kelvin force
        if (has_kelvin)
        {
            const auto M_cell = ux_cell->as_dof_handler_iterator(*M_dof_handler);
            fe_values_M->reinit(M_cell);
            fe_values_M->get_function_values(Mx_relevant, Mx_vals);
            fe_values_M->get_function_values(My_relevant, My_vals);
            fe_values_M->get_function_gradients(Mx_relevant, grad_Mx_vals);
            fe_values_M->get_function_gradients(My_relevant, grad_My_vals);

            const auto phi_cell = ux_cell->as_dof_handler_iterator(*phi_dof_handler);
            fe_values_phi->reinit(phi_cell);
            fe_values_phi->get_function_gradients(phi_relevant, phi_grad_vals);
            fe_values_phi->get_function_hessians(phi_relevant, phi_hess_vals);
        }

        // Zero local matrices
        local_ux_ux = 0.0;  local_ux_uy = 0.0;
        local_uy_ux = 0.0;  local_uy_uy = 0.0;
        local_ux_p = 0.0;   local_uy_p = 0.0;
        local_p_ux = 0.0;   local_p_uy = 0.0;
        local_p_p = 0.0;
        local_rhs_ux = 0.0; local_rhs_uy = 0.0;
        local_rhs_p = 0.0;

        for (unsigned int q = 0; q < n_q; ++q)
        {
            const auto& x_q = fe_values_vel.quadrature_point(q);
            const double JxW = fe_values_vel.JxW(q);

            // Old velocity at q
            dealii::Tensor<1, dim> U_old;
            double div_U_old = 0.0;
            if (has_old && include_convection)
            {
                U_old[0] = ux_old_vals[q];
                U_old[1] = uy_old_vals[q];
                div_U_old = grad_ux_old[q][0] + grad_uy_old[q][1];
            }

            // MMS source
            dealii::Tensor<1, dim> f_mms;
            if (params_.enable_mms && mms_source_)
            {
                dealii::Tensor<1, dim> U_old_disc;
                if (has_old)
                {
                    U_old_disc[0] = ux_old_vals[q];
                    U_old_disc[1] = uy_old_vals[q];
                }
                f_mms = mms_source_(x_q, current_time,
                                     current_time - dt, nu_eff,
                                     U_old_disc, div_U_old,
                                     include_convection);
            }

            for (unsigned int i = 0; i < ux_dpc; ++i)
            {
                const double phi_ux_i = fe_values_vel.shape_value(i, q);
                const auto& grad_phi_ux_i = fe_values_vel.shape_grad(i, q);
                const auto T_Vx_i = T_test_ux<dim>(grad_phi_ux_i);
                const auto T_Vy_i = T_test_uy<dim>(grad_phi_ux_i);

                // RHS: old mass + MMS source
                if (has_time && has_old)
                {
                    local_rhs_ux(i) += mass_coeff * ux_old_vals[q] * phi_ux_i * JxW;
                    local_rhs_uy(i) += mass_coeff * uy_old_vals[q] * phi_ux_i * JxW;
                }

                local_rhs_ux(i) += f_mms[0] * phi_ux_i * JxW;
                local_rhs_uy(i) += f_mms[1] * phi_ux_i * JxW;

                // Micropolar coupling: 2ν_r(w^{k-1}, ∇×v)
                // In 2D: v=(vx,0) → ∇×v = -∂vx/∂y
                //         v=(0,vy) → ∇×v = ∂vy/∂x
                if (has_w)
                {
                    local_rhs_ux(i) += 2.0 * nu_r * w_vals[q]
                                       * (-grad_phi_ux_i[1]) * JxW;
                    local_rhs_uy(i) += 2.0 * nu_r * w_vals[q]
                                       * grad_phi_ux_i[0] * JxW;
                }

                // Kelvin force (cell kernel): μ₀·B_h^m(v, h, m)
                // Cell integrand: (M·∇)H · V + ½(∇·M)(H·V)
                // H = ∇φ (total field; h_a encoded via Poisson RHS)
                if (has_kelvin)
                {
                    dealii::Tensor<1, dim> M_q;
                    M_q[0] = Mx_vals[q];
                    M_q[1] = My_vals[q];

                    const dealii::Tensor<1, dim>& H_q = phi_grad_vals[q];
                    const dealii::Tensor<2, dim>& grad_H = phi_hess_vals[q];

                    double kelvin_ux, kelvin_uy;
                    KelvinForce::cell_kernel_full<dim>(
                        M_q, grad_H,
                        grad_Mx_vals[q], grad_My_vals[q],
                        H_q,
                        phi_ux_i, phi_ux_i,
                        kelvin_ux, kelvin_uy);

                    local_rhs_ux(i) += mu_0 * kelvin_ux * JxW;
                    local_rhs_uy(i) += mu_0 * kelvin_uy * JxW;
                }

                for (unsigned int j = 0; j < ux_dpc; ++j)
                {
                    const double phi_ux_j = fe_values_vel.shape_value(j, q);
                    const auto& grad_phi_ux_j = fe_values_vel.shape_grad(j, q);
                    const auto T_Ux_j = T_test_ux<dim>(grad_phi_ux_j);
                    const auto T_Uy_j = T_test_uy<dim>(grad_phi_ux_j);

                    // Mass: (1/τ)(u, v)
                    double mass = 0.0;
                    if (has_time)
                        mass = mass_coeff * phi_ux_j * phi_ux_i;

                    // Viscous: (ν_eff)(D(u), D(v)) = (ν_eff/4)(T(u):T(v))
                    const double visc_ux_ux = (nu_eff / 4.0) * (T_Ux_j * T_Vx_i);
                    const double visc_uy_ux = (nu_eff / 4.0) * (T_Uy_j * T_Vx_i);
                    const double visc_ux_uy = (nu_eff / 4.0) * (T_Ux_j * T_Vy_i);
                    const double visc_uy_uy = (nu_eff / 4.0) * (T_Uy_j * T_Vy_i);

                    // Convection: b_h(u_old; u, v) — scalar Temam form
                    double conv_ux = 0.0, conv_uy = 0.0;
                    if (has_old && include_convection)
                    {
                        conv_ux = skew_magnetic_cell_value_scalar<dim>(
                            U_old, div_U_old, phi_ux_j, grad_phi_ux_j, phi_ux_i);
                        conv_uy = skew_magnetic_cell_value_scalar<dim>(
                            U_old, div_U_old, phi_ux_j, grad_phi_ux_j, phi_ux_i);
                    }

                    local_ux_ux(i, j) += (mass + visc_ux_ux + conv_ux) * JxW;
                    local_ux_uy(i, j) += visc_ux_uy * JxW;
                    local_uy_ux(i, j) += visc_uy_ux * JxW;
                    local_uy_uy(i, j) += (mass + visc_uy_uy + conv_uy) * JxW;
                }

                // Pressure coupling: -(p, ∇·v) and (∇·u, q)
                for (unsigned int j = 0; j < p_dpc; ++j)
                {
                    const double phi_p_j = fe_values_p.shape_value(j, q);

                    // B^T: -(p, ∂vx/∂x) for ux test, -(p, ∂vy/∂y) for uy test
                    local_ux_p(i, j) += (-phi_p_j * grad_phi_ux_i[0]) * JxW;
                    local_uy_p(i, j) += (-phi_p_j * grad_phi_ux_i[1]) * JxW;

                    // B: (∂ux/∂x, q) and (∂uy/∂y, q)
                    local_p_ux(j, i) += (grad_phi_ux_i[0] * phi_p_j) * JxW;
                    local_p_uy(j, i) += (grad_phi_ux_i[1] * phi_p_j) * JxW;
                }
            }
        }

        // ================================================================
        // Distribute local matrices to global coupled system
        // ================================================================
        // Map local DoFs to coupled system indices
        std::vector<dealii::types::global_dof_index> coupled_ux(ux_dpc);
        std::vector<dealii::types::global_dof_index> coupled_uy(uy_dpc);
        std::vector<dealii::types::global_dof_index> coupled_p(p_dpc);

        for (unsigned int i = 0; i < ux_dpc; ++i)
            coupled_ux[i] = ux_dofs[i];
        for (unsigned int i = 0; i < uy_dpc; ++i)
            coupled_uy[i] = n_ux_ + uy_dofs[i];
        for (unsigned int i = 0; i < p_dpc; ++i)
            coupled_p[i] = n_ux_ + n_uy_ + p_dofs[i];

        // Velocity-velocity blocks
        ns_constraints_.distribute_local_to_global(
            local_ux_ux, local_rhs_ux, coupled_ux, ns_matrix_, ns_rhs_);
        ns_constraints_.distribute_local_to_global(
            local_ux_uy, coupled_ux, coupled_uy, ns_matrix_);
        ns_constraints_.distribute_local_to_global(
            local_uy_ux, coupled_uy, coupled_ux, ns_matrix_);
        ns_constraints_.distribute_local_to_global(
            local_uy_uy, local_rhs_uy, coupled_uy, ns_matrix_, ns_rhs_);

        // Velocity-pressure and pressure-velocity blocks
        ns_constraints_.distribute_local_to_global(
            local_ux_p, coupled_ux, coupled_p, ns_matrix_);
        ns_constraints_.distribute_local_to_global(
            local_uy_p, coupled_uy, coupled_p, ns_matrix_);
        ns_constraints_.distribute_local_to_global(
            local_p_ux, coupled_p, coupled_ux, ns_matrix_);
        ns_constraints_.distribute_local_to_global(
            local_p_uy, coupled_p, coupled_uy, ns_matrix_);

        // Pressure-pressure block (zero matrix, but needed so that
        // distribute_local_to_global sets diagonal = 1 for pinned pressure DoF)
        ns_constraints_.distribute_local_to_global(
            local_p_p, local_rhs_p, coupled_p, ns_matrix_, ns_rhs_);
    }

    ns_matrix_.compress(dealii::VectorOperation::add);
    ns_rhs_.compress(dealii::VectorOperation::add);

    timer.stop();
    last_assemble_time_ = timer.wall_time();
}

// Explicit instantiations
template void NavierStokesSubsystem<2>::assemble(
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::TrilinosWrappers::MPI::Vector&,
    double, double, bool,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::DoFHandler<2>&,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::DoFHandler<2>*,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::DoFHandler<2>*);
template void NavierStokesSubsystem<3>::assemble(
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::TrilinosWrappers::MPI::Vector&,
    double, double, bool,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::DoFHandler<3>&,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::DoFHandler<3>*,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::DoFHandler<3>*);
