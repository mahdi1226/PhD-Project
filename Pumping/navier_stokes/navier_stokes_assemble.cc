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
#include "physics/material_properties.h"

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
    const dealii::DoFHandler<dim>* phi_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& ch_solution_relevant,
    const dealii::DoFHandler<dim>* ch_dof_handler)
{
    dealii::Timer timer;
    timer.start();

    const double nu_const = params_.physics.nu + params_.physics.nu_r;  // constant fallback
    const double nu_r = params_.physics.nu_r;
    const double nu_carrier = params_.physics.nu_carrier;
    const double nu_ferro = params_.physics.nu_ferro;
    const double mu_0 = params_.physics.mu_0;
    const double mass_coeff = 1.0 / dt;

    const bool has_time = (dt > 0.0 && dt < 1e30);
    const bool has_old = (ux_old_relevant.size() > 0);
    const bool has_w = (w_relevant.size() > 0);
    const bool has_kelvin = (Mx_relevant.size() > 0
                             && M_dof_handler != nullptr
                             && phi_relevant.size() > 0
                             && phi_dof_handler != nullptr);
    const bool has_capillary = (ch_solution_relevant.size() > 0
                                && ch_dof_handler != nullptr);
    const double ch_sigma = params_.cahn_hilliard_params.sigma;

    const unsigned int vel_degree = fe_velocity_.degree;
    const dealii::QGauss<dim> quadrature(vel_degree + 1);
    const unsigned int n_q = quadrature.size();

    // Face quadrature for Kelvin force face integral (Eq. 38, 2nd line)
    const dealii::QGauss<dim - 1> face_quadrature(vel_degree + 1);
    const unsigned int n_face_q = face_quadrature.size();

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

    // FEFaceValues for Kelvin force face integral (Eq. 38, 2nd line):
    // −Σ_F ∫_F (V·n⁻) [[H]]·{M} ds
    std::unique_ptr<dealii::FEFaceValues<dim>> fe_face_vel;
    std::unique_ptr<dealii::FEFaceValues<dim>> fe_face_phi;
    std::unique_ptr<dealii::FEFaceValues<dim>> fe_face_phi_neighbor;
    std::unique_ptr<dealii::FEFaceValues<dim>> fe_face_M;
    std::unique_ptr<dealii::FEFaceValues<dim>> fe_face_M_neighbor;
    if (has_kelvin)
    {
        fe_face_vel = std::make_unique<dealii::FEFaceValues<dim>>(
            fe_velocity_, face_quadrature,
            dealii::update_values | dealii::update_normal_vectors
            | dealii::update_JxW_values);
        fe_face_phi = std::make_unique<dealii::FEFaceValues<dim>>(
            phi_dof_handler->get_fe(), face_quadrature,
            dealii::update_gradients);
        fe_face_phi_neighbor = std::make_unique<dealii::FEFaceValues<dim>>(
            phi_dof_handler->get_fe(), face_quadrature,
            dealii::update_gradients);
        fe_face_M = std::make_unique<dealii::FEFaceValues<dim>>(
            M_dof_handler->get_fe(), face_quadrature,
            dealii::update_values);
        fe_face_M_neighbor = std::make_unique<dealii::FEFaceValues<dim>>(
            M_dof_handler->get_fe(), face_quadrature,
            dealii::update_values);
    }

    // FEValues for Cahn-Hilliard (FESystem: phi=component 0, mu=component 1)
    // Used for capillary force: σ μ ∇φ
    std::unique_ptr<dealii::FEValues<dim>> fe_values_ch;
    if (has_capillary)
    {
        fe_values_ch = std::make_unique<dealii::FEValues<dim>>(
            ch_dof_handler->get_fe(), quadrature,
            dealii::update_values | dealii::update_gradients);
    }

    // Capillary force + phase-dependent viscosity buffers
    std::vector<double> ch_mu_vals(n_q);
    std::vector<double> ch_theta_vals(n_q);
    std::vector<dealii::Tensor<1, dim>> ch_grad_phi_vals(n_q);

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

    // Pressure mass matrix local (for block-Schur preconditioner)
    dealii::FullMatrix<double> local_Mp(p_dpc, p_dpc);

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

    // Kelvin face integral buffers (Eq. 38, 2nd line)
    std::vector<dealii::Tensor<1, dim>> phi_grad_face_minus(n_face_q);
    std::vector<dealii::Tensor<1, dim>> phi_grad_face_plus(n_face_q);
    std::vector<double> Mx_face_minus(n_face_q), My_face_minus(n_face_q);
    std::vector<double> Mx_face_plus(n_face_q), My_face_plus(n_face_q);

    // Zero global system
    ns_matrix_ = 0.0;
    ns_rhs_ = 0.0;

    if (use_block_schur_)
    {
        A_ux_ux_ = 0.0;
        A_uy_uy_ = 0.0;
        Bt_ux_ = 0.0;
        Bt_uy_ = 0.0;
        B_ux_ = 0.0;
        B_uy_ = 0.0;
        M_p_ = 0.0;
    }

    // Kelvin force diagnostics accumulators (mesh-dependence tracking)
    double local_kelvin_cell_sq = 0.0;  // Σ |f_cell|² JxW
    double local_kelvin_face_sq = 0.0;  // Σ |f_face|² JxW
    double local_kelvin_Fx = 0.0;       // Σ f_cell_x JxW + face
    double local_kelvin_Fy = 0.0;       // Σ f_cell_y JxW + face

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

        // Get CH solution for capillary force + phase-dependent viscosity
        if (has_capillary)
        {
            const auto ch_cell = ux_cell->as_dof_handler_iterator(
                *ch_dof_handler);
            fe_values_ch->reinit(ch_cell);

            const dealii::FEValuesExtractors::Scalar ch_phi(0);
            const dealii::FEValuesExtractors::Scalar ch_mu(1);
            (*fe_values_ch)[ch_phi].get_function_values(
                ch_solution_relevant, ch_theta_vals);
            (*fe_values_ch)[ch_mu].get_function_values(
                ch_solution_relevant, ch_mu_vals);
            (*fe_values_ch)[ch_phi].get_function_gradients(
                ch_solution_relevant, ch_grad_phi_vals);
        }

        // Zero local matrices
        local_ux_ux = 0.0;  local_ux_uy = 0.0;
        local_uy_ux = 0.0;  local_uy_uy = 0.0;
        local_ux_p = 0.0;   local_uy_p = 0.0;
        local_p_ux = 0.0;   local_p_uy = 0.0;
        local_p_p = 0.0;    local_Mp = 0.0;
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

            // Phase-dependent viscosity: ν(φ) + ν_r
            // In carrier (φ=-1): ν = ν_carrier + ν_r
            // In ferrofluid (φ=+1): ν = ν_ferro + ν_r
            const double nu_eff_q = has_capillary
                ? viscosity(ch_theta_vals[q], nu_carrier, nu_ferro) + nu_r
                : nu_const;

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
                                     current_time - dt, nu_eff_q,
                                     U_old_disc, div_U_old,
                                     include_convection);
            }

            // Kelvin cell force density diagnostic (independent of test function)
            // f_cell = μ₀[(M·∇)H + ½ div(M) H]
            if (has_kelvin)
            {
                dealii::Tensor<1, dim> M_q;
                M_q[0] = Mx_vals[q];
                M_q[1] = My_vals[q];

                const dealii::Tensor<1, dim>& H_q = phi_grad_vals[q];
                const dealii::Tensor<2, dim>& grad_H = phi_hess_vals[q];

                // (M·∇)H
                dealii::Tensor<1, dim> M_grad_H;
                for (unsigned int d = 0; d < dim; ++d)
                    for (unsigned int e = 0; e < dim; ++e)
                        M_grad_H[d] += M_q[e] * grad_H[d][e];

                const double div_M = grad_Mx_vals[q][0] + grad_My_vals[q][1];

                dealii::Tensor<1, dim> f_cell;
                for (unsigned int d = 0; d < dim; ++d)
                    f_cell[d] = mu_0 * (M_grad_H[d] + 0.5 * div_M * H_q[d]);

                local_kelvin_cell_sq += (f_cell * f_cell) * JxW;
                local_kelvin_Fx += f_cell[0] * JxW;
                local_kelvin_Fy += f_cell[1] * JxW;
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

                // Capillary force (Phase B): σ μ ∇φ · v
                // from Cahn-Hilliard coupling
                if (has_capillary)
                {
                    local_rhs_ux(i) += ch_sigma * ch_mu_vals[q]
                                       * ch_grad_phi_vals[q][0]
                                       * phi_ux_i * JxW;
                    local_rhs_uy(i) += ch_sigma * ch_mu_vals[q]
                                       * ch_grad_phi_vals[q][1]
                                       * phi_ux_i * JxW;
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
                    const double visc_ux_ux = (nu_eff_q / 4.0) * (T_Ux_j * T_Vx_i);
                    const double visc_uy_ux = (nu_eff_q / 4.0) * (T_Uy_j * T_Vx_i);
                    const double visc_ux_uy = (nu_eff_q / 4.0) * (T_Ux_j * T_Vy_i);
                    const double visc_uy_uy = (nu_eff_q / 4.0) * (T_Uy_j * T_Vy_i);

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

            // Pressure mass matrix (for Schur complement approximation)
            if (use_block_schur_)
            {
                for (unsigned int i = 0; i < p_dpc; ++i)
                {
                    const double phi_p_i = fe_values_p.shape_value(i, q);
                    for (unsigned int j = 0; j < p_dpc; ++j)
                    {
                        const double phi_p_j = fe_values_p.shape_value(j, q);
                        local_Mp(i, j) += phi_p_i * phi_p_j * JxW;
                    }
                }
            }
        }

        // ================================================================
        // Face loop: Kelvin force face integral — Eq. 38, 2nd line
        // −Σ_F ∫_F (V·n⁻) [[H]]·{M} ds
        //
        // Process each interior face ONCE (dedup by CellId).
        // CG velocity: evaluate test functions from one side only.
        // RHS-only contribution (no matrix modification).
        // ================================================================
        if (has_kelvin)
        {
            for (unsigned int f = 0;
                 f < dealii::GeometryInfo<dim>::faces_per_cell; ++f)
            {
                if (ux_cell->face(f)->at_boundary())
                    continue;
                if (ux_cell->neighbor_is_coarser(f))
                    continue;

                const auto neighbor = ux_cell->neighbor(f);

                // Dedup: process each face once (smaller CellId)
                if (ux_cell->level() == neighbor->level()
                    && neighbor->id() < ux_cell->id())
                    continue;

                if (!neighbor->is_ghost() && !neighbor->is_locally_owned())
                    continue;

                const unsigned int nf = ux_cell->neighbor_of_neighbor(f);

                // Velocity test functions on minus side
                fe_face_vel->reinit(ux_cell, f);

                // φ gradients on both sides → [[∇φ]]
                const auto phi_minus = ux_cell->as_dof_handler_iterator(
                    *phi_dof_handler);
                const auto phi_plus = neighbor->as_dof_handler_iterator(
                    *phi_dof_handler);
                fe_face_phi->reinit(phi_minus, f);
                fe_face_phi_neighbor->reinit(phi_plus, nf);
                fe_face_phi->get_function_gradients(
                    phi_relevant, phi_grad_face_minus);
                fe_face_phi_neighbor->get_function_gradients(
                    phi_relevant, phi_grad_face_plus);

                // M values on both sides → {M}
                const auto M_minus = ux_cell->as_dof_handler_iterator(
                    *M_dof_handler);
                const auto M_plus = neighbor->as_dof_handler_iterator(
                    *M_dof_handler);
                fe_face_M->reinit(M_minus, f);
                fe_face_M_neighbor->reinit(M_plus, nf);
                fe_face_M->get_function_values(Mx_relevant, Mx_face_minus);
                fe_face_M->get_function_values(My_relevant, My_face_minus);
                fe_face_M_neighbor->get_function_values(
                    Mx_relevant, Mx_face_plus);
                fe_face_M_neighbor->get_function_values(
                    My_relevant, My_face_plus);

                for (unsigned int q = 0; q < n_face_q; ++q)
                {
                    const double JxW = fe_face_vel->JxW(q);
                    const auto& normal = fe_face_vel->normal_vector(q);

                    // [[H]] = ∇φ⁻ − ∇φ⁺
                    dealii::Tensor<1, dim> jump_H;
                    jump_H = phi_grad_face_minus[q]
                           - phi_grad_face_plus[q];

                    // {M} = ½(M⁻ + M⁺)
                    dealii::Tensor<1, dim> avg_M;
                    avg_M[0] = 0.5 * (Mx_face_minus[q] + Mx_face_plus[q]);
                    avg_M[1] = 0.5 * (My_face_minus[q] + My_face_plus[q]);

                    // Face force diagnostic: f_face = -μ₀ n [[H]]·{M}
                    {
                        const double jump_dot_avg = jump_H * avg_M;
                        // |f_face|² = μ₀² (n[0]² + n[1]²)(jump·avg)²
                        //            = μ₀² (jump·avg)²  (|n|=1)
                        local_kelvin_face_sq += mu_0 * mu_0
                            * jump_dot_avg * jump_dot_avg * JxW;
                        local_kelvin_Fx += -mu_0 * normal[0]
                            * jump_dot_avg * JxW;
                        local_kelvin_Fy += -mu_0 * normal[1]
                            * jump_dot_avg * JxW;
                    }

                    for (unsigned int i = 0; i < ux_dpc; ++i)
                    {
                        const double phi_i =
                            fe_face_vel->shape_value(i, q);

                        double kelvin_ux, kelvin_uy;
                        KelvinForce::face_kernel<dim>(
                            phi_i, phi_i, normal, jump_H, avg_M,
                            kelvin_ux, kelvin_uy);

                        local_rhs_ux(i) += mu_0 * kelvin_ux * JxW;
                        local_rhs_uy(i) += mu_0 * kelvin_uy * JxW;
                    }
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

        // ================================================================
        // Distribute to separate block matrices (for Schur preconditioner)
        // ================================================================
        if (use_block_schur_)
        {
            // Velocity diagonal blocks
            ux_constraints_.distribute_local_to_global(
                local_ux_ux, ux_dofs, A_ux_ux_);
            uy_constraints_.distribute_local_to_global(
                local_uy_uy, uy_dofs, A_uy_uy_);

            // B^T blocks (velocity × pressure): rows=vel, cols=p
            // Use raw add since cross-block constraints are handled by monolithic
            for (unsigned int i = 0; i < ux_dpc; ++i)
                for (unsigned int j = 0; j < p_dpc; ++j)
                {
                    if (local_ux_p(i, j) != 0.0)
                        Bt_ux_.add(ux_dofs[i], p_dofs[j], local_ux_p(i, j));
                    if (local_uy_p(i, j) != 0.0)
                        Bt_uy_.add(uy_dofs[i], p_dofs[j], local_uy_p(i, j));
                }

            // B blocks (pressure × velocity): rows=p, cols=vel
            for (unsigned int i = 0; i < p_dpc; ++i)
                for (unsigned int j = 0; j < ux_dpc; ++j)
                {
                    if (local_p_ux(i, j) != 0.0)
                        B_ux_.add(p_dofs[i], ux_dofs[j], local_p_ux(i, j));
                    if (local_p_uy(i, j) != 0.0)
                        B_uy_.add(p_dofs[i], uy_dofs[j], local_p_uy(i, j));
                }

            // Pressure mass matrix
            p_constraints_.distribute_local_to_global(
                local_Mp, p_dofs, M_p_);
        }
    }

    ns_matrix_.compress(dealii::VectorOperation::add);
    ns_rhs_.compress(dealii::VectorOperation::add);

    if (use_block_schur_)
    {
        A_ux_ux_.compress(dealii::VectorOperation::add);
        A_uy_uy_.compress(dealii::VectorOperation::add);
        Bt_ux_.compress(dealii::VectorOperation::add);
        Bt_uy_.compress(dealii::VectorOperation::add);
        B_ux_.compress(dealii::VectorOperation::add);
        B_uy_.compress(dealii::VectorOperation::add);
        M_p_.compress(dealii::VectorOperation::add);
    }

    // Kelvin force diagnostics: MPI reduce and store
    last_kelvin_cell_L2_sq_ = dealii::Utilities::MPI::sum(
        local_kelvin_cell_sq, mpi_comm_);
    last_kelvin_face_L2_sq_ = dealii::Utilities::MPI::sum(
        local_kelvin_face_sq, mpi_comm_);
    last_kelvin_Fx_ = dealii::Utilities::MPI::sum(
        local_kelvin_Fx, mpi_comm_);
    last_kelvin_Fy_ = dealii::Utilities::MPI::sum(
        local_kelvin_Fy, mpi_comm_);

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
    const dealii::DoFHandler<3>*,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::DoFHandler<3>*);
