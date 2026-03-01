// ============================================================================
// magnetization/magnetization_assemble.cc - DG Assembly
//
// PAPER EQUATION 42c (Nochetto, Salgado & Tomas, arXiv:1511.04381):
//
//   (m^k/τ, z) + σ·a_h^m(m^k, z) + B_h^m(u^{k-1}; m^k, z) + (1/𝒯)(m^k, z)
//     = (1/𝒯)(κ₀·h^k, z) + (m^{k-1}/τ, z) + f_mms
//
// DG forms:
//   a_h^m (Eq. 63-65): Symmetric interior penalty for diffusion
//   B_h^m (Eq. 62):    Skew-symmetric transport + upwind
//
// Matrix (same for Mx and My):
//   (1/τ + 1/𝒯)(m, z)        — mass
//   σ·a_h^m(m, z)             — DG diffusion (SIP, if σ > 0)
//   B_h^m(u^{k-1}; m, z)     — DG transport (skew + upwind)
//
// RHS (separate for Mx, My):
//   (1/τ)(m^{k-1}, z)         — old-time mass
//   (1/𝒯)(κ₀·h^k, z)         — relaxation source (h = ∇φ, total field)
//   (f_mms, z)                — MMS source (if enabled)
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015)
// ============================================================================

#include "magnetization/magnetization.h"
#include "physics/skew_forms.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/meshworker/mesh_loop.h>

template <int dim>
void MagnetizationSubsystem<dim>::assemble(
    const dealii::TrilinosWrappers::MPI::Vector& Mx_old_relevant,
    const dealii::TrilinosWrappers::MPI::Vector& My_old_relevant,
    const dealii::TrilinosWrappers::MPI::Vector& phi_relevant,
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& ux_relevant,
    const dealii::TrilinosWrappers::MPI::Vector& uy_relevant,
    const dealii::DoFHandler<dim>& u_dof_handler,
    double dt,
    double current_time)
{
    dealii::Timer timer;
    timer.start();

    const double tau = dt;
    const double tau_M = params_.physics.T_relax;
    const double sigma = params_.physics.sigma;
    const double kappa_0 = params_.physics.kappa_0;
    const double eta = params_.dg.penalty_parameter;
    const double mass_coeff = 1.0 / tau + 1.0 / tau_M;
    const double old_coeff = 1.0 / tau;
    const double relax_coeff = kappa_0 / tau_M;

    const bool has_velocity = (ux_relevant.size() > 0);
    const bool has_phi = (phi_relevant.size() > 0);

    const unsigned int degree = fe_.degree;
    const dealii::QGauss<dim> quadrature(degree + 1);
    const dealii::QGauss<dim - 1> face_quadrature(degree + 1);

    const unsigned int n_q_points = quadrature.size();
    const unsigned int n_face_q_points = face_quadrature.size();
    const unsigned int dpc = fe_.n_dofs_per_cell();

    // ========================================================================
    // FEValues for DG magnetization (cell + face)
    // ========================================================================
    dealii::FEValues<dim> fe_values_M(fe_, quadrature,
        dealii::update_values | dealii::update_gradients |
        dealii::update_quadrature_points | dealii::update_JxW_values);

    dealii::FEFaceValues<dim> fe_face_M(fe_, face_quadrature,
        dealii::update_values | dealii::update_gradients |
        dealii::update_normal_vectors | dealii::update_JxW_values);

    dealii::FEFaceValues<dim> fe_face_M_neighbor(fe_, face_quadrature,
        dealii::update_values | dealii::update_gradients);

    dealii::FESubfaceValues<dim> fe_subface_M(fe_, face_quadrature,
        dealii::update_values | dealii::update_gradients);

    // FEValues for Poisson (CG) — evaluate ∇φ at DG quadrature points
    std::unique_ptr<dealii::FEValues<dim>> fe_values_phi;
    if (has_phi)
    {
        fe_values_phi = std::make_unique<dealii::FEValues<dim>>(
            phi_dof_handler.get_fe(), quadrature,
            dealii::update_gradients);
    }

    // FEValues for velocity (CG) — evaluate U and div(U)
    std::unique_ptr<dealii::FEValues<dim>> fe_values_U;
    if (has_velocity)
    {
        fe_values_U = std::make_unique<dealii::FEValues<dim>>(
            u_dof_handler.get_fe(), quadrature,
            dealii::update_values | dealii::update_gradients);
    }

    std::unique_ptr<dealii::FEFaceValues<dim>> fe_face_U;
    if (has_velocity)
    {
        fe_face_U = std::make_unique<dealii::FEFaceValues<dim>>(
            u_dof_handler.get_fe(), face_quadrature,
            dealii::update_values);
    }

    // ========================================================================
    // Local storage
    // ========================================================================
    dealii::FullMatrix<double> cell_matrix(dpc, dpc);
    dealii::Vector<double> cell_rhs_x(dpc), cell_rhs_y(dpc);
    std::vector<dealii::types::global_dof_index> local_dofs(dpc);

    // Face coupling matrices (here-here, here-there, there-here, there-there)
    dealii::FullMatrix<double> face_hh(dpc, dpc), face_ht(dpc, dpc);
    dealii::FullMatrix<double> face_th(dpc, dpc), face_tt(dpc, dpc);

    std::vector<dealii::types::global_dof_index> local_dofs_neighbor(dpc);

    // Quadrature point value buffers
    std::vector<double> Mx_old_vals(n_q_points), My_old_vals(n_q_points);
    std::vector<dealii::Tensor<1, dim>> grad_phi_vals(n_q_points);
    std::vector<double> ux_vals(n_q_points), uy_vals(n_q_points);
    std::vector<dealii::Tensor<1, dim>> grad_ux_vals(n_q_points), grad_uy_vals(n_q_points);

    // Face velocity buffers
    std::vector<double> ux_face_vals(n_face_q_points);
    std::vector<double> uy_face_vals(n_face_q_points);

    // ========================================================================
    // Zero global system
    // ========================================================================
    system_matrix_ = 0.0;
    Mx_rhs_ = 0.0;
    My_rhs_ = 0.0;

    // ========================================================================
    // Cell loop
    // ========================================================================
    for (const auto& cell : dof_handler_.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values_M.reinit(cell);
        cell_matrix = 0.0;
        cell_rhs_x = 0.0;
        cell_rhs_y = 0.0;
        cell->get_dof_indices(local_dofs);

        // Get old M values
        fe_values_M.get_function_values(Mx_old_relevant, Mx_old_vals);
        fe_values_M.get_function_values(My_old_relevant, My_old_vals);

        // Get ∇φ (h = ∇φ is total field; h_a already encoded via Poisson RHS)
        if (has_phi)
        {
            const auto phi_cell = cell->as_dof_handler_iterator(phi_dof_handler);
            fe_values_phi->reinit(phi_cell);
            fe_values_phi->get_function_gradients(phi_relevant, grad_phi_vals);
        }

        // Get velocity U and grad U
        if (has_velocity)
        {
            const auto u_cell = cell->as_dof_handler_iterator(u_dof_handler);
            fe_values_U->reinit(u_cell);
            fe_values_U->get_function_values(ux_relevant, ux_vals);
            fe_values_U->get_function_values(uy_relevant, uy_vals);
            fe_values_U->get_function_gradients(ux_relevant, grad_ux_vals);
            fe_values_U->get_function_gradients(uy_relevant, grad_uy_vals);
        }

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const auto& x_q = fe_values_M.quadrature_point(q);
            const double JxW = fe_values_M.JxW(q);

            // Total field h = ∇φ (h_a is encoded into φ via Poisson RHS)
            dealii::Tensor<1, dim> H;
            if (has_phi)
                H = grad_phi_vals[q];

            // Velocity
            dealii::Tensor<1, dim> U;
            double div_U = 0.0;
            if (has_velocity)
            {
                U[0] = ux_vals[q];
                U[1] = uy_vals[q];
                div_U = grad_ux_vals[q][0] + grad_uy_vals[q][1];
            }

            for (unsigned int i = 0; i < dpc; ++i)
            {
                const double z_i = fe_values_M.shape_value(i, q);
                const auto& grad_z_i = fe_values_M.shape_grad(i, q);

                // --- RHS ---
                // (1/τ)(m^{k-1}, z) + (κ₀/𝒯)(h^k, z) + f_mms
                cell_rhs_x(i) += (old_coeff * Mx_old_vals[q] * z_i
                                   + relax_coeff * H[0] * z_i) * JxW;
                cell_rhs_y(i) += (old_coeff * My_old_vals[q] * z_i
                                   + relax_coeff * H[1] * z_i) * JxW;

                // MMS source
                if (params_.enable_mms && mms_source_)
                {
                    dealii::Tensor<1, dim> M_old_q;
                    M_old_q[0] = Mx_old_vals[q];
                    M_old_q[1] = My_old_vals[q];
                    const auto f = mms_source_(x_q, current_time,
                                               current_time - dt,
                                               tau_M, kappa_0, H, U, div_U,
                                               M_old_q);
                    cell_rhs_x(i) += f[0] * z_i * JxW;
                    cell_rhs_y(i) += f[1] * z_i * JxW;
                }

                for (unsigned int j = 0; j < dpc; ++j)
                {
                    const double m_j = fe_values_M.shape_value(j, q);
                    const auto& grad_m_j = fe_values_M.shape_grad(j, q);

                    // Mass: (1/τ + 1/𝒯)(m, z)
                    double val = mass_coeff * m_j * z_i;

                    // Diffusion: σ(∇m, ∇z) — cell interior
                    if (sigma > 0.0)
                        val += sigma * (grad_m_j * grad_z_i);

                    // Transport: B_h^m cell term
                    if (has_velocity)
                    {
                        val += skew_magnetic_cell_value_scalar<dim>(
                            U, div_U, m_j, grad_m_j, z_i);
                    }

                    cell_matrix(i, j) += val * JxW;
                }
            }
        }

        // ================================================================
        // Face loop (interior faces only)
        // ================================================================
        for (unsigned int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell; ++f)
        {
            const auto face = cell->face(f);

            // Skip boundary faces (DG: natural BCs = zero flux)
            if (face->at_boundary())
                continue;

            // Skip if neighbor is coarser (handled from coarser side)
            if (cell->neighbor_is_coarser(f))
                continue;

            // Determine neighbor
            const auto neighbor = cell->neighbor(f);

            // Each internal face is visited by BOTH adjacent cells.
            // Process only once: the cell with the smaller CellId handles it.
            // This prevents double-counting of face flux terms.
            if (cell->level() == neighbor->level()
                && neighbor->id() < cell->id())
                continue;

            if (neighbor->is_ghost() || neighbor->is_locally_owned())
            {
                // Same-level or coarser processing
                const unsigned int neighbor_face = cell->neighbor_of_neighbor(f);

                fe_face_M.reinit(cell, f);
                fe_face_M_neighbor.reinit(neighbor, neighbor_face);

                neighbor->get_dof_indices(local_dofs_neighbor);

                // Evaluate velocity at face quadrature points
                if (has_velocity)
                {
                    const auto u_cell = cell->as_dof_handler_iterator(u_dof_handler);
                    fe_face_U->reinit(u_cell, f);
                    fe_face_U->get_function_values(ux_relevant, ux_face_vals);
                    fe_face_U->get_function_values(uy_relevant, uy_face_vals);
                }

                face_hh = 0.0;
                face_ht = 0.0;
                face_th = 0.0;
                face_tt = 0.0;

                const double h_face = face->diameter();
                const double penalty = eta * degree * (degree + 1) / h_face;

                for (unsigned int q = 0; q < n_face_q_points; ++q)
                {
                    const double JxW = fe_face_M.JxW(q);
                    const auto& normal = fe_face_M.normal_vector(q);

                    // Velocity at face quadrature point
                    double U_dot_n = 0.0;
                    if (has_velocity)
                    {
                        U_dot_n = ux_face_vals[q] * normal[0]
                                + uy_face_vals[q] * normal[1];
                    }

                    for (unsigned int i = 0; i < dpc; ++i)
                    {
                        const double z_h = fe_face_M.shape_value(i, q);
                        const double z_t = fe_face_M_neighbor.shape_value(i, q);
                        const auto& grad_z_h = fe_face_M.shape_grad(i, q);
                        const auto& grad_z_t = fe_face_M_neighbor.shape_grad(i, q);

                        for (unsigned int j = 0; j < dpc; ++j)
                        {
                            const double m_h = fe_face_M.shape_value(j, q);
                            const double m_t = fe_face_M_neighbor.shape_value(j, q);
                            const auto& grad_m_h = fe_face_M.shape_grad(j, q);
                            const auto& grad_m_t = fe_face_M_neighbor.shape_grad(j, q);

                            // SIP diffusion face terms (if σ > 0)
                            if (sigma > 0.0)
                            {
                                // Penalty: +η/h [[m]][[z]]
                                // Here-here: +penalty * m_h * z_h
                                face_hh(i, j) += sigma * penalty * m_h * z_h * JxW;
                                face_ht(i, j) -= sigma * penalty * m_t * z_h * JxW;
                                face_th(i, j) -= sigma * penalty * m_h * z_t * JxW;
                                face_tt(i, j) += sigma * penalty * m_t * z_t * JxW;

                                // Consistency: -{∂m/∂n}[[z]]
                                const double avg_grad_m_n_h = 0.5 * (grad_m_h * normal);
                                const double avg_grad_m_n_t = 0.5 * (grad_m_t * normal);
                                face_hh(i, j) -= sigma * avg_grad_m_n_h * z_h * JxW;
                                face_ht(i, j) -= sigma * avg_grad_m_n_t * z_h * JxW;
                                face_th(i, j) += sigma * avg_grad_m_n_h * z_t * JxW;
                                face_tt(i, j) += sigma * avg_grad_m_n_t * z_t * JxW;

                                // Symmetry: -[[m]]{∂z/∂n}
                                const double avg_grad_z_n_h = 0.5 * (grad_z_h * normal);
                                const double avg_grad_z_n_t = 0.5 * (grad_z_t * normal);
                                face_hh(i, j) -= sigma * m_h * avg_grad_z_n_h * JxW;
                                face_ht(i, j) += sigma * m_t * avg_grad_z_n_h * JxW;
                                face_th(i, j) -= sigma * m_h * avg_grad_z_n_t * JxW;
                                face_tt(i, j) += sigma * m_t * avg_grad_z_n_t * JxW;
                            }

                            // Transport face terms: -U·n [[m]] {z} + ½|U·n| [[m]][[z]]
                            if (has_velocity && std::abs(U_dot_n) > 1e-14)
                            {
                                const double abs_Un = std::abs(U_dot_n);

                                // Central: -U·n (m_h - m_t) · ½(z_h + z_t)
                                const double central_hh = -U_dot_n * m_h * 0.5 * z_h;
                                const double central_ht = -U_dot_n * (-m_t) * 0.5 * z_h;
                                const double central_th = -U_dot_n * m_h * 0.5 * z_t;
                                const double central_tt = -U_dot_n * (-m_t) * 0.5 * z_t;

                                // Upwind: ½|U·n| (m_h - m_t)(z_h - z_t)
                                const double upwind_hh = 0.5 * abs_Un * m_h * z_h;
                                const double upwind_ht = 0.5 * abs_Un * (-m_t) * z_h;
                                const double upwind_th = 0.5 * abs_Un * m_h * (-z_t);
                                const double upwind_tt = 0.5 * abs_Un * (-m_t) * (-z_t);

                                face_hh(i, j) += (central_hh + upwind_hh) * JxW;
                                face_ht(i, j) += (central_ht + upwind_ht) * JxW;
                                face_th(i, j) += (central_th + upwind_th) * JxW;
                                face_tt(i, j) += (central_tt + upwind_tt) * JxW;
                            }
                        }
                    }
                }

                // Distribute face matrices to global
                constraints_.distribute_local_to_global(
                    face_hh, local_dofs, system_matrix_);
                constraints_.distribute_local_to_global(
                    face_ht, local_dofs, local_dofs_neighbor, system_matrix_);
                constraints_.distribute_local_to_global(
                    face_th, local_dofs_neighbor, local_dofs, system_matrix_);
                constraints_.distribute_local_to_global(
                    face_tt, local_dofs_neighbor, system_matrix_);
            }
        }

        // Distribute cell contributions
        constraints_.distribute_local_to_global(
            cell_matrix, local_dofs, system_matrix_);
        constraints_.distribute_local_to_global(
            cell_rhs_x, local_dofs, Mx_rhs_);
        constraints_.distribute_local_to_global(
            cell_rhs_y, local_dofs, My_rhs_);
    }

    system_matrix_.compress(dealii::VectorOperation::add);
    Mx_rhs_.compress(dealii::VectorOperation::add);
    My_rhs_.compress(dealii::VectorOperation::add);

    timer.stop();
    last_assemble_time_ = timer.wall_time();
}

// Explicit instantiations
template void MagnetizationSubsystem<2>::assemble(
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::DoFHandler<2>&,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::DoFHandler<2>&,
    double, double);

template void MagnetizationSubsystem<3>::assemble(
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::DoFHandler<3>&,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::DoFHandler<3>&,
    double, double);
