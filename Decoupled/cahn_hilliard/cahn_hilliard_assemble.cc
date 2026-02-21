// ============================================================================
// cahn_hilliard/cahn_hilliard_assemble.cc - Coupled θ-ψ Assembly
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Equations 42a-42b (discrete scheme), p.505
//
// Paper's discrete scheme:
//   Eq 42a: (δθ^k/τ, Λ) - (U^{n-1} θ^{n-1}, ∇Λ) - γ(∇ψ^k, ∇Λ) = 0
//   Eq 42b: (ψ^k, Υ) + ε(∇θ^k, ∇Υ) + (1/ε)(f(θ^{n-1}), Υ) + (1/η)(δθ^k, Υ) = 0
//
// where η = ε (stabilization parameter)
//
// Assembly layout in local matrix (2*dofs_per_cell × 2*dofs_per_cell):
//   Row indices [0, dpc)     = θ test functions Λ_i
//   Row indices [dpc, 2*dpc) = ψ test functions Υ_i
//   Col indices [0, dpc)     = θ trial functions θ_j
//   Col indices [dpc, 2*dpc) = ψ trial functions ψ_j
//
// MPI-SAFE: Uses triangulation-based cell iteration to synchronize
// all DoFHandlers (θ, ψ, U) on the same physical cell.
// ============================================================================

#include "cahn_hilliard/cahn_hilliard.h"
#include "physics/material_properties.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/full_matrix.h>

#include <chrono>
#include <mpi.h>

template <int dim>
void CahnHilliardSubsystem<dim>::assemble_system(
    const dealii::TrilinosWrappers::MPI::Vector& theta_old_relevant,
    const std::vector<const dealii::TrilinosWrappers::MPI::Vector*>& velocity_components,
    const dealii::DoFHandler<dim>& u_dof_handler,
    double dt,
    double current_time)
{
    auto t_start = std::chrono::high_resolution_clock::now();

    Assert(velocity_components.size() == dim,
           dealii::ExcMessage("velocity_components must have exactly dim entries"));

    system_matrix_ = 0;
    system_rhs_ = 0;

    const unsigned int dofs_per_cell = fe_.n_dofs_per_cell();
    const unsigned int quad_degree = fe_.degree + 2;
    dealii::QGauss<dim> quadrature(quad_degree);
    const unsigned int n_q = quadrature.size();

    // FEValues for θ (values + gradients + quadrature points + JxW)
    dealii::FEValues<dim> theta_fe_values(fe_, quadrature,
        dealii::update_values | dealii::update_gradients |
        dealii::update_quadrature_points | dealii::update_JxW_values);

    // FEValues for ψ (same FE, values + gradients only)
    dealii::FEValues<dim> psi_fe_values(fe_, quadrature,
        dealii::update_values | dealii::update_gradients);

    // FEValues for velocity (different FE, values only)
    const auto& fe_vel = u_dof_handler.get_fe();
    dealii::FEValues<dim> vel_fe_values(fe_vel, quadrature,
        dealii::update_values);

    // Local system
    dealii::FullMatrix<double> local_matrix(2 * dofs_per_cell, 2 * dofs_per_cell);
    dealii::Vector<double> local_rhs(2 * dofs_per_cell);

    // DoF index arrays
    std::vector<dealii::types::global_dof_index> theta_local_dofs(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> psi_local_dofs(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> coupled_dof_indices(2 * dofs_per_cell);

    // Quadrature-point value arrays
    std::vector<double> theta_old_vals(n_q);

    // Velocity components at quadrature points: u_vals[d][q]
    std::vector<std::vector<double>> u_vals(dim, std::vector<double>(n_q, 0.0));

    // Physics parameters
    const double eps = params_.physics.epsilon;
    const double gamma = params_.physics.mobility;
    const double eta = eps;  // stabilization η = ε (paper convention)

    // Check if velocity is available (avoid expensive linfty_norm MPI reduction)
    bool use_convection = false;
    for (unsigned int d = 0; d < dim; ++d)
    {
        if (velocity_components[d] != nullptr &&
            velocity_components[d]->size() > 0)
        {
            use_convection = true;
            break;
        }
    }

    // ========================================================================
    // Cell loop — iterate over θ DoFHandler, construct matching cells for others
    // ========================================================================
    for (const auto& theta_cell : theta_dof_handler_.active_cell_iterators())
    {
        if (!theta_cell->is_locally_owned())
            continue;

        // Construct matching cells from the same triangulation cell
        const typename dealii::DoFHandler<dim>::active_cell_iterator psi_cell(
            &triangulation_, theta_cell->level(), theta_cell->index(),
            &psi_dof_handler_);

        theta_fe_values.reinit(theta_cell);
        psi_fe_values.reinit(psi_cell);

        // Get θ^{n-1} at quadrature points
        theta_fe_values.get_function_values(theta_old_relevant, theta_old_vals);

        // Get velocity at quadrature points (all dim components)
        if (use_convection)
        {
            const typename dealii::DoFHandler<dim>::active_cell_iterator vel_cell(
                &triangulation_, theta_cell->level(), theta_cell->index(),
                &u_dof_handler);
            vel_fe_values.reinit(vel_cell);

            for (unsigned int d = 0; d < dim; ++d)
            {
                if (velocity_components[d] != nullptr)
                    vel_fe_values.get_function_values(*velocity_components[d], u_vals[d]);
                else
                    std::fill(u_vals[d].begin(), u_vals[d].end(), 0.0);
            }
        }
        // Note: u_vals stays zero-initialized from line 80 when !use_convection

        local_matrix = 0;
        local_rhs = 0;

        // ====================================================================
        // Quadrature point loop
        // ====================================================================
        for (unsigned int q = 0; q < n_q; ++q)
        {
            const double JxW = theta_fe_values.JxW(q);
            const double theta_old_q = theta_old_vals[q];
            const auto& x_q = theta_fe_values.quadrature_point(q);

            // Velocity vector (all dim components)
            dealii::Tensor<1, dim> U;
            for (unsigned int d = 0; d < dim; ++d)
                U[d] = u_vals[d][q];

            // Nonlinearity: f(θ^{n-1}) = (θ^{n-1})³ - θ^{n-1}
            const double f_old = double_well_derivative(theta_old_q);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                // Test functions
                const double Lambda_i = theta_fe_values.shape_value(i, q);
                const auto& grad_Lambda_i = theta_fe_values.shape_grad(i, q);
                const double Upsilon_i = psi_fe_values.shape_value(i, q);
                const auto& grad_Upsilon_i = psi_fe_values.shape_grad(i, q);

                // Local indices in coupled system
                const unsigned int i_theta = i;
                const unsigned int i_psi = dofs_per_cell + i;

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    const double theta_j = theta_fe_values.shape_value(j, q);
                    const auto& grad_theta_j = theta_fe_values.shape_grad(j, q);
                    const double psi_j = psi_fe_values.shape_value(j, q);
                    const auto& grad_psi_j = psi_fe_values.shape_grad(j, q);

                    const unsigned int j_theta = j;
                    const unsigned int j_psi = dofs_per_cell + j;

                    // --------------------------------------------------------
                    // Eq 42a LHS
                    // --------------------------------------------------------
                    // (1/τ)(θ^n, Λ)  — mass term
                    local_matrix(i_theta, j_theta) +=
                        (1.0 / dt) * theta_j * Lambda_i * JxW;

                    // -γ(∇ψ^n, ∇Λ)  — diffusion of chemical potential
                    local_matrix(i_theta, j_psi) -=
                        gamma * (grad_psi_j * grad_Lambda_i) * JxW;

                    // --------------------------------------------------------
                    // Eq 42b LHS
                    // --------------------------------------------------------
                    // (ψ^n, Υ)  — mass term
                    local_matrix(i_psi, j_psi) +=
                        psi_j * Upsilon_i * JxW;

                    // ε(∇θ^n, ∇Υ)  — Laplacian of θ
                    local_matrix(i_psi, j_theta) +=
                        eps * (grad_theta_j * grad_Upsilon_i) * JxW;

                    // (1/η)(θ^n, Υ)  — stabilization
                    local_matrix(i_psi, j_theta) +=
                        (1.0 / eta) * theta_j * Upsilon_i * JxW;
                }

                // ============================================================
                // Eq 42a RHS
                // ============================================================
                // (1/τ)(θ^{n-1}, Λ)
                local_rhs(i_theta) +=
                    (1.0 / dt) * theta_old_q * Lambda_i * JxW;

                // Convection: (U^{n-1} · ∇Λ) θ^{n-1}  (LAGGED transport)
                // Note: this is -(U·∇θ, Λ) after IBP → +(θ, U·∇Λ)
                local_rhs(i_theta) +=
                    theta_old_q * (U * grad_Lambda_i) * JxW;

                // ============================================================
                // Eq 42b RHS
                // ============================================================
                // -(1/ε)(f(θ^{n-1}), Υ)
                local_rhs(i_psi) -=
                    (1.0 / eps) * f_old * Upsilon_i * JxW;

                // (1/η)(θ^{n-1}, Υ)  — stabilization
                local_rhs(i_psi) +=
                    (1.0 / eta) * theta_old_q * Upsilon_i * JxW;

                // MMS source terms (merged into main loop to avoid
                // duplicate quadrature iteration)
                if (mms_source_theta_)
                    local_rhs(i_theta) +=
                        mms_source_theta_(x_q, current_time) * Lambda_i * JxW;
                if (mms_source_psi_)
                    local_rhs(i_psi) +=
                        mms_source_psi_(x_q, current_time) * Upsilon_i * JxW;
            }
        }

        // ====================================================================
        // Distribute to global system with coupled constraints
        // ====================================================================
        theta_cell->get_dof_indices(theta_local_dofs);
        psi_cell->get_dof_indices(psi_local_dofs);

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            coupled_dof_indices[i] = theta_to_ch_map_[theta_local_dofs[i]];
            coupled_dof_indices[dofs_per_cell + i] = psi_to_ch_map_[psi_local_dofs[i]];
        }

        ch_constraints_.distribute_local_to_global(
            local_matrix, local_rhs,
            coupled_dof_indices,
            system_matrix_, system_rhs_);
    }

    // Compress for MPI
    system_matrix_.compress(dealii::VectorOperation::add);
    system_rhs_.compress(dealii::VectorOperation::add);

    auto t_end = std::chrono::high_resolution_clock::now();
    last_assemble_time_ =
        std::chrono::duration<double>(t_end - t_start).count();
}

// ============================================================================
// assemble_sav — Zhang's energy-stable SAV scheme
//
// Modified CH with:
//   Eq 42a LHS: (1/tau + S1)(theta^{n+1}, Lambda) - gamma(grad psi, grad Lambda)
//   Eq 42a RHS: (1/tau + S1)(theta^{n-1}, Lambda) + (U^{n-1} . grad Lambda) theta^{n-1}
//   Eq 42b LHS: (psi, Upsilon) + eps(grad theta, grad Upsilon) + (1/eta)(theta, Upsilon)
//   Eq 42b RHS: -sav_factor*(1/eps)*f(theta^{n-1}) * Upsilon + (1/eta)(theta^{n-1}, Upsilon)
//
// sav_factor = r_n / sqrt(E1_old + C0)  (1.0 = standard scheme, no SAV)
// S1 >= 1/eps is the CH stabilization constant
// ============================================================================
template <int dim>
void CahnHilliardSubsystem<dim>::assemble_sav(
    const dealii::TrilinosWrappers::MPI::Vector& theta_old_relevant,
    const std::vector<const dealii::TrilinosWrappers::MPI::Vector*>& velocity_components,
    const dealii::DoFHandler<dim>& u_dof_handler,
    double dt,
    double current_time,
    double S1,
    double sav_factor)
{
    auto t_start = std::chrono::high_resolution_clock::now();

    Assert(velocity_components.size() == dim,
           dealii::ExcMessage("velocity_components must have exactly dim entries"));

    system_matrix_ = 0;
    system_rhs_ = 0;

    const unsigned int dofs_per_cell = fe_.n_dofs_per_cell();
    const unsigned int quad_degree = fe_.degree + 2;
    dealii::QGauss<dim> quadrature(quad_degree);
    const unsigned int n_q = quadrature.size();

    dealii::FEValues<dim> theta_fe_values(fe_, quadrature,
        dealii::update_values | dealii::update_gradients |
        dealii::update_quadrature_points | dealii::update_JxW_values);

    dealii::FEValues<dim> psi_fe_values(fe_, quadrature,
        dealii::update_values | dealii::update_gradients);

    const auto& fe_vel = u_dof_handler.get_fe();
    dealii::FEValues<dim> vel_fe_values(fe_vel, quadrature,
        dealii::update_values);

    dealii::FullMatrix<double> local_matrix(2 * dofs_per_cell, 2 * dofs_per_cell);
    dealii::Vector<double> local_rhs(2 * dofs_per_cell);

    std::vector<dealii::types::global_dof_index> theta_local_dofs(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> psi_local_dofs(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> coupled_dof_indices(2 * dofs_per_cell);

    std::vector<double> theta_old_vals(n_q);
    std::vector<std::vector<double>> u_vals(dim, std::vector<double>(n_q, 0.0));

    const double eps = params_.physics.epsilon;
    const double lambda = params_.physics.lambda;
    const double gamma = params_.physics.mobility;
    // Zhang Eq 3.5-3.6: SAV scheme uses S₁ in the θ equation only.
    // NO Douglas-Dupont (1/η) in ψ equation. λ multiplies ε and 1/ε terms in ψ.

    // Effective mass coefficient: 1/dt + S1
    const double mass_coeff = 1.0 / dt + S1;

    bool use_convection = false;
    for (unsigned int d = 0; d < dim; ++d)
    {
        if (velocity_components[d] != nullptr &&
            velocity_components[d]->size() > 0)
        {
            use_convection = true;
            break;
        }
    }

    for (const auto& theta_cell : theta_dof_handler_.active_cell_iterators())
    {
        if (!theta_cell->is_locally_owned())
            continue;

        const typename dealii::DoFHandler<dim>::active_cell_iterator psi_cell(
            &triangulation_, theta_cell->level(), theta_cell->index(),
            &psi_dof_handler_);

        theta_fe_values.reinit(theta_cell);
        psi_fe_values.reinit(psi_cell);

        theta_fe_values.get_function_values(theta_old_relevant, theta_old_vals);

        if (use_convection)
        {
            const typename dealii::DoFHandler<dim>::active_cell_iterator vel_cell(
                &triangulation_, theta_cell->level(), theta_cell->index(),
                &u_dof_handler);
            vel_fe_values.reinit(vel_cell);

            for (unsigned int d = 0; d < dim; ++d)
            {
                if (velocity_components[d] != nullptr)
                    vel_fe_values.get_function_values(*velocity_components[d], u_vals[d]);
                else
                    std::fill(u_vals[d].begin(), u_vals[d].end(), 0.0);
            }
        }

        local_matrix = 0;
        local_rhs = 0;

        for (unsigned int q = 0; q < n_q; ++q)
        {
            const double JxW = theta_fe_values.JxW(q);
            const double theta_old_q = theta_old_vals[q];

            dealii::Tensor<1, dim> U;
            for (unsigned int d = 0; d < dim; ++d)
                U[d] = u_vals[d][q];

            // Nonlinearity: f(theta^{n-1})
            const double f_old = double_well_derivative(theta_old_q);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const double Lambda_i = theta_fe_values.shape_value(i, q);
                const auto& grad_Lambda_i = theta_fe_values.shape_grad(i, q);
                const double Upsilon_i = psi_fe_values.shape_value(i, q);
                const auto& grad_Upsilon_i = psi_fe_values.shape_grad(i, q);

                const unsigned int i_theta = i;
                const unsigned int i_psi = dofs_per_cell + i;

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    const double theta_j = theta_fe_values.shape_value(j, q);
                    const auto& grad_theta_j = theta_fe_values.shape_grad(j, q);
                    const double psi_j = psi_fe_values.shape_value(j, q);
                    const auto& grad_psi_j = psi_fe_values.shape_grad(j, q);

                    const unsigned int j_theta = j;
                    const unsigned int j_psi = dofs_per_cell + j;

                    // Eq 42a LHS: (1/tau + S1)(theta^n, Lambda)
                    local_matrix(i_theta, j_theta) +=
                        mass_coeff * theta_j * Lambda_i * JxW;

                    // Eq 42a LHS: -gamma(grad psi, grad Lambda)
                    local_matrix(i_theta, j_psi) -=
                        gamma * (grad_psi_j * grad_Lambda_i) * JxW;

                    // Eq 42b LHS: (psi, Upsilon)
                    local_matrix(i_psi, j_psi) +=
                        psi_j * Upsilon_i * JxW;

                    // Zhang Eq 3.6 LHS: λε(∇θ^{n+1}, ∇Υ)
                    local_matrix(i_psi, j_theta) +=
                        lambda * eps * (grad_theta_j * grad_Upsilon_i) * JxW;
                }

                // Eq 42a RHS: (1/tau + S1)(theta^{n-1}, Lambda)
                local_rhs(i_theta) +=
                    mass_coeff * theta_old_q * Lambda_i * JxW;

                // Eq 42a RHS: convection (U^{n-1} . grad Lambda) theta^{n-1}
                local_rhs(i_theta) +=
                    theta_old_q * (U * grad_Lambda_i) * JxW;

                // Zhang Eq 3.6 RHS: -(r/√(E₁+C₀)) · (λ/ε) · f(θ^{n-1}) · Υ
                local_rhs(i_psi) -=
                    sav_factor * (lambda / eps) * f_old * Upsilon_i * JxW;

                // MMS sources (if any)
                if (mms_source_theta_)
                {
                    const auto& x_q = theta_fe_values.quadrature_point(q);
                    local_rhs(i_theta) +=
                        mms_source_theta_(x_q, current_time) * Lambda_i * JxW;
                }
                if (mms_source_psi_)
                {
                    const auto& x_q = theta_fe_values.quadrature_point(q);
                    local_rhs(i_psi) +=
                        mms_source_psi_(x_q, current_time) * Upsilon_i * JxW;
                }
            }
        }

        theta_cell->get_dof_indices(theta_local_dofs);
        psi_cell->get_dof_indices(psi_local_dofs);

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            coupled_dof_indices[i] = theta_to_ch_map_[theta_local_dofs[i]];
            coupled_dof_indices[dofs_per_cell + i] = psi_to_ch_map_[psi_local_dofs[i]];
        }

        ch_constraints_.distribute_local_to_global(
            local_matrix, local_rhs,
            coupled_dof_indices,
            system_matrix_, system_rhs_);
    }

    system_matrix_.compress(dealii::VectorOperation::add);
    system_rhs_.compress(dealii::VectorOperation::add);

    auto t_end = std::chrono::high_resolution_clock::now();
    last_assemble_time_ =
        std::chrono::duration<double>(t_end - t_start).count();
}

// ============================================================================
// compute_bulk_energy — E1(theta) = (lambda/eps) * integral F(theta) dOmega
//
// F(theta) = (1/4)(theta^2 - 1)^2  (double-well potential)
// This is used for the SAV variable: r(t) = sqrt(E1(theta) + C0)
// ============================================================================
template <int dim>
double CahnHilliardSubsystem<dim>::compute_bulk_energy(
    const dealii::TrilinosWrappers::MPI::Vector& theta_relevant) const
{
    const unsigned int quad_degree = fe_.degree + 2;
    dealii::QGauss<dim> quadrature(quad_degree);

    dealii::FEValues<dim> fe_values(fe_, quadrature,
        dealii::update_values | dealii::update_JxW_values);

    const unsigned int n_q = quadrature.size();
    std::vector<double> theta_vals(n_q);

    const double lambda = params_.physics.lambda;
    const double eps = params_.physics.epsilon;
    const double coeff = lambda / eps;

    double local_energy = 0.0;

    for (const auto& cell : theta_dof_handler_.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);
        fe_values.get_function_values(theta_relevant, theta_vals);

        for (unsigned int q = 0; q < n_q; ++q)
        {
            const double F_q = double_well_potential(theta_vals[q]);
            local_energy += coeff * F_q * fe_values.JxW(q);
        }
    }

    double global_energy = 0.0;
    MPI_Allreduce(&local_energy, &global_energy, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_);

    return global_energy;
}

// ============================================================================
// compute_sav_inner_product
//
// Computes: (1/eps) * integral f(theta^n) * (theta^{n+1} - theta^n) dOmega
//
// Used for SAV update: r^{n+1} = r^n + result / (2 * sqrt(E1_old + C0))
// ============================================================================
template <int dim>
double CahnHilliardSubsystem<dim>::compute_sav_inner_product(
    const dealii::TrilinosWrappers::MPI::Vector& theta_new_relevant,
    const dealii::TrilinosWrappers::MPI::Vector& theta_old_relevant) const
{
    const unsigned int quad_degree = fe_.degree + 2;
    dealii::QGauss<dim> quadrature(quad_degree);

    dealii::FEValues<dim> fe_values(fe_, quadrature,
        dealii::update_values | dealii::update_JxW_values);

    const unsigned int n_q = quadrature.size();
    std::vector<double> theta_new_vals(n_q);
    std::vector<double> theta_old_vals(n_q);

    const double eps = params_.physics.epsilon;
    const double lambda = params_.physics.lambda;
    const double coeff = lambda / eps;  // δE₁/δθ = (λ/ε)f(θ)
    double local_sum = 0.0;

    for (const auto& cell : theta_dof_handler_.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);
        fe_values.get_function_values(theta_new_relevant, theta_new_vals);
        fe_values.get_function_values(theta_old_relevant, theta_old_vals);

        for (unsigned int q = 0; q < n_q; ++q)
        {
            const double f_old = double_well_derivative(theta_old_vals[q]);
            const double delta_theta = theta_new_vals[q] - theta_old_vals[q];
            local_sum += coeff * f_old * delta_theta * fe_values.JxW(q);
        }
    }

    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_);

    return global_sum;
}

// ============================================================================
// Explicit instantiations
// ============================================================================
template class CahnHilliardSubsystem<2>;
template class CahnHilliardSubsystem<3>;
