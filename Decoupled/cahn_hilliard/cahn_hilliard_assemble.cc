// ============================================================================
// cahn_hilliard/cahn_hilliard_assemble.cc - Coupled θ-ψ Assembly
//
// Reference: Zhang, He & Yang, SIAM J. Sci. Comput. 43(1), 2021, B167-B193
// Algorithm 3.1, Step 1: Cahn-Hilliard with stabilization
//
// Zhang uses Φ∈{0,1} with F(Φ) = (1/(4ε))Φ²(Φ-1)².  Our θ = 2Φ-1 ∈ {-1,+1}.
// Chemical potential: W = -(1/2)ψ.  Conversion gives:
//
//   Eq 3.9 → θ: (δθ/τ, Λ) - (U(θ+1), ∇Λ)
//              - [M + (δt/8)(θ+1)²](∇ψ, ∇Λ) = 0
//   Eq 3.10 → ψ: (ψ, Υ) + λε(∇θ, ∇Υ) + S(δθ, Υ) + (λ/ε)(f_θ, Υ) = 0
//
// where f_θ(θ) = (θ³−θ)/4,  F_θ(θ) = (θ²−1)²/16.
// γ = M = 2e-4 (mobility — direct from Zhang, NO factor-4 conversion).
// SUPG: (dt/2)·λ_code·(θ+1)² = (dt/8)(θ+1)² with λ_code = 0.25 = λ_Φ/4.
// S = λ_code/(4ε) = 12.5 (Zhang p.B182, with λ_code = λ_Φ/4 = 0.25).
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
    const double lambda = params_.physics.lambda;
    // Stabilization: Zhang p.B182 uses S₁ = λ_Φ·L/2 where L = max|f''_Φ| = 1/(2ε).
    // So S₁ = λ_Φ/(4ε).  In θ-space with ψ equation scaled by λ_code = λ_Φ/4:
    //   S_θ = λ_code/(4ε) = λ_Φ/(16ε).
    // But matching Zhang exactly: S = λ/(4ε) where λ = lambda (= 0.25 for Rosensweig).
    const double S_stab = lambda / (4.0 * eps);

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

            // Nonlinearity: f(θ^{n-1}) = ((θ^{n-1})³ - θ^{n-1})/4
            const double f_old = double_well_derivative(theta_old_q);

            // Zhang Eq 3.9 decoupled stabilization: (δt/2)(Φ∇W, Φ∇Λ)
            // In θ-space: (δt·λ/2)(θ^old+1)²  added to ψ diffusion coefficient.
            // This ensures energy stability of the CH-NS decoupled splitting.
            const double stab_coupling =
                0.5 * dt * lambda * (theta_old_q + 1.0) * (theta_old_q + 1.0);

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
                    // -(δt·λ/2)(θ^old+1)²(∇ψ^n, ∇Λ)  — decoupled stabilization
                    //   Zhang Eq 3.9: (δt/2)(Φ∇W, Φ∇Λ) → θ-space adds
                    //   (δt·λ/2)(θ+1)² to the ψ diffusion coefficient.
                    //   In bulk (θ=1): extra = 2δtλ = 5e-4, γ_eff ≈ 7e-4.
                    local_matrix(i_theta, j_psi) -=
                        (gamma + stab_coupling) * (grad_psi_j * grad_Lambda_i) * JxW;

                    // --------------------------------------------------------
                    // Eq 42b LHS
                    // --------------------------------------------------------
                    // (ψ^n, Υ)  — mass term
                    local_matrix(i_psi, j_psi) +=
                        psi_j * Upsilon_i * JxW;

                    // λε(∇θ^n, ∇Υ)  — Laplacian of θ (Zhang Eq 3.10)
                    local_matrix(i_psi, j_theta) +=
                        lambda * eps * (grad_theta_j * grad_Upsilon_i) * JxW;

                    // S(θ^n, Υ)  — stabilization (S = λ/(4ε), Zhang p.B182)
                    local_matrix(i_psi, j_theta) +=
                        S_stab * theta_j * Upsilon_i * JxW;
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
                // -(λ/ε)(f(θ^{n-1}), Υ)  (Zhang Eq 3.10)
                local_rhs(i_psi) -=
                    (lambda / eps) * f_old * Upsilon_i * JxW;

                // S(θ^{n-1}, Υ)  — stabilization (S = λ/(4ε))
                local_rhs(i_psi) +=
                    S_stab * theta_old_q * Upsilon_i * JxW;

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
// Explicit instantiations
// ============================================================================
template class CahnHilliardSubsystem<2>;
template class CahnHilliardSubsystem<3>;
