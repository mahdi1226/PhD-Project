// ============================================================================
// cahn_hilliard/cahn_hilliard_assemble.cc - Monolithic Cell Assembly
//
// SPLIT CH (stabilized linear, backward Euler):
//
// Standard CH formulation for θ ∈ {-1, +1}, F(θ) = (θ²-1)²/16:
//
//   μ = -ε·Δθ + (1/ε)·F'(θ)     (chemical potential)
//   θ_t + ∇·(uθ) = γ·Δμ         (phase evolution)
//
// Stabilized linearization (Eyre-type):
//   μ^n = -ε·Δθ^n + (1/ε)[F'(θ^{n-1}) + S(θ^n − θ^{n-1})]
//   where S = max|F''(θ)| = 1/2  (conservative choice)
//
// Block structure:
//   Block(1,1) [phi test, phi trial]: (1/dt)(phi_j, v_i) + b_h(u; phi_j, v_i)
//   Block(1,2) [phi test, mu trial]:  gamma * (grad mu_j, grad v_i)
//   Block(2,1) [mu test, phi trial]:  -(S/ε)(phi_j, w_i) - ε(grad phi_j, grad w_i)
//   Block(2,2) [mu test, mu trial]:   (mu_j, w_i)
//
//   RHS_phi: (1/dt)(phi_old, v_i) [+ MMS source]
//   RHS_mu:  ((1/ε)(Psi'(phi_old) - S*phi_old), w_i) [+ MMS source]
//
//   b_h = skew-symmetric convection for energy neutrality
//
// Reference: Nochetto, Salgado & Tomas, CMAME 2016, arXiv:1601.06824
//            Zhang, He, Yang (2021), SIAM J. Sci. Comput.
// ============================================================================

#include "cahn_hilliard/cahn_hilliard.h"
#include "physics/material_properties.h"
#include "physics/skew_forms.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/fe/fe_values.h>

#include <cmath>

template <int dim>
void CahnHilliardSubsystem<dim>::assemble(
    const dealii::TrilinosWrappers::MPI::Vector& old_solution_relevant,
    double dt,
    const dealii::TrilinosWrappers::MPI::Vector& ux_relevant,
    const dealii::TrilinosWrappers::MPI::Vector& uy_relevant,
    const dealii::DoFHandler<dim>& vel_dof_handler)
{
    dealii::Timer timer;
    timer.start();

    const double epsilon = params_.cahn_hilliard_params.epsilon;
    const double gamma   = params_.cahn_hilliard_params.gamma;

    // Standard CH: μ = -ε·Δθ + (1/ε)·F'(θ)
    // Stabilized: μ^n includes S·(θ^n − θ^{n-1})/ε with S = max|F''| = 1/2
    const double grad_coeff = epsilon;                   // coeff of (∇θ, ∇w)
    const double pot_coeff  = 1.0 / epsilon;             // coeff of F'(θ)·w
    const double S_stab     = 0.5;                       // ≥ max|F''(θ)| = 1/2
    const double S_eff      = pot_coeff * S_stab;        // = 1/(2ε)
    const double mass_coeff = 1.0 / dt;

    const bool has_old = (old_solution_relevant.size() > 0);
    const bool has_vel = (ux_relevant.size() > 0);
    const bool has_mms = static_cast<bool>(mms_source_fn_);

    const double t_new = params_.time.dt; // MMS: overridden by caller via source fn

    const unsigned int degree = params_.fe.degree_cahn_hilliard;
    const dealii::QGauss<dim> quadrature(degree + 1);
    const unsigned int n_q = quadrature.size();
    const unsigned int dpc = fe_.n_dofs_per_cell();

    // FEValues for CH system (phi = component 0, mu = component 1)
    dealii::FEValues<dim> fe_values(fe_, quadrature,
        dealii::update_values | dealii::update_gradients |
        dealii::update_quadrature_points | dealii::update_JxW_values);

    const dealii::FEValuesExtractors::Scalar phi_extract(0);
    const dealii::FEValuesExtractors::Scalar mu_extract(1);

    // Cross-mesh FEValues for velocity
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

    // Buffers for old solution
    std::vector<double> phi_old_vals(n_q);

    // Buffers for velocity
    std::vector<double> ux_vals(n_q), uy_vals(n_q);
    std::vector<dealii::Tensor<1, dim>> grad_ux_vals(n_q), grad_uy_vals(n_q);

    // Zero global system
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

        // Get old phi values at quadrature points (component 0 only)
        if (has_old)
            fe_values[phi_extract].get_function_values(
                old_solution_relevant, phi_old_vals);

        // Get velocity at quadrature points (cross-mesh)
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

            // Old phi at this quadrature point
            const double phi_old_q = has_old ? phi_old_vals[q] : 0.0;

            // Velocity at quadrature point
            dealii::Tensor<1, dim> U_q;
            double div_U_q = 0.0;
            if (has_vel)
            {
                U_q[0] = ux_vals[q];
                U_q[1] = uy_vals[q];
                div_U_q = grad_ux_vals[q][0] + grad_uy_vals[q][1];
            }

            // Psi'(phi_old) = double_well_derivative(phi_old)
            const double psi_prime = double_well_derivative(phi_old_q);

            // MMS source at this quadrature point
            double f_phi = 0.0, f_mu = 0.0;
            if (has_mms)
            {
                const auto& x_q = fe_values.quadrature_point(q);
                auto [fp, fm] = mms_source_fn_(x_q, t_new, dt, phi_old_q);
                f_phi = fp;
                f_mu  = fm;
            }

            // ==============================================================
            // Assemble local matrix and RHS
            // ==============================================================
            for (unsigned int i = 0; i < dpc; ++i)
            {
                // Test functions (either phi-test or mu-test is nonzero)
                const double v_i     = fe_values[phi_extract].value(i, q);
                const auto& grad_v_i = fe_values[phi_extract].gradient(i, q);
                const double w_i     = fe_values[mu_extract].value(i, q);
                const auto& grad_w_i = fe_values[mu_extract].gradient(i, q);

                // ---- RHS ----

                // Phi equation RHS: (1/dt)(phi_old, v_i) + (f_phi, v_i)
                cell_rhs(i) += mass_coeff * phi_old_q * v_i * JxW;
                if (has_mms)
                    cell_rhs(i) += f_phi * v_i * JxW;

                // Mu equation RHS: ((1/ε)(Psi'(phi_old) - S*phi_old), w_i) + (f_mu, w_i)
                cell_rhs(i) += pot_coeff * (psi_prime - S_stab * phi_old_q)
                               * w_i * JxW;
                if (has_mms)
                    cell_rhs(i) += f_mu * w_i * JxW;

                // ---- Matrix ----
                for (unsigned int j = 0; j < dpc; ++j)
                {
                    // Trial functions
                    const double phi_j     = fe_values[phi_extract].value(j, q);
                    const auto& grad_phi_j = fe_values[phi_extract].gradient(j, q);
                    const double mu_j      = fe_values[mu_extract].value(j, q);
                    const auto& grad_mu_j  = fe_values[mu_extract].gradient(j, q);

                    double entry = 0.0;

                    // Block(1,1): (1/dt)(phi_j, v_i) + convection
                    entry += mass_coeff * phi_j * v_i;
                    if (has_vel)
                    {
                        entry += skew_angular_convection_scalar<dim>(
                            U_q, div_U_q, phi_j, grad_phi_j, v_i);
                    }

                    // Block(1,2): gamma * (grad mu_j, grad v_i)
                    entry += gamma * (grad_mu_j * grad_v_i);

                    // Block(2,1): -(S/ε)(phi_j, w_i) - ε(grad phi_j, grad w_i)
                    entry += -S_eff * phi_j * w_i;
                    entry += -grad_coeff * (grad_phi_j * grad_w_i);

                    // Block(2,2): (mu_j, w_i)
                    entry += mu_j * w_i;

                    cell_matrix(i, j) += entry * JxW;
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
template void CahnHilliardSubsystem<2>::assemble(
    const dealii::TrilinosWrappers::MPI::Vector&,
    double,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::DoFHandler<2>&);
template void CahnHilliardSubsystem<3>::assemble(
    const dealii::TrilinosWrappers::MPI::Vector&,
    double,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::DoFHandler<3>&);
