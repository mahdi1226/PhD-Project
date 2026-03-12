// ============================================================================
// magnetization/magnetization_assemble.cc — CG Cell Assembly
//
// Private methods:
//   assemble_system_internal()  — unified matrix+RHS or RHS-only
//   initialize_preconditioner() — ILU from current matrix
//
// CELL INTEGRALS (Zhang Eq 3.14/3.17):
//   LHS:  (1/τ + 1/τ_M)(M^k, Z) + (U·∇M^k)·Z + ½(∇·U)(M^k·Z)
//   RHS:  (1/τ_M)(χ(θ)H^k, Z) + (1/τ)(M^{n-1}, Z)
//         + ½(∇×U × M^{n-1}, Z)     (spin-vorticity coupling, explicit)
//         + β[M(M·H) - H|M|²]·Z     (Landau-Lifshitz, explicit)
//         + (F_mms, Z)               (MMS source, testing only)
//
// CG: No face integrals needed (continuity enforced by FE space).
//
// β-TERM (Zhang-He-Yang 2021, 2D identity):
//   β M×(M×H) = β[M(M·H) - H|M|²]
//   Explicit treatment using M^{n-1}, H^k on RHS.
//
// Reference: Zhang, He & Yang, SIAM J. Sci. Comput. 43(1) (2021) B167-B193
// ============================================================================

#include "magnetization/magnetization.h"
#include "physics/skew_forms.h"
#include "physics/material_properties.h"
#include "physics/applied_field.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/full_matrix.h>

using namespace dealii;

// ============================================================================
// assemble_system_internal() — unified assembly
//
// matrix_and_rhs = true:  build matrix + RHS (call once per timestep)
// matrix_and_rhs = false: build RHS only (Picard iteration, matrix frozen)
//
// explicit_transport:
//   true  = Step 5 (Zhang Eq 3.14): mass-only matrix, explicit transport on RHS.
//           Transport uses FULL divergence: -[(U·∇)M_old + (∇·U)M_old]·Z
//           (coefficient 1, not ½ as in skew form). No implicit transport.
//   false = Step 6 (Zhang Eq 3.17): full CG transport matrix (implicit).
//           Standard bilinear form b(U, M, Z) on LHS.
// ============================================================================
template <int dim>
void MagnetizationSubsystem<dim>::assemble_system_internal(
    const TrilinosWrappers::MPI::Vector& Mx_old_relevant,
    const TrilinosWrappers::MPI::Vector& My_old_relevant,
    const TrilinosWrappers::MPI::Vector& phi_relevant,
    const DoFHandler<dim>&               phi_dof_handler,
    const TrilinosWrappers::MPI::Vector& theta_relevant,
    const DoFHandler<dim>&               theta_dof_handler,
    const TrilinosWrappers::MPI::Vector& ux_relevant,
    const TrilinosWrappers::MPI::Vector& uy_relevant,
    const DoFHandler<dim>&               u_dof_handler,
    double dt,
    double current_time,
    bool matrix_and_rhs,
    bool explicit_transport)
{
    Timer timer;
    timer.start();

    const double t_old = current_time - dt;

    const FiniteElement<dim>& fe_M     = dof_handler_.get_fe();
    const FiniteElement<dim>& fe_phi   = phi_dof_handler.get_fe();
    const FiniteElement<dim>& fe_theta = theta_dof_handler.get_fe();

    const unsigned int dofs_per_cell = fe_M.n_dofs_per_cell();

    // Quadrature
    QGauss<dim> quadrature_cell(fe_M.degree + 2);
    const unsigned int n_q_cell = quadrature_cell.size();

    // ========================================================================
    // FEValues — cell integrals
    // ========================================================================
    FEValues<dim> fe_values_M(fe_M, quadrature_cell,
        update_values | update_gradients |
        update_quadrature_points | update_JxW_values);
    FEValues<dim> fe_values_phi(fe_phi, quadrature_cell,
        update_gradients | update_quadrature_points);
    FEValues<dim> fe_values_theta(fe_theta, quadrature_cell,
        update_values);

    // Pre-allocated scratch for cell field values
    std::vector<Tensor<1, dim>> grad_phi_vals(n_q_cell);
    std::vector<double>         theta_vals(n_q_cell);
    std::vector<double>         Mx_old_vals(n_q_cell);
    std::vector<double>         My_old_vals(n_q_cell);

    // Gradients of M^{n-1} — needed for explicit transport (Step 5, Eq 3.14)
    std::vector<Tensor<1, dim>> grad_Mx_old(n_q_cell);
    std::vector<Tensor<1, dim>> grad_My_old(n_q_cell);

    // Velocity FEValues + scratch (only needed for full assembly)
    std::unique_ptr<FEValues<dim>> fe_values_U_ptr;
    std::vector<double>            Ux_vals(n_q_cell);
    std::vector<double>            Uy_vals(n_q_cell);
    std::vector<Tensor<1, dim>>    grad_Ux(n_q_cell);
    std::vector<Tensor<1, dim>>    grad_Uy(n_q_cell);

    if (matrix_and_rhs)
    {
        const FiniteElement<dim>& fe_U = u_dof_handler.get_fe();
        fe_values_U_ptr = std::make_unique<FEValues<dim>>(
            fe_U, quadrature_cell,
            update_values | update_gradients);
    }

    // ========================================================================
    // Local contributions
    // ========================================================================
    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     local_rhs_x(dofs_per_cell);
    Vector<double>     local_rhs_y(dofs_per_cell);
    Vector<double>     local_sv_x(dofs_per_cell);  // spin-vorticity cache
    Vector<double>     local_sv_y(dofs_per_cell);
    Vector<double>     local_et_x(dofs_per_cell);  // explicit transport cache
    Vector<double>     local_et_y(dofs_per_cell);
    std::vector<types::global_dof_index> local_dofs(dofs_per_cell);

    // ========================================================================
    // Coefficients
    // ========================================================================
    const double tau       = dt;
    const double tau_M_val = params_.physics.tau_M;
    const double mass_coeff  = (tau_M_val > 0.0)
        ? (1.0 / tau + 1.0 / tau_M_val) : 1.0 / tau;
    const double relax_coeff = (tau_M_val > 0.0)
        ? 1.0 / tau_M_val : 0.0;
    const double old_coeff   = 1.0 / tau;

    const bool beta_active = params_.physics.enable_beta_term
                          && (std::abs(params_.physics.beta) > 1e-14);
    const double beta_val  = params_.physics.beta;

    const bool mms_active = static_cast<bool>(mms_source_);

    // ========================================================================
    // Zero outputs
    // ========================================================================
    if (matrix_and_rhs)
    {
        system_matrix_ = 0;
        spin_vort_rhs_x_ = 0;
        spin_vort_rhs_y_ = 0;
        if (explicit_transport)
        {
            explicit_transport_rhs_x_ = 0;
            explicit_transport_rhs_y_ = 0;
        }
    }
    Mx_rhs_ = 0;
    My_rhs_ = 0;

    // ========================================================================
    // CELL LOOP — synchronized iteration across DoFHandlers
    // ========================================================================
    auto cell_M     = dof_handler_.begin_active();
    auto cell_phi   = phi_dof_handler.begin_active();
    auto cell_theta = theta_dof_handler.begin_active();
    auto cell_U     = u_dof_handler.begin_active();

    for (; cell_M != dof_handler_.end();
         ++cell_M, ++cell_phi, ++cell_theta, ++cell_U)
    {
        if (!cell_M->is_locally_owned())
            continue;

        fe_values_M.reinit(cell_M);
        fe_values_phi.reinit(cell_phi);
        fe_values_theta.reinit(cell_theta);

        if (matrix_and_rhs)
        {
            fe_values_U_ptr->reinit(cell_U);
            fe_values_U_ptr->get_function_values(ux_relevant, Ux_vals);
            fe_values_U_ptr->get_function_values(uy_relevant, Uy_vals);
            fe_values_U_ptr->get_function_gradients(ux_relevant, grad_Ux);
            fe_values_U_ptr->get_function_gradients(uy_relevant, grad_Uy);
        }

        // Always-needed field values
        fe_values_phi.get_function_gradients(phi_relevant, grad_phi_vals);
        fe_values_theta.get_function_values(theta_relevant, theta_vals);
        fe_values_M.get_function_values(Mx_old_relevant, Mx_old_vals);
        fe_values_M.get_function_values(My_old_relevant, My_old_vals);

        // Evaluate M_old gradients for explicit transport (Step 5)
        if (explicit_transport && matrix_and_rhs)
        {
            fe_values_M.get_function_gradients(Mx_old_relevant, grad_Mx_old);
            fe_values_M.get_function_gradients(My_old_relevant, grad_My_old);
        }

        if (matrix_and_rhs)
        {
            local_matrix = 0;
            local_sv_x = 0;
            local_sv_y = 0;
            if (explicit_transport)
            {
                local_et_x = 0;
                local_et_y = 0;
            }
        }
        local_rhs_x = 0;
        local_rhs_y = 0;

        // ====================================================================
        // Cell quadrature: mass + transport (LHS) + relaxation + β (RHS)
        // ====================================================================
        for (unsigned int q = 0; q < n_q_cell; ++q)
        {
            const double JxW = fe_values_M.JxW(q);
            const Point<dim>& x_q = fe_values_M.quadrature_point(q);

            // H = ∇φ (total field — Poisson encodes h_a via natural BCs)
            Tensor<1, dim> H;
            if (params_.use_reduced_magnetic_field)
            {
                H = compute_applied_field<dim>(x_q, params_, current_time);
            }
            else
            {
                for (unsigned int d = 0; d < dim; ++d)
                    H[d] = grad_phi_vals[q][d];
            }

            // χ(θ)
            const double chi_theta = susceptibility(
                theta_vals[q], params_.physics.chi_0);

            // Velocity + divergence (only for matrix assembly)
            Tensor<1, dim> U_q;
            double div_U_q = 0.0;
            if (matrix_and_rhs)
            {
                U_q[0]  = Ux_vals[q];
                if constexpr (dim > 1)
                    U_q[1] = Uy_vals[q];
                div_U_q = grad_Ux[q][0];
                if constexpr (dim > 1)
                    div_U_q += grad_Uy[q][1];
            }

            // ================================================================
            // Spin-vorticity: +½(∇×U × M^{n-1}, Z)  (explicit, RHS)
            // ================================================================
            double spin_vort_x = 0.0;
            double spin_vort_y = 0.0;
            if (matrix_and_rhs)
            {
                const double omega_z = grad_Uy[q][0] - grad_Ux[q][1];
                const double Mx = Mx_old_vals[q];
                const double My = (dim > 1) ? My_old_vals[q] : 0.0;
                spin_vort_x = 0.5 * omega_z * (-My);
                if constexpr (dim > 1)
                    spin_vort_y = 0.5 * omega_z * Mx;
            }

            // ================================================================
            // β-term: Eq 2.8 has +β m×(m×h) on LHS → RHS = -β m×(m×h)
            //   m×(m×h) = m(m·h) - h|m|²   (vector triple product)
            //   So RHS contribution = -β[m(m·h) - h|m|²]
            //                       = β[h|m|² - m(m·h)]
            // Equivalently: +β(m×h, m×n)  (Zhang discrete form, Eq 3.14)
            // ================================================================
            Tensor<1, dim> beta_term;
            if (beta_active)
            {
                const double Mx = Mx_old_vals[q];
                const double My = (dim > 1) ? My_old_vals[q] : 0.0;
                const double MdotH = Mx * H[0]
                    + ((dim > 1) ? My * H[1] : 0.0);
                const double M_sq = Mx * Mx + My * My;

                // -β[m(m·h) - h|m|²] = β[h|m|² - m(m·h)]
                beta_term[0] = beta_val * (H[0] * M_sq - Mx * MdotH);
                if constexpr (dim > 1)
                    beta_term[1] = beta_val * (H[1] * M_sq - My * MdotH);
            }

            // ================================================================
            // MMS source (full assembly only, when callback is set)
            // ================================================================
            Tensor<1, dim> F_mms;
            if (mms_active)
            {
                F_mms = mms_source_(x_q, current_time, t_old,
                                     tau_M_val, chi_theta,
                                     H, U_q, div_U_q);
            }

            // ================================================================
            // Explicit transport: -[(U·∇)M_old + (∇·U)M_old]
            //
            // Zhang Eq 3.14: coefficient of (∇·U) is 1 (FULL divergence),
            // NOT ½ as in the CG skew form. Required for energy stability.
            // ================================================================
            double et_x = 0.0, et_y = 0.0;
            if (explicit_transport && matrix_and_rhs)
            {
                et_x = -(U_q * grad_Mx_old[q] + div_U_q * Mx_old_vals[q]);
                if constexpr (dim > 1)
                    et_y = -(U_q * grad_My_old[q] + div_U_q * My_old_vals[q]);
            }

            // ================================================================
            // Assemble cell contributions
            // ================================================================
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const double Z_i = fe_values_M.shape_value(i, q);
                const Tensor<1, dim>& grad_Z_i = fe_values_M.shape_grad(i, q);

                // -- Matrix: mass + optional CG cell transport --
                if (matrix_and_rhs)
                {
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                        const double M_j = fe_values_M.shape_value(j, q);

                        // Mass: (1/τ + 1/τ_M)(M_j, Z_i)
                        double val = mass_coeff * M_j * Z_i;

                        // CG skew transport (Step 6 only):
                        // b(U, M, Z) = (U·∇M)Z + ½(∇·U)(MZ)
                        if (!explicit_transport)
                        {
                            val += -skew_magnetic_cell_value_scalar<dim>(
                                U_q, div_U_q, Z_i, grad_Z_i, M_j);
                        }

                        local_matrix(i, j) += val * JxW;
                    }
                }

                // -- RHS: relaxation + old-time + β-term + MMS --
                double rhs_x_val = relax_coeff * chi_theta * H[0]
                    + old_coeff * Mx_old_vals[q];
                double rhs_y_val = (dim > 1)
                    ? relax_coeff * chi_theta * H[1]
                      + old_coeff * My_old_vals[q]
                    : 0.0;

                // Explicit transport on RHS (Step 5 only)
                if (explicit_transport && matrix_and_rhs)
                {
                    rhs_x_val += et_x;
                    local_et_x(i) += et_x * Z_i * JxW;
                    if constexpr (dim > 1)
                    {
                        rhs_y_val += et_y;
                        local_et_y(i) += et_y * Z_i * JxW;
                    }
                }

                // Spin-vorticity on RHS
                if (matrix_and_rhs)
                {
                    rhs_x_val += spin_vort_x;
                    local_sv_x(i) += spin_vort_x * Z_i * JxW;
                    if constexpr (dim > 1)
                    {
                        rhs_y_val += spin_vort_y;
                        local_sv_y(i) += spin_vort_y * Z_i * JxW;
                    }
                }

                if (beta_active)
                {
                    rhs_x_val += beta_term[0];
                    if constexpr (dim > 1)
                        rhs_y_val += beta_term[1];
                }

                if (mms_active)
                {
                    rhs_x_val += F_mms[0];
                    if constexpr (dim > 1)
                        rhs_y_val += F_mms[1];
                }

                local_rhs_x(i) += rhs_x_val * Z_i * JxW;
                local_rhs_y(i) += rhs_y_val * Z_i * JxW;
            }
        }

        // ====================================================================
        // Distribute cell contributions via constraints
        // ====================================================================
        cell_M->get_dof_indices(local_dofs);

        if (matrix_and_rhs)
        {
            // Matrix + Mx RHS together (constraints modify both consistently)
            constraints_.distribute_local_to_global(
                local_matrix, local_rhs_x, local_dofs,
                system_matrix_, Mx_rhs_);

            // My RHS separately (same constraints, RHS-only distribution)
            constraints_.distribute_local_to_global(
                local_rhs_y, local_dofs, My_rhs_);

            // Distribute spin-vorticity cache (unconstrained — just add)
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                spin_vort_rhs_x_[local_dofs[i]] += local_sv_x(i);
                spin_vort_rhs_y_[local_dofs[i]] += local_sv_y(i);
            }

            // Distribute explicit transport cache (Step 5 only)
            if (explicit_transport)
            {
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    explicit_transport_rhs_x_[local_dofs[i]] += local_et_x(i);
                    explicit_transport_rhs_y_[local_dofs[i]] += local_et_y(i);
                }
            }
        }
        else
        {
            // RHS-only mode: distribute both RHS vectors via constraints
            constraints_.distribute_local_to_global(
                local_rhs_x, local_dofs, Mx_rhs_);
            constraints_.distribute_local_to_global(
                local_rhs_y, local_dofs, My_rhs_);
        }
    }

    // ========================================================================
    // Compress (synchronize MPI contributions)
    // ========================================================================
    if (matrix_and_rhs)
    {
        system_matrix_.compress(VectorOperation::add);
        spin_vort_rhs_x_.compress(VectorOperation::add);
        spin_vort_rhs_y_.compress(VectorOperation::add);
        if (explicit_transport)
        {
            explicit_transport_rhs_x_.compress(VectorOperation::add);
            explicit_transport_rhs_y_.compress(VectorOperation::add);
        }
    }
    Mx_rhs_.compress(VectorOperation::add);
    My_rhs_.compress(VectorOperation::add);

    // ========================================================================
    // In RHS-only mode, add cached spin-vorticity and transport contributions.
    // ========================================================================
    if (!matrix_and_rhs)
    {
        Mx_rhs_.add(1.0, spin_vort_rhs_x_);
        My_rhs_.add(1.0, spin_vort_rhs_y_);
        if (explicit_transport)
        {
            Mx_rhs_.add(1.0, explicit_transport_rhs_x_);
            My_rhs_.add(1.0, explicit_transport_rhs_y_);
        }
    }

    timer.stop();
    last_assemble_time_ = timer.wall_time();

    pcout_ << "[Magnetization] Assembly ("
           << (matrix_and_rhs ? "matrix+RHS" : "RHS-only")
           << (explicit_transport ? ", Step5-explicit" : ", Step6-implicit")
           << "): " << last_assemble_time_ << " s" << std::endl;
}

// ============================================================================
// initialize_preconditioner() — ILU from current matrix
// ============================================================================
template <int dim>
void MagnetizationSubsystem<dim>::initialize_preconditioner()
{
    TrilinosWrappers::PreconditionILU::AdditionalData ilu_data;
    ilu_data.ilu_fill = 1;
    ilu_data.ilu_atol = 1e-12;
    ilu_data.ilu_rtol = 1.0;

    ilu_preconditioner_.initialize(system_matrix_, ilu_data);
    preconditioner_initialized_ = true;

    pcout_ << "[Magnetization] ILU preconditioner initialized" << std::endl;
}

// ============================================================================
// Explicit instantiations
// ============================================================================
template void MagnetizationSubsystem<2>::assemble_system_internal(
    const TrilinosWrappers::MPI::Vector&,
    const TrilinosWrappers::MPI::Vector&,
    const TrilinosWrappers::MPI::Vector&,
    const DoFHandler<2>&,
    const TrilinosWrappers::MPI::Vector&,
    const DoFHandler<2>&,
    const TrilinosWrappers::MPI::Vector&,
    const TrilinosWrappers::MPI::Vector&,
    const DoFHandler<2>&,
    double, double, bool, bool);

template void MagnetizationSubsystem<2>::initialize_preconditioner();

template void MagnetizationSubsystem<3>::assemble_system_internal(
    const TrilinosWrappers::MPI::Vector&,
    const TrilinosWrappers::MPI::Vector&,
    const TrilinosWrappers::MPI::Vector&,
    const DoFHandler<3>&,
    const TrilinosWrappers::MPI::Vector&,
    const DoFHandler<3>&,
    const TrilinosWrappers::MPI::Vector&,
    const TrilinosWrappers::MPI::Vector&,
    const DoFHandler<3>&,
    double, double, bool, bool);

template void MagnetizationSubsystem<3>::initialize_preconditioner();
