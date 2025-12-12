// ============================================================================
// assembly/ns_assembler.cc - Navier-Stokes System Assembler
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Equations 42e-f, p.505
// ============================================================================

#include "ns_assembler.h"
#include "output/logger.h"
#include "physics/material_properties.h"

#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/full_matrix.h>

template <int dim>
NSAssembler<dim>::NSAssembler(PhaseFieldProblem<dim>& problem)
    : problem_(problem)
{
    Logger::info("      NSAssembler constructed");
}

// ============================================================================
// assemble()
//
// Block system (Eq. 42e-f):
//   [A   B^T] [U]   [f]
//   [B   0  ] [P] = [0]
//
// A = (1/τ)M + C(U_old) + K(ν_θ)
// B = divergence operator
// f = (1/τ)M U_old + Kelvin + Capillary + Gravity
// ============================================================================
template <int dim>
void NSAssembler<dim>::assemble(double dt, double current_time)
{
    (void)current_time;  // Used for time-dependent forces if needed

    const double inv_tau = 1.0 / dt;
    const double nu_w    = problem_.params_.ns.nu_water;
    const double nu_f    = problem_.params_.ns.nu_ferro;
    const double mu_0    = problem_.params_.ns.mu_0;
    const double lambda  = problem_.params_.ch.lambda;
    const double epsilon = problem_.params_.ch.epsilon;

    const unsigned int n_u = problem_.ux_dof_handler_.n_dofs();  // Q2
    const unsigned int n_p = problem_.p_dof_handler_.n_dofs();   // Q1

    problem_.ns_matrix_ = 0;
    problem_.ns_rhs_    = 0;

    // Q2 for velocity
    const dealii::QGauss<dim> quadrature_Q2(problem_.fe_Q2_.degree + 1);
    dealii::FEValues<dim> fe_values_Q2(problem_.fe_Q2_, quadrature_Q2,
                                        dealii::update_values |
                                        dealii::update_gradients |
                                        dealii::update_quadrature_points |
                                        dealii::update_JxW_values);

    // Q1 for pressure
    const dealii::QGauss<dim> quadrature_Q1(problem_.fe_Q1_.degree + 1);
    dealii::FEValues<dim> fe_values_Q1(problem_.fe_Q1_, quadrature_Q1,
                                        dealii::update_values |
                                        dealii::update_gradients |
                                        dealii::update_JxW_values);

    const unsigned int dofs_per_cell_Q2 = problem_.fe_Q2_.n_dofs_per_cell();
    const unsigned int dofs_per_cell_Q1 = problem_.fe_Q1_.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_Q2.size();

    // Local matrices for velocity block
    dealii::FullMatrix<double> local_A(dofs_per_cell_Q2, dofs_per_cell_Q2);
    dealii::Vector<double> local_rhs_x(dofs_per_cell_Q2);
    dealii::Vector<double> local_rhs_y(dofs_per_cell_Q2);

    // DoF indices
    std::vector<dealii::types::global_dof_index> ux_dofs(dofs_per_cell_Q2);
    std::vector<dealii::types::global_dof_index> uy_dofs(dofs_per_cell_Q2);
    std::vector<dealii::types::global_dof_index> p_dofs(dofs_per_cell_Q1);
    std::vector<dealii::types::global_dof_index> theta_dofs(dofs_per_cell_Q2);
    std::vector<dealii::types::global_dof_index> psi_dofs(dofs_per_cell_Q2);
    std::vector<dealii::types::global_dof_index> mx_dofs(dofs_per_cell_Q2);
    std::vector<dealii::types::global_dof_index> my_dofs(dofs_per_cell_Q2);
    std::vector<dealii::types::global_dof_index> phi_dofs(dofs_per_cell_Q2);

    // Field values at quadrature points
    std::vector<double> theta_vals(n_q_points);
    std::vector<double> ux_old_vals(n_q_points), uy_old_vals(n_q_points);
    std::vector<double> mx_vals(n_q_points), my_vals(n_q_points);
    std::vector<dealii::Tensor<1, dim>> grad_psi_vals(n_q_points);
    std::vector<dealii::Tensor<1, dim>> grad_phi_vals(n_q_points);

    for (const auto& cell : problem_.ux_dof_handler_.active_cell_iterators())
    {
        fe_values_Q2.reinit(cell);
        local_A = 0;
        local_rhs_x = 0;
        local_rhs_y = 0;

        cell->get_dof_indices(ux_dofs);

        // Get corresponding cells for other fields
        typename dealii::DoFHandler<dim>::active_cell_iterator
            uy_cell(&problem_.triangulation_, cell->level(), cell->index(), &problem_.uy_dof_handler_);
        uy_cell->get_dof_indices(uy_dofs);

        typename dealii::DoFHandler<dim>::active_cell_iterator
            p_cell(&problem_.triangulation_, cell->level(), cell->index(), &problem_.p_dof_handler_);

        typename dealii::DoFHandler<dim>::active_cell_iterator
            theta_cell(&problem_.triangulation_, cell->level(), cell->index(), &problem_.theta_dof_handler_);
        theta_cell->get_dof_indices(theta_dofs);

        typename dealii::DoFHandler<dim>::active_cell_iterator
            psi_cell(&problem_.triangulation_, cell->level(), cell->index(), &problem_.psi_dof_handler_);
        psi_cell->get_dof_indices(psi_dofs);

        typename dealii::DoFHandler<dim>::active_cell_iterator
            mx_cell(&problem_.triangulation_, cell->level(), cell->index(), &problem_.mx_dof_handler_);
        mx_cell->get_dof_indices(mx_dofs);

        typename dealii::DoFHandler<dim>::active_cell_iterator
            my_cell(&problem_.triangulation_, cell->level(), cell->index(), &problem_.my_dof_handler_);
        my_cell->get_dof_indices(my_dofs);

        typename dealii::DoFHandler<dim>::active_cell_iterator
            phi_cell(&problem_.triangulation_, cell->level(), cell->index(), &problem_.phi_dof_handler_);
        phi_cell->get_dof_indices(phi_dofs);

        // Evaluate fields at quadrature points
        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            theta_vals[q] = 0;
            ux_old_vals[q] = 0;
            uy_old_vals[q] = 0;
            mx_vals[q] = 0;
            my_vals[q] = 0;
            grad_psi_vals[q] = 0;
            grad_phi_vals[q] = 0;

            for (unsigned int i = 0; i < dofs_per_cell_Q2; ++i)
            {
                const double phi_i = fe_values_Q2.shape_value(i, q);
                const auto grad_phi_i = fe_values_Q2.shape_grad(i, q);

                theta_vals[q] += problem_.theta_old_solution_[theta_dofs[i]] * phi_i;
                ux_old_vals[q] += problem_.ux_old_solution_[ux_dofs[i]] * phi_i;
                uy_old_vals[q] += problem_.uy_old_solution_[uy_dofs[i]] * phi_i;
                mx_vals[q] += problem_.mx_solution_[mx_dofs[i]] * phi_i;
                my_vals[q] += problem_.my_solution_[my_dofs[i]] * phi_i;
                grad_psi_vals[q] += problem_.psi_solution_[psi_dofs[i]] * grad_phi_i;
                grad_phi_vals[q] += problem_.phi_solution_[phi_dofs[i]] * grad_phi_i;
            }
        }

        // Quadrature loop
        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = fe_values_Q2.JxW(q);

            // Phase-dependent viscosity: ν_θ = ν_w + (ν_f - ν_w) H(θ/ε)
            const double H_theta = MaterialProperties::heaviside(theta_vals[q] / epsilon);
            const double nu_theta = nu_w + (nu_f - nu_w) * H_theta;

            // Convecting velocity
            dealii::Tensor<1, dim> U_old;
            U_old[0] = ux_old_vals[q];
            if constexpr (dim >= 2) U_old[1] = uy_old_vals[q];

            // h = ∇φ (induced field)
            dealii::Tensor<1, dim> h = grad_phi_vals[q];

            // Kelvin force: μ₀ (m·∇)h (simplified form)
            // Full form: μ₀ B_h^m(V, H, M) but we use explicit treatment
            dealii::Tensor<1, dim> kelvin_force;
            kelvin_force[0] = mu_0 * (mx_vals[q] * h[0] + my_vals[q] * h[1]);
            if constexpr (dim >= 2)
                kelvin_force[1] = mu_0 * (mx_vals[q] * h[0] + my_vals[q] * h[1]);

            // Capillary force: (λ/ε) θ ∇ψ
            dealii::Tensor<1, dim> capillary_force;
            capillary_force = (lambda / epsilon) * theta_vals[q] * grad_psi_vals[q];

            // Gravity: (1 + r H(θ/ε)) g ê_y
            dealii::Tensor<1, dim> gravity_force;
            if (problem_.params_.gravity.enabled)
            {
                const double r = problem_.params_.ns.r;
                const double g = problem_.params_.gravity.magnitude;
                gravity_force[0] = 0;
                if constexpr (dim >= 2)
                    gravity_force[1] = -(1.0 + r * H_theta) * g;  // Downward
            }

            for (unsigned int i = 0; i < dofs_per_cell_Q2; ++i)
            {
                const double phi_i = fe_values_Q2.shape_value(i, q);
                const auto grad_phi_i = fe_values_Q2.shape_grad(i, q);

                // RHS: (1/τ) u_old + forces
                local_rhs_x(i) += (inv_tau * ux_old_vals[q] * phi_i
                                  + kelvin_force[0] * phi_i
                                  + capillary_force[0] * phi_i
                                  + gravity_force[0] * phi_i) * JxW;

                local_rhs_y(i) += (inv_tau * uy_old_vals[q] * phi_i
                                  + kelvin_force[1] * phi_i
                                  + capillary_force[1] * phi_i
                                  + gravity_force[1] * phi_i) * JxW;

                for (unsigned int j = 0; j < dofs_per_cell_Q2; ++j)
                {
                    const double phi_j = fe_values_Q2.shape_value(j, q);
                    const auto grad_phi_j = fe_values_Q2.shape_grad(j, q);

                    // Mass: (1/τ) (u, v)
                    const double mass = inv_tau * phi_i * phi_j;

                    // Convection: (U_old · ∇u, v) - simplified, explicit
                    const double convection = (U_old * grad_phi_j) * phi_i;

                    // Viscous: ν_θ (∇u : ∇v)
                    const double viscous = nu_theta * (grad_phi_i * grad_phi_j);

                    local_A(i, j) += (mass + convection + viscous) * JxW;
                }
            }
        }

        // Distribute velocity block to global (diagonal blocks u_x-u_x, u_y-u_y)
        for (unsigned int i = 0; i < dofs_per_cell_Q2; ++i)
        {
            // u_x block: [0, n_u)
            problem_.ns_rhs_(ux_dofs[i]) += local_rhs_x(i);
            // u_y block: [n_u, 2*n_u)
            problem_.ns_rhs_(n_u + uy_dofs[i]) += local_rhs_y(i);

            for (unsigned int j = 0; j < dofs_per_cell_Q2; ++j)
            {
                problem_.ns_matrix_.add(ux_dofs[i], ux_dofs[j], local_A(i, j));
                problem_.ns_matrix_.add(n_u + uy_dofs[i], n_u + uy_dofs[j], local_A(i, j));
            }
        }

        // Pressure-velocity coupling (B and B^T blocks)
        // Need Q1 FE values on same cell
        dealii::FEValues<dim> fe_values_Q1_cell(problem_.fe_Q1_, quadrature_Q2,
                                                 dealii::update_values);
        fe_values_Q1_cell.reinit(p_cell);
        p_cell->get_dof_indices(p_dofs);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = fe_values_Q2.JxW(q);

            for (unsigned int i = 0; i < dofs_per_cell_Q2; ++i)
            {
                const auto grad_phi_i = fe_values_Q2.shape_grad(i, q);

                for (unsigned int j = 0; j < dofs_per_cell_Q1; ++j)
                {
                    const double psi_j = fe_values_Q1_cell.shape_value(j, q);

                    // B^T: -(p, div v) → -p * ∂v_x/∂x, -p * ∂v_y/∂y
                    // B: (q, div u) → q * ∂u_x/∂x, q * ∂u_y/∂y

                    const unsigned int p_global = 2 * n_u + p_dofs[j];

                    // B^T block (velocity equations, pressure column)
                    problem_.ns_matrix_.add(ux_dofs[i], p_global, -grad_phi_i[0] * psi_j * JxW);
                    problem_.ns_matrix_.add(n_u + uy_dofs[i], p_global, -grad_phi_i[1] * psi_j * JxW);

                    // B block (continuity equation, velocity columns)
                    problem_.ns_matrix_.add(p_global, ux_dofs[i], grad_phi_i[0] * psi_j * JxW);
                    problem_.ns_matrix_.add(p_global, n_u + uy_dofs[i], grad_phi_i[1] * psi_j * JxW);
                }
            }
        }
    }

    // Apply velocity boundary conditions (no-slip already in constraints)
    // Pin one pressure DoF for uniqueness
    const unsigned int p_pin = 2 * n_u;  // First pressure DoF
    problem_.ns_matrix_.set(p_pin, p_pin, problem_.ns_matrix_(p_pin, p_pin) + 1e10);
    problem_.ns_rhs_(p_pin) = 0;
}

template class NSAssembler<2>;
// template class NSAssembler<3>;