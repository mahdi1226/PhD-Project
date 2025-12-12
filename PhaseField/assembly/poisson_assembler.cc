// ============================================================================
// assembly/poisson_assembler.cc - Poisson (Magnetostatics) Assembler
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Equation 42d, p.505
// ============================================================================

#include "poisson_assembler.h"
#include "output/logger.h"
#include "physics/applied_field.h"

#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/full_matrix.h>

template <int dim>
PoissonAssembler<dim>::PoissonAssembler(PhaseFieldProblem<dim>& problem)
    : problem_(problem)
{
    Logger::info("      PoissonAssembler constructed");
}

// ============================================================================
// assemble()
//
// Discrete scheme (Eq. 42d, p.505):
//   (∇Φ^k, ∇X) = (h_a^k - M^k, ∇X)
//
// LHS: Laplacian stiffness matrix  ∫_Ω ∇φ · ∇χ dx
// RHS: Source term                 ∫_Ω (h_a - m) · ∇χ dx
//
// Pure Neumann problem → pin first DoF to zero
// ============================================================================
template <int dim>
void PoissonAssembler<dim>::assemble(double current_time)
{
    // Create applied field calculator
    AppliedField<dim> h_a(problem_.params_.dipoles.positions,
                          problem_.params_.dipoles.direction,
                          problem_.params_.dipoles.intensity_max,
                          problem_.params_.dipoles.ramp_time);

    problem_.poisson_matrix_ = 0;
    problem_.poisson_rhs_    = 0;

    const auto& fe = problem_.fe_Q2_;
    const dealii::QGauss<dim> quadrature(fe.degree + 1);

    dealii::FEValues<dim> fe_values(fe, quadrature,
                                     dealii::update_values |
                                     dealii::update_gradients |
                                     dealii::update_quadrature_points |
                                     dealii::update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature.size();

    dealii::FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    dealii::Vector<double>     local_rhs(dofs_per_cell);

    std::vector<dealii::types::global_dof_index> phi_dofs(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> mx_dofs(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> my_dofs(dofs_per_cell);

    std::vector<double> mx_vals(n_q_points);
    std::vector<double> my_vals(n_q_points);

    for (const auto& cell : problem_.phi_dof_handler_.active_cell_iterators())
    {
        fe_values.reinit(cell);
        local_matrix = 0;
        local_rhs    = 0;

        cell->get_dof_indices(phi_dofs);

        // Get m values at quadrature points
        typename dealii::DoFHandler<dim>::active_cell_iterator
            mx_cell(&problem_.triangulation_, cell->level(), cell->index(),
                    &problem_.mx_dof_handler_);
        mx_cell->get_dof_indices(mx_dofs);

        typename dealii::DoFHandler<dim>::active_cell_iterator
            my_cell(&problem_.triangulation_, cell->level(), cell->index(),
                    &problem_.my_dof_handler_);
        my_cell->get_dof_indices(my_dofs);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            mx_vals[q] = 0;
            my_vals[q] = 0;
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const double phi = fe_values.shape_value(i, q);
                mx_vals[q] += problem_.mx_solution_[mx_dofs[i]] * phi;
                my_vals[q] += problem_.my_solution_[my_dofs[i]] * phi;
            }
        }

        // Quadrature loop
        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = fe_values.JxW(q);
            const auto& x_q = fe_values.quadrature_point(q);

            // Compute h_a at quadrature point
            dealii::Tensor<1, dim> h_a_q = h_a.compute_field(x_q, current_time);

            // Magnetization at quadrature point
            dealii::Tensor<1, dim> m_q;
            m_q[0] = mx_vals[q];
            if constexpr (dim >= 2) m_q[1] = my_vals[q];

            // Source: h_a - m
            dealii::Tensor<1, dim> source = h_a_q - m_q;

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const auto grad_phi_i = fe_values.shape_grad(i, q);

                // RHS: (h_a - m) · ∇χ
                local_rhs(i) += (source * grad_phi_i) * JxW;

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    const auto grad_phi_j = fe_values.shape_grad(j, q);

                    // LHS: ∇φ · ∇χ
                    local_matrix(i, j) += (grad_phi_i * grad_phi_j) * JxW;
                }
            }
        }

        // Distribute to global
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            problem_.poisson_rhs_(phi_dofs[i]) += local_rhs(i);
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
                problem_.poisson_matrix_.add(phi_dofs[i], phi_dofs[j], local_matrix(i, j));
        }
    }

    // Pin first DoF to handle pure Neumann (φ determined up to constant)
    const double large = 1e10;
    problem_.poisson_matrix_.set(0, 0, problem_.poisson_matrix_(0, 0) + large);
    problem_.poisson_rhs_(0) = 0;
}

template class PoissonAssembler<2>;
//template class PoissonAssembler<3>;