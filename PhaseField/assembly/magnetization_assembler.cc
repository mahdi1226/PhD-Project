// ============================================================================
// assembly/magnetization_assembler.cc - Magnetization Equation Assembler
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Equation 42c, p.505
// ============================================================================

#include "magnetization_assembler.h"
#include "output/logger.h"
#include "physics/applied_field.h"
#include "physics/material_properties.h"

#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/numerics/vector_tools.h>

template <int dim>
MagnetizationAssembler<dim>::MagnetizationAssembler(PhaseFieldProblem<dim>& problem)
    : problem_(problem)
{
    Logger::info("      MagnetizationAssembler constructed");
}

// ============================================================================
// assemble()
//
// Equilibrium approximation (T → 0):
//   m = χ_θ h
//
// ASSUMPTION: Use h_a (applied field) instead of total field h = h_a + ∇φ
// BASIS: Avoids circular dependency (m needs h, Poisson needs m).
//        Paper Section 5 (p.510): when χ₀ << 1, can use h ≈ h_a.
// QUESTION: Should we iterate Mag/Poisson to get self-consistent m and φ?
//
// Susceptibility (Eq. 17, p.501):
//   χ_θ = χ₀ H(θ/ε)
// ============================================================================
template <int dim>
void MagnetizationAssembler<dim>::assemble(double /*dt*/, double current_time)
{
    const double chi_0   = problem_.params_.magnetization.chi_0;
    const double epsilon = problem_.params_.ch.epsilon;

    // Create applied field calculator
    AppliedField<dim> h_a(problem_.params_.dipoles.positions,
                          problem_.params_.dipoles.direction,
                          problem_.params_.dipoles.intensity_max,
                          problem_.params_.dipoles.ramp_time);

    const auto& fe = problem_.fe_Q2_;
    const dealii::QGauss<dim> quadrature(fe.degree + 1);

    dealii::FEValues<dim> fe_values(fe, quadrature,
                                     dealii::update_values |
                                     dealii::update_quadrature_points |
                                     dealii::update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature.size();

    // We'll do L2 projection: M m = rhs where M is mass matrix
    // For simplicity, use lumped mass (diagonal) approximation
    // m_i = Σ_q (χ_θ h_a)_q φ_i(q) JxW / Σ_q φ_i(q)² JxW

    // Actually simpler: loop over cells, accumulate weighted values
    dealii::Vector<double> mx_numerator(problem_.mx_solution_.size());
    dealii::Vector<double> my_numerator(problem_.my_solution_.size());
    dealii::Vector<double> mass_diagonal(problem_.mx_solution_.size());

    std::vector<dealii::types::global_dof_index> theta_dofs(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> m_dofs(dofs_per_cell);
    std::vector<double> theta_vals(n_q_points);

    for (const auto& cell : problem_.theta_dof_handler_.active_cell_iterators())
    {
        fe_values.reinit(cell);
        cell->get_dof_indices(theta_dofs);

        // Get m DoFs (same mesh)
        typename dealii::DoFHandler<dim>::active_cell_iterator
            m_cell(&problem_.triangulation_, cell->level(), cell->index(),
                   &problem_.mx_dof_handler_);
        m_cell->get_dof_indices(m_dofs);

        // Evaluate θ at quadrature points
        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            theta_vals[q] = 0;
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                theta_vals[q] += problem_.theta_solution_[theta_dofs[i]]
                               * fe_values.shape_value(i, q);
        }

        // Quadrature loop
        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = fe_values.JxW(q);
            const auto& x_q = fe_values.quadrature_point(q);

            // χ_θ = χ₀ H(θ/ε)
            const double chi_theta = chi_0 * MaterialProperties::heaviside(theta_vals[q] / epsilon);

            // h_a at quadrature point
            dealii::Tensor<1, dim> h_a_q = h_a.compute_field(x_q, current_time);

            // m = χ_θ h_a
            const double mx_q = chi_theta * h_a_q[0];
            const double my_q = (dim >= 2) ? chi_theta * h_a_q[1] : 0.0;

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const double phi_i = fe_values.shape_value(i, q);
                const double weight = phi_i * JxW;

                mx_numerator[m_dofs[i]] += mx_q * weight;
                my_numerator[m_dofs[i]] += my_q * weight;
                mass_diagonal[m_dofs[i]] += phi_i * weight;
            }
        }
    }

    // Divide to get nodal values (lumped mass projection)
    for (unsigned int i = 0; i < problem_.mx_solution_.size(); ++i)
    {
        if (mass_diagonal[i] > 1e-14)
        {
            problem_.mx_solution_[i] = mx_numerator[i] / mass_diagonal[i];
            problem_.my_solution_[i] = my_numerator[i] / mass_diagonal[i];
        }
    }

    problem_.mx_constraints_.distribute(problem_.mx_solution_);
    problem_.my_constraints_.distribute(problem_.my_solution_);
}

template class MagnetizationAssembler<2>;
// template class MagnetizationAssembler<3>;  // Parameters uses Point<2>