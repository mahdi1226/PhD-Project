// ============================================================================
// assembly/poisson_assembler.cc - Magnetostatic Poisson Assembly Implementation
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "assembly/poisson_assembler.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/numerics/vector_tools.h>

#include <iostream>

// ============================================================================
// DipolePotentialFunction - Wrapper for deal.II boundary interpolation
// ============================================================================
template <int dim>
class DipolePotentialFunction : public dealii::Function<dim>
{
public:
    DipolePotentialFunction(const Parameters& params, double time)
        : dealii::Function<dim>(1)
        , params_(params)
        , time_(time)
    {}

    virtual double value(const dealii::Point<dim>& p,
                         const unsigned int component = 0) const override
    {
        (void)component;
        return compute_dipole_potential<dim>(p, params_, time_);
    }

private:
    const Parameters& params_;
    double time_;
};

// ============================================================================
// Apply Dirichlet BCs for Poisson from dipole field
// ============================================================================
template <int dim>
void apply_poisson_dirichlet_bcs(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const Parameters& params,
    double current_time,
    dealii::AffineConstraints<double>& phi_constraints)
{
    // Note: phi_constraints should already have hanging node constraints
    // We add Dirichlet BCs on top

    DipolePotentialFunction<dim> dipole_potential(params, current_time);

    // Apply to all boundary IDs (0, 1, 2, 3 for rectangular domain)
    for (unsigned int boundary_id = 0; boundary_id <= 3; ++boundary_id)
    {
        dealii::VectorTools::interpolate_boundary_values(
            phi_dof_handler,
            boundary_id,
            dipole_potential,
            phi_constraints);
    }
}

// ============================================================================
// Assemble the magnetostatic Poisson system
// ============================================================================
template <int dim>
void assemble_poisson_system(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::Vector<double>& theta_solution,
    const Parameters& params,
    dealii::SparseMatrix<double>& phi_matrix,
    dealii::Vector<double>& phi_rhs,
    const dealii::AffineConstraints<double>& phi_constraints)
{
    phi_matrix = 0;
    phi_rhs = 0;

    const auto& phi_fe = phi_dof_handler.get_fe();
    const auto& theta_fe = theta_dof_handler.get_fe();
    const unsigned int dofs_per_cell = phi_fe.n_dofs_per_cell();

    // Quadrature
    dealii::QGauss<dim> quadrature(params.fe.degree_potential + 2);
    const unsigned int n_q_points = quadrature.size();

    dealii::FEValues<dim> phi_fe_values(phi_fe, quadrature,
        dealii::update_values | dealii::update_gradients |
        dealii::update_quadrature_points | dealii::update_JxW_values);

    dealii::FEValues<dim> theta_fe_values(theta_fe, quadrature,
        dealii::update_values);

    dealii::FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    dealii::Vector<double> local_rhs(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    // Storage for θ values at quadrature points
    std::vector<double> theta_values(n_q_points);

    // Parameters
    const double kappa_0 = params.magnetization.chi_0;  // Susceptibility κ₀
    const double epsilon = params.ch.epsilon;           // Interface thickness

    // Diagnostic tracking
    double max_mu = 0.0, min_mu = 1e10;

    // Iterate over cells (both DoFHandlers share the same mesh)
    auto phi_cell = phi_dof_handler.begin_active();
    auto theta_cell = theta_dof_handler.begin_active();

    for (; phi_cell != phi_dof_handler.end(); ++phi_cell, ++theta_cell)
    {
        phi_fe_values.reinit(phi_cell);
        theta_fe_values.reinit(theta_cell);

        local_matrix = 0;
        local_rhs = 0;  // RHS is zero (no interior source)

        // Get phase field values at quadrature points
        theta_fe_values.get_function_values(theta_solution, theta_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = phi_fe_values.JxW(q);

            // Compute permeability using sigmoid interpolation [Eq. 17]
            const double theta = theta_values[q];
            const double mu = compute_permeability(theta, epsilon, kappa_0);

            max_mu = std::max(max_mu, mu);
            min_mu = std::min(min_mu, mu);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const auto& grad_psi_i = phi_fe_values.shape_grad(i, q);

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    const auto& grad_psi_j = phi_fe_values.shape_grad(j, q);

                    // (μ(θ)∇φ, ∇ψ)
                    local_matrix(i, j) += mu * (grad_psi_i * grad_psi_j) * JxW;
                }
                // RHS is zero (BCs provide the field)
            }
        }

        phi_cell->get_dof_indices(local_dof_indices);

        // Distribute with constraints (applies Dirichlet BCs)
        phi_constraints.distribute_local_to_global(
            local_matrix, local_rhs, local_dof_indices,
            phi_matrix, phi_rhs);
    }

    // Debug output
    if (params.output.verbose)
    {
        static unsigned int call_count = 0;
        if (call_count % 100 == 0)  // Print every 100 calls
        {
            std::cout << "[Poisson] μ range: [" << min_mu << ", " << max_mu << "]\n";
        }
        ++call_count;
    }
}

// ============================================================================
// Explicit instantiations
// ============================================================================
template void apply_poisson_dirichlet_bcs<2>(
    const dealii::DoFHandler<2>&,
    const Parameters&,
    double,
    dealii::AffineConstraints<double>&);

template void assemble_poisson_system<2>(
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    const dealii::Vector<double>&,
    const Parameters&,
    dealii::SparseMatrix<double>&,
    dealii::Vector<double>&,
    const dealii::AffineConstraints<double>&);