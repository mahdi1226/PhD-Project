// ============================================================================
// assembly/poisson_assembler.cc - Magnetostatic Poisson assembly
//
// CORRECTED to match Nochetto, Salgado & Tomas (2016):
// "A diffuse interface model for two-phase ferrofluid flows"
// Comput. Methods Appl. Mech. Engrg. 309 (2016) 497-531
//
// Solves: -∇·(μ(θ)∇φ) = 0  in Ω
//         φ = φ_dipole      on ∂Ω
//
// where μ(θ) = 1 + κ₀ H(θ/ε) is the phase-dependent permeability [Eq 17]
//
// KEY FIX: The dipole field provides Dirichlet BCs on the boundary.
// Inside the domain, the field is distorted by the ferrofluid (high μ)
// compared to water (μ≈1). This creates the attraction force.
//
// Previously, we were solving with φ=0 BCs which gave zero field inside!
// ============================================================================
#include "assembly/poisson_assembler.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>

#include <iostream>

// ============================================================================
// Dipole potential function class for deal.II interpolation
// ============================================================================
template <int dim>
class DipolePotentialFunction : public dealii::Function<dim>
{
public:
    DipolePotentialFunction(const NSCHParameters& params, double time)
        : dealii::Function<dim>(1)
        , params_(params)
        , time_(time)
    {}

    virtual double value(const dealii::Point<dim>& p,
                         const unsigned int component = 0) const override
    {
        (void)component;
        return compute_dipole_potential(p[0], p[1], params_, time_);
    }

private:
    const NSCHParameters& params_;
    double time_;
};

// ============================================================================
// Set up Dirichlet boundary conditions from dipole field
// ============================================================================
template <int dim>
void setup_poisson_dirichlet_bcs(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const NSCHParameters&          params,
    double                         current_time,
    dealii::AffineConstraints<double>& phi_constraints)
{
    phi_constraints.clear();

    // First add hanging node constraints (for AMR)
    dealii::DoFTools::make_hanging_node_constraints(phi_dof_handler, phi_constraints);

    // Apply Dirichlet BCs: φ = φ_dipole on all boundaries
    DipolePotentialFunction<dim> dipole_potential(params, current_time);

    // Apply to all boundary IDs (0 = all boundaries in deal.II default)
    dealii::VectorTools::interpolate_boundary_values(
        phi_dof_handler,
        0,  // boundary_id = 0 (all boundaries)
        dipole_potential,
        phi_constraints);

    // Also apply to other boundary IDs that might exist (1, 2, 3 for rectangular domain)
    for (unsigned int boundary_id = 1; boundary_id <= 3; ++boundary_id)
    {
        dealii::VectorTools::interpolate_boundary_values(
            phi_dof_handler,
            boundary_id,
            dipole_potential,
            phi_constraints);
    }

    phi_constraints.close();
}

// ============================================================================
// Assemble the magnetostatic Poisson system
// ============================================================================
template <int dim>
void assemble_poisson_system_scalar(
    const dealii::DoFHandler<dim>&           phi_dof_handler,
    const dealii::DoFHandler<dim>&           c_dof_handler,
    const dealii::Vector<double>&            c_solution,
    const NSCHParameters&                    params,
    dealii::SparseMatrix<double>&            phi_matrix,
    dealii::Vector<double>&                  phi_rhs,
    const dealii::AffineConstraints<double>& phi_constraints)
{
    phi_matrix = 0;
    phi_rhs    = 0;

    const auto& phi_fe = phi_dof_handler.get_fe();
    const auto& c_fe = c_dof_handler.get_fe();
    const unsigned int dofs_per_cell = phi_fe.n_dofs_per_cell();

    // Use same quadrature degree as phase field
    dealii::QGauss<dim> quadrature(params.fe_degree_potential + 2);
    const unsigned int n_q_points = quadrature.size();

    dealii::FEValues<dim> phi_fe_values(phi_fe, quadrature,
        dealii::update_values | dealii::update_gradients |
        dealii::update_quadrature_points | dealii::update_JxW_values);

    dealii::FEValues<dim> c_fe_values(c_fe, quadrature,
        dealii::update_values);

    dealii::FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    dealii::Vector<double>     local_rhs(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    // Storage for θ (concentration) values at quadrature points
    std::vector<double> theta_values(n_q_points);

    // Nochetto parameters
    const double kappa_0 = params.chi_m;   // κ₀: susceptibility
    const double epsilon = params.epsilon; // ε: interface thickness

    // Diagnostic tracking
    double max_mu = 0.0, min_mu = 1e10;

    // Iterate over cells (both DoFHandlers share the same mesh)
    auto phi_cell = phi_dof_handler.begin_active();
    auto c_cell = c_dof_handler.begin_active();

    for (; phi_cell != phi_dof_handler.end(); ++phi_cell, ++c_cell)
    {
        phi_fe_values.reinit(phi_cell);
        c_fe_values.reinit(c_cell);

        local_matrix = 0;
        local_rhs    = 0;  // RHS is zero (no interior source)

        // Get phase field values at quadrature points
        c_fe_values.get_function_values(c_solution, theta_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = phi_fe_values.JxW(q);

            // ================================================================
            // Compute permeability using Nochetto's sigmoid interpolation [Eq 17]
            //   μ(θ) = 1 + κ_θ = 1 + κ₀ H(θ/ε)
            // ================================================================
            const double theta = theta_values[q];
            const double mu = compute_permeability_nochetto(theta, epsilon, kappa_0);

            max_mu = std::max(max_mu, mu);
            min_mu = std::min(min_mu, mu);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const dealii::Tensor<1, dim> grad_psi_i =
                    phi_fe_values.shape_grad(i, q);

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    const dealii::Tensor<1, dim> grad_psi_j =
                        phi_fe_values.shape_grad(j, q);

                    // (μ(θ) ∇φ, ∇ψ)
                    local_matrix(i, j) += mu * (grad_psi_i * grad_psi_j) * JxW;
                }

                // RHS is zero (no source term, BCs provide the field)
            }
        }

        phi_cell->get_dof_indices(local_dof_indices);

        // Distribute with constraints (applies Dirichlet BCs)
        phi_constraints.distribute_local_to_global(
            local_matrix, local_rhs, local_dof_indices,
            phi_matrix, phi_rhs);
    }

    // Debug output (once per major time interval)
    static double last_report_time = -1.0;
    if (params.verbose && std::abs(params.current_time - last_report_time) > 0.05)
    {
        std::cout << "[POISSON] μ range: [" << min_mu << ", " << max_mu << "]" << std::endl;
        last_report_time = params.current_time;
    }
}

// ============================================================================
// Explicit instantiations
// ============================================================================
template void setup_poisson_dirichlet_bcs<2>(
    const dealii::DoFHandler<2>&,
    const NSCHParameters&,
    double,
    dealii::AffineConstraints<double>&);

template void assemble_poisson_system_scalar<2>(
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    const dealii::Vector<double>&,
    const NSCHParameters&,
    dealii::SparseMatrix<double>&,
    dealii::Vector<double>&,
    const dealii::AffineConstraints<double>&);