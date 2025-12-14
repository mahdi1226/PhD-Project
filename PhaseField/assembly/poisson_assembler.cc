// ============================================================================
// assembly/poisson_assembler.cc - Magnetostatic Poisson Assembly (CORRECTED)
//
// CORRECTED to match paper Eq. 42d, 63:
//   (∇φ, ∇χ) = (h_a - m, ∇χ)  with Neumann BC
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
// Setup Neumann constraints (pin one DoF to fix constant)
// ============================================================================
template <int dim>
void setup_poisson_neumann_constraints(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    dealii::AffineConstraints<double>& phi_constraints)
{
    // For pure Neumann problem, solution is unique up to a constant.
    // We fix this by constraining DoF 0 to zero.

    phi_constraints.clear();

    // First, add hanging node constraints
    dealii::DoFTools::make_hanging_node_constraints(phi_dof_handler, phi_constraints);

    // Pin DoF 0 to zero to fix the constant
    if (phi_dof_handler.n_dofs() > 0)
    {
        if (!phi_constraints.is_constrained(0))
        {
            phi_constraints.add_line(0);
            phi_constraints.set_inhomogeneity(0, 0.0);
        }
    }

    phi_constraints.close();
}

// ============================================================================
// Assemble the magnetostatic Poisson system (CORRECTED)
//
// Variational form (Eq. 42d):
//   (∇φ, ∇χ) = (h_a - m, ∇χ)
//
// LHS: Simple Laplacian stiffness (no μ(θ)!)
// RHS: (h_a - m) · ∇χ integrated over domain
// BC:  Pure Neumann (natural BC)
// ============================================================================
template <int dim>
void assemble_poisson_system(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::Vector<double>& theta_solution,
    const dealii::Vector<double>* mx_solution,
    const dealii::Vector<double>* my_solution,
    const Parameters& params,
    double current_time,
    dealii::SparseMatrix<double>& phi_matrix,
    dealii::Vector<double>& phi_rhs,
    const dealii::AffineConstraints<double>& phi_constraints,
    bool use_simplified)
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

    // Storage for values at quadrature points
    std::vector<double> theta_values(n_q_points);
    std::vector<double> mx_values(n_q_points);
    std::vector<double> my_values(n_q_points);

    // Parameters
    const double chi_0 = params.magnetization.chi_0;
    const double epsilon = params.ch.epsilon;

    // Check if we have magnetization data
    const bool have_magnetization = (mx_solution != nullptr) &&
                                     (my_solution != nullptr) &&
                                     !use_simplified;

    // Diagnostic tracking
    double max_rhs_contrib = 0.0;
    double max_h_a_magnitude = 0.0;

    // Iterate over cells
    auto phi_cell = phi_dof_handler.begin_active();
    auto theta_cell = theta_dof_handler.begin_active();

    for (; phi_cell != phi_dof_handler.end(); ++phi_cell, ++theta_cell)
    {
        phi_fe_values.reinit(phi_cell);
        theta_fe_values.reinit(theta_cell);

        local_matrix = 0;
        local_rhs = 0;

        // Get phase field values
        theta_fe_values.get_function_values(theta_solution, theta_values);

        // Get magnetization values if available
        if (have_magnetization)
        {
            theta_fe_values.get_function_values(*mx_solution, mx_values);
            theta_fe_values.get_function_values(*my_solution, my_values);
        }

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = phi_fe_values.JxW(q);
            const dealii::Point<dim>& x_q = phi_fe_values.quadrature_point(q);

            // Compute applied field h_a at this quadrature point
            dealii::Tensor<1, dim> h_a = compute_applied_field(x_q, params, current_time);
            max_h_a_magnitude = std::max(max_h_a_magnitude, h_a.norm());

            // Compute (h_a - m) for RHS
            dealii::Tensor<1, dim> h_a_minus_m;

            if (have_magnetization)
            {
                // Full model: use actual magnetization
                h_a_minus_m[0] = h_a[0] - mx_values[q];
                h_a_minus_m[1] = h_a[1] - my_values[q];
            }
            else if (use_simplified)
            {
                // Simplified model (Section 5): m ≈ 0, so h_a - m ≈ h_a
                h_a_minus_m[0] = h_a[0];
                h_a_minus_m[1] = h_a[1];
            }
            else
            {
                // Quasi-equilibrium: m ≈ χ_θ h_a (when we don't have explicit m)
                const double theta = theta_values[q];
                const double chi_theta = compute_susceptibility(theta, epsilon, chi_0);

                // m ≈ χ_θ h_a => h_a - m ≈ (1 - χ_θ) h_a
                h_a_minus_m[0] = h_a[0] * (1.0 - chi_theta);
                h_a_minus_m[1] = h_a[1] * (1.0 - chi_theta);
            }

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const auto& grad_chi_i = phi_fe_values.shape_grad(i, q);

                // RHS: (h_a - m, ∇χ)
                double rhs_contrib = (h_a_minus_m * grad_chi_i) * JxW;
                local_rhs(i) += rhs_contrib;
                max_rhs_contrib = std::max(max_rhs_contrib, std::abs(rhs_contrib));

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    const auto& grad_chi_j = phi_fe_values.shape_grad(j, q);

                    // LHS: (∇φ, ∇χ) - simple Laplacian, NO μ(θ)!
                    local_matrix(i, j) += (grad_chi_i * grad_chi_j) * JxW;
                }
            }
        }

        phi_cell->get_dof_indices(local_dof_indices);

        // Distribute with constraints
        phi_constraints.distribute_local_to_global(
            local_matrix, local_rhs, local_dof_indices,
            phi_matrix, phi_rhs);
    }

    // Debug output
    if (params.output.verbose)
    {
        static unsigned int call_count = 0;
        if (call_count % 100 == 0)
        {
            std::cout << "[Poisson] |h_a|_max = " << max_h_a_magnitude
                      << ", RHS contrib max = " << max_rhs_contrib
                      << ", RHS norm = " << phi_rhs.l2_norm() << "\n";
        }
        ++call_count;
    }
}

// ============================================================================
// BACKWARD COMPATIBILITY: Old 7-argument interface
// NOTE: This uses params.current_time - make sure to set it before calling!
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
    // WARNING: params.current_time must be set by caller!
    if (params.current_time == 0.0 && params.dipoles.ramp_time > 0.0)
    {
        // This is likely a bug - current_time not updated
        static bool warned = false;
        if (!warned)
        {
            std::cerr << "[Poisson] WARNING: current_time = 0. "
                      << "Did you forget to set params.current_time?\n";
            warned = true;
        }
    }

    // Call new implementation
    assemble_poisson_system<dim>(
        phi_dof_handler,
        theta_dof_handler,
        theta_solution,
        nullptr,              // no mx
        nullptr,              // no my
        params,
        params.current_time,
        phi_matrix,
        phi_rhs,
        phi_constraints,
        false);
}

// ============================================================================
// BACKWARD COMPATIBILITY: 8-argument interface with explicit time
// ============================================================================
template <int dim>
void assemble_poisson_system(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::Vector<double>& theta_solution,
    const Parameters& params,
    double current_time,
    dealii::SparseMatrix<double>& phi_matrix,
    dealii::Vector<double>& phi_rhs,
    const dealii::AffineConstraints<double>& phi_constraints)
{
    // Call new implementation with explicit time
    assemble_poisson_system<dim>(
        phi_dof_handler,
        theta_dof_handler,
        theta_solution,
        nullptr,              // no mx
        nullptr,              // no my
        params,
        current_time,         // use provided time
        phi_matrix,
        phi_rhs,
        phi_constraints,
        false);
}

// ============================================================================
// BACKWARD COMPATIBILITY: Deprecated Dirichlet BC setup (now a no-op)
// ============================================================================
template <int dim>
void apply_poisson_dirichlet_bcs(
    const dealii::DoFHandler<dim>& /*phi_dof_handler*/,
    const Parameters& /*params*/,
    double /*current_time*/,
    dealii::AffineConstraints<double>& /*phi_constraints*/)
{
    // NO-OP: We now use Neumann BC, not Dirichlet
    // The constant is fixed by pinning DoF 0 in setup_poisson_neumann_constraints()
    //
    // WARNING: If you see this function called, the code should be updated
    // to use setup_poisson_neumann_constraints() instead.
}

// ============================================================================
// Explicit instantiations
// ============================================================================
template void setup_poisson_neumann_constraints<2>(
    const dealii::DoFHandler<2>&,
    dealii::AffineConstraints<double>&);

template void assemble_poisson_system<2>(
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    const dealii::Vector<double>&,
    const dealii::Vector<double>*,
    const dealii::Vector<double>*,
    const Parameters&,
    double,
    dealii::SparseMatrix<double>&,
    dealii::Vector<double>&,
    const dealii::AffineConstraints<double>&,
    bool);

// Old interface
template void assemble_poisson_system<2>(
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    const dealii::Vector<double>&,
    const Parameters&,
    dealii::SparseMatrix<double>&,
    dealii::Vector<double>&,
    const dealii::AffineConstraints<double>&);

// 8-arg interface with explicit time
template void assemble_poisson_system<2>(
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    const dealii::Vector<double>&,
    const Parameters&,
    double,
    dealii::SparseMatrix<double>&,
    dealii::Vector<double>&,
    const dealii::AffineConstraints<double>&);

template void apply_poisson_dirichlet_bcs<2>(
    const dealii::DoFHandler<2>&,
    const Parameters&,
    double,
    dealii::AffineConstraints<double>&);