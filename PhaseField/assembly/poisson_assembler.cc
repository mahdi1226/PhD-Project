// ============================================================================
// assembly/poisson_assembler.cc - Magnetostatic Poisson Assembly (FIXED)
//
// PAPER EQUATION 42d:
//   (∇φ, ∇χ) = (h_a - M^k, ∇χ)  ∀χ ∈ X_h
//
// For QUASI-EQUILIBRIUM (M = χH = χ∇φ):
//   ((1 + χ(θ))∇φ, ∇χ) = (h_a, ∇χ)
//   (μ(θ)∇φ, ∇χ) = (h_a, ∇χ)  where μ(θ) = 1 + χ(θ)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "assembly/poisson_assembler.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/exceptions.h>
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
    phi_constraints.clear();

    // Hanging nodes (important for AMR)
    dealii::DoFTools::make_hanging_node_constraints(phi_dof_handler, phi_constraints);

    // Pure Neumann: fix constant by pinning first unconstrained DoF
    const unsigned int n_dofs = phi_dof_handler.n_dofs();
    Assert(n_dofs > 0, dealii::ExcMessage("phi_dof_handler has zero DoFs."));

    dealii::types::global_dof_index pinned = dealii::numbers::invalid_dof_index;
    for (dealii::types::global_dof_index i = 0; i < n_dofs; ++i)
    {
        if (!phi_constraints.is_constrained(i))
        {
            pinned = i;
            break;
        }
    }

    Assert(pinned != dealii::numbers::invalid_dof_index,
           dealii::ExcMessage("Could not find an unconstrained DoF to pin for Neumann Poisson."));

    phi_constraints.add_line(pinned);
    phi_constraints.set_inhomogeneity(pinned, 0.0);

    phi_constraints.close();
}

// ============================================================================
// Assemble the magnetostatic Poisson system (FULL MODEL with DG M)
//
// Paper Eq. 42d:
//   (∇φ, ∇χ) = (h_a - M^k, ∇χ)  ∀χ ∈ X_h
// ============================================================================
template <int dim>
void assemble_poisson_system(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::DoFHandler<dim>& M_dof_handler,
    const dealii::Vector<double>& mx_solution,
    const dealii::Vector<double>& my_solution,
    const Parameters& params,
    double current_time,
    dealii::SparseMatrix<double>& phi_matrix,
    dealii::Vector<double>& phi_rhs,
    const dealii::AffineConstraints<double>& phi_constraints)
{
    // Verify triangulations match
    Assert(&phi_dof_handler.get_triangulation() == &M_dof_handler.get_triangulation(),
           dealii::ExcMessage("phi and M DoFHandlers must share the same triangulation"));

    phi_matrix = 0;
    phi_rhs = 0;

    const auto& phi_fe = phi_dof_handler.get_fe();
    const auto& M_fe = M_dof_handler.get_fe();
    const unsigned int dofs_per_cell = phi_fe.n_dofs_per_cell();

    const unsigned int quad_degree = std::max(phi_fe.degree, M_fe.degree) + 2;
    dealii::QGauss<dim> quadrature(quad_degree);
    const unsigned int n_q_points = quadrature.size();

    dealii::FEValues<dim> phi_fe_values(phi_fe, quadrature,
        dealii::update_values | dealii::update_gradients |
        dealii::update_quadrature_points | dealii::update_JxW_values);

    dealii::FEValues<dim> M_fe_values(M_fe, quadrature,
        dealii::update_values);

    dealii::FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    dealii::Vector<double> local_rhs(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<double> mx_values(n_q_points);
    std::vector<double> my_values(n_q_points);

    double max_h_a_magnitude = 0.0;
    double max_M_magnitude = 0.0;

    auto phi_cell = phi_dof_handler.begin_active();
    auto M_cell = M_dof_handler.begin_active();

    for (; phi_cell != phi_dof_handler.end(); ++phi_cell, ++M_cell)
    {
        phi_fe_values.reinit(phi_cell);
        M_fe_values.reinit(M_cell);

        local_matrix = 0;
        local_rhs = 0;

        M_fe_values.get_function_values(mx_solution, mx_values);
        M_fe_values.get_function_values(my_solution, my_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = phi_fe_values.JxW(q);
            const dealii::Point<dim>& x_q = phi_fe_values.quadrature_point(q);

            dealii::Tensor<1, dim> h_a = compute_applied_field(x_q, params, current_time);
            max_h_a_magnitude = std::max(max_h_a_magnitude, h_a.norm());

            dealii::Tensor<1, dim> M;
            M[0] = mx_values[q];
            M[1] = my_values[q];
            max_M_magnitude = std::max(max_M_magnitude, M.norm());

            dealii::Tensor<1, dim> h_a_minus_M;
            h_a_minus_M[0] = h_a[0] - M[0];
            h_a_minus_M[1] = h_a[1] - M[1];

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const auto& grad_chi_i = phi_fe_values.shape_grad(i, q);

                // RHS: (h_a - M^k, ∇χ)
                local_rhs(i) += (h_a_minus_M * grad_chi_i) * JxW;

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    const auto& grad_chi_j = phi_fe_values.shape_grad(j, q);
                    // LHS: (∇φ, ∇χ)
                    local_matrix(i, j) += (grad_chi_i * grad_chi_j) * JxW;
                }
            }
        }

        phi_cell->get_dof_indices(local_dof_indices);
        phi_constraints.distribute_local_to_global(
            local_matrix, local_rhs, local_dof_indices,
            phi_matrix, phi_rhs);
    }

    if (params.output.verbose)
    {
        static unsigned int call_count = 0;
        if (call_count % 100 == 0)
        {
            std::cout << "[Poisson FULL] |h_a|_max = " << max_h_a_magnitude
                      << ", |M|_max = " << max_M_magnitude
                      << ", RHS norm = " << phi_rhs.l2_norm() << "\n";
        }
        ++call_count;
    }
}

// ============================================================================
// Assemble Poisson for QUASI-EQUILIBRIUM (M = χ(θ)H)
//
// For quasi-equilibrium, M = χ(θ)∇φ. Substituting into Eq. 42d:
//   (∇φ, ∇χ) = (h_a - χ(θ)∇φ, ∇χ)
//   (∇φ, ∇χ) + (χ(θ)∇φ, ∇χ) = (h_a, ∇χ)
//   ((1 + χ(θ))∇φ, ∇χ) = (h_a, ∇χ)
//   (μ(θ)∇φ, ∇χ) = (h_a, ∇χ)  where μ(θ) = 1 + χ(θ)
//
// This is the CORRECT quasi-equilibrium formulation!
// ============================================================================
template <int dim>
void assemble_poisson_system_quasi_equilibrium(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::Vector<double>& theta_solution,
    const Parameters& params,
    double current_time,
    dealii::SparseMatrix<double>& phi_matrix,
    dealii::Vector<double>& phi_rhs,
    const dealii::AffineConstraints<double>& phi_constraints)
{
    phi_matrix = 0;
    phi_rhs = 0;

    const auto& phi_fe = phi_dof_handler.get_fe();
    const auto& theta_fe = theta_dof_handler.get_fe();
    const unsigned int dofs_per_cell = phi_fe.n_dofs_per_cell();

    const unsigned int quad_degree = std::max(phi_fe.degree, theta_fe.degree) + 2;
    dealii::QGauss<dim> quadrature(quad_degree);
    const unsigned int n_q_points = quadrature.size();

    dealii::FEValues<dim> phi_fe_values(phi_fe, quadrature,
        dealii::update_values | dealii::update_gradients |
        dealii::update_quadrature_points | dealii::update_JxW_values);

    dealii::FEValues<dim> theta_fe_values(theta_fe, quadrature,
        dealii::update_values);

    dealii::FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    dealii::Vector<double> local_rhs(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<double> theta_values(n_q_points);

    const double epsilon = params.ch.epsilon;
    const double chi_0 = params.magnetization.chi_0;

    // Debug tracking
    double max_h_a_magnitude = 0.0;
    double max_mu = 0.0;
    double min_mu = 1e10;

    // Debug: Check dipole configuration once
    static bool printed_dipole_info = false;
    if (!printed_dipole_info && params.output.verbose)
    {
        std::cout << "[Poisson] Dipole configuration:\n";
        std::cout << "  Number of dipoles: " << params.dipoles.positions.size() << "\n";
        std::cout << "  Intensity max: " << params.dipoles.intensity_max << "\n";
        std::cout << "  Ramp time: " << params.dipoles.ramp_time << "\n";
        std::cout << "  Direction: (" << params.dipoles.direction[0] << ", "
                  << params.dipoles.direction[1] << ")\n";
        if (!params.dipoles.positions.empty())
        {
            std::cout << "  First dipole at: (" << params.dipoles.positions[0][0]
                      << ", " << params.dipoles.positions[0][1] << ")\n";
        }
        printed_dipole_info = true;
    }

    auto phi_cell = phi_dof_handler.begin_active();
    auto theta_cell = theta_dof_handler.begin_active();

    for (; phi_cell != phi_dof_handler.end(); ++phi_cell, ++theta_cell)
    {
        phi_fe_values.reinit(phi_cell);
        theta_fe_values.reinit(theta_cell);

        local_matrix = 0;
        local_rhs = 0;

        theta_fe_values.get_function_values(theta_solution, theta_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = phi_fe_values.JxW(q);
            const dealii::Point<dim>& x_q = phi_fe_values.quadrature_point(q);

            // Compute applied field h_a from dipoles
            dealii::Tensor<1, dim> h_a = compute_applied_field(x_q, params, current_time);
            max_h_a_magnitude = std::max(max_h_a_magnitude, h_a.norm());

            // Compute phase-dependent permeability μ(θ) = 1 + χ(θ)
            const double theta_q = theta_values[q];
            const double chi_theta = compute_susceptibility(theta_q, epsilon, chi_0);
            const double mu_theta = 1.0 + chi_theta;

            max_mu = std::max(max_mu, mu_theta);
            min_mu = std::min(min_mu, mu_theta);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const auto& grad_chi_i = phi_fe_values.shape_grad(i, q);

                // RHS: (h_a, ∇χ)
                local_rhs(i) += (h_a * grad_chi_i) * JxW;

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    const auto& grad_chi_j = phi_fe_values.shape_grad(j, q);

                    // LHS: (μ(θ)∇φ, ∇χ) - phase-dependent permeability!
                    local_matrix(i, j) += mu_theta * (grad_chi_i * grad_chi_j) * JxW;
                }
            }
        }

        phi_cell->get_dof_indices(local_dof_indices);
        phi_constraints.distribute_local_to_global(
            local_matrix, local_rhs, local_dof_indices,
            phi_matrix, phi_rhs);
    }

    if (params.output.verbose)
    {
        static unsigned int call_count = 0;
        if (call_count % 100 == 0)
        {
            std::cout << "[Poisson QUASI-EQ] t=" << current_time
                      << ", |h_a|_max=" << max_h_a_magnitude
                      << ", μ∈[" << min_mu << "," << max_mu << "]"
                      << ", RHS=" << phi_rhs.l2_norm() << "\n";
        }
        ++call_count;
    }

    // Check for zero RHS (indicates problem with dipoles)
    if (phi_rhs.l2_norm() < 1e-14 && current_time > 1e-10)
    {
        std::cerr << "[Poisson] WARNING: RHS is zero at t=" << current_time << "!\n";
        std::cerr << "  This usually means dipoles are not configured.\n";
        std::cerr << "  Number of dipoles: " << params.dipoles.positions.size() << "\n";
        std::cerr << "  Intensity max: " << params.dipoles.intensity_max << "\n";
        std::cerr << "  Ramp factor: " << current_time / params.dipoles.ramp_time << "\n";
    }
}

// ============================================================================
// Assemble the magnetostatic Poisson system (SIMPLIFIED MODEL - M = 0)
//
// Section 5 simplified model: ignore magnetization, just use h_a
//   (∇φ, ∇χ) = (h_a, ∇χ)
// ============================================================================
template <int dim>
void assemble_poisson_system_simplified(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const Parameters& params,
    double current_time,
    dealii::SparseMatrix<double>& phi_matrix,
    dealii::Vector<double>& phi_rhs,
    const dealii::AffineConstraints<double>& phi_constraints)
{
    phi_matrix = 0;
    phi_rhs = 0;

    const auto& phi_fe = phi_dof_handler.get_fe();
    const unsigned int dofs_per_cell = phi_fe.n_dofs_per_cell();

    dealii::QGauss<dim> quadrature(phi_fe.degree + 2);
    const unsigned int n_q_points = quadrature.size();

    dealii::FEValues<dim> phi_fe_values(phi_fe, quadrature,
        dealii::update_values | dealii::update_gradients |
        dealii::update_quadrature_points | dealii::update_JxW_values);

    dealii::FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    dealii::Vector<double> local_rhs(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    double max_h_a = 0.0;

    for (const auto& cell : phi_dof_handler.active_cell_iterators())
    {
        phi_fe_values.reinit(cell);

        local_matrix = 0;
        local_rhs = 0;

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = phi_fe_values.JxW(q);
            const dealii::Point<dim>& x_q = phi_fe_values.quadrature_point(q);

            dealii::Tensor<1, dim> h_a = compute_applied_field(x_q, params, current_time);
            max_h_a = std::max(max_h_a, h_a.norm());

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const auto& grad_chi_i = phi_fe_values.shape_grad(i, q);

                // RHS: (h_a, ∇χ)
                local_rhs(i) += (h_a * grad_chi_i) * JxW;

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    const auto& grad_chi_j = phi_fe_values.shape_grad(j, q);
                    // LHS: (∇φ, ∇χ) - simple Laplacian (μ = 1)
                    local_matrix(i, j) += (grad_chi_i * grad_chi_j) * JxW;
                }
            }
        }

        cell->get_dof_indices(local_dof_indices);
        phi_constraints.distribute_local_to_global(
            local_matrix, local_rhs, local_dof_indices,
            phi_matrix, phi_rhs);
    }

    if (params.output.verbose)
    {
        std::cout << "[Poisson SIMPLIFIED] t=" << current_time
                  << ", |h_a|_max=" << max_h_a
                  << ", RHS=" << phi_rhs.l2_norm() << "\n";
    }
}

// ============================================================================
// BACKWARD COMPATIBILITY: Old interface with optional M vectors
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
    if (use_simplified)
    {
        // Section 5 simplified model: M = 0, μ = 1
        assemble_poisson_system_simplified<dim>(
            phi_dof_handler,
            params,
            current_time,
            phi_matrix,
            phi_rhs,
            phi_constraints);
    }
    else if (mx_solution == nullptr || my_solution == nullptr)
    {
        // Quasi-equilibrium: M = χ(θ)H, use μ(θ) = 1 + χ(θ)
        assemble_poisson_system_quasi_equilibrium<dim>(
            phi_dof_handler,
            theta_dof_handler,
            theta_solution,
            params,
            current_time,
            phi_matrix,
            phi_rhs,
            phi_constraints);
    }
    else
    {
        // Full DG transport model - need M_dof_handler
        static bool warned = false;
        if (!warned)
        {
            std::cerr << "[Poisson] WARNING: Using theta_dof_handler as M_dof_handler proxy.\n"
                      << "  If M is on a different DG space, results may be incorrect.\n"
                      << "  Please migrate to the explicit M_dof_handler interface.\n";
            warned = true;
        }

        assemble_poisson_system<dim>(
            phi_dof_handler,
            theta_dof_handler,
            *mx_solution,
            *my_solution,
            params,
            current_time,
            phi_matrix,
            phi_rhs,
            phi_constraints);
    }
}

// ============================================================================
// BACKWARD COMPATIBILITY: 7-argument interface (no time parameter)
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
    // Use quasi-equilibrium model (NOT simplified!)
    assemble_poisson_system_quasi_equilibrium<dim>(
        phi_dof_handler,
        theta_dof_handler,
        theta_solution,
        params,
        params.current_time,
        phi_matrix,
        phi_rhs,
        phi_constraints);
}

// ============================================================================
// BACKWARD COMPATIBILITY: 8-argument interface with explicit time
//
// FIXED: Now uses quasi-equilibrium (NOT simplified!)
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
    // FIXED: Use quasi-equilibrium, NOT simplified!
    // The simplified model (M=0, μ=1) is only for Section 5 comparisons.
    // Physical ferrofluid simulations need the full quasi-equilibrium model.
    assemble_poisson_system_quasi_equilibrium<dim>(
        phi_dof_handler,
        theta_dof_handler,
        theta_solution,
        params,
        current_time,
        phi_matrix,
        phi_rhs,
        phi_constraints);
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
}

// ============================================================================
// Explicit instantiations
// ============================================================================

template void setup_poisson_neumann_constraints<2>(
    const dealii::DoFHandler<2>&,
    dealii::AffineConstraints<double>&);

// Full model (DG M)
template void assemble_poisson_system<2>(
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    const dealii::Vector<double>&,
    const dealii::Vector<double>&,
    const Parameters&,
    double,
    dealii::SparseMatrix<double>&,
    dealii::Vector<double>&,
    const dealii::AffineConstraints<double>&);

// Quasi-equilibrium
template void assemble_poisson_system_quasi_equilibrium<2>(
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    const dealii::Vector<double>&,
    const Parameters&,
    double,
    dealii::SparseMatrix<double>&,
    dealii::Vector<double>&,
    const dealii::AffineConstraints<double>&);

// Simplified
template void assemble_poisson_system_simplified<2>(
    const dealii::DoFHandler<2>&,
    const Parameters&,
    double,
    dealii::SparseMatrix<double>&,
    dealii::Vector<double>&,
    const dealii::AffineConstraints<double>&);

// Backward compatibility interfaces
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

template void assemble_poisson_system<2>(
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    const dealii::Vector<double>&,
    const Parameters&,
    dealii::SparseMatrix<double>&,
    dealii::Vector<double>&,
    const dealii::AffineConstraints<double>&);

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