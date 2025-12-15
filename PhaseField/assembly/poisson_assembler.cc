// ============================================================================
// assembly/poisson_assembler.cc - Magnetostatic Poisson Assembly (CORRECTED)
//
// PAPER EQUATION 42d:
//   (∇φ, ∇χ) = (h_a - M^k, ∇χ)  ∀χ ∈ X_h
//
// CRITICAL FIX: M is on a DG DoFHandler (FE_DGP), separate from φ (CG).
//               We must create FEValues from M's DoFHandler to properly
//               extract magnetization values at quadrature points.
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
//
// NOTE: The Neumann BC ∂φ/∂n = (h_a - m)·n is a NATURAL BC, enforced
//       implicitly by the weak form. This function only handles:
//       - Hanging node constraints (for AMR)
//       - Pinning one DoF to fix the nullspace of pure Neumann
// ============================================================================
template <int dim>
void setup_poisson_neumann_constraints(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    dealii::AffineConstraints<double>& phi_constraints)
{
    using namespace dealii;

    phi_constraints.clear();

    // Hanging nodes (important for AMR)
    DoFTools::make_hanging_node_constraints(phi_dof_handler, phi_constraints);

    // Pure Neumann: fix constant by pinning first unconstrained DoF
    const unsigned int n_dofs = phi_dof_handler.n_dofs();
    Assert(n_dofs > 0, ExcMessage("phi_dof_handler has zero DoFs."));

    types::global_dof_index pinned = numbers::invalid_dof_index;
    for (types::global_dof_index i = 0; i < n_dofs; ++i)
    {
        if (!phi_constraints.is_constrained(i))
        {
            pinned = i;
            break;
        }
    }

    Assert(pinned != numbers::invalid_dof_index,
           ExcMessage("Could not find an unconstrained DoF to pin for Neumann Poisson."));

    phi_constraints.add_line(pinned);
    phi_constraints.set_inhomogeneity(pinned, 0.0);

    phi_constraints.close();
}

// ============================================================================
// Assemble the magnetostatic Poisson system (FULL MODEL)
//
// Paper Eq. 42d:
//   (∇φ, ∇χ) = (h_a - M^k, ∇χ)  ∀χ ∈ X_h
//
// LHS: Simple Laplacian (∇φ, ∇χ)
// RHS: (h_a - M^k) · ∇χ
// BC:  Pure Neumann (natural)
//
// CRITICAL: Uses M_dof_handler to create FEValues for extracting M values.
//           M is DG (FE_DGP), φ is CG - they have different DoFHandlers!
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
    using namespace dealii;

    // Verify triangulations match
    Assert(&phi_dof_handler.get_triangulation() == &M_dof_handler.get_triangulation(),
           ExcMessage("phi and M DoFHandlers must share the same triangulation"));

    phi_matrix = 0;
    phi_rhs = 0;

    const auto& phi_fe = phi_dof_handler.get_fe();
    const auto& M_fe = M_dof_handler.get_fe();
    const unsigned int dofs_per_cell = phi_fe.n_dofs_per_cell();

    // Quadrature - use sufficient order for both spaces
    const unsigned int quad_degree = std::max(phi_fe.degree, M_fe.degree) + 2;
    QGauss<dim> quadrature(quad_degree);
    const unsigned int n_q_points = quadrature.size();

    // FEValues for φ (CG) - for shape functions and gradients
    FEValues<dim> phi_fe_values(phi_fe, quadrature,
        update_values | update_gradients |
        update_quadrature_points | update_JxW_values);

    // FEValues for M (DG) - for extracting magnetization values
    // CRITICAL: This must be built from M's DoFHandler, not θ's!
    FEValues<dim> M_fe_values(M_fe, quadrature,
        update_values);

    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> local_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // Storage for M values at quadrature points
    std::vector<double> mx_values(n_q_points);
    std::vector<double> my_values(n_q_points);

    // Diagnostic tracking
    double max_rhs_contrib = 0.0;
    double max_h_a_magnitude = 0.0;
    double max_M_magnitude = 0.0;

    // Iterate over cells - must iterate both DoFHandlers in sync
    auto phi_cell = phi_dof_handler.begin_active();
    auto M_cell = M_dof_handler.begin_active();

    for (; phi_cell != phi_dof_handler.end(); ++phi_cell, ++M_cell)
    {
        phi_fe_values.reinit(phi_cell);
        M_fe_values.reinit(M_cell);

        local_matrix = 0;
        local_rhs = 0;

        // Get magnetization values using M's FEValues (CORRECT!)
        M_fe_values.get_function_values(mx_solution, mx_values);
        M_fe_values.get_function_values(my_solution, my_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = phi_fe_values.JxW(q);
            const Point<dim>& x_q = phi_fe_values.quadrature_point(q);

            // Compute applied field h_a at this quadrature point
            Tensor<1, dim> h_a = compute_applied_field(x_q, params, current_time);
            max_h_a_magnitude = std::max(max_h_a_magnitude, h_a.norm());

            // Get M^k from transport solution
            Tensor<1, dim> M;
            M[0] = mx_values[q];
            M[1] = my_values[q];
            max_M_magnitude = std::max(max_M_magnitude, M.norm());

            // RHS source: (h_a - M^k)
            Tensor<1, dim> h_a_minus_M;
            h_a_minus_M[0] = h_a[0] - M[0];
            h_a_minus_M[1] = h_a[1] - M[1];

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const auto& grad_chi_i = phi_fe_values.shape_grad(i, q);

                // RHS: (h_a - M^k, ∇χ)
                double rhs_contrib = (h_a_minus_M * grad_chi_i) * JxW;
                local_rhs(i) += rhs_contrib;
                max_rhs_contrib = std::max(max_rhs_contrib, std::abs(rhs_contrib));

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    const auto& grad_chi_j = phi_fe_values.shape_grad(j, q);

                    // LHS: (∇φ, ∇χ) - simple Laplacian
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
            std::cout << "[Poisson FULL] |h_a|_max = " << max_h_a_magnitude
                      << ", |M|_max = " << max_M_magnitude
                      << ", RHS norm = " << phi_rhs.l2_norm() << "\n";
        }
        ++call_count;
    }
}

// ============================================================================
// Assemble the magnetostatic Poisson system (SIMPLIFIED MODEL)
//
// Simplified model: M = 0, so RHS = (h_a, ∇χ)
//
// This is for comparison/debugging. In the true simplified model (Section 5),
// you wouldn't solve Poisson at all - you'd just set H = h_a.
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
    using namespace dealii;

    phi_matrix = 0;
    phi_rhs = 0;

    const auto& phi_fe = phi_dof_handler.get_fe();
    const unsigned int dofs_per_cell = phi_fe.n_dofs_per_cell();

    QGauss<dim> quadrature(phi_fe.degree + 2);
    const unsigned int n_q_points = quadrature.size();

    FEValues<dim> phi_fe_values(phi_fe, quadrature,
        update_values | update_gradients |
        update_quadrature_points | update_JxW_values);

    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> local_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto& cell : phi_dof_handler.active_cell_iterators())
    {
        phi_fe_values.reinit(cell);

        local_matrix = 0;
        local_rhs = 0;

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = phi_fe_values.JxW(q);
            const Point<dim>& x_q = phi_fe_values.quadrature_point(q);

            // Compute applied field h_a (M = 0 in simplified model)
            Tensor<1, dim> h_a = compute_applied_field(x_q, params, current_time);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const auto& grad_chi_i = phi_fe_values.shape_grad(i, q);

                // RHS: (h_a, ∇χ)  [since M = 0]
                local_rhs(i) += (h_a * grad_chi_i) * JxW;

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    const auto& grad_chi_j = phi_fe_values.shape_grad(j, q);

                    // LHS: (∇φ, ∇χ)
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
        std::cout << "[Poisson SIMPLIFIED] RHS norm = " << phi_rhs.l2_norm() << "\n";
    }
}

// ============================================================================
// BACKWARD COMPATIBILITY: Old interface with theta_dof_handler
//
// WARNING: This is INCORRECT if M is on a different (DG) space than θ!
// The old code used theta_fe_values to extract M values, which fails
// when M and θ have different DoFHandlers.
//
// This wrapper attempts to work around the issue by assuming M and θ
// share the same triangulation and using theta_dof_handler as a proxy.
// THIS IS A BUG - please migrate to the new interface!
// ============================================================================
template <int dim>
void assemble_poisson_system(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::Vector<double>& /*theta_solution*/,
    const dealii::Vector<double>* mx_solution,
    const dealii::Vector<double>* my_solution,
    const Parameters& params,
    double current_time,
    dealii::SparseMatrix<double>& phi_matrix,
    dealii::Vector<double>& phi_rhs,
    const dealii::AffineConstraints<double>& phi_constraints,
    bool use_simplified)
{
    if (use_simplified || mx_solution == nullptr || my_solution == nullptr)
    {
        // Simplified model: ignore M, use h_a only
        assemble_poisson_system_simplified<dim>(
            phi_dof_handler,
            params,
            current_time,
            phi_matrix,
            phi_rhs,
            phi_constraints);
    }
    else
    {
        // WARNING: This is a compatibility shim that assumes M is stored
        // on theta_dof_handler, which is INCORRECT for DG magnetization!
        //
        // The correct approach is to pass M_dof_handler explicitly.
        // This code path will produce wrong results for DG M.
        static bool warned = false;
        if (!warned)
        {
            std::cerr << "[Poisson] WARNING: Using deprecated interface.\n"
                      << "  If M is on a DG DoFHandler (different from theta),\n"
                      << "  the results will be INCORRECT!\n"
                      << "  Please migrate to: assemble_poisson_system(phi_dof, M_dof, mx, my, ...)\n";
            warned = true;
        }

        // Fall back to using theta_dof_handler as proxy for M
        // This is WRONG for DG M, but matches the old (buggy) behavior
        assemble_poisson_system<dim>(
            phi_dof_handler,
            theta_dof_handler,  // BUG: should be M_dof_handler
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
// BACKWARD COMPATIBILITY: Old 7-argument interface
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
    // Call simplified model (no M)
    assemble_poisson_system<dim>(
        phi_dof_handler,
        theta_dof_handler,
        theta_solution,
        nullptr,
        nullptr,
        params,
        params.current_time,
        phi_matrix,
        phi_rhs,
        phi_constraints,
        true);  // use_simplified = true
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
    // Call simplified model (no M)
    assemble_poisson_system<dim>(
        phi_dof_handler,
        theta_dof_handler,
        theta_solution,
        nullptr,
        nullptr,
        params,
        current_time,
        phi_matrix,
        phi_rhs,
        phi_constraints,
        true);  // use_simplified = true
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
    // The constant is fixed by pinning a DoF in setup_poisson_neumann_constraints()
}

// ============================================================================
// Explicit instantiations
// ============================================================================

// New interface (FULL MODEL)
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

// Simplified model
template void assemble_poisson_system_simplified<2>(
    const dealii::DoFHandler<2>&,
    const Parameters&,
    double,
    dealii::SparseMatrix<double>&,
    dealii::Vector<double>&,
    const dealii::AffineConstraints<double>&);

// Neumann constraints
template void setup_poisson_neumann_constraints<2>(
    const dealii::DoFHandler<2>&,
    dealii::AffineConstraints<double>&);

// Backward compatibility interfaces (valid for quasi-equilibrium)
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