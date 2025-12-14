// ============================================================================
// setup/poisson_setup.cc - Magnetostatic Poisson System Setup (CORRECTED)
//
// CORRECTED for pure Neumann problem.
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "setup/poisson_setup.h"

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <iostream>

// ============================================================================
// Setup constraints for pure Neumann Poisson problem
// ============================================================================
template <int dim>
void setup_poisson_constraints(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    dealii::AffineConstraints<double>& phi_constraints)
{
    phi_constraints.clear();

    // Add hanging node constraints (for AMR)
    dealii::DoFTools::make_hanging_node_constraints(phi_dof_handler, phi_constraints);

    // For pure Neumann problem, solution is unique up to a constant.
    // Fix the constant by pinning DoF 0 to zero.
    //
    // Alternative approaches:
    // 1. Mean-zero constraint: harder to implement, requires Lagrange multiplier
    // 2. Pin a corner DoF: may cause issues with hanging nodes
    // 3. Pin DoF 0: simple and robust
    //
    // We use approach 3.
    if (phi_dof_handler.n_dofs() > 0)
    {
        // Check if DoF 0 is already constrained (e.g., by hanging nodes)
        if (!phi_constraints.is_constrained(0))
        {
            phi_constraints.add_line(0);
            phi_constraints.set_inhomogeneity(0, 0.0);
        }
    }

    phi_constraints.close();
}

// ============================================================================
// Build sparsity pattern
// ============================================================================
template <int dim>
void build_poisson_sparsity(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::AffineConstraints<double>& phi_constraints,
    dealii::SparsityPattern& phi_sparsity,
    bool verbose)
{
    const unsigned int n_dofs = phi_dof_handler.n_dofs();

    // Build sparsity pattern
    dealii::DynamicSparsityPattern dsp(n_dofs, n_dofs);
    dealii::DoFTools::make_sparsity_pattern(phi_dof_handler, dsp, phi_constraints, false);

    // Copy to final sparsity pattern
    phi_sparsity.copy_from(dsp);

    if (verbose)
    {
        std::cout << "[Setup] Poisson DoFs: " << n_dofs
                  << ", sparsity: " << phi_sparsity.n_nonzero_elements()
                  << " nonzeros\n";
    }
}

// ============================================================================
// Explicit instantiations
// ============================================================================
template void setup_poisson_constraints<2>(
    const dealii::DoFHandler<2>&,
    dealii::AffineConstraints<double>&);

template void build_poisson_sparsity<2>(
    const dealii::DoFHandler<2>&,
    const dealii::AffineConstraints<double>&,
    dealii::SparsityPattern&,
    bool);