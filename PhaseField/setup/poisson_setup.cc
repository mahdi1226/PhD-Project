// ============================================================================
// setup/poisson_setup.cc - Magnetostatic Poisson System Setup
//
// Free function for Poisson system setup following ch_setup.cc pattern.
//
// Pure Neumann problem: (μ∇φ, ∇χ) = (h_a - M, ∇χ)
// Requires pinning one DoF to fix the constant (nullspace).
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531, Eq. 42c
// ============================================================================

#include "setup/poisson_setup.h"

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <iostream>

// ============================================================================
// setup_poisson_constraints_and_sparsity
//
// Creates constraints and sparsity pattern for Poisson system.
// This was previously hardcoded in phase_field_setup.cc.
// ============================================================================
template <int dim>
void setup_poisson_constraints_and_sparsity(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    dealii::AffineConstraints<double>& phi_constraints,
    dealii::SparsityPattern& phi_sparsity,
    bool verbose)
{
    // ========================================================================
    // Step 1: Build constraints (hanging nodes + nullspace fix)
    // ========================================================================
    phi_constraints.clear();

    // Hanging node constraints (for AMR)
    dealii::DoFTools::make_hanging_node_constraints(phi_dof_handler, phi_constraints);

    // Pin DoF 0 to zero (fixes the constant for pure Neumann)
    // This is the standard approach for pure Neumann problems.
    if (phi_dof_handler.n_dofs() > 0 && !phi_constraints.is_constrained(0))
    {
        phi_constraints.add_line(0);
        phi_constraints.set_inhomogeneity(0, 0.0);
    }

    phi_constraints.close();

    // ========================================================================
    // Step 2: Build sparsity pattern (with constraints eliminated)
    // ========================================================================
    dealii::DynamicSparsityPattern dsp(phi_dof_handler.n_dofs());
    dealii::DoFTools::make_sparsity_pattern(phi_dof_handler, dsp,
                                             phi_constraints,
                                             /*keep_constrained_dofs=*/false);
    phi_sparsity.copy_from(dsp);

    if (verbose)
    {
        std::cout << "[Setup] Poisson: " << phi_dof_handler.n_dofs() << " DoFs, "
                  << phi_sparsity.n_nonzero_elements() << " nonzeros\n";
    }
}

// ============================================================================
// Explicit instantiations
// ============================================================================
template void setup_poisson_constraints_and_sparsity<2>(
    const dealii::DoFHandler<2>&,
    dealii::AffineConstraints<double>&,
    dealii::SparsityPattern&,
    bool);

template void setup_poisson_constraints_and_sparsity<3>(
    const dealii::DoFHandler<3>&,
    dealii::AffineConstraints<double>&,
    dealii::SparsityPattern&,
    bool);