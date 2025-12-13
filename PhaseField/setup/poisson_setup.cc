// ============================================================================
// setup/poisson_setup.cc - Magnetostatic Poisson System Setup Implementation
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "setup/poisson_setup.h"

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <iostream>

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
        std::cout << "[Setup] Poisson sparsity: " << phi_sparsity.n_nonzero_elements()
                  << " nonzeros\n";
    }
}

// ============================================================================
// Explicit instantiation
// ============================================================================
template void build_poisson_sparsity<2>(
    const dealii::DoFHandler<2>&,
    const dealii::AffineConstraints<double>&,
    dealii::SparsityPattern&,
    bool);