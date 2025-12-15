// ============================================================================
// setup/poisson_setup.cc - Magnetostatic Poisson System Setup
//
// Pure Neumann problem: -Δφ = ∇·(m - h_a), ∂φ/∂n = (h_a - m)·n
//
// NOTE: Constraint setup (hanging nodes + nullspace fix) is in
//       poisson_assembler.cc: setup_poisson_neumann_constraints()
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531, Eq. 42d
// ============================================================================

#include "setup/poisson_setup.h"

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <iostream>

// ============================================================================
// Sparsity pattern (use constraints to eliminate constrained entries)
// ============================================================================
template <int dim>
void build_poisson_sparsity(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::AffineConstraints<double>& phi_constraints,
    dealii::SparsityPattern& phi_sparsity,
    bool verbose)
{
    using namespace dealii;

    DynamicSparsityPattern dsp(phi_dof_handler.n_dofs(), phi_dof_handler.n_dofs());

    DoFTools::make_sparsity_pattern(phi_dof_handler, dsp, phi_constraints,
                                    /*keep_constrained_dofs=*/false);

    phi_sparsity.copy_from(dsp);

    if (verbose)
    {
        std::cout << "[PoissonSetup] DoFs: " << phi_dof_handler.n_dofs()
                  << ", nnz: " << phi_sparsity.n_nonzero_elements() << std::endl;
    }
}

// ============================================================================
// Explicit instantiations
// ============================================================================
template void build_poisson_sparsity<2>(
    const dealii::DoFHandler<2>&,
    const dealii::AffineConstraints<double>&,
    dealii::SparsityPattern&,
    bool);

template void build_poisson_sparsity<3>(
    const dealii::DoFHandler<3>&,
    const dealii::AffineConstraints<double>&,
    dealii::SparsityPattern&,
    bool);