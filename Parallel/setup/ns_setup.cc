// ============================================================================
// ns_setup.cc - Parallel Navier-Stokes Setup Implementation
//
// Uses deal.II 9.7.x API (no deprecated functions)
// Cell-based sparsity pattern construction for robustness
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "ns_setup.h"

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/function.h>

template <int dim>
void setup_ns_coupled_system_parallel(
    const dealii::DoFHandler<dim>& ux_dof_handler,
    const dealii::DoFHandler<dim>& uy_dof_handler,
    const dealii::DoFHandler<dim>& p_dof_handler,
    const dealii::AffineConstraints<double>& ux_constraints,
    const dealii::AffineConstraints<double>& uy_constraints,
    const dealii::AffineConstraints<double>& p_constraints,
    std::vector<dealii::types::global_dof_index>& ux_to_ns_map,
    std::vector<dealii::types::global_dof_index>& uy_to_ns_map,
    std::vector<dealii::types::global_dof_index>& p_to_ns_map,
    dealii::IndexSet& ns_owned,
    dealii::IndexSet& ns_relevant,
    dealii::AffineConstraints<double>& ns_constraints,
    dealii::TrilinosWrappers::SparsityPattern& ns_sparsity,
    MPI_Comm mpi_comm,
    dealii::ConditionalOStream& pcout)
{
    // ========================================================================
    // Step 1: Get sizes and IndexSets from individual DoFHandlers
    // ========================================================================
    const dealii::types::global_dof_index n_ux = ux_dof_handler.n_dofs();
    const dealii::types::global_dof_index n_uy = uy_dof_handler.n_dofs();
    const dealii::types::global_dof_index n_p = p_dof_handler.n_dofs();
    const dealii::types::global_dof_index n_total = n_ux + n_uy + n_p;

    const dealii::IndexSet ux_owned = ux_dof_handler.locally_owned_dofs();
    const dealii::IndexSet uy_owned = uy_dof_handler.locally_owned_dofs();
    const dealii::IndexSet p_owned = p_dof_handler.locally_owned_dofs();

    // deal.II 9.7.x API: returns IndexSet directly
    const dealii::IndexSet ux_relevant =
        dealii::DoFTools::extract_locally_relevant_dofs(ux_dof_handler);
    const dealii::IndexSet uy_relevant =
        dealii::DoFTools::extract_locally_relevant_dofs(uy_dof_handler);
    const dealii::IndexSet p_relevant =
        dealii::DoFTools::extract_locally_relevant_dofs(p_dof_handler);

    // ========================================================================
    // Step 2: Build global index maps
    // ========================================================================
    // Layout: [ux | uy | p] with global offsets
    // ux: [0, n_ux)
    // uy: [n_ux, n_ux + n_uy)
    // p:  [n_ux + n_uy, n_total)

    ux_to_ns_map.clear();
    uy_to_ns_map.clear();
    p_to_ns_map.clear();

    ux_to_ns_map.resize(n_ux);
    uy_to_ns_map.resize(n_uy);
    p_to_ns_map.resize(n_p);

    for (dealii::types::global_dof_index i = 0; i < n_ux; ++i)
        ux_to_ns_map[i] = i;

    for (dealii::types::global_dof_index i = 0; i < n_uy; ++i)
        uy_to_ns_map[i] = n_ux + i;

    for (dealii::types::global_dof_index i = 0; i < n_p; ++i)
        p_to_ns_map[i] = n_ux + n_uy + i;

    // ========================================================================
    // Step 3: Build combined IndexSets
    // ========================================================================
    ns_owned.clear();
    ns_owned.set_size(n_total);

    // Add ux owned (no offset)
    for (auto it = ux_owned.begin(); it != ux_owned.end(); ++it)
        ns_owned.add_index(*it);

    // Add uy owned (offset by n_ux)
    for (auto it = uy_owned.begin(); it != uy_owned.end(); ++it)
        ns_owned.add_index(n_ux + *it);

    // Add p owned (offset by n_ux + n_uy)
    for (auto it = p_owned.begin(); it != p_owned.end(); ++it)
        ns_owned.add_index(n_ux + n_uy + *it);

    ns_owned.compress();

    ns_relevant.clear();
    ns_relevant.set_size(n_total);

    // Add ux relevant
    for (auto it = ux_relevant.begin(); it != ux_relevant.end(); ++it)
        ns_relevant.add_index(*it);

    // Add uy relevant
    for (auto it = uy_relevant.begin(); it != uy_relevant.end(); ++it)
        ns_relevant.add_index(n_ux + *it);

    // Add p relevant
    for (auto it = p_relevant.begin(); it != p_relevant.end(); ++it)
        ns_relevant.add_index(n_ux + n_uy + *it);

    ns_relevant.compress();

    // ========================================================================
    // Step 4: Build combined constraints
    // ========================================================================
    ns_constraints.clear();
    // deal.II 9.7.x API: reinit with (owned, relevant)
    ns_constraints.reinit(ns_owned, ns_relevant);

    // Helper lambda to map constraints from individual field to coupled system
    auto map_constraints = [&](const dealii::AffineConstraints<double>& src,
                               const std::vector<dealii::types::global_dof_index>& index_map,
                               const dealii::IndexSet& relevant)
    {
        for (auto it = relevant.begin(); it != relevant.end(); ++it)
        {
            const auto local_dof = *it;
            if (src.is_constrained(local_dof))
            {
                const auto coupled_i = index_map[local_dof];
                const auto* entries = src.get_constraint_entries(local_dof);
                const double inhom = src.get_inhomogeneity(local_dof);

                ns_constraints.add_line(coupled_i);

                if (entries != nullptr && !entries->empty())
                {
                    for (const auto& entry : *entries)
                    {
                        const auto coupled_j = index_map[entry.first];
                        ns_constraints.add_entry(coupled_i, coupled_j, entry.second);
                    }
                }
                ns_constraints.set_inhomogeneity(coupled_i, inhom);
            }
        }
    };

    map_constraints(ux_constraints, ux_to_ns_map, ux_relevant);
    map_constraints(uy_constraints, uy_to_ns_map, uy_relevant);
    map_constraints(p_constraints, p_to_ns_map, p_relevant);

    ns_constraints.close();

    // ========================================================================
    // Step 5: Build Trilinos sparsity pattern BY CELL ITERATION
    // ========================================================================
    // This is the robust approach for coupled systems in parallel.
    // We iterate over all locally relevant cells and add all 9-block couplings.

    dealii::DynamicSparsityPattern dsp(ns_relevant);

    const unsigned int dofs_per_cell_Q2 = ux_dof_handler.get_fe().n_dofs_per_cell();
    const unsigned int dofs_per_cell_Q1 = p_dof_handler.get_fe().n_dofs_per_cell();

    std::vector<dealii::types::global_dof_index> ux_dof_indices(dofs_per_cell_Q2);
    std::vector<dealii::types::global_dof_index> uy_dof_indices(dofs_per_cell_Q2);
    std::vector<dealii::types::global_dof_index> p_dof_indices(dofs_per_cell_Q1);

    auto ux_cell = ux_dof_handler.begin_active();
    auto uy_cell = uy_dof_handler.begin_active();
    auto p_cell = p_dof_handler.begin_active();

    for (; ux_cell != ux_dof_handler.end(); ++ux_cell, ++uy_cell, ++p_cell)
    {
        // Process all locally relevant cells (not just owned)
        // This ensures we capture ghost couplings needed for assembly
        if (!ux_cell->is_artificial())
        {
            ux_cell->get_dof_indices(ux_dof_indices);
            uy_cell->get_dof_indices(uy_dof_indices);
            p_cell->get_dof_indices(p_dof_indices);

            // All 9 blocks of couplings
            for (unsigned int i = 0; i < dofs_per_cell_Q2; ++i)
            {
                const auto row_ux = ux_to_ns_map[ux_dof_indices[i]];
                const auto row_uy = uy_to_ns_map[uy_dof_indices[i]];

                // ux row couplings
                for (unsigned int j = 0; j < dofs_per_cell_Q2; ++j)
                {
                    dsp.add(row_ux, ux_to_ns_map[ux_dof_indices[j]]);  // ux-ux
                    dsp.add(row_ux, uy_to_ns_map[uy_dof_indices[j]]);  // ux-uy
                }
                for (unsigned int j = 0; j < dofs_per_cell_Q1; ++j)
                {
                    dsp.add(row_ux, p_to_ns_map[p_dof_indices[j]]);    // ux-p
                }

                // uy row couplings
                for (unsigned int j = 0; j < dofs_per_cell_Q2; ++j)
                {
                    dsp.add(row_uy, ux_to_ns_map[ux_dof_indices[j]]);  // uy-ux
                    dsp.add(row_uy, uy_to_ns_map[uy_dof_indices[j]]);  // uy-uy
                }
                for (unsigned int j = 0; j < dofs_per_cell_Q1; ++j)
                {
                    dsp.add(row_uy, p_to_ns_map[p_dof_indices[j]]);    // uy-p
                }
            }

            // p row couplings
            for (unsigned int i = 0; i < dofs_per_cell_Q1; ++i)
            {
                const auto row_p = p_to_ns_map[p_dof_indices[i]];

                for (unsigned int j = 0; j < dofs_per_cell_Q2; ++j)
                {
                    dsp.add(row_p, ux_to_ns_map[ux_dof_indices[j]]);   // p-ux
                    dsp.add(row_p, uy_to_ns_map[uy_dof_indices[j]]);   // p-uy
                }
                // p-p diagonal (needed for constraint handling)
                for (unsigned int j = 0; j < dofs_per_cell_Q1; ++j)
                {
                    dsp.add(row_p, p_to_ns_map[p_dof_indices[j]]);     // p-p
                }
            }
        }
    }

    // Apply constraints to sparsity
    ns_constraints.condense(dsp);

    // Create Trilinos sparsity
    ns_sparsity.reinit(ns_owned, ns_owned, dsp, mpi_comm);

    pcout << "[NS Setup] ux: " << n_ux << ", uy: " << n_uy << ", p: " << n_p
          << ", total: " << n_total
          << ", locally_owned: " << ns_owned.n_elements()
          << ", nnz: " << ns_sparsity.n_nonzero_elements() << "\n";
}

// ============================================================================
// Velocity constraints (MMS: homogeneous Dirichlet on all boundaries)
// ============================================================================
template <int dim>
void setup_ns_velocity_constraints_parallel(
    const dealii::DoFHandler<dim>& ux_dof_handler,
    const dealii::DoFHandler<dim>& uy_dof_handler,
    dealii::AffineConstraints<double>& ux_constraints,
    dealii::AffineConstraints<double>& uy_constraints)
{
    // deal.II 9.7.x API
    const dealii::IndexSet ux_owned = ux_dof_handler.locally_owned_dofs();
    const dealii::IndexSet uy_owned = uy_dof_handler.locally_owned_dofs();
    const dealii::IndexSet ux_relevant =
        dealii::DoFTools::extract_locally_relevant_dofs(ux_dof_handler);
    const dealii::IndexSet uy_relevant =
        dealii::DoFTools::extract_locally_relevant_dofs(uy_dof_handler);

    ux_constraints.clear();
    ux_constraints.reinit(ux_owned, ux_relevant);
    uy_constraints.clear();
    uy_constraints.reinit(uy_owned, uy_relevant);

    // Hanging node constraints
    dealii::DoFTools::make_hanging_node_constraints(ux_dof_handler, ux_constraints);
    dealii::DoFTools::make_hanging_node_constraints(uy_dof_handler, uy_constraints);

    // Homogeneous Dirichlet on all boundaries (boundary IDs 0-3)
    for (dealii::types::boundary_id bid = 0; bid <= 3; ++bid)
    {
        dealii::VectorTools::interpolate_boundary_values(
            ux_dof_handler,
            bid,
            dealii::Functions::ZeroFunction<dim>(),
            ux_constraints);

        dealii::VectorTools::interpolate_boundary_values(
            uy_dof_handler,
            bid,
            dealii::Functions::ZeroFunction<dim>(),
            uy_constraints);
    }

    ux_constraints.close();
    uy_constraints.close();
}

// ============================================================================
// Pressure constraints for DG pressure (Paper requirement A1)
// DG elements have no hanging node constraints
// Pin DoF 0 to fix pressure constant (null space)
// ============================================================================
template <int dim>
void setup_ns_pressure_constraints_parallel(
    const dealii::DoFHandler<dim>& p_dof_handler,
    dealii::AffineConstraints<double>& p_constraints)
{
    // deal.II 9.7.x API
    const dealii::IndexSet p_owned = p_dof_handler.locally_owned_dofs();
    const dealii::IndexSet p_relevant =
        dealii::DoFTools::extract_locally_relevant_dofs(p_dof_handler);

    p_constraints.clear();
    p_constraints.reinit(p_owned, p_relevant);

    // DG pressure: NO hanging node constraints (no inter-element continuity)
    // This is key for paper requirement (A1) - element-local incompressibility

    // Pin pressure DoF 0 to fix the constant (same as serial code)
    // For pure Neumann problem (Stokes), pressure is determined up to a constant.
    // Error computation subtracts mean from both numerical and exact pressure.
    if (p_dof_handler.locally_owned_dofs().is_element(0))
    {
        if (!p_constraints.is_constrained(0))
        {
            p_constraints.add_line(0);
            p_constraints.set_inhomogeneity(0, 0.0);
        }
    }

    p_constraints.close();
}

// ============================================================================
// Explicit instantiations
// ============================================================================
template void setup_ns_coupled_system_parallel<2>(
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    const dealii::AffineConstraints<double>&,
    const dealii::AffineConstraints<double>&,
    const dealii::AffineConstraints<double>&,
    std::vector<dealii::types::global_dof_index>&,
    std::vector<dealii::types::global_dof_index>&,
    std::vector<dealii::types::global_dof_index>&,
    dealii::IndexSet&,
    dealii::IndexSet&,
    dealii::AffineConstraints<double>&,
    dealii::TrilinosWrappers::SparsityPattern&,
    MPI_Comm,
    dealii::ConditionalOStream&);

template void setup_ns_velocity_constraints_parallel<2>(
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    dealii::AffineConstraints<double>&,
    dealii::AffineConstraints<double>&);

template void setup_ns_pressure_constraints_parallel<2>(
    const dealii::DoFHandler<2>&,
    dealii::AffineConstraints<double>&);