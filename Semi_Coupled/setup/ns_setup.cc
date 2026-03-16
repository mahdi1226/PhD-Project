// ============================================================================
// ns_setup.cc - Parallel Navier-Stokes Setup Implementation
//
// Uses deal.II 9.7.x API (no deprecated functions)
// Cell-based sparsity pattern construction for robustness
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
//
// PARALLEL FIX (2026-01-17):
//   Trilinos Epetra requires CONTIGUOUS row ownership per MPI rank.
//   The naive block layout [ux|uy|p] with global offsets creates non-contiguous
//   ownership (each rank owns scattered pieces of each block).
//
//   Solution: Renumber so each rank owns a CONTIGUOUS block of the coupled system.
//   New global layout: [rank0_dofs | rank1_dofs | rank2_dofs | ...]
//   where rank_i_dofs = [ux_owned_by_i | uy_owned_by_i | p_owned_by_i]
// ============================================================================

#include "ns_setup.h"

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/mpi.h>

#include <cmath>

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
    dealii::ConditionalOStream& pcout,
    bool interleave_velocity)
{
    const unsigned int my_rank = dealii::Utilities::MPI::this_mpi_process(mpi_comm);
    const unsigned int n_ranks = dealii::Utilities::MPI::n_mpi_processes(mpi_comm);

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

    const dealii::IndexSet ux_relevant =
        dealii::DoFTools::extract_locally_relevant_dofs(ux_dof_handler);
    const dealii::IndexSet uy_relevant =
        dealii::DoFTools::extract_locally_relevant_dofs(uy_dof_handler);
    const dealii::IndexSet p_relevant =
        dealii::DoFTools::extract_locally_relevant_dofs(p_dof_handler);

    // ========================================================================
    // Step 2: Build CONTIGUOUS renumbering for Trilinos compatibility
    // ========================================================================
    // Each rank will own a contiguous block of the coupled system.
    // Local ordering within each rank: [ux_local | uy_local | p_local]

    const dealii::types::global_dof_index n_ux_local = ux_owned.n_elements();
    const dealii::types::global_dof_index n_uy_local = uy_owned.n_elements();
    const dealii::types::global_dof_index n_p_local = p_owned.n_elements();
    const dealii::types::global_dof_index n_local = n_ux_local + n_uy_local + n_p_local;

    // Gather n_local from all ranks to compute global offsets
    // Portable: global_dof_index may be 32-bit (unsigned int) or 64-bit (uint64_t)
    const MPI_Datatype mpi_dof_type =
        (sizeof(dealii::types::global_dof_index) == sizeof(unsigned int))
            ? MPI_UNSIGNED : MPI_UNSIGNED_LONG_LONG;

    std::vector<dealii::types::global_dof_index> all_n_local(n_ranks);
    {
        dealii::types::global_dof_index my_n_local = n_local;
        MPI_Allgather(&my_n_local, 1, mpi_dof_type,
                      all_n_local.data(), 1, mpi_dof_type, mpi_comm);
    }

    // Compute global starting index for this rank (exclusive prefix sum)
    dealii::types::global_dof_index global_start = 0;
    for (unsigned int r = 0; r < my_rank; ++r)
        global_start += all_n_local[r];

    // ========================================================================
    // Step 2b: Build local-to-global maps for owned DoFs
    // ========================================================================
    // Initialize maps with invalid index
    ux_to_ns_map.assign(n_ux, dealii::numbers::invalid_dof_index);
    uy_to_ns_map.assign(n_uy, dealii::numbers::invalid_dof_index);
    p_to_ns_map.assign(n_p, dealii::numbers::invalid_dof_index);

    // Fill in mappings for owned DoFs
    dealii::types::global_dof_index coupled_idx = global_start;

    if (interleave_velocity)
    {
        // Node-wise interleaving: [ux_0,uy_0, ux_1,uy_1, ..., p_0,p_1,...]
        // Puts velocity DoFs from the same mesh node adjacent (offset by 1).
        // Since ux and uy use the same FE on the same mesh, owned DoFs match.
        auto ux_it = ux_owned.begin();
        auto uy_it = uy_owned.begin();
        for (; ux_it != ux_owned.end(); ++ux_it, ++uy_it)
        {
            ux_to_ns_map[*ux_it] = coupled_idx++;
            uy_to_ns_map[*uy_it] = coupled_idx++;
        }
        // Pressure DoFs appended after velocity
        for (auto it = p_owned.begin(); it != p_owned.end(); ++it)
            p_to_ns_map[*it] = coupled_idx++;

        pcout << "[NS Setup] Using node-wise interleaved velocity ordering\n";
    }
    else
    {
        // Block ordering: [all_ux | all_uy | all_p] (default)
        for (auto it = ux_owned.begin(); it != ux_owned.end(); ++it)
            ux_to_ns_map[*it] = coupled_idx++;

        for (auto it = uy_owned.begin(); it != uy_owned.end(); ++it)
            uy_to_ns_map[*it] = coupled_idx++;

        for (auto it = p_owned.begin(); it != p_owned.end(); ++it)
            p_to_ns_map[*it] = coupled_idx++;
    }

    // ========================================================================
    // Step 2c: Exchange mappings for ghost DoFs via Trilinos ghost import
    // ========================================================================
    // Uses point-to-point MPI (Trilinos Import) to exchange only ghost DoF
    // mappings. Scales as O(n_ghost) per rank pair, not O(n_total) like
    // the previous Allreduce approach.
    // Precision note: double has 53-bit mantissa, sufficient for DoF indices
    // up to 2^53 ~ 9e15, well beyond any practical mesh size.
    {
        auto exchange_ghost_map = [&](const dealii::IndexSet& owned,
                                      const dealii::IndexSet& relevant,
                                      std::vector<dealii::types::global_dof_index>& map)
        {
            dealii::TrilinosWrappers::MPI::Vector owned_vec;
            owned_vec.reinit(owned, mpi_comm);

            for (auto it = owned.begin(); it != owned.end(); ++it)
                owned_vec(*it) = static_cast<double>(map[*it]);

            dealii::TrilinosWrappers::MPI::Vector ghosted_vec;
            ghosted_vec.reinit(owned, relevant, mpi_comm);
            ghosted_vec = owned_vec;

            for (auto it = relevant.begin(); it != relevant.end(); ++it)
            {
                if (!owned.is_element(*it))
                    map[*it] = static_cast<dealii::types::global_dof_index>(
                        std::round(ghosted_vec(*it)));
            }
        };

        exchange_ghost_map(ux_owned, ux_relevant, ux_to_ns_map);
        exchange_ghost_map(uy_owned, uy_relevant, uy_to_ns_map);
        exchange_ghost_map(p_owned, p_relevant, p_to_ns_map);
    }

    // ========================================================================
    // Step 3: Build combined IndexSets (NOW CONTIGUOUS per rank!)
    // ========================================================================
    ns_owned.clear();
    ns_owned.set_size(n_total);
    ns_owned.add_range(global_start, global_start + n_local);
    ns_owned.compress();

    // Build relevant IndexSet by mapping ghost DoFs
    // Skip any invalid indices (shouldn't happen but defensive)
    ns_relevant.clear();
    ns_relevant.set_size(n_total);

    for (auto it = ux_relevant.begin(); it != ux_relevant.end(); ++it)
    {
        const auto mapped = ux_to_ns_map[*it];
        if (mapped != dealii::numbers::invalid_dof_index)
            ns_relevant.add_index(mapped);
    }

    for (auto it = uy_relevant.begin(); it != uy_relevant.end(); ++it)
    {
        const auto mapped = uy_to_ns_map[*it];
        if (mapped != dealii::numbers::invalid_dof_index)
            ns_relevant.add_index(mapped);
    }

    for (auto it = p_relevant.begin(); it != p_relevant.end(); ++it)
    {
        const auto mapped = p_to_ns_map[*it];
        if (mapped != dealii::numbers::invalid_dof_index)
            ns_relevant.add_index(mapped);
    }

    ns_relevant.compress();

    // ========================================================================
    // Step 4: Build combined constraints (using new numbering)
    // ========================================================================
    ns_constraints.clear();
    ns_constraints.reinit(ns_owned, ns_relevant);

    // Helper lambda to map constraints
    auto map_constraints = [&](const dealii::AffineConstraints<double>& src,
                               const std::vector<dealii::types::global_dof_index>& index_map,
                               const dealii::IndexSet& relevant)
    {
        for (auto it = relevant.begin(); it != relevant.end(); ++it)
        {
            const auto orig_dof = *it;
            if (src.is_constrained(orig_dof))
            {
                const auto coupled_i = index_map[orig_dof];
                const auto* entries = src.get_constraint_entries(orig_dof);
                const double inhom = src.get_inhomogeneity(orig_dof);

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
    // Step 5: Build sparsity pattern by cell iteration
    // ========================================================================
    dealii::DynamicSparsityPattern dsp(n_total, n_total, ns_relevant);

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
        // Skip artificial cells - they have no valid DoF information
        if (ux_cell->is_artificial())
            continue;

        ux_cell->get_dof_indices(ux_dof_indices);
        uy_cell->get_dof_indices(uy_dof_indices);
        p_cell->get_dof_indices(p_dof_indices);

        // All 9 blocks of couplings (using mapped indices)
        for (unsigned int i = 0; i < dofs_per_cell_Q2; ++i)
        {
            const auto row_ux = ux_to_ns_map[ux_dof_indices[i]];
            const auto row_uy = uy_to_ns_map[uy_dof_indices[i]];

            for (unsigned int j = 0; j < dofs_per_cell_Q2; ++j)
            {
                dsp.add(row_ux, ux_to_ns_map[ux_dof_indices[j]]);  // ux-ux
                dsp.add(row_ux, uy_to_ns_map[uy_dof_indices[j]]);  // ux-uy
                dsp.add(row_uy, ux_to_ns_map[ux_dof_indices[j]]);  // uy-ux
                dsp.add(row_uy, uy_to_ns_map[uy_dof_indices[j]]);  // uy-uy
            }
            for (unsigned int j = 0; j < dofs_per_cell_Q1; ++j)
            {
                dsp.add(row_ux, p_to_ns_map[p_dof_indices[j]]);    // ux-p
                dsp.add(row_uy, p_to_ns_map[p_dof_indices[j]]);    // uy-p
            }
        }

        for (unsigned int i = 0; i < dofs_per_cell_Q1; ++i)
        {
            const auto row_p = p_to_ns_map[p_dof_indices[i]];

            for (unsigned int j = 0; j < dofs_per_cell_Q2; ++j)
            {
                dsp.add(row_p, ux_to_ns_map[ux_dof_indices[j]]);   // p-ux
                dsp.add(row_p, uy_to_ns_map[uy_dof_indices[j]]);   // p-uy
            }
            for (unsigned int j = 0; j < dofs_per_cell_Q1; ++j)
            {
                dsp.add(row_p, p_to_ns_map[p_dof_indices[j]]);     // p-p
            }
        }
    }

    // Apply constraints
    ns_constraints.condense(dsp);

    // Distribute sparsity pattern for parallel (exchange off-processor entries)
    dealii::SparsityTools::distribute_sparsity_pattern(dsp, ns_owned, mpi_comm, ns_relevant);

    // Create Trilinos sparsity - NOW ns_owned IS CONTIGUOUS!
    ns_sparsity.reinit(ns_owned, ns_owned, dsp, mpi_comm);

    pcout << "[NS Setup] ux: " << n_ux << ", uy: " << n_uy << ", p: " << n_p
          << ", total: " << n_total
          << ", locally_owned: " << ns_owned.n_elements()
          << ", range: [" << global_start << ", " << (global_start + n_local) << ")"
          << ", nnz: " << ns_sparsity.n_nonzero_elements() << "\n";
}

// ============================================================================
// Velocity constraints
// ============================================================================
template <int dim>
void setup_ns_velocity_constraints_parallel(
    const dealii::DoFHandler<dim>& ux_dof_handler,
    const dealii::DoFHandler<dim>& uy_dof_handler,
    dealii::AffineConstraints<double>& ux_constraints,
    dealii::AffineConstraints<double>& uy_constraints)
{
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

    dealii::DoFTools::make_hanging_node_constraints(ux_dof_handler, ux_constraints);
    dealii::DoFTools::make_hanging_node_constraints(uy_dof_handler, uy_constraints);

    for (dealii::types::boundary_id bid = 0; bid <= 3; ++bid)
    {
        dealii::VectorTools::interpolate_boundary_values(
            ux_dof_handler, bid,
            dealii::Functions::ZeroFunction<dim>(),
            ux_constraints);

        dealii::VectorTools::interpolate_boundary_values(
            uy_dof_handler, bid,
            dealii::Functions::ZeroFunction<dim>(),
            uy_constraints);
    }

    ux_constraints.close();
    uy_constraints.close();
}

// ============================================================================
// Pressure constraints
// ============================================================================
template <int dim>
void setup_ns_pressure_constraints_parallel(
    const dealii::DoFHandler<dim>& p_dof_handler,
    dealii::AffineConstraints<double>& p_constraints)
{
    const dealii::IndexSet p_owned = p_dof_handler.locally_owned_dofs();
    const dealii::IndexSet p_relevant =
        dealii::DoFTools::extract_locally_relevant_dofs(p_dof_handler);

    p_constraints.clear();
    p_constraints.reinit(p_owned, p_relevant);

    // CG pressure: add hanging node constraints for AMR
    dealii::DoFTools::make_hanging_node_constraints(p_dof_handler, p_constraints);

    // Pin DoF 0 to fix pressure constant
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
    dealii::ConditionalOStream&,
    bool);

template void setup_ns_velocity_constraints_parallel<2>(
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    dealii::AffineConstraints<double>&,
    dealii::AffineConstraints<double>&);

template void setup_ns_pressure_constraints_parallel<2>(
    const dealii::DoFHandler<2>&,
    dealii::AffineConstraints<double>&);