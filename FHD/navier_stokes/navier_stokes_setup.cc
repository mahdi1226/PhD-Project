// ============================================================================
// navier_stokes/navier_stokes_setup.cc - DoFs, Constraints, Coupled System
//
// Three scalar DoFHandlers (ux, uy, p) assembled into a monolithic
// saddle-point system via index maps.
//
// FE spaces (Nochetto Section 4.3):
//   Velocity: CG Q_ℓ (FE_Q)
//   Pressure: DG P_{ℓ-1} (FE_DGP, discontinuous polynomial)
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015)
// ============================================================================

#include "navier_stokes/navier_stokes.h"

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/function.h>

template <int dim>
void NavierStokesSubsystem<dim>::distribute_dofs()
{
    ux_dof_handler_.distribute_dofs(fe_velocity_);
    uy_dof_handler_.distribute_dofs(fe_velocity_);
    p_dof_handler_.distribute_dofs(fe_pressure_);

    ux_locally_owned_ = ux_dof_handler_.locally_owned_dofs();
    uy_locally_owned_ = uy_dof_handler_.locally_owned_dofs();
    p_locally_owned_  = p_dof_handler_.locally_owned_dofs();

    ux_locally_relevant_ = dealii::DoFTools::extract_locally_relevant_dofs(ux_dof_handler_);
    uy_locally_relevant_ = dealii::DoFTools::extract_locally_relevant_dofs(uy_dof_handler_);
    p_locally_relevant_  = dealii::DoFTools::extract_locally_relevant_dofs(p_dof_handler_);

    n_ux_ = ux_dof_handler_.n_dofs();
    n_uy_ = uy_dof_handler_.n_dofs();
    n_p_  = p_dof_handler_.n_dofs();
}

template <int dim>
void NavierStokesSubsystem<dim>::build_constraints()
{
    // Velocity: hanging nodes + homogeneous Dirichlet on all boundaries
    ux_constraints_.clear();
    ux_constraints_.reinit(ux_locally_owned_, ux_locally_relevant_);
    dealii::DoFTools::make_hanging_node_constraints(ux_dof_handler_, ux_constraints_);
    dealii::VectorTools::interpolate_boundary_values(
        ux_dof_handler_, 0, dealii::Functions::ZeroFunction<dim>(), ux_constraints_);
    dealii::VectorTools::interpolate_boundary_values(
        ux_dof_handler_, 1, dealii::Functions::ZeroFunction<dim>(), ux_constraints_);
    dealii::VectorTools::interpolate_boundary_values(
        ux_dof_handler_, 2, dealii::Functions::ZeroFunction<dim>(), ux_constraints_);
    dealii::VectorTools::interpolate_boundary_values(
        ux_dof_handler_, 3, dealii::Functions::ZeroFunction<dim>(), ux_constraints_);
    ux_constraints_.close();

    uy_constraints_.clear();
    uy_constraints_.reinit(uy_locally_owned_, uy_locally_relevant_);
    dealii::DoFTools::make_hanging_node_constraints(uy_dof_handler_, uy_constraints_);
    dealii::VectorTools::interpolate_boundary_values(
        uy_dof_handler_, 0, dealii::Functions::ZeroFunction<dim>(), uy_constraints_);
    dealii::VectorTools::interpolate_boundary_values(
        uy_dof_handler_, 1, dealii::Functions::ZeroFunction<dim>(), uy_constraints_);
    dealii::VectorTools::interpolate_boundary_values(
        uy_dof_handler_, 2, dealii::Functions::ZeroFunction<dim>(), uy_constraints_);
    dealii::VectorTools::interpolate_boundary_values(
        uy_dof_handler_, 3, dealii::Functions::ZeroFunction<dim>(), uy_constraints_);
    uy_constraints_.close();

    // Pressure: DG has no hanging nodes; pin DoF 0 for uniqueness
    p_constraints_.clear();
    p_constraints_.reinit(p_locally_owned_, p_locally_relevant_);
    if (p_locally_owned_.is_element(0))
        p_constraints_.add_constraint(0, {}, 0.0);
    p_constraints_.close();
}

template <int dim>
void NavierStokesSubsystem<dim>::build_coupled_system()
{
    // ================================================================
    // Build index maps: component DoF → coupled system DoF
    //
    // Global layout: [ux DoFs | uy DoFs | p DoFs]
    //                [0..n_ux) [n_ux..n_ux+n_uy) [n_ux+n_uy..total)
    // ================================================================
    const dealii::types::global_dof_index total_dofs = n_ux_ + n_uy_ + n_p_;

    // Build owned and relevant index sets for coupled system
    ns_locally_owned_.clear();
    ns_locally_owned_.set_size(total_dofs);

    ns_locally_relevant_.clear();
    ns_locally_relevant_.set_size(total_dofs);

    // Add ux indices (offset = 0)
    for (auto it = ux_locally_owned_.begin(); it != ux_locally_owned_.end(); ++it)
        ns_locally_owned_.add_index(*it);
    for (auto it = ux_locally_relevant_.begin(); it != ux_locally_relevant_.end(); ++it)
        ns_locally_relevant_.add_index(*it);

    // Add uy indices (offset = n_ux_)
    for (auto it = uy_locally_owned_.begin(); it != uy_locally_owned_.end(); ++it)
        ns_locally_owned_.add_index(n_ux_ + *it);
    for (auto it = uy_locally_relevant_.begin(); it != uy_locally_relevant_.end(); ++it)
        ns_locally_relevant_.add_index(n_ux_ + *it);

    // Add p indices (offset = n_ux_ + n_uy_)
    for (auto it = p_locally_owned_.begin(); it != p_locally_owned_.end(); ++it)
        ns_locally_owned_.add_index(n_ux_ + n_uy_ + *it);
    for (auto it = p_locally_relevant_.begin(); it != p_locally_relevant_.end(); ++it)
        ns_locally_relevant_.add_index(n_ux_ + n_uy_ + *it);

    ns_locally_owned_.compress();
    ns_locally_relevant_.compress();

    // ================================================================
    // Build coupled constraints
    // ================================================================
    ns_constraints_.clear();
    ns_constraints_.reinit(ns_locally_owned_, ns_locally_relevant_);

    // Transfer ux constraints (offset = 0)
    for (auto it = ux_locally_relevant_.begin(); it != ux_locally_relevant_.end(); ++it)
    {
        const auto dof = *it;
        if (ux_constraints_.is_constrained(dof))
        {
            const auto* entries = ux_constraints_.get_constraint_entries(dof);
            const double inhomogeneity = ux_constraints_.get_inhomogeneity(dof);
            std::vector<std::pair<dealii::types::global_dof_index, double>> mapped;
            if (entries)
                for (const auto& e : *entries)
                    mapped.emplace_back(e.first, e.second);
            ns_constraints_.add_constraint(dof, mapped, inhomogeneity);
        }
    }

    // Transfer uy constraints (offset = n_ux_)
    for (auto it = uy_locally_relevant_.begin(); it != uy_locally_relevant_.end(); ++it)
    {
        const auto dof = *it;
        if (uy_constraints_.is_constrained(dof))
        {
            const auto* entries = uy_constraints_.get_constraint_entries(dof);
            const double inhomogeneity = uy_constraints_.get_inhomogeneity(dof);
            std::vector<std::pair<dealii::types::global_dof_index, double>> mapped;
            if (entries)
                for (const auto& e : *entries)
                    mapped.emplace_back(n_ux_ + e.first, e.second);
            ns_constraints_.add_constraint(n_ux_ + dof, mapped, inhomogeneity);
        }
    }

    // Transfer p constraints (offset = n_ux_ + n_uy_)
    for (auto it = p_locally_relevant_.begin(); it != p_locally_relevant_.end(); ++it)
    {
        const auto dof = *it;
        if (p_constraints_.is_constrained(dof))
        {
            const auto* entries = p_constraints_.get_constraint_entries(dof);
            const double inhomogeneity = p_constraints_.get_inhomogeneity(dof);
            std::vector<std::pair<dealii::types::global_dof_index, double>> mapped;
            if (entries)
                for (const auto& e : *entries)
                    mapped.emplace_back(n_ux_ + n_uy_ + e.first, e.second);
            ns_constraints_.add_constraint(n_ux_ + n_uy_ + dof, mapped, inhomogeneity);
        }
    }

    ns_constraints_.close();

    // ================================================================
    // Build sparsity pattern for coupled system
    // ================================================================
    const unsigned int ux_dpc = fe_velocity_.n_dofs_per_cell();
    const unsigned int uy_dpc = fe_velocity_.n_dofs_per_cell();
    const unsigned int p_dpc  = fe_pressure_.n_dofs_per_cell();

    std::vector<dealii::types::global_dof_index> ux_dofs(ux_dpc);
    std::vector<dealii::types::global_dof_index> uy_dofs(uy_dpc);
    std::vector<dealii::types::global_dof_index> p_dofs(p_dpc);

    dealii::TrilinosWrappers::SparsityPattern sparsity(
        ns_locally_owned_, ns_locally_owned_,
        ns_locally_relevant_, mpi_comm_);

    for (const auto& cell : ux_dof_handler_.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        cell->get_dof_indices(ux_dofs);

        const auto uy_cell = cell->as_dof_handler_iterator(uy_dof_handler_);
        uy_cell->get_dof_indices(uy_dofs);

        const auto p_cell = cell->as_dof_handler_iterator(p_dof_handler_);
        p_cell->get_dof_indices(p_dofs);

        // Map to coupled system indices
        std::vector<dealii::types::global_dof_index> coupled_ux(ux_dpc);
        std::vector<dealii::types::global_dof_index> coupled_uy(uy_dpc);
        std::vector<dealii::types::global_dof_index> coupled_p(p_dpc);

        for (unsigned int i = 0; i < ux_dpc; ++i)
            coupled_ux[i] = ux_dofs[i];
        for (unsigned int i = 0; i < uy_dpc; ++i)
            coupled_uy[i] = n_ux_ + uy_dofs[i];
        for (unsigned int i = 0; i < p_dpc; ++i)
            coupled_p[i] = n_ux_ + n_uy_ + p_dofs[i];

        // All block couplings
        std::vector<dealii::types::global_dof_index> all_dofs;
        all_dofs.insert(all_dofs.end(), coupled_ux.begin(), coupled_ux.end());
        all_dofs.insert(all_dofs.end(), coupled_uy.begin(), coupled_uy.end());
        all_dofs.insert(all_dofs.end(), coupled_p.begin(), coupled_p.end());

        ns_constraints_.add_entries_local_to_global(all_dofs, sparsity);
    }

    sparsity.compress();
    ns_matrix_.reinit(sparsity);
}

template <int dim>
void NavierStokesSubsystem<dim>::allocate_vectors()
{
    // Component vectors
    ux_solution_.reinit(ux_locally_owned_, mpi_comm_);
    uy_solution_.reinit(uy_locally_owned_, mpi_comm_);
    p_solution_.reinit(p_locally_owned_, mpi_comm_);

    ux_relevant_.reinit(ux_locally_owned_, ux_locally_relevant_, mpi_comm_);
    uy_relevant_.reinit(uy_locally_owned_, uy_locally_relevant_, mpi_comm_);
    p_relevant_.reinit(p_locally_owned_, p_locally_relevant_, mpi_comm_);

    // Monolithic system
    ns_rhs_.reinit(ns_locally_owned_, mpi_comm_);
    ns_solution_.reinit(ns_locally_owned_, mpi_comm_);
}

template <int dim>
void NavierStokesSubsystem<dim>::build_block_sparsity_patterns()
{
    const unsigned int vel_dpc = fe_velocity_.n_dofs_per_cell();
    const unsigned int p_dpc  = fe_pressure_.n_dofs_per_cell();

    std::vector<dealii::types::global_dof_index> ux_dofs(vel_dpc);
    std::vector<dealii::types::global_dof_index> uy_dofs(vel_dpc);
    std::vector<dealii::types::global_dof_index> p_dofs(p_dpc);

    // A_ux_ux: velocity × velocity (ux block)
    {
        dealii::TrilinosWrappers::SparsityPattern sp(
            ux_locally_owned_, ux_locally_owned_,
            ux_locally_relevant_, mpi_comm_);

        for (const auto& cell : ux_dof_handler_.active_cell_iterators())
        {
            if (!cell->is_locally_owned()) continue;
            cell->get_dof_indices(ux_dofs);
            ux_constraints_.add_entries_local_to_global(ux_dofs, sp);
        }
        sp.compress();
        A_ux_ux_.reinit(sp);
    }

    // A_uy_uy: velocity × velocity (uy block) — same sparsity as ux
    {
        dealii::TrilinosWrappers::SparsityPattern sp(
            uy_locally_owned_, uy_locally_owned_,
            uy_locally_relevant_, mpi_comm_);

        for (const auto& cell : uy_dof_handler_.active_cell_iterators())
        {
            if (!cell->is_locally_owned()) continue;
            cell->get_dof_indices(uy_dofs);
            uy_constraints_.add_entries_local_to_global(uy_dofs, sp);
        }
        sp.compress();
        A_uy_uy_.reinit(sp);
    }

    // B_ux: pressure × ux velocity (divergence operator, ux component)
    // B^T_ux: ux velocity × pressure (pressure gradient, ux component)
    {
        dealii::TrilinosWrappers::SparsityPattern sp_B(
            p_locally_owned_, ux_locally_owned_,
            p_locally_relevant_, mpi_comm_);

        dealii::TrilinosWrappers::SparsityPattern sp_Bt(
            ux_locally_owned_, p_locally_owned_,
            ux_locally_relevant_, mpi_comm_);

        for (const auto& cell : ux_dof_handler_.active_cell_iterators())
        {
            if (!cell->is_locally_owned()) continue;
            cell->get_dof_indices(ux_dofs);

            const auto p_cell = cell->as_dof_handler_iterator(p_dof_handler_);
            p_cell->get_dof_indices(p_dofs);

            // B_ux sparsity: rows = p, cols = ux
            for (unsigned int i = 0; i < p_dpc; ++i)
                for (unsigned int j = 0; j < vel_dpc; ++j)
                    sp_B.add(p_dofs[i], ux_dofs[j]);

            // Bt_ux sparsity: rows = ux, cols = p
            for (unsigned int i = 0; i < vel_dpc; ++i)
                for (unsigned int j = 0; j < p_dpc; ++j)
                    sp_Bt.add(ux_dofs[i], p_dofs[j]);
        }
        sp_B.compress();
        sp_Bt.compress();
        B_ux_.reinit(sp_B);
        Bt_ux_.reinit(sp_Bt);
    }

    // B_uy, Bt_uy: same structure but for uy
    {
        dealii::TrilinosWrappers::SparsityPattern sp_B(
            p_locally_owned_, uy_locally_owned_,
            p_locally_relevant_, mpi_comm_);

        dealii::TrilinosWrappers::SparsityPattern sp_Bt(
            uy_locally_owned_, p_locally_owned_,
            uy_locally_relevant_, mpi_comm_);

        for (const auto& cell : uy_dof_handler_.active_cell_iterators())
        {
            if (!cell->is_locally_owned()) continue;
            cell->get_dof_indices(uy_dofs);

            const auto p_cell = cell->as_dof_handler_iterator(p_dof_handler_);
            p_cell->get_dof_indices(p_dofs);

            for (unsigned int i = 0; i < p_dpc; ++i)
                for (unsigned int j = 0; j < vel_dpc; ++j)
                    sp_B.add(p_dofs[i], uy_dofs[j]);

            for (unsigned int i = 0; i < vel_dpc; ++i)
                for (unsigned int j = 0; j < p_dpc; ++j)
                    sp_Bt.add(uy_dofs[i], p_dofs[j]);
        }
        sp_B.compress();
        sp_Bt.compress();
        B_uy_.reinit(sp_B);
        Bt_uy_.reinit(sp_Bt);
    }

    // M_p: pressure mass matrix (p × p)
    {
        dealii::TrilinosWrappers::SparsityPattern sp(
            p_locally_owned_, p_locally_owned_,
            p_locally_relevant_, mpi_comm_);

        for (const auto& cell : p_dof_handler_.active_cell_iterators())
        {
            if (!cell->is_locally_owned()) continue;
            cell->get_dof_indices(p_dofs);
            p_constraints_.add_entries_local_to_global(p_dofs, sp);
        }
        sp.compress();
        M_p_.reinit(sp);
    }
}

// Explicit instantiations
template void NavierStokesSubsystem<2>::distribute_dofs();
template void NavierStokesSubsystem<2>::build_constraints();
template void NavierStokesSubsystem<2>::build_coupled_system();
template void NavierStokesSubsystem<2>::build_block_sparsity_patterns();
template void NavierStokesSubsystem<2>::allocate_vectors();

template void NavierStokesSubsystem<3>::distribute_dofs();
template void NavierStokesSubsystem<3>::build_constraints();
template void NavierStokesSubsystem<3>::build_coupled_system();
template void NavierStokesSubsystem<3>::build_block_sparsity_patterns();
template void NavierStokesSubsystem<3>::allocate_vectors();
