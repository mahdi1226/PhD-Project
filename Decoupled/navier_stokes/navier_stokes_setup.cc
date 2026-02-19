// ============================================================================
// navier_stokes/navier_stokes_setup.cc — Setup Implementation
//
// Implements NSSubsystem<dim>::setup():
//   1. Distribute DoFs for ux (Q2), uy (Q2), p (DG Q1)
//   2. Extract per-component index sets
//   3. Build velocity constraints (hanging nodes + Dirichlet u=0)
//   4. Build pressure constraints (pin DoF 0 for uniqueness)
//   5. Build coupled saddle-point system with contiguous MPI renumbering
//   6. Allocate matrices and vectors (owned + ghosted)
//   7. Assemble pressure mass matrix (for Schur preconditioner)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
//
// PARALLEL FIX: Trilinos Epetra requires CONTIGUOUS row ownership per
// MPI rank. The naive block layout [ux|uy|p] with global offsets creates
// non-contiguous ownership. Solution: renumber so each rank owns a
// contiguous block: [rank0_dofs | rank1_dofs | rank2_dofs | ...]
// where rank_i_dofs = [ux_owned_by_i | uy_owned_by_i | p_owned_by_i]
// ============================================================================

// --- Pressure null-space fix selector ---
// 0 = original (both pins, BROKEN)
// 1 = constraint pin only
// 2 = direct pin only
// 3 = no pin + post-solve mean subtraction (RECOMMENDED)
#ifndef NS_PRESSURE_FIX
#define NS_PRESSURE_FIX 2
#endif

#include "navier_stokes/navier_stokes.h"

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/full_matrix.h>

// ============================================================================
// setup() — call once after mesh is ready
// ============================================================================
template <int dim>
void NSSubsystem<dim>::setup()
{
    // ========================================================================
    // Step 1: Distribute DoFs
    //
    // Velocity:  FE_Q<dim>(degree_velocity) — Q2 continuous (Taylor-Hood)
    // Pressure:  FE_DGP<dim>(degree_pressure) — DG P1 discontinuous
    //            Paper requirement A1: P_{k-1}^{dc} total-degree polynomials
    //            Ensures inf-sup stability with Q_k velocity
    // ========================================================================
    ux_dof_handler_.distribute_dofs(fe_velocity_);
    uy_dof_handler_.distribute_dofs(fe_velocity_);
    p_dof_handler_.distribute_dofs(fe_pressure_);

    // ========================================================================
    // Step 2: Extract per-component index sets
    // ========================================================================
    ux_locally_owned_    = ux_dof_handler_.locally_owned_dofs();
    ux_locally_relevant_ = dealii::DoFTools::extract_locally_relevant_dofs(ux_dof_handler_);
    uy_locally_owned_    = uy_dof_handler_.locally_owned_dofs();
    uy_locally_relevant_ = dealii::DoFTools::extract_locally_relevant_dofs(uy_dof_handler_);
    p_locally_owned_     = p_dof_handler_.locally_owned_dofs();
    p_locally_relevant_  = dealii::DoFTools::extract_locally_relevant_dofs(p_dof_handler_);

    pcout_ << "[NS Setup] DoFs: ux=" << ux_dof_handler_.n_dofs()
           << ", uy=" << uy_dof_handler_.n_dofs()
           << ", p=" << p_dof_handler_.n_dofs() << "\n";

    // ========================================================================
    // Step 3: Build velocity constraints
    //
    // Hanging nodes + homogeneous Dirichlet u=0 on all boundaries (IDs 0-3).
    // ========================================================================
    ux_constraints_.clear();
    ux_constraints_.reinit(ux_locally_owned_, ux_locally_relevant_);
    uy_constraints_.clear();
    uy_constraints_.reinit(uy_locally_owned_, uy_locally_relevant_);

    dealii::DoFTools::make_hanging_node_constraints(ux_dof_handler_, ux_constraints_);
    dealii::DoFTools::make_hanging_node_constraints(uy_dof_handler_, uy_constraints_);

    for (dealii::types::boundary_id bid = 0; bid <= 3; ++bid)
    {
        dealii::VectorTools::interpolate_boundary_values(
            ux_dof_handler_, bid,
            dealii::Functions::ZeroFunction<dim>(), ux_constraints_);
        dealii::VectorTools::interpolate_boundary_values(
            uy_dof_handler_, bid,
            dealii::Functions::ZeroFunction<dim>(), uy_constraints_);
    }

    ux_constraints_.close();
    uy_constraints_.close();

    // ========================================================================
    // Step 4: Build pressure constraints
    //
    // DG pressure: no hanging node constraints (no inter-element continuity).
    //
    // NS_PRESSURE_FIX controls null-space handling:
    //   0 or undefined: original (constraint pin + direct pin in solver) [BROKEN]
    //   1: constraint pin only (no direct pin in solver)
    //   2: direct pin only (no constraint pin here)
    //   3: no pin at all (post-solve mean subtraction in solver)
    // ========================================================================
    p_constraints_.clear();
    p_constraints_.reinit(p_locally_owned_, p_locally_relevant_);
    p_constraints_.close();

    // ========================================================================
    // Step 5: Build coupled saddle-point system
    //
    // Creates monolithic system structure with contiguous per-rank numbering
    // (required by Trilinos Epetra).
    //
    // Layout per rank: [ux_owned | uy_owned | p_owned]
    // Global:          [rank0_block | rank1_block | ... | rankN_block]
    // ========================================================================
    build_coupled_system();

    // ========================================================================
    // Step 6: Allocate matrices and vectors
    // ========================================================================

    // --- Component solutions (owned) ---
    ux_solution_.reinit(ux_locally_owned_, mpi_comm_);
    ux_old_solution_.reinit(ux_locally_owned_, mpi_comm_);
    uy_solution_.reinit(uy_locally_owned_, mpi_comm_);
    uy_old_solution_.reinit(uy_locally_owned_, mpi_comm_);
    p_solution_.reinit(p_locally_owned_, mpi_comm_);

    // --- Ghosted vectors (for assembly and inter-subsystem reads) ---
    ux_relevant_.reinit(ux_locally_owned_, ux_locally_relevant_, mpi_comm_);
    uy_relevant_.reinit(uy_locally_owned_, uy_locally_relevant_, mpi_comm_);
    p_relevant_.reinit(p_locally_owned_, p_locally_relevant_, mpi_comm_);
    ux_old_relevant_.reinit(ux_locally_owned_, ux_locally_relevant_, mpi_comm_);
    uy_old_relevant_.reinit(uy_locally_owned_, uy_locally_relevant_, mpi_comm_);

    // ========================================================================
    // Step 7: Assemble pressure mass matrix (for Schur preconditioner)
    //
    // M_p(i,j) = ∫ φ_i φ_j dx   (DG Q1 mass matrix)
    // Assembled once, reused every solve.
    // Schur complement: S ≈ α M_p, α = ν + 1/Δt
    // ========================================================================
    assemble_pressure_mass_matrix();
}

// ============================================================================
// build_coupled_system() — Build monolithic saddle-point system
//
// Creates index maps, combined constraints, sparsity pattern, and
// allocates the coupled matrix/vectors.
//
// PARALLEL: Contiguous renumbering for Trilinos Epetra compatibility.
// ============================================================================
template <int dim>
void NSSubsystem<dim>::build_coupled_system()
{
    const unsigned int my_rank = dealii::Utilities::MPI::this_mpi_process(mpi_comm_);
    const unsigned int n_ranks = dealii::Utilities::MPI::n_mpi_processes(mpi_comm_);

    const dealii::types::global_dof_index n_ux = ux_dof_handler_.n_dofs();
    const dealii::types::global_dof_index n_uy = uy_dof_handler_.n_dofs();
    const dealii::types::global_dof_index n_p  = p_dof_handler_.n_dofs();
    const dealii::types::global_dof_index n_total = n_ux + n_uy + n_p;

    // --- Per-rank local counts ---
    const dealii::types::global_dof_index n_ux_local = ux_locally_owned_.n_elements();
    const dealii::types::global_dof_index n_uy_local = uy_locally_owned_.n_elements();
    const dealii::types::global_dof_index n_p_local  = p_locally_owned_.n_elements();
    const dealii::types::global_dof_index n_local = n_ux_local + n_uy_local + n_p_local;

    // --- Gather n_local from all ranks to compute global offsets ---
    std::vector<dealii::types::global_dof_index> all_n_local(n_ranks);
    {
        dealii::types::global_dof_index my_n_local = n_local;
        MPI_Allgather(&my_n_local, 1, MPI_UNSIGNED,
                      all_n_local.data(), 1, MPI_UNSIGNED, mpi_comm_);
    }

    // --- Global starting index for this rank (exclusive prefix sum) ---
    dealii::types::global_dof_index global_start = 0;
    for (unsigned int r = 0; r < my_rank; ++r)
        global_start += all_n_local[r];

    // --- Build local-to-global maps for owned DoFs ---
    ux_to_ns_map_.assign(n_ux, dealii::numbers::invalid_dof_index);
    uy_to_ns_map_.assign(n_uy, dealii::numbers::invalid_dof_index);
    p_to_ns_map_.assign(n_p,  dealii::numbers::invalid_dof_index);

    dealii::types::global_dof_index coupled_idx = global_start;

    for (auto it = ux_locally_owned_.begin(); it != ux_locally_owned_.end(); ++it)
        ux_to_ns_map_[*it] = coupled_idx++;

    for (auto it = uy_locally_owned_.begin(); it != uy_locally_owned_.end(); ++it)
        uy_to_ns_map_[*it] = coupled_idx++;

    for (auto it = p_locally_owned_.begin(); it != p_locally_owned_.end(); ++it)
        p_to_ns_map_[*it] = coupled_idx++;

    // --- Exchange mappings for ghost DoFs via MPI_Allreduce(MIN) ---
    // Invalid index is max value, so MPI_MIN picks the valid one.
    {
        std::vector<dealii::types::global_dof_index> recv_ux(n_ux);
        std::vector<dealii::types::global_dof_index> recv_uy(n_uy);
        std::vector<dealii::types::global_dof_index> recv_p(n_p);

        MPI_Allreduce(ux_to_ns_map_.data(), recv_ux.data(),
                      static_cast<int>(n_ux), MPI_UNSIGNED, MPI_MIN, mpi_comm_);
        MPI_Allreduce(uy_to_ns_map_.data(), recv_uy.data(),
                      static_cast<int>(n_uy), MPI_UNSIGNED, MPI_MIN, mpi_comm_);
        MPI_Allreduce(p_to_ns_map_.data(), recv_p.data(),
                      static_cast<int>(n_p), MPI_UNSIGNED, MPI_MIN, mpi_comm_);

        ux_to_ns_map_ = std::move(recv_ux);
        uy_to_ns_map_ = std::move(recv_uy);
        p_to_ns_map_  = std::move(recv_p);
    }

    // --- Build combined IndexSets (contiguous per rank) ---
    ns_locally_owned_.clear();
    ns_locally_owned_.set_size(n_total);
    ns_locally_owned_.add_range(global_start, global_start + n_local);
    ns_locally_owned_.compress();

    ns_locally_relevant_.clear();
    ns_locally_relevant_.set_size(n_total);

    for (auto it = ux_locally_relevant_.begin(); it != ux_locally_relevant_.end(); ++it)
    {
        const auto mapped = ux_to_ns_map_[*it];
        if (mapped != dealii::numbers::invalid_dof_index)
            ns_locally_relevant_.add_index(mapped);
    }
    for (auto it = uy_locally_relevant_.begin(); it != uy_locally_relevant_.end(); ++it)
    {
        const auto mapped = uy_to_ns_map_[*it];
        if (mapped != dealii::numbers::invalid_dof_index)
            ns_locally_relevant_.add_index(mapped);
    }
    for (auto it = p_locally_relevant_.begin(); it != p_locally_relevant_.end(); ++it)
    {
        const auto mapped = p_to_ns_map_[*it];
        if (mapped != dealii::numbers::invalid_dof_index)
            ns_locally_relevant_.add_index(mapped);
    }
    ns_locally_relevant_.compress();

    // --- Build combined constraints (using coupled numbering) ---
    ns_constraints_.clear();
    ns_constraints_.reinit(ns_locally_owned_, ns_locally_relevant_);

    // Helper lambda to map constraints from component to coupled system
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

                ns_constraints_.add_line(coupled_i);

                if (entries != nullptr && !entries->empty())
                {
                    for (const auto& entry : *entries)
                    {
                        const auto coupled_j = index_map[entry.first];
                        ns_constraints_.add_entry(coupled_i, coupled_j, entry.second);
                    }
                }
                ns_constraints_.set_inhomogeneity(coupled_i, inhom);
            }
        }
    };

    map_constraints(ux_constraints_, ux_to_ns_map_, ux_locally_relevant_);
    map_constraints(uy_constraints_, uy_to_ns_map_, uy_locally_relevant_);
    map_constraints(p_constraints_,  p_to_ns_map_,  p_locally_relevant_);

    ns_constraints_.close();

    // --- Build sparsity pattern by cell iteration ---
    dealii::DynamicSparsityPattern dsp(n_total, n_total, ns_locally_relevant_);

    const unsigned int dofs_per_cell_vel = fe_velocity_.n_dofs_per_cell();
    const unsigned int dofs_per_cell_p   = fe_pressure_.n_dofs_per_cell();

    std::vector<dealii::types::global_dof_index> ux_dof_indices(dofs_per_cell_vel);
    std::vector<dealii::types::global_dof_index> uy_dof_indices(dofs_per_cell_vel);
    std::vector<dealii::types::global_dof_index> p_dof_indices(dofs_per_cell_p);

    auto ux_cell = ux_dof_handler_.begin_active();
    auto uy_cell = uy_dof_handler_.begin_active();
    auto p_cell  = p_dof_handler_.begin_active();

    for (; ux_cell != ux_dof_handler_.end(); ++ux_cell, ++uy_cell, ++p_cell)
    {
        if (ux_cell->is_artificial())
            continue;

        ux_cell->get_dof_indices(ux_dof_indices);
        uy_cell->get_dof_indices(uy_dof_indices);
        p_cell->get_dof_indices(p_dof_indices);

        // All 9 blocks of couplings (velocity-velocity, velocity-pressure, etc.)
        for (unsigned int i = 0; i < dofs_per_cell_vel; ++i)
        {
            const auto row_ux = ux_to_ns_map_[ux_dof_indices[i]];
            const auto row_uy = uy_to_ns_map_[uy_dof_indices[i]];

            for (unsigned int j = 0; j < dofs_per_cell_vel; ++j)
            {
                dsp.add(row_ux, ux_to_ns_map_[ux_dof_indices[j]]);  // ux-ux
                dsp.add(row_ux, uy_to_ns_map_[uy_dof_indices[j]]);  // ux-uy
                dsp.add(row_uy, ux_to_ns_map_[ux_dof_indices[j]]);  // uy-ux
                dsp.add(row_uy, uy_to_ns_map_[uy_dof_indices[j]]);  // uy-uy
            }
            for (unsigned int j = 0; j < dofs_per_cell_p; ++j)
            {
                dsp.add(row_ux, p_to_ns_map_[p_dof_indices[j]]);    // ux-p
                dsp.add(row_uy, p_to_ns_map_[p_dof_indices[j]]);    // uy-p
            }
        }

        for (unsigned int i = 0; i < dofs_per_cell_p; ++i)
        {
            const auto row_p = p_to_ns_map_[p_dof_indices[i]];

            for (unsigned int j = 0; j < dofs_per_cell_vel; ++j)
            {
                dsp.add(row_p, ux_to_ns_map_[ux_dof_indices[j]]);   // p-ux
                dsp.add(row_p, uy_to_ns_map_[uy_dof_indices[j]]);   // p-uy
            }
            for (unsigned int j = 0; j < dofs_per_cell_p; ++j)
            {
                dsp.add(row_p, p_to_ns_map_[p_dof_indices[j]]);     // p-p
            }
        }
    }

    ns_constraints_.condense(dsp);

    dealii::SparsityTools::distribute_sparsity_pattern(
        dsp, ns_locally_owned_, mpi_comm_, ns_locally_relevant_);

    dealii::TrilinosWrappers::SparsityPattern ns_sparsity;
    ns_sparsity.reinit(ns_locally_owned_, ns_locally_owned_, dsp, mpi_comm_);

    // --- Allocate coupled system ---
    ns_matrix_.reinit(ns_sparsity);
    ns_rhs_.reinit(ns_locally_owned_, mpi_comm_);
    ns_solution_.reinit(ns_locally_owned_, mpi_comm_);

    pcout_ << "[NS Setup] total=" << n_total
           << ", locally_owned=" << ns_locally_owned_.n_elements()
           << ", range=[" << global_start << ", " << (global_start + n_local) << ")"
           << ", nnz=" << ns_sparsity.n_nonzero_elements() << "\n";
}

// ============================================================================
// assemble_pressure_mass_matrix() — M_p(i,j) = ∫ φ_i φ_j dx
//
// DG Q1 mass matrix for Schur complement preconditioner.
// Assembled once in setup(), reused every solve.
// ============================================================================
template <int dim>
void NSSubsystem<dim>::assemble_pressure_mass_matrix()
{
    const unsigned int dofs_per_cell = fe_pressure_.n_dofs_per_cell();

    // --- Build pressure-only sparsity ---
    dealii::DynamicSparsityPattern dsp(p_locally_relevant_);
    dealii::DoFTools::make_sparsity_pattern(p_dof_handler_, dsp, p_constraints_, false);

    dealii::TrilinosWrappers::SparsityPattern sp;
    sp.reinit(p_locally_owned_, p_locally_owned_, dsp, mpi_comm_);

    pressure_mass_matrix_.reinit(sp);

    // --- Assemble mass matrix ---
    dealii::QGauss<dim> quadrature(fe_pressure_.degree + 1);
    dealii::FEValues<dim> fe_values(fe_pressure_, quadrature,
        dealii::update_values | dealii::update_JxW_values);

    const unsigned int n_q_points = quadrature.size();
    dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto& cell : p_dof_handler_.active_cell_iterators())
    {
        if (cell->is_locally_owned())
        {
            fe_values.reinit(cell);
            cell_matrix = 0;

            for (unsigned int q = 0; q < n_q_points; ++q)
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        cell_matrix(i, j) += fe_values.shape_value(i, q) *
                                             fe_values.shape_value(j, q) *
                                             fe_values.JxW(q);

            cell->get_dof_indices(local_dof_indices);
            p_constraints_.distribute_local_to_global(
                cell_matrix, local_dof_indices, pressure_mass_matrix_);
        }
    }

    pressure_mass_matrix_.compress(dealii::VectorOperation::add);
}

// ============================================================================
// Explicit instantiations (for methods defined in THIS file only)
// ============================================================================
template void NSSubsystem<2>::setup();
template void NSSubsystem<3>::setup();
template void NSSubsystem<2>::build_coupled_system();
template void NSSubsystem<3>::build_coupled_system();
template void NSSubsystem<2>::assemble_pressure_mass_matrix();
template void NSSubsystem<3>::assemble_pressure_mass_matrix();