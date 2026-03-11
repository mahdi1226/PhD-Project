// ============================================================================
// utilities/amr.h — Adaptive Mesh Refinement for Decoupled Ferrofluid Solver
//
// Header-only template implementation (instantiated in the driver).
// Adapted from Semi_Coupled/core/phase_field_amr.cc (Nochetto et al.).
//
// Algorithm:
//   1. Kelly error estimation on theta (interface field)
//   2. Mark cells for refinement/coarsening (fixed-fraction)
//   2b. Absolute error threshold — don't refine cells with tiny error
//   2c. Force bulk coarsening — cells with |theta| > 0.95 everywhere
//   3. Enforce global level limits (max/min)
//   4. Interface protection — never coarsen near phase boundary
//   5. Prepare triangulation
//   6. Create SolutionTransfer objects and prepare
//   7. Execute mesh refinement
//   8. Re-setup all subsystems on new mesh
//   9. Interpolate solutions to new mesh
//  10. Clamp theta to [-1, 1]
//  11. Recompute psi = theta^3 - theta nodally
//  12. Update all ghost vectors
//  13. Log diagnostics (level distribution)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef AMR_H
#define AMR_H

#include "utilities/parameters.h"
#include "cahn_hilliard/cahn_hilliard.h"
#include "navier_stokes/navier_stokes.h"
#include "poisson/poisson.h"
#include "magnetization/magnetization.h"

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/numerics/error_estimator.h>

#include <deal.II/lac/trilinos_vector.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

template <int dim>
void perform_amr(
    dealii::parallel::distributed::Triangulation<dim>& triangulation,
    const Parameters& params,
    MPI_Comm mpi_comm,
    CahnHilliardSubsystem<dim>& ch,
    NSSubsystem<dim>& ns,
    PoissonSubsystem<dim>& poisson,
    MagnetizationSubsystem<dim>& mag,
    bool ns_enabled,
    bool mag_enabled)
{
    using namespace dealii;
    using VectorType = TrilinosWrappers::MPI::Vector;

    ConditionalOStream pcout(std::cout,
        Utilities::MPI::this_mpi_process(mpi_comm) == 0);

    const unsigned int n_cells_before = triangulation.n_global_active_cells();

    // ====================================================================
    // Step 1: Kelly error estimation on theta (interface field)
    // ====================================================================
    Vector<float> indicators(triangulation.n_active_cells());
    KellyErrorEstimator<dim>::estimate(
        ch.get_theta_dof_handler(),
        QGauss<dim - 1>(ch.get_theta_dof_handler().get_fe().degree + 1),
        {},                           // no Neumann BCs
        ch.get_theta_relevant(),      // ghosted theta
        indicators);

    // Log indicator statistics (from Semi_Coupled)
    {
        float local_max = 0.0f, local_sum = 0.0f;
        unsigned int local_count = 0;
        for (const auto& cell : triangulation.active_cell_iterators())
        {
            if (!cell->is_locally_owned()) continue;
            float val = indicators[cell->active_cell_index()];
            local_max = std::max(local_max, val);
            local_sum += val;
            ++local_count;
        }
        float global_max = Utilities::MPI::max(local_max, mpi_comm);
        float global_sum = Utilities::MPI::sum(local_sum, mpi_comm);
        unsigned int global_count = Utilities::MPI::sum(local_count, mpi_comm);
        pcout << "[AMR] Kelly indicators: max=" << global_max
              << " mean=" << (global_count > 0 ? global_sum/global_count : 0.0f) << "\n";
    }

    // ====================================================================
    // Step 2: Mark cells for refinement/coarsening (fixed-fraction)
    // ====================================================================
    parallel::distributed::GridRefinement::refine_and_coarsen_fixed_fraction(
        triangulation,
        indicators,
        params.mesh.amr_upper_fraction,
        params.mesh.amr_lower_fraction);

    // ====================================================================
    // Step 2b: Absolute error threshold — don't refine cells with
    //   small error. Prevents gratuitous refinement of flat interface.
    // ====================================================================
    if (params.mesh.amr_refine_threshold > 0.0)
    {
        unsigned int cleared = 0;
        for (const auto& cell : triangulation.active_cell_iterators())
        {
            if (!cell->is_locally_owned())
                continue;
            if (cell->refine_flag_set() &&
                indicators[cell->active_cell_index()] < params.mesh.amr_refine_threshold)
            {
                cell->clear_refine_flag();
                ++cleared;
            }
        }
        unsigned int global_cleared = Utilities::MPI::sum(cleared, mpi_comm);
        if (global_cleared > 0)
            pcout << "[AMR] Threshold cleared " << global_cleared << " refine flags\n";
    }

    // ====================================================================
    // Step 2c: Force bulk coarsening — cells where ALL DoFs satisfy
    //   |theta| > 0.95 are deep in the bulk (far from interface).
    //   Force coarsening to prevent refine-coarsen oscillation.
    //   (From Semi_Coupled/core/phase_field_amr.cc)
    // ====================================================================
    {
        const double bulk_threshold = 0.95;
        unsigned int force_coarsened = 0;

        std::vector<types::global_dof_index> bulk_dof_indices(
            ch.get_theta_dof_handler().get_fe().n_dofs_per_cell());

        for (const auto& cell : ch.get_theta_dof_handler().active_cell_iterators())
        {
            if (!cell->is_locally_owned())
                continue;

            if (cell->level() <= static_cast<int>(params.mesh.amr_min_level))
                continue;

            cell->get_dof_indices(bulk_dof_indices);

            bool is_bulk = true;
            for (const auto idx : bulk_dof_indices)
            {
                if (std::abs(ch.get_theta_relevant()[idx]) <= bulk_threshold)
                {
                    is_bulk = false;
                    break;
                }
            }

            if (is_bulk)
            {
                cell->clear_refine_flag();
                cell->set_coarsen_flag();
                ++force_coarsened;
            }
        }

        unsigned int global_force_coarsened =
            Utilities::MPI::sum(force_coarsened, mpi_comm);
        if (global_force_coarsened > 0)
            pcout << "[AMR] Force-coarsened " << global_force_coarsened
                  << " bulk cells (|theta| > " << bulk_threshold << ")\n";
    }

    // ====================================================================
    // Step 3: Enforce global refinement level limits
    //
    // amr_max_level and amr_min_level are auto-computed in parameters.cc
    // finalization if not explicitly set:
    //   max_level = initial_refinement + 2
    //   min_level = max(1, initial_refinement - 2)
    // ====================================================================
    for (const auto& cell : triangulation.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;
        if (cell->level() >= static_cast<int>(params.mesh.amr_max_level))
            cell->clear_refine_flag();
        if (cell->level() <= static_cast<int>(params.mesh.amr_min_level))
            cell->clear_coarsen_flag();
    }

    // ====================================================================
    // Step 4: Interface protection — never coarsen near phase boundary
    // ====================================================================
    {
        const double threshold = params.mesh.interface_coarsen_threshold;
        const unsigned int interface_min_level = std::max(
            params.mesh.amr_min_level,
            params.mesh.initial_refinement);

        std::vector<types::global_dof_index> dof_indices(
            ch.get_theta_dof_handler().get_fe().n_dofs_per_cell());

        for (const auto& cell : ch.get_theta_dof_handler().active_cell_iterators())
        {
            if (!cell->is_locally_owned())
                continue;

            cell->get_dof_indices(dof_indices);
            double min_abs_theta = std::numeric_limits<double>::max();
            for (const auto idx : dof_indices)
            {
                const double v = std::min(std::abs(ch.get_theta_relevant()[idx]), 1.0);
                min_abs_theta = std::min(min_abs_theta, v);
            }

            if (min_abs_theta < threshold)
            {
                // Interface cell: protect from coarsening
                cell->clear_coarsen_flag();
                // Force refinement if below initial level
                if (cell->level() < static_cast<int>(interface_min_level))
                    cell->set_refine_flag();
            }
        }
    }

    // ====================================================================
    // Step 5: Prepare triangulation
    // ====================================================================
    triangulation.prepare_coarsening_and_refinement();

    // ====================================================================
    // Step 6: Create SolutionTransfer objects and prepare
    //
    // Pattern: prepare() takes GHOSTED vectors, interpolate() writes
    // to NON-GHOSTED vectors. For non-ghosted old solutions, create
    // temporary ghosted copies.
    // ====================================================================

    // --- 6a. CH: theta (theta DoFHandler) ---
    parallel::distributed::SolutionTransfer<dim, VectorType>
        theta_trans(ch.get_theta_dof_handler_mutable());
    theta_trans.prepare_for_coarsening_and_refinement(ch.get_theta_relevant());

    // --- 6b. CH: psi (psi DoFHandler) ---
    parallel::distributed::SolutionTransfer<dim, VectorType>
        psi_trans(ch.get_psi_dof_handler_mutable());
    psi_trans.prepare_for_coarsening_and_refinement(ch.get_psi_relevant());

    // --- 6c. NS (if enabled): ux, ux_old, uy, uy_old, p ---
    std::unique_ptr<parallel::distributed::SolutionTransfer<dim, VectorType>>
        ux_trans, uy_trans, p_trans;
    VectorType ux_old_ghost, uy_old_ghost;

    if (ns_enabled)
    {
        ux_trans = std::make_unique<parallel::distributed::SolutionTransfer<dim, VectorType>>(
            ns.get_ux_dof_handler_mutable());
        uy_trans = std::make_unique<parallel::distributed::SolutionTransfer<dim, VectorType>>(
            ns.get_uy_dof_handler_mutable());
        p_trans = std::make_unique<parallel::distributed::SolutionTransfer<dim, VectorType>>(
            ns.get_p_dof_handler_mutable());

        // Create ghosted copies of old velocity (for prepare)
        const IndexSet ux_owned = ns.get_ux_dof_handler().locally_owned_dofs();
        IndexSet ux_relevant;
        DoFTools::extract_locally_relevant_dofs(ns.get_ux_dof_handler(), ux_relevant);
        ux_old_ghost.reinit(ux_owned, ux_relevant, mpi_comm);
        ux_old_ghost = ns.get_ux_old_solution_mutable();  // copy owned → ghosted

        const IndexSet uy_owned = ns.get_uy_dof_handler().locally_owned_dofs();
        IndexSet uy_relevant;
        DoFTools::extract_locally_relevant_dofs(ns.get_uy_dof_handler(), uy_relevant);
        uy_old_ghost.reinit(uy_owned, uy_relevant, mpi_comm);
        uy_old_ghost = ns.get_uy_old_solution_mutable();

        // Pack current + old into single transfer per DoFHandler
        std::vector<const VectorType*> ux_pre = {&ns.get_ux_relevant(), &ux_old_ghost};
        ux_trans->prepare_for_coarsening_and_refinement(ux_pre);

        std::vector<const VectorType*> uy_pre = {&ns.get_uy_relevant(), &uy_old_ghost};
        uy_trans->prepare_for_coarsening_and_refinement(uy_pre);

        p_trans->prepare_for_coarsening_and_refinement(ns.get_p_relevant());
    }

    // --- 6d. Poisson (if magnetic enabled): phi ---
    std::unique_ptr<parallel::distributed::SolutionTransfer<dim, VectorType>> phi_trans;
    if (mag_enabled)
    {
        phi_trans = std::make_unique<parallel::distributed::SolutionTransfer<dim, VectorType>>(
            poisson.get_dof_handler_mutable());
        phi_trans->prepare_for_coarsening_and_refinement(
            poisson.get_solution_relevant());
    }

    // --- 6e. Magnetization (if enabled): Mx, My, Mx_old, My_old ---
    std::unique_ptr<parallel::distributed::SolutionTransfer<dim, VectorType>> M_trans;
    if (mag_enabled)
    {
        M_trans = std::make_unique<parallel::distributed::SolutionTransfer<dim, VectorType>>(
            mag.get_dof_handler_mutable());

        // Mag old vectors are already ghosted
        std::vector<const VectorType*> M_pre = {
            &mag.get_Mx_relevant(),
            &mag.get_My_relevant(),
            &mag.get_Mx_old_relevant(),
            &mag.get_My_old_relevant()
        };
        M_trans->prepare_for_coarsening_and_refinement(M_pre);
    }

    // ====================================================================
    // Step 7: Execute mesh refinement
    // ====================================================================
    triangulation.execute_coarsening_and_refinement();

    // ====================================================================
    // Step 8: Re-setup all subsystems on new mesh
    //
    // Each setup() rebuilds: DoFs, constraints, index maps, sparsity,
    // matrices, vectors (zeroed). The vectors will be overwritten in
    // Step 9 by SolutionTransfer::interpolate().
    //
    // CRITICAL: invalidate_ghosts() after each setup() because setup()
    // creates new zeroed ghosted vectors but does NOT reset ghosts_valid_.
    // Without this, update_ghosts() in Step 12 would skip the copy
    // (thinking ghosts are still valid) and leave theta_relevant_ = 0.
    // ====================================================================
    ch.setup();
    ch.invalidate_ghosts();
    if (ns_enabled)
    {
        ns.setup();
        ns.invalidate_ghosts();
    }
    if (mag_enabled)
    {
        poisson.setup();
        poisson.invalidate_ghosts();
        mag.setup();
        mag.invalidate_ghosts();
    }

    // ====================================================================
    // Step 9: Interpolate solutions to new mesh
    // ====================================================================

    // --- 9a. CH: theta ---
    theta_trans.interpolate(ch.get_theta_solution());

    // --- 9b. CH: psi ---
    psi_trans.interpolate(ch.get_psi_solution());

    // --- 9c. NS ---
    if (ns_enabled)
    {
        std::vector<VectorType*> ux_post = {
            &ns.get_ux_solution_mutable(),
            &ns.get_ux_old_solution_mutable()
        };
        ux_trans->interpolate(ux_post);

        std::vector<VectorType*> uy_post = {
            &ns.get_uy_solution_mutable(),
            &ns.get_uy_old_solution_mutable()
        };
        uy_trans->interpolate(uy_post);

        p_trans->interpolate(ns.get_p_solution_mutable());
    }

    // --- 9d. Poisson ---
    if (mag_enabled)
    {
        phi_trans->interpolate(poisson.get_solution_mutable());
    }

    // --- 9e. Magnetization ---
    if (mag_enabled)
    {
        // Mag has 4 vectors packed: Mx, My, Mx_old, My_old
        // Old vectors are ghosted — need temp non-ghosted for interpolate
        const IndexSet M_owned = mag.get_dof_handler().locally_owned_dofs();
        VectorType Mx_tmp(M_owned, mpi_comm);
        VectorType My_tmp(M_owned, mpi_comm);
        VectorType Mx_old_tmp(M_owned, mpi_comm);
        VectorType My_old_tmp(M_owned, mpi_comm);

        std::vector<VectorType*> M_post = {&Mx_tmp, &My_tmp, &Mx_old_tmp, &My_old_tmp};
        M_trans->interpolate(M_post);

        // Copy into subsystem vectors
        mag.get_Mx_solution_mutable() = Mx_tmp;
        mag.get_My_solution_mutable() = My_tmp;
        // For old vectors (ghosted): copy owned part, ghost update in step 12
        mag.get_Mx_old_relevant_mutable() = Mx_old_tmp;
        mag.get_My_old_relevant_mutable() = My_old_tmp;
    }

    // ====================================================================
    // Step 10: Clamp theta to [-1, 1]
    //
    // Interpolation causes overshoot (theta > 1 or < -1).
    // W'(theta) = theta^3 - theta grows rapidly outside [-1,1].
    // ====================================================================
    {
        const IndexSet& theta_owned = ch.get_theta_dof_handler().locally_owned_dofs();
        VectorType& theta_vec = ch.get_theta_solution();

        double local_min = std::numeric_limits<double>::max();
        double local_max = std::numeric_limits<double>::lowest();
        unsigned int n_clamped = 0;

        for (auto idx = theta_owned.begin(); idx != theta_owned.end(); ++idx)
        {
            const double val = theta_vec[*idx];
            local_min = std::min(local_min, val);
            local_max = std::max(local_max, val);

            if (val < -1.0)
            {
                theta_vec[*idx] = -1.0;
                ++n_clamped;
            }
            else if (val > 1.0)
            {
                theta_vec[*idx] = 1.0;
                ++n_clamped;
            }
        }

        const double global_min = Utilities::MPI::min(local_min, mpi_comm);
        const double global_max = Utilities::MPI::max(local_max, mpi_comm);
        const unsigned int total_clamped = Utilities::MPI::sum(n_clamped, mpi_comm);

        pcout << "[AMR] Pre-clamp theta bounds: ["
              << global_min << ", " << global_max << "]";
        if (total_clamped > 0)
            pcout << ", clamped " << total_clamped << " DoFs";
        pcout << "\n";
    }

    // ====================================================================
    // Step 11: Recompute psi = theta^3 - theta nodally
    //
    // The interpolated psi violates the constitutive relation.
    // Cell-local approach: loop over cells, read theta DoF values,
    // compute f(theta) = theta^3 - theta, write to psi DoFs.
    // Same FE and mesh → same local DoF ordering per cell.
    // ====================================================================
    {
        const IndexSet& psi_owned = ch.get_psi_dof_handler().locally_owned_dofs();
        VectorType& psi_vec = ch.get_psi_solution();

        // Ghost theta for reading across cells
        const IndexSet theta_owned = ch.get_theta_dof_handler().locally_owned_dofs();
        IndexSet theta_relevant;
        DoFTools::extract_locally_relevant_dofs(ch.get_theta_dof_handler(), theta_relevant);
        VectorType theta_ghost(theta_owned, theta_relevant, mpi_comm);
        theta_ghost = ch.get_theta_solution();

        const unsigned int dofs_per_cell = ch.get_theta_dof_handler().get_fe().n_dofs_per_cell();
        std::vector<types::global_dof_index> theta_dof_indices(dofs_per_cell);
        std::vector<types::global_dof_index> psi_dof_indices(dofs_per_cell);

        auto theta_cell = ch.get_theta_dof_handler().begin_active();
        auto psi_cell = ch.get_psi_dof_handler().begin_active();
        const auto theta_end = ch.get_theta_dof_handler().end();

        for (; theta_cell != theta_end; ++theta_cell, ++psi_cell)
        {
            if (!theta_cell->is_locally_owned())
                continue;

            theta_cell->get_dof_indices(theta_dof_indices);
            psi_cell->get_dof_indices(psi_dof_indices);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                if (psi_owned.is_element(psi_dof_indices[i]))
                {
                    const double th = theta_ghost[theta_dof_indices[i]];
                    psi_vec[psi_dof_indices[i]] = th * th * th - th;
                }
            }
        }
    }

    // ====================================================================
    // Step 12: Update all ghost vectors
    // ====================================================================
    ch.update_ghosts();
    if (ns_enabled)
        ns.update_ghosts();
    if (mag_enabled)
    {
        poisson.update_ghosts();
        mag.update_ghosts();
    }

    // ====================================================================
    // Step 13: Log diagnostics — cell count + level distribution
    // ====================================================================
    const unsigned int n_cells_after = triangulation.n_global_active_cells();

    {
        int local_min_level = std::numeric_limits<int>::max();
        int local_max_level = 0;
        std::vector<unsigned int> level_counts(20, 0);
        for (const auto& cell : triangulation.active_cell_iterators())
        {
            if (!cell->is_locally_owned()) continue;
            int lev = cell->level();
            local_min_level = std::min(local_min_level, lev);
            local_max_level = std::max(local_max_level, lev);
            if (lev < 20) level_counts[lev]++;
        }
        int global_min = Utilities::MPI::min(local_min_level, mpi_comm);
        int global_max = Utilities::MPI::max(local_max_level, mpi_comm);

        // Sum level counts across MPI
        std::vector<unsigned int> global_counts(20, 0);
        for (int l = 0; l < 20; l++)
            global_counts[l] = Utilities::MPI::sum(level_counts[l], mpi_comm);

        pcout << "[AMR] Cells: " << n_cells_before << " -> " << n_cells_after
              << "  levels=[" << global_min << "," << global_max << "]  (";
        bool first = true;
        for (int l = global_min; l <= global_max; l++)
        {
            if (!first) pcout << "/";
            pcout << "L" << l << ":" << global_counts[l];
            first = false;
        }
        pcout << ")  theta DoFs: " << ch.get_theta_dof_handler().n_dofs() << "\n";
    }
}

#endif // AMR_H
