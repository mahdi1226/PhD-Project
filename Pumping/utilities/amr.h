// ============================================================================
// utilities/amr.h — Adaptive Mesh Refinement for FHD + CH Solver
//
// Header-only template implementation (instantiated in the driver).
//
// Algorithm:
//   1. Kelly error estimation on theta (phase field, CH component 0)
//   2. Mark cells for refinement/coarsening (fixed-fraction)
//   3. Enforce global level limits (max/min)
//   4. Interface protection — never coarsen near phase boundary
//   5. Prepare triangulation
//   6. Create SolutionTransfer objects and prepare
//   7. Execute mesh refinement
//   8. Re-setup all subsystems on new mesh
//   9. Interpolate solutions to new mesh
//  10. Clamp theta to [-1, 1] (prevent W'(theta) explosion)
//  11. Update all ghost vectors
//  12. Reinitialize driver-local old vectors with new distribution
//  13. Log diagnostics
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016), Section 6.1, Eq. 99
// ============================================================================
#ifndef FHD_AMR_H
#define FHD_AMR_H

#include "cahn_hilliard/cahn_hilliard.h"
#include "navier_stokes/navier_stokes.h"
#include "angular_momentum/angular_momentum.h"
#include "poisson/poisson.h"
#include "magnetization/magnetization.h"
#include "utilities/parameters.h"

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/component_mask.h>

#include <deal.II/numerics/error_estimator.h>

#include <deal.II/lac/trilinos_vector.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

// ============================================================================
// perform_amr: Refine/coarsen mesh adaptively based on phase field gradient.
//
// Takes references to all subsystems and driver-local old solution vectors.
// After AMR:
//   - All subsystems are re-setup on the new mesh
//   - All solutions (current + old) are interpolated
//   - Driver-local old vectors are resized and populated
//   - Ghost vectors are updated
//
// Note: driver-local old vectors (ux_old_rel, etc.) are re-initialized
//       with new index sets and interpolated values.
// ============================================================================
template <int dim>
void perform_amr(
    dealii::parallel::distributed::Triangulation<dim>& triangulation,
    const Parameters& params,
    MPI_Comm mpi_comm,
    // Subsystems
    CahnHilliardSubsystem<dim>& ch,
    NavierStokesSubsystem<dim>& ns,
    AngularMomentumSubsystem<dim>& am,
    PoissonSubsystem<dim>& poisson,
    MagnetizationSubsystem<dim>& mag,
    // Driver-local old solutions (will be resized + interpolated)
    dealii::TrilinosWrappers::MPI::Vector& ux_old_rel,
    dealii::TrilinosWrappers::MPI::Vector& uy_old_rel,
    dealii::TrilinosWrappers::MPI::Vector& w_old_rel,
    dealii::TrilinosWrappers::MPI::Vector& Mx_old,
    dealii::TrilinosWrappers::MPI::Vector& My_old,
    dealii::TrilinosWrappers::MPI::Vector& Mx_relaxed,
    dealii::TrilinosWrappers::MPI::Vector& My_relaxed)
{
    using namespace dealii;
    using VectorType = TrilinosWrappers::MPI::Vector;

    ConditionalOStream pcout(std::cout,
        Utilities::MPI::this_mpi_process(mpi_comm) == 0);

    const unsigned int n_cells_before = triangulation.n_global_active_cells();

    // ====================================================================
    // Step 0: Compute theta mass BEFORE AMR for mass correction
    //
    // ∫ theta dx must be conserved. Interpolation during coarsening is
    // NOT mass-conserving (Q2→coarser Q2 loses accuracy). We compute
    // the mass before and after AMR and add a uniform correction.
    // ====================================================================
    double mass_before = 0.0;
    {
        const QGauss<dim> quadrature(ch.get_fe().degree + 1);
        FEValues<dim> fe_vals(ch.get_fe(), quadrature,
            update_values | update_JxW_values);

        const FEValuesExtractors::Scalar theta_ext(0);  // component 0
        std::vector<double> theta_vals(quadrature.size());

        for (const auto& cell : ch.get_dof_handler().active_cell_iterators())
        {
            if (!cell->is_locally_owned())
                continue;
            fe_vals.reinit(cell);
            fe_vals[theta_ext].get_function_values(ch.get_relevant(), theta_vals);
            for (unsigned int q = 0; q < quadrature.size(); ++q)
                mass_before += theta_vals[q] * fe_vals.JxW(q);
        }
        mass_before = Utilities::MPI::sum(mass_before, mpi_comm);
    }

    // ====================================================================
    // Step 1: Kelly error estimation on theta (CH component 0)
    //
    // CH uses FESystem(FE_Q, 2) with [theta, mu]. We estimate error
    // on theta only using ComponentMask.
    // ====================================================================
    Vector<float> indicators(triangulation.n_active_cells());

    ComponentMask theta_mask(2, false);
    theta_mask.set(0, true);  // component 0 = theta

    KellyErrorEstimator<dim>::estimate(
        ch.get_dof_handler(),
        QGauss<dim - 1>(ch.get_fe().degree + 1),
        {},                      // no Neumann BCs
        ch.get_relevant(),       // ghosted solution [theta, mu]
        indicators,
        theta_mask);

    // ====================================================================
    // Step 2: Mark cells for refinement/coarsening (fixed-fraction)
    // ====================================================================
    parallel::distributed::GridRefinement::refine_and_coarsen_fixed_fraction(
        triangulation,
        indicators,
        params.mesh.amr_upper_fraction,
        params.mesh.amr_lower_fraction);

    // ====================================================================
    // Step 3: Enforce global refinement level limits
    // ====================================================================
    const unsigned int max_level = (params.mesh.amr_max_level > 0)
        ? params.mesh.amr_max_level
        : params.mesh.initial_refinement + 4;
    // Default min_level = initial_refinement: never coarsen below the
    // initial uniform mesh.  Coarsening the bulk loses interface resolution
    // (ε/h drops below ~1) and causes mass loss + instability.
    const unsigned int min_level = (params.mesh.amr_min_level > 0)
        ? params.mesh.amr_min_level
        : params.mesh.initial_refinement;

    for (const auto& cell : triangulation.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;
        if (cell->level() >= static_cast<int>(max_level))
            cell->clear_refine_flag();
        if (cell->level() <= static_cast<int>(min_level))
            cell->clear_coarsen_flag();
    }

    // ====================================================================
    // Step 4: Interface protection — never coarsen near phase boundary
    //
    // Cells where |theta| < threshold (near the diffuse interface)
    // must not be coarsened — this would smear the interface.
    // ====================================================================
    {
        const double threshold = params.mesh.interface_coarsen_threshold;
        const unsigned int interface_min_level = std::max(
            min_level, params.mesh.initial_refinement);

        std::vector<types::global_dof_index> dof_indices(
            ch.get_fe().n_dofs_per_cell());

        for (const auto& cell : ch.get_dof_handler().active_cell_iterators())
        {
            if (!cell->is_locally_owned())
                continue;

            cell->get_dof_indices(dof_indices);
            double min_abs_theta = std::numeric_limits<double>::max();

            // Check only theta DoFs (component 0) — every other DoF in FESystem
            for (unsigned int i = 0; i < dof_indices.size(); ++i)
            {
                if (ch.get_fe().system_to_component_index(i).first == 0)
                {
                    const double v = std::abs(ch.get_relevant()[dof_indices[i]]);
                    min_abs_theta = std::min(min_abs_theta, std::min(v, 1.0));
                }
            }

            if (min_abs_theta < threshold)
            {
                cell->clear_coarsen_flag();
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
    // Pattern: prepare() takes GHOSTED vectors.
    //          interpolate() writes to NON-GHOSTED vectors.
    //
    // For driver-local old vectors (already ghosted), use them directly.
    // ====================================================================

    // --- 6a. CH: solution [theta, mu] + old_solution [theta_old, mu_old] ---
    parallel::distributed::SolutionTransfer<dim, VectorType>
        ch_trans(ch.get_dof_handler());
    // CH old_solution_relevant_ is ghosted, pass both
    std::vector<const VectorType*> ch_in = {
        &ch.get_relevant(),         // current [theta, mu]
        &ch.get_old_relevant()      // old [theta_old, mu_old]
    };
    ch_trans.prepare_for_coarsening_and_refinement(ch_in);

    // --- 6b. NS: ux, uy, p (current) + ux_old, uy_old (driver) ---
    parallel::distributed::SolutionTransfer<dim, VectorType>
        ux_trans(ns.get_ux_dof_handler());
    parallel::distributed::SolutionTransfer<dim, VectorType>
        uy_trans(ns.get_uy_dof_handler());
    parallel::distributed::SolutionTransfer<dim, VectorType>
        p_trans(ns.get_p_dof_handler());

    std::vector<const VectorType*> ux_in = {
        &ns.get_ux_relevant(), &ux_old_rel};
    ux_trans.prepare_for_coarsening_and_refinement(ux_in);

    std::vector<const VectorType*> uy_in = {
        &ns.get_uy_relevant(), &uy_old_rel};
    uy_trans.prepare_for_coarsening_and_refinement(uy_in);

    p_trans.prepare_for_coarsening_and_refinement(ns.get_p_relevant());

    // --- 6c. Angular Momentum: w (current) + w_old (driver) ---
    parallel::distributed::SolutionTransfer<dim, VectorType>
        am_trans(am.get_dof_handler());
    std::vector<const VectorType*> am_in = {
        &am.get_relevant(), &w_old_rel};
    am_trans.prepare_for_coarsening_and_refinement(am_in);

    // --- 6d. Poisson: phi_mag (current) ---
    parallel::distributed::SolutionTransfer<dim, VectorType>
        phi_trans(poisson.get_dof_handler());
    phi_trans.prepare_for_coarsening_and_refinement(
        poisson.get_solution_relevant());

    // --- 6e. Magnetization: Mx, My (current) + Mx_old, My_old (driver) ---
    parallel::distributed::SolutionTransfer<dim, VectorType>
        mag_trans(mag.get_dof_handler());
    std::vector<const VectorType*> mag_in = {
        &mag.get_Mx_relevant(), &mag.get_My_relevant(),
        &Mx_old, &My_old, &Mx_relaxed, &My_relaxed};
    mag_trans.prepare_for_coarsening_and_refinement(mag_in);

    // ====================================================================
    // Step 7: Execute mesh refinement
    // ====================================================================
    triangulation.execute_coarsening_and_refinement();

    // ====================================================================
    // Step 8: Re-setup all subsystems on new mesh
    //
    // Each setup() rebuilds: DoFs, constraints, sparsity patterns,
    // matrices, vectors. Vectors are zeroed — will be overwritten below.
    //
    // CRITICAL: invalidate_ghosts() after setup() so that update_ghosts()
    // will actually copy owned→ghosted (otherwise it might skip the copy
    // thinking ghosts are still valid).
    // ====================================================================
    ch.setup();       ch.invalidate_ghosts();
    ns.setup();       ns.invalidate_ghosts();
    am.setup();       am.invalidate_ghosts();
    poisson.setup();  poisson.invalidate_ghosts();
    mag.setup();      mag.invalidate_ghosts();

    // ====================================================================
    // Step 9: Interpolate solutions to new mesh
    //
    // Create temporary non-ghosted vectors with new DoF distribution,
    // interpolate, then copy into subsystem internal vectors.
    // ====================================================================

    // ====================================================================
    // Step 10: Clamp theta to [-1, 1] on OWNED vectors
    //
    // Must clamp BEFORE copying to subsystem, because old_solution is
    // stored as a ghosted vector — writes to ghosted vectors are forbidden.
    // Interpolation can cause overshoot; W'(theta) = theta^3 - theta
    // grows rapidly outside [-1,1], causing instability.
    //
    // CH uses FESystem(FE_Q, 2) with [theta, mu]. We identify theta
    // DoFs via fe.system_to_component_index(local_i).first == 0.
    // ====================================================================
    auto clamp_theta = [&](VectorType& sol, const char* label)
    {
        const IndexSet& owned = ch.get_dof_handler().locally_owned_dofs();
        const unsigned int dpc = ch.get_fe().n_dofs_per_cell();

        std::vector<types::global_dof_index> dof_indices(dpc);
        unsigned int n_clamped = 0;
        double local_min = std::numeric_limits<double>::max();
        double local_max = std::numeric_limits<double>::lowest();

        for (const auto& cell : ch.get_dof_handler().active_cell_iterators())
        {
            if (!cell->is_locally_owned())
                continue;

            cell->get_dof_indices(dof_indices);
            for (unsigned int i = 0; i < dpc; ++i)
            {
                if (ch.get_fe().system_to_component_index(i).first != 0)
                    continue;

                const auto idx = dof_indices[i];
                if (!owned.is_element(idx))
                    continue;

                const double val = sol[idx];
                local_min = std::min(local_min, val);
                local_max = std::max(local_max, val);

                if (val < -1.0) { sol[idx] = -1.0; ++n_clamped; }
                else if (val > 1.0) { sol[idx] = 1.0; ++n_clamped; }
            }
        }

        const unsigned int total = Utilities::MPI::sum(n_clamped, mpi_comm);
        if (total > 0)
        {
            pcout << "[AMR] " << label << " theta pre-clamp: ["
                  << Utilities::MPI::min(local_min, mpi_comm) << ", "
                  << Utilities::MPI::max(local_max, mpi_comm) << "], clamped "
                  << total << " DoFs\n";
        }
    };

    // --- 9a. CH (with theta clamping on owned vectors) ---
    {
        const IndexSet ch_owned = ch.get_dof_handler().locally_owned_dofs();
        VectorType ch_new(ch_owned, mpi_comm);
        VectorType ch_old_new(ch_owned, mpi_comm);
        std::vector<VectorType*> ch_out = {&ch_new, &ch_old_new};
        ch_trans.interpolate(ch_out);

        // Clamp theta on owned (non-ghosted) vectors before subsystem copy
        clamp_theta(ch_new, "current");
        clamp_theta(ch_old_new, "old");

        // ------------------------------------------------------------------
        // Mass correction: ∫theta must be conserved after AMR
        //
        // Compute mass on new mesh, then add uniform correction delta to
        // theta DoFs so that ∫theta_new = ∫theta_before.
        // ------------------------------------------------------------------
        {
            // Copy to subsystem + update ghosts to compute mass integral
            ch.get_solution_mutable() = ch_new;
            ch.invalidate_ghosts();
            ch.update_ghosts();

            double mass_after = 0.0;
            double domain_area = 0.0;
            {
                const QGauss<dim> quadrature(ch.get_fe().degree + 1);
                FEValues<dim> fe_vals(ch.get_fe(), quadrature,
                    update_values | update_JxW_values);
                const FEValuesExtractors::Scalar theta_ext(0);
                std::vector<double> theta_vals(quadrature.size());

                for (const auto& cell :
                     ch.get_dof_handler().active_cell_iterators())
                {
                    if (!cell->is_locally_owned())
                        continue;
                    fe_vals.reinit(cell);
                    fe_vals[theta_ext].get_function_values(
                        ch.get_relevant(), theta_vals);
                    for (unsigned int q = 0; q < quadrature.size(); ++q)
                    {
                        mass_after += theta_vals[q] * fe_vals.JxW(q);
                        domain_area += fe_vals.JxW(q);
                    }
                }
                mass_after = Utilities::MPI::sum(mass_after, mpi_comm);
                domain_area = Utilities::MPI::sum(domain_area, mpi_comm);
            }

            const double delta = (mass_before - mass_after) / domain_area;
            if (std::abs(delta) > 1e-15)
            {
                pcout << "[AMR] Mass correction: delta = "
                      << std::scientific << std::setprecision(3) << delta
                      << " (mass_before=" << mass_before
                      << ", mass_after=" << mass_after << ")\n";

                // Add delta to theta DoFs in current solution
                const IndexSet& owned = ch.get_dof_handler().locally_owned_dofs();
                const unsigned int dpc = ch.get_fe().n_dofs_per_cell();
                std::vector<types::global_dof_index> dof_indices(dpc);

                for (const auto& cell :
                     ch.get_dof_handler().active_cell_iterators())
                {
                    if (!cell->is_locally_owned())
                        continue;
                    cell->get_dof_indices(dof_indices);
                    for (unsigned int i = 0; i < dpc; ++i)
                    {
                        if (ch.get_fe().system_to_component_index(i).first != 0)
                            continue;
                        const auto idx = dof_indices[i];
                        if (owned.is_element(idx))
                            ch_new[idx] += delta;
                    }
                }

                // Also correct old solution
                for (const auto& cell :
                     ch.get_dof_handler().active_cell_iterators())
                {
                    if (!cell->is_locally_owned())
                        continue;
                    cell->get_dof_indices(dof_indices);
                    for (unsigned int i = 0; i < dpc; ++i)
                    {
                        if (ch.get_fe().system_to_component_index(i).first != 0)
                            continue;
                        const auto idx = dof_indices[i];
                        if (owned.is_element(idx))
                            ch_old_new[idx] += delta;
                    }
                }
            }
        }

        // Final copy to subsystem
        ch.get_solution_mutable() = ch_new;
        ch.get_old_solution_mutable() = ch_old_new;
    }

    // --- 9b. NS ---
    {
        const IndexSet ux_owned = ns.get_ux_dof_handler().locally_owned_dofs();
        const IndexSet uy_owned = ns.get_uy_dof_handler().locally_owned_dofs();
        const IndexSet p_owned = ns.get_p_dof_handler().locally_owned_dofs();

        VectorType ux_new(ux_owned, mpi_comm);
        VectorType ux_old_new(ux_owned, mpi_comm);
        std::vector<VectorType*> ux_out = {&ux_new, &ux_old_new};
        ux_trans.interpolate(ux_out);
        ns.get_ux_solution_mutable() = ux_new;

        VectorType uy_new(uy_owned, mpi_comm);
        VectorType uy_old_new(uy_owned, mpi_comm);
        std::vector<VectorType*> uy_out = {&uy_new, &uy_old_new};
        uy_trans.interpolate(uy_out);
        ns.get_uy_solution_mutable() = uy_new;

        VectorType p_new(p_owned, mpi_comm);
        p_trans.interpolate(p_new);
        ns.get_p_solution_mutable() = p_new;

        // Reinitialize driver-local old vectors with new distribution
        const IndexSet ux_rel_set =
            DoFTools::extract_locally_relevant_dofs(ns.get_ux_dof_handler());
        ux_old_rel.reinit(ux_owned, ux_rel_set, mpi_comm);
        ux_old_rel = ux_old_new;

        const IndexSet uy_rel_set =
            DoFTools::extract_locally_relevant_dofs(ns.get_uy_dof_handler());
        uy_old_rel.reinit(uy_owned, uy_rel_set, mpi_comm);
        uy_old_rel = uy_old_new;
    }

    // --- 9c. Angular Momentum ---
    {
        const IndexSet w_owned = am.get_dof_handler().locally_owned_dofs();
        VectorType w_new(w_owned, mpi_comm);
        VectorType w_old_new(w_owned, mpi_comm);
        std::vector<VectorType*> am_out = {&w_new, &w_old_new};
        am_trans.interpolate(am_out);
        am.get_solution_mutable() = w_new;

        const IndexSet w_rel_set =
            DoFTools::extract_locally_relevant_dofs(am.get_dof_handler());
        w_old_rel.reinit(w_owned, w_rel_set, mpi_comm);
        w_old_rel = w_old_new;
    }

    // --- 9d. Poisson ---
    {
        const IndexSet phi_owned = poisson.get_dof_handler().locally_owned_dofs();
        VectorType phi_new(phi_owned, mpi_comm);
        phi_trans.interpolate(phi_new);
        poisson.get_solution_mutable() = phi_new;
    }

    // --- 9e. Magnetization ---
    {
        const IndexSet M_owned = mag.get_dof_handler().locally_owned_dofs();
        VectorType Mx_new(M_owned, mpi_comm);
        VectorType My_new(M_owned, mpi_comm);
        VectorType Mx_old_new(M_owned, mpi_comm);
        VectorType My_old_new(M_owned, mpi_comm);
        VectorType Mx_rel_new(M_owned, mpi_comm);
        VectorType My_rel_new(M_owned, mpi_comm);

        std::vector<VectorType*> mag_out = {
            &Mx_new, &My_new,
            &Mx_old_new, &My_old_new,
            &Mx_rel_new, &My_rel_new};
        mag_trans.interpolate(mag_out);

        mag.get_Mx_solution_mutable() = Mx_new;
        mag.get_My_solution_mutable() = My_new;

        // Reinitialize driver-local mag old vectors
        const IndexSet M_rel_set =
            DoFTools::extract_locally_relevant_dofs(mag.get_dof_handler());

        Mx_old.reinit(M_owned, M_rel_set, mpi_comm);
        My_old.reinit(M_owned, M_rel_set, mpi_comm);
        Mx_relaxed.reinit(M_owned, M_rel_set, mpi_comm);
        My_relaxed.reinit(M_owned, M_rel_set, mpi_comm);

        Mx_old = Mx_old_new;
        My_old = My_old_new;
        Mx_relaxed = Mx_rel_new;
        My_relaxed = My_rel_new;
    }

    // ====================================================================
    // Step 11: Update all ghost vectors
    // ====================================================================
    ch.update_ghosts();
    ns.update_ghosts();
    am.update_ghosts();
    poisson.update_ghosts();
    mag.update_ghosts();

    // ====================================================================
    // Step 12: Log diagnostics
    // ====================================================================
    const unsigned int n_cells_after = triangulation.n_global_active_cells();
    const unsigned int n_dofs_ch = ch.get_dof_handler().n_dofs();

    pcout << "[AMR] Cells: " << n_cells_before << " -> " << n_cells_after
          << " (CH DoFs: " << n_dofs_ch << ")\n";
}

#endif // FHD_AMR_H
