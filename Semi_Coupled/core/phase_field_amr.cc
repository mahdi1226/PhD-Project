// ============================================================================
// core/phase_field_amr.cc - Adaptive Mesh Refinement (PARALLEL)
//
// Implements AMR for parallel::distributed::Triangulation with Trilinos vectors.
//
// Algorithm:
//   1. Kelly error estimation on θ (interface indicator)
//   2. Mark cells with parallel GridRefinement
//   3. Enforce level limits + interface protection
//   4. SolutionTransfer all fields to new mesh
//   5. Re-setup all subsystems
//   6. Post-process: clamp θ, update ψ
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Section 6.1, Eq. 99 (Kelly-type error indicator)
// ============================================================================

#include "core/phase_field.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/numerics/error_estimator.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

// ============================================================================
// refine_mesh() - Main AMR routine (parallel)
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::refine_mesh()
{
    if (!params_.mesh.use_amr)
        return;

    pcout_ << "[AMR] Starting mesh refinement...\n";

    const unsigned int old_n_cells = triangulation_.n_global_active_cells();

    // =========================================================================
    // Step 1: Compute error indicators (Kelly estimator, Paper Eq. 99)
    // =========================================================================
    dealii::Vector<float> indicators(triangulation_.n_active_cells());

    dealii::KellyErrorEstimator<dim>::estimate(
        theta_dof_handler_,
        dealii::QGauss<dim - 1>(fe_Q2_.degree + 1),
        {},
        theta_relevant_,
        indicators);

    // Log indicator statistics
    {
        float local_max = 0.0f, local_sum = 0.0f;
        unsigned int local_count = 0;
        for (const auto& cell : triangulation_.active_cell_iterators())
        {
            if (!cell->is_locally_owned()) continue;
            float val = indicators[cell->active_cell_index()];
            local_max = std::max(local_max, val);
            local_sum += val;
            ++local_count;
        }
        float global_max = dealii::Utilities::MPI::max(local_max, mpi_communicator_);
        float global_sum = dealii::Utilities::MPI::sum(local_sum, mpi_communicator_);
        unsigned int global_count = dealii::Utilities::MPI::sum(local_count, mpi_communicator_);
        pcout_ << "[AMR] Kelly indicators: max=" << global_max
               << " mean=" << (global_count > 0 ? global_sum/global_count : 0.0f) << "\n";
    }

    // =========================================================================
    // Step 2: Mark cells for refinement/coarsening (parallel version)
    // =========================================================================
    dealii::parallel::distributed::GridRefinement::refine_and_coarsen_fixed_fraction(
        triangulation_,
        indicators,
        params_.mesh.amr_upper_fraction,
        params_.mesh.amr_lower_fraction);

    // =========================================================================
    // Step 2b: Absolute error threshold — don't refine cells with small error
    //   Prevents gratuitous refinement of a flat interface when nothing is happening
    // =========================================================================
    if (params_.mesh.amr_refine_threshold > 0.0)
    {
        unsigned int cleared = 0;
        for (const auto& cell : triangulation_.active_cell_iterators())
        {
            if (!cell->is_locally_owned())
                continue;
            if (cell->refine_flag_set() &&
                indicators[cell->active_cell_index()] < params_.mesh.amr_refine_threshold)
            {
                cell->clear_refine_flag();
                ++cleared;
            }
        }
        unsigned int global_cleared = dealii::Utilities::MPI::sum(cleared, mpi_communicator_);
        if (global_cleared > 0)
            pcout_ << "[AMR] Threshold cleared " << global_cleared << " refine flags\n";
    }

    // =========================================================================
    // Step 2c: Force bulk coarsening — override Kelly estimator for pure bulk
    //   Cells where ALL DoF values satisfy |θ| > threshold are deep in the bulk
    //   (far from interface). Force coarsening to amr_min_level.
    //
    //   IMPORTANT: bulk_threshold must be >= interface_coarsen_threshold (Step 4)
    //   to avoid an oscillation dead zone. If bulk=0.95 but interface=0.9, cells
    //   with DoFs in [0.9, 0.95] get coarsened→interpolated→refined every cycle.
    // =========================================================================
    {
        const double bulk_threshold = 0.99;
        unsigned int force_coarsened = 0;

        std::vector<dealii::types::global_dof_index> bulk_dof_indices(
            theta_dof_handler_.get_fe().n_dofs_per_cell());

        for (const auto& cell : theta_dof_handler_.active_cell_iterators())
        {
            if (!cell->is_locally_owned())
                continue;

            if (cell->level() <= static_cast<int>(params_.mesh.amr_min_level))
                continue;

            cell->get_dof_indices(bulk_dof_indices);

            bool is_bulk = true;
            for (const auto idx : bulk_dof_indices)
            {
                if (std::abs(theta_relevant_[idx]) <= bulk_threshold)
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
            dealii::Utilities::MPI::sum(force_coarsened, mpi_communicator_);
        if (global_force_coarsened > 0)
            pcout_ << "[AMR] Force-coarsened " << global_force_coarsened
                   << " bulk cells (|theta| > " << bulk_threshold << ")\n";
    }

    // =========================================================================
    // Step 3: Enforce min/max refinement levels
    // =========================================================================
    if (params_.mesh.amr_max_level > 0)
    {
        for (const auto& cell : triangulation_.active_cell_iterators())
        {
            if (!cell->is_locally_owned())
                continue;
            if (cell->level() >= static_cast<int>(params_.mesh.amr_max_level))
                cell->clear_refine_flag();
            if (cell->level() <= static_cast<int>(params_.mesh.amr_min_level))
                cell->clear_coarsen_flag();
        }
    }

    // =========================================================================
    // Step 4: Interface protection - never coarsen near interface
    // =========================================================================
    {
        std::vector<dealii::types::global_dof_index> dof_indices(
            theta_dof_handler_.get_fe().n_dofs_per_cell());

        const unsigned int interface_min_level = std::max(
            params_.mesh.amr_min_level,
            static_cast<unsigned int>(params_.mesh.initial_refinement));

        const double interface_threshold = params_.mesh.interface_coarsen_threshold;

        for (const auto& cell : theta_dof_handler_.active_cell_iterators())
        {
            if (!cell->is_locally_owned())
                continue;

            cell->get_dof_indices(dof_indices);

            double min_abs_theta = std::numeric_limits<double>::max();
            for (const auto idx : dof_indices)
            {
                const double v = std::min(std::abs(theta_relevant_[idx]), 1.0);
                min_abs_theta = std::min(min_abs_theta, v);
            }

            if (min_abs_theta < interface_threshold)
            {
                cell->clear_coarsen_flag();
                if (cell->level() < static_cast<int>(interface_min_level))
                    cell->set_refine_flag();
            }
        }
    }

    // =========================================================================
    // Step 5: Prepare triangulation for refinement
    // =========================================================================
    triangulation_.prepare_coarsening_and_refinement();

    // =========================================================================
    // Step 6: Create SolutionTransfer objects and prepare
    //
    // For parallel::distributed::SolutionTransfer, we need ghosted (locally
    // relevant) vectors for prepare_for_coarsening_and_refinement.
    // =========================================================================

    // --- CH fields: θ, θ_old on theta_dof_handler_ ---
    dealii::parallel::distributed::SolutionTransfer<
        dim, dealii::TrilinosWrappers::MPI::Vector>
        theta_trans(theta_dof_handler_);

    // theta_relevant_ and theta_old_relevant_ are already ghosted and up-to-date
    std::vector<const dealii::TrilinosWrappers::MPI::Vector*> theta_pre = {
        &theta_relevant_, &theta_old_relevant_};
    theta_trans.prepare_for_coarsening_and_refinement(theta_pre);

    // --- ψ on psi_dof_handler_ ---
    dealii::parallel::distributed::SolutionTransfer<
        dim, dealii::TrilinosWrappers::MPI::Vector>
        psi_trans(psi_dof_handler_);
    psi_trans.prepare_for_coarsening_and_refinement(psi_relevant_);

    // --- Monolithic magnetics (M+φ) on mag_dof_handler_ ---
    std::unique_ptr<dealii::parallel::distributed::SolutionTransfer<
        dim, dealii::TrilinosWrappers::MPI::Vector>>
        mag_trans;
    if (params_.enable_magnetic)
    {
        mag_trans = std::make_unique<dealii::parallel::distributed::SolutionTransfer<
            dim, dealii::TrilinosWrappers::MPI::Vector>>(mag_dof_handler_);

        // mag_relevant_ and mag_old_relevant_ are already ghosted
        std::vector<const dealii::TrilinosWrappers::MPI::Vector*> mag_pre = {
            &mag_relevant_, &mag_old_relevant_};
        mag_trans->prepare_for_coarsening_and_refinement(mag_pre);
    }

    // --- NS fields on separate dof handlers ---
    std::unique_ptr<dealii::parallel::distributed::SolutionTransfer<
        dim, dealii::TrilinosWrappers::MPI::Vector>>
        ux_trans, uy_trans, p_trans;
    dealii::TrilinosWrappers::MPI::Vector ux_old_ghost, uy_old_ghost;

    if (params_.enable_ns)
    {
        ux_trans = std::make_unique<dealii::parallel::distributed::SolutionTransfer<
            dim, dealii::TrilinosWrappers::MPI::Vector>>(ux_dof_handler_);
        uy_trans = std::make_unique<dealii::parallel::distributed::SolutionTransfer<
            dim, dealii::TrilinosWrappers::MPI::Vector>>(uy_dof_handler_);
        p_trans = std::make_unique<dealii::parallel::distributed::SolutionTransfer<
            dim, dealii::TrilinosWrappers::MPI::Vector>>(p_dof_handler_);

        // Create temporary ghosted copies for old velocity
        ux_old_ghost.reinit(ux_locally_owned_, ux_locally_relevant_, mpi_communicator_);
        uy_old_ghost.reinit(uy_locally_owned_, uy_locally_relevant_, mpi_communicator_);
        ux_old_ghost = ux_old_solution_;
        uy_old_ghost = uy_old_solution_;

        std::vector<const dealii::TrilinosWrappers::MPI::Vector*> ux_pre = {
            &ux_relevant_, &ux_old_ghost};
        ux_trans->prepare_for_coarsening_and_refinement(ux_pre);

        std::vector<const dealii::TrilinosWrappers::MPI::Vector*> uy_pre = {
            &uy_relevant_, &uy_old_ghost};
        uy_trans->prepare_for_coarsening_and_refinement(uy_pre);

        p_trans->prepare_for_coarsening_and_refinement(p_relevant_);
    }

    // =========================================================================
    // Step 7: Execute refinement
    // =========================================================================
    triangulation_.execute_coarsening_and_refinement();

    // =========================================================================
    // Step 8: Re-setup all systems on new mesh
    //
    // This redistributes DoFs, rebuilds constraints, sparsity patterns,
    // matrices, and reinits all vectors (to correct size, zeroed).
    // =========================================================================
    setup_dof_handlers();
    setup_ch_system();

    if (params_.enable_magnetic)
        setup_magnetic_system();

    if (params_.enable_ns)
        setup_ns_system();

    // =========================================================================
    // Step 9: Interpolate solutions to new mesh
    //
    // The setup methods have already reinit'd vectors to correct size.
    // interpolate() overwrites them with transferred data.
    // =========================================================================

    // --- θ, θ_old ---
    {
        std::vector<dealii::TrilinosWrappers::MPI::Vector*> theta_post = {
            &theta_solution_, &theta_old_solution_};
        theta_trans.interpolate(theta_post);
    }

    // --- ψ ---
    psi_trans.interpolate(psi_solution_);

    // --- Monolithic magnetics (M+φ) ---
    if (params_.enable_magnetic)
    {
        std::vector<dealii::TrilinosWrappers::MPI::Vector*> mag_post = {
            &mag_solution_, &mag_old_solution_};
        mag_trans->interpolate(mag_post);
    }

    // --- ux, ux_old ---
    if (params_.enable_ns)
    {
        {
            std::vector<dealii::TrilinosWrappers::MPI::Vector*> ux_post = {
                &ux_solution_, &ux_old_solution_};
            ux_trans->interpolate(ux_post);
        }
        {
            std::vector<dealii::TrilinosWrappers::MPI::Vector*> uy_post = {
                &uy_solution_, &uy_old_solution_};
            uy_trans->interpolate(uy_post);
        }
        p_trans->interpolate(p_solution_);
    }

    // =========================================================================
    // Step 10: Apply constraints
    // =========================================================================
    theta_constraints_.distribute(theta_solution_);
    theta_constraints_.distribute(theta_old_solution_);
    psi_constraints_.distribute(psi_solution_);

    if (params_.enable_magnetic)
    {
        mag_constraints_.distribute(mag_solution_);
        mag_constraints_.distribute(mag_old_solution_);
    }

    if (params_.enable_ns)
    {
        ux_constraints_.distribute(ux_solution_);
        ux_constraints_.distribute(ux_old_solution_);
        uy_constraints_.distribute(uy_solution_);
        uy_constraints_.distribute(uy_old_solution_);
        p_constraints_.distribute(p_solution_);
    }

    // =========================================================================
    // Step 11: Post-process θ — clamp to [-1, 1]
    //
    // Interpolation causes overshoot. For |θ| > 1:
    //   W'(θ) = θ³ - θ grows rapidly → CH instability
    // =========================================================================
    {
        // Log pre-clamping bounds to quantify interpolation overshoot
        double local_min = std::numeric_limits<double>::max();
        double local_max = std::numeric_limits<double>::lowest();
        for (auto idx = theta_locally_owned_.begin(); idx != theta_locally_owned_.end(); ++idx)
        {
            const double val = theta_solution_[*idx];
            local_min = std::min(local_min, val);
            local_max = std::max(local_max, val);
        }
        const double global_min = dealii::Utilities::MPI::min(local_min, mpi_communicator_);
        const double global_max = dealii::Utilities::MPI::max(local_max, mpi_communicator_);
        (void)global_min; (void)global_max;  // suppress unused warnings in non-verbose mode

        pcout_ << "[AMR] Pre-clamp θ bounds: [" << global_min << ", " << global_max << "]\n";

        auto clamp_locally_owned = [](dealii::TrilinosWrappers::MPI::Vector& vec,
                                      const dealii::IndexSet& owned)
        {
            unsigned int count = 0;
            for (auto idx = owned.begin(); idx != owned.end(); ++idx)
            {
                if (vec[*idx] < -1.0)
                {
                    vec[*idx] = -1.0;
                    ++count;
                }
                else if (vec[*idx] > 1.0)
                {
                    vec[*idx] = 1.0;
                    ++count;
                }
            }
            return count;
        };

        unsigned int n_clamped_theta = clamp_locally_owned(theta_solution_, theta_locally_owned_);
        unsigned int n_clamped_old = clamp_locally_owned(theta_old_solution_, theta_locally_owned_);

        // Re-apply constraints after clamping
        theta_constraints_.distribute(theta_solution_);
        theta_constraints_.distribute(theta_old_solution_);

        unsigned int total_clamped = dealii::Utilities::MPI::sum(
            n_clamped_theta + n_clamped_old, mpi_communicator_);
        if (total_clamped > 0)
            pcout_ << "[AMR] Clamped " << total_clamped << " theta DoFs to [-1,1]\n";
    }

    // =========================================================================
    // Step 12: Set ψ = W'(θ) nodally (simple approximation)
    //
    // The full L² reprojection is unnecessary because the very next CH solve
    // will produce a consistent (θ, ψ) pair from the weak form.
    // =========================================================================
    {
        for (auto idx = psi_locally_owned_.begin(); idx != psi_locally_owned_.end(); ++idx)
        {
            const double theta_val = theta_solution_[*idx];
            psi_solution_[*idx] = theta_val * theta_val * theta_val - theta_val;
        }
        psi_constraints_.distribute(psi_solution_);
    }

    // =========================================================================
    // Step 13: Update all ghosted vectors
    // =========================================================================
    theta_relevant_ = theta_solution_;
    theta_old_relevant_ = theta_old_solution_;
    psi_relevant_ = psi_solution_;

    if (params_.enable_magnetic)
    {
        mag_relevant_ = mag_solution_;
        mag_old_relevant_ = mag_old_solution_;
        extract_magnetic_components();  // fills phi/Mx/My auxiliary vectors
    }

    if (params_.enable_ns)
    {
        ux_relevant_ = ux_solution_;
        uy_relevant_ = uy_solution_;
        p_relevant_ = p_solution_;
    }

    // =========================================================================
    // Step 14: Rebuild NS combined solution vector
    // =========================================================================
    if (params_.enable_ns)
    {
        ns_solution_.reinit(ns_locally_owned_, mpi_communicator_);
        for (unsigned int i = 0; i < ux_to_ns_map_.size(); ++i)
            if (ux_locally_owned_.is_element(i))
                ns_solution_[ux_to_ns_map_[i]] = ux_solution_[i];
        for (unsigned int i = 0; i < uy_to_ns_map_.size(); ++i)
            if (uy_locally_owned_.is_element(i))
                ns_solution_[uy_to_ns_map_[i]] = uy_solution_[i];
        for (unsigned int i = 0; i < p_to_ns_map_.size(); ++i)
            if (p_locally_owned_.is_element(i))
                ns_solution_[p_to_ns_map_[i]] = p_solution_[i];

        ns_solution_.compress(dealii::VectorOperation::insert);
        ns_constraints_.distribute(ns_solution_);
    }

    const unsigned int new_n_cells = triangulation_.n_global_active_cells();

    // Level distribution diagnostic
    {
        int local_min_level = std::numeric_limits<int>::max();
        int local_max_level = 0;
        std::vector<unsigned int> level_counts(20, 0);
        for (const auto& cell : triangulation_.active_cell_iterators())
        {
            if (!cell->is_locally_owned()) continue;
            int lev = cell->level();
            local_min_level = std::min(local_min_level, lev);
            local_max_level = std::max(local_max_level, lev);
            if (lev < 20) level_counts[lev]++;
        }
        int global_min = dealii::Utilities::MPI::min(local_min_level, mpi_communicator_);
        int global_max = dealii::Utilities::MPI::max(local_max_level, mpi_communicator_);
        pcout_ << "[AMR] Cells: " << old_n_cells << " -> " << new_n_cells
               << "  levels=[" << global_min << "," << global_max << "]";
        // Sum level counts across MPI
        std::vector<unsigned int> global_counts(20, 0);
        for (int l = 0; l < 20; l++)
            global_counts[l] = dealii::Utilities::MPI::sum(level_counts[l], mpi_communicator_);
        pcout_ << "  (";
        bool first = true;
        for (int l = global_min; l <= global_max; l++)
        {
            if (!first) pcout_ << "/";
            pcout_ << "L" << l << ":" << global_counts[l];
            first = false;
        }
        pcout_ << ")\n";
    }
    pcout_ << "[AMR] Mesh refinement complete.\n";
}

// Explicit instantiation
template class PhaseFieldProblem<2>;
