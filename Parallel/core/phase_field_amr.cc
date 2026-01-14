// ============================================================================
// core/phase_field_amr.cc - Adaptive Mesh Refinement for PhaseFieldProblem
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Section 6.1, Eq. 99 (Kelly-type error indicator)
//
// After AMR, the AMRPostProcessor handles:
//   1. Clamping θ to [-1, 1] (prevents W'(θ) explosion)
//   2. Reprojecting ψ from θ (restores constitutive relation)
// ============================================================================

#include "core/phase_field.h"
#include "core/amr_postprocess.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/error_estimator.h>

#include <iostream>
#include <memory>
#include <limits>
#include <algorithm>

// ============================================================================
// refine_mesh() - Main AMR routine
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::refine_mesh()
{
    if (!params_.mesh.use_amr)
        return;

    if (params_.output.verbose)
        std::cout << "[AMR] Starting mesh refinement...\n";

    // =========================================================================
    // Step 1: Compute error indicators (Kelly estimator, Paper Eq. 99)
    // =========================================================================
    dealii::Vector<float> indicators(triangulation_.n_active_cells());

    dealii::KellyErrorEstimator<dim>::estimate(
        theta_dof_handler_,
        dealii::QGauss<dim - 1>(theta_dof_handler_.get_fe().degree + 1),
        {},
        theta_solution_,
        indicators);

    // =========================================================================
    // Step 2: Mark cells for refinement/coarsening
    // =========================================================================
    dealii::GridRefinement::refine_and_coarsen_fixed_fraction(
        triangulation_,
        indicators,
        params_.mesh.amr_upper_fraction,
        params_.mesh.amr_lower_fraction);

    // Enforce min/max refinement levels
    for (const auto& cell : triangulation_.active_cell_iterators())
    {
        if (cell->level() >= static_cast<int>(params_.mesh.amr_max_level))
            cell->clear_refine_flag();
        if (cell->level() <= static_cast<int>(params_.mesh.amr_min_level))
            cell->clear_coarsen_flag();
    }

    // =========================================================================
    // Step 2b: Interface protection - never coarsen near interface
    // =========================================================================
    std::vector<dealii::types::global_dof_index> dof_indices(
        theta_dof_handler_.get_fe().n_dofs_per_cell());

    const unsigned int interface_min_level = std::max(
        params_.mesh.amr_min_level,
        static_cast<unsigned int>(params_.mesh.initial_refinement));

    const double interface_threshold = params_.mesh.interface_coarsen_threshold;

    for (const auto& cell : theta_dof_handler_.active_cell_iterators())
    {
        cell->get_dof_indices(dof_indices);

        double min_abs_theta = std::numeric_limits<double>::max();
        for (const auto idx : dof_indices)
        {
            const double v = std::min(std::abs(theta_solution_[idx]), 1.0);
            min_abs_theta = std::min(min_abs_theta, v);
        }

        if (min_abs_theta < interface_threshold)
        {
            cell->clear_coarsen_flag();
            if (cell->level() < static_cast<int>(interface_min_level))
                cell->set_refine_flag();
        }
    }

    // =========================================================================
    // Step 2c: Limit coarsening rate to 10%
    // =========================================================================
    const double max_coarsen_rate = 0.10;
    unsigned int n_coarsen_flagged = 0;
    for (const auto& cell : triangulation_.active_cell_iterators())
        if (cell->coarsen_flag_set())
            ++n_coarsen_flagged;

    const unsigned int n_active = triangulation_.n_active_cells();
    const double coarsen_rate = static_cast<double>(n_coarsen_flagged) / n_active;

    if (coarsen_rate > max_coarsen_rate)
    {
        const unsigned int max_coarsen_cells =
            static_cast<unsigned int>(max_coarsen_rate * n_active);

        std::vector<std::pair<float, typename dealii::Triangulation<dim>::active_cell_iterator>>
            coarsen_candidates;

        unsigned int cell_idx = 0;
        for (auto cell = triangulation_.begin_active();
             cell != triangulation_.end(); ++cell, ++cell_idx)
        {
            if (cell->coarsen_flag_set())
                coarsen_candidates.emplace_back(indicators[cell_idx], cell);
        }

        std::sort(coarsen_candidates.begin(), coarsen_candidates.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        for (unsigned int i = max_coarsen_cells; i < coarsen_candidates.size(); ++i)
            coarsen_candidates[i].second->clear_coarsen_flag();
    }

    // Check if any changes needed
    bool any_change = false;
    for (const auto& cell : triangulation_.active_cell_iterators())
    {
        if (cell->refine_flag_set() || cell->coarsen_flag_set())
        {
            any_change = true;
            break;
        }
    }

    if (!any_change)
    {
        if (params_.output.verbose)
            std::cout << "[AMR] No mesh changes needed\n";
        return;
    }

    const unsigned int old_n_cells = triangulation_.n_active_cells();

    // =========================================================================
    // Step 3: Create SolutionTransfer objects
    // NOTE: No psi_transfer - ψ will be reprojected by AMRPostProcessor
    // =========================================================================
    dealii::SolutionTransfer<dim, dealii::Vector<double>> theta_transfer(theta_dof_handler_);
    dealii::SolutionTransfer<dim, dealii::Vector<double>> theta_old_transfer(theta_dof_handler_);

    std::unique_ptr<dealii::SolutionTransfer<dim, dealii::Vector<double>>> phi_transfer;
    if (params_.enable_magnetic)
        phi_transfer = std::make_unique<dealii::SolutionTransfer<dim, dealii::Vector<double>>>(phi_dof_handler_);

    std::unique_ptr<dealii::SolutionTransfer<dim, dealii::Vector<double>>> mx_transfer;
    std::unique_ptr<dealii::SolutionTransfer<dim, dealii::Vector<double>>> my_transfer;
    std::unique_ptr<dealii::SolutionTransfer<dim, dealii::Vector<double>>> mx_old_transfer;
    std::unique_ptr<dealii::SolutionTransfer<dim, dealii::Vector<double>>> my_old_transfer;
    if (params_.enable_magnetic && params_.use_dg_transport)
    {
        mx_transfer = std::make_unique<dealii::SolutionTransfer<dim, dealii::Vector<double>>>(mx_dof_handler_);
        my_transfer = std::make_unique<dealii::SolutionTransfer<dim, dealii::Vector<double>>>(my_dof_handler_);
        mx_old_transfer = std::make_unique<dealii::SolutionTransfer<dim, dealii::Vector<double>>>(mx_dof_handler_);
        my_old_transfer = std::make_unique<dealii::SolutionTransfer<dim, dealii::Vector<double>>>(my_dof_handler_);
    }

    std::unique_ptr<dealii::SolutionTransfer<dim, dealii::Vector<double>>> ux_transfer;
    std::unique_ptr<dealii::SolutionTransfer<dim, dealii::Vector<double>>> uy_transfer;
    std::unique_ptr<dealii::SolutionTransfer<dim, dealii::Vector<double>>> ux_old_transfer;
    std::unique_ptr<dealii::SolutionTransfer<dim, dealii::Vector<double>>> uy_old_transfer;
    std::unique_ptr<dealii::SolutionTransfer<dim, dealii::Vector<double>>> p_transfer;
    if (params_.enable_ns)
    {
        ux_transfer = std::make_unique<dealii::SolutionTransfer<dim, dealii::Vector<double>>>(ux_dof_handler_);
        uy_transfer = std::make_unique<dealii::SolutionTransfer<dim, dealii::Vector<double>>>(uy_dof_handler_);
        ux_old_transfer = std::make_unique<dealii::SolutionTransfer<dim, dealii::Vector<double>>>(ux_dof_handler_);
        uy_old_transfer = std::make_unique<dealii::SolutionTransfer<dim, dealii::Vector<double>>>(uy_dof_handler_);
        p_transfer = std::make_unique<dealii::SolutionTransfer<dim, dealii::Vector<double>>>(p_dof_handler_);
    }

    // =========================================================================
    // Step 4: Prepare for coarsening and refinement
    // =========================================================================
    triangulation_.prepare_coarsening_and_refinement();

    theta_transfer.prepare_for_coarsening_and_refinement(theta_solution_);
    theta_old_transfer.prepare_for_coarsening_and_refinement(theta_old_solution_);

    if (params_.enable_magnetic)
        phi_transfer->prepare_for_coarsening_and_refinement(phi_solution_);

    if (params_.enable_magnetic && params_.use_dg_transport)
    {
        mx_transfer->prepare_for_coarsening_and_refinement(mx_solution_);
        my_transfer->prepare_for_coarsening_and_refinement(my_solution_);
        mx_old_transfer->prepare_for_coarsening_and_refinement(mx_old_solution_);
        my_old_transfer->prepare_for_coarsening_and_refinement(my_old_solution_);
    }

    if (params_.enable_ns)
    {
        ux_transfer->prepare_for_coarsening_and_refinement(ux_solution_);
        uy_transfer->prepare_for_coarsening_and_refinement(uy_solution_);
        ux_old_transfer->prepare_for_coarsening_and_refinement(ux_old_solution_);
        uy_old_transfer->prepare_for_coarsening_and_refinement(uy_old_solution_);
        p_transfer->prepare_for_coarsening_and_refinement(p_solution_);
    }

    // =========================================================================
    // Step 5: Execute refinement
    // =========================================================================
    triangulation_.execute_coarsening_and_refinement();

    if (params_.output.verbose)
    {
        const unsigned int new_n_cells = triangulation_.n_active_cells();
        std::cout << "[AMR] Cells: " << old_n_cells << " -> " << new_n_cells << "\n";
    }

    // =========================================================================
    // Step 6: Setup systems on new mesh
    // =========================================================================
    setup_dof_handlers();
    setup_constraints();
    setup_ch_system();

    if (params_.enable_magnetic)
    {
        setup_poisson_system();
        if (params_.use_dg_transport)
            setup_magnetization_system();
    }

    if (params_.enable_ns)
        setup_ns_system();

    // =========================================================================
    // Step 7: Interpolate solutions to new mesh
    // =========================================================================
    {
        dealii::Vector<double> tmp(theta_dof_handler_.n_dofs());
        theta_transfer.interpolate(tmp);
        theta_solution_ = tmp;
    }
    {
        dealii::Vector<double> tmp(theta_dof_handler_.n_dofs());
        theta_old_transfer.interpolate(tmp);
        theta_old_solution_ = tmp;
    }

    if (params_.enable_magnetic)
    {
        dealii::Vector<double> tmp(phi_dof_handler_.n_dofs());
        phi_transfer->interpolate(tmp);
        phi_solution_ = tmp;
    }

    if (params_.enable_magnetic && params_.use_dg_transport)
    {
        {
            dealii::Vector<double> tmp(mx_dof_handler_.n_dofs());
            mx_transfer->interpolate(tmp);
            mx_solution_ = tmp;
        }
        {
            dealii::Vector<double> tmp(mx_dof_handler_.n_dofs());
            mx_old_transfer->interpolate(tmp);
            mx_old_solution_ = tmp;
        }
        {
            dealii::Vector<double> tmp(my_dof_handler_.n_dofs());
            my_transfer->interpolate(tmp);
            my_solution_ = tmp;
        }
        {
            dealii::Vector<double> tmp(my_dof_handler_.n_dofs());
            my_old_transfer->interpolate(tmp);
            my_old_solution_ = tmp;
        }
    }

    if (params_.enable_ns)
    {
        {
            dealii::Vector<double> tmp(ux_dof_handler_.n_dofs());
            ux_transfer->interpolate(tmp);
            ux_solution_ = tmp;
        }
        {
            dealii::Vector<double> tmp(ux_dof_handler_.n_dofs());
            ux_old_transfer->interpolate(tmp);
            ux_old_solution_ = tmp;
        }
        {
            dealii::Vector<double> tmp(uy_dof_handler_.n_dofs());
            uy_transfer->interpolate(tmp);
            uy_solution_ = tmp;
        }
        {
            dealii::Vector<double> tmp(uy_dof_handler_.n_dofs());
            uy_old_transfer->interpolate(tmp);
            uy_old_solution_ = tmp;
        }
        {
            dealii::Vector<double> tmp(p_dof_handler_.n_dofs());
            p_transfer->interpolate(tmp);
            p_solution_ = tmp;
        }
    }

    // =========================================================================
    // Step 8: Apply constraints
    // =========================================================================
    theta_constraints_.distribute(theta_solution_);
    theta_constraints_.distribute(theta_old_solution_);

    if (params_.enable_magnetic)
        phi_constraints_.distribute(phi_solution_);

    if (params_.enable_ns)
    {
        ux_constraints_.distribute(ux_solution_);
        ux_constraints_.distribute(ux_old_solution_);
        uy_constraints_.distribute(uy_solution_);
        uy_constraints_.distribute(uy_old_solution_);
        p_constraints_.distribute(p_solution_);
    }

    // =========================================================================
    // Step 9: POST-AMR CORRECTIONS (CRITICAL FOR STABILITY)
    //
    // This does:
    //   1. Clamps θ and θ_old to [-1, 1]
    //   2. Reprojects ψ from θ (restores ψ = W'(θ) - ε²Δθ)
    // =========================================================================
    {
        AMRPostProcessor<dim> postproc(params_, params_.output.verbose);
        postproc.process(
            theta_solution_,
            theta_old_solution_,
            psi_solution_,
            theta_dof_handler_,
            psi_dof_handler_,
            theta_constraints_,
            psi_constraints_);
    }

    // =========================================================================
    // Step 10: Rebuild NS combined solution vector
    // =========================================================================
    if (params_.enable_ns)
    {
        const unsigned int n_ns = ux_dof_handler_.n_dofs() +
            uy_dof_handler_.n_dofs() +
            p_dof_handler_.n_dofs();

        ns_solution_.reinit(n_ns);

        for (unsigned int i = 0; i < ux_to_ns_map_.size(); ++i)
            ns_solution_[ux_to_ns_map_[i]] = ux_solution_[i];
        for (unsigned int i = 0; i < uy_to_ns_map_.size(); ++i)
            ns_solution_[uy_to_ns_map_[i]] = uy_solution_[i];
        for (unsigned int i = 0; i < p_to_ns_map_.size(); ++i)
            ns_solution_[p_to_ns_map_[i]] = p_solution_[i];

        ns_combined_constraints_.distribute(ns_solution_);

        // Use direct solver for K steps after AMR (paper recommendation)
        direct_solve_countdown_ = std::max(direct_solve_countdown_, 9);

        // Invalidate Schur preconditioner
        schur_preconditioner_.reset();

        // =================================================================
        // RESTORE DIVERGENCE-FREE VELOCITY
        // Interpolated velocity is NOT div-free. Solve NS to project it.
        // =================================================================
        if (params_.output.verbose)
            std::cout << "[AMR] Restoring divergence-free velocity...\n";

        // Set old = current -> time derivative term becomes zero
        ux_old_solution_ = ux_solution_;
        uy_old_solution_ = uy_solution_;

        // Solve NS to restore div-free constraint
        solve_ns(/*projection_only=*/true);


        // Update old with corrected velocity
        ux_old_solution_ = ux_solution_;
        uy_old_solution_ = uy_solution_;

        if (params_.output.verbose)
            std::cout << "[AMR] Divergence-free velocity restored.\n";

        if (!first_amr_occurred_)
            first_amr_occurred_ = true;
    }

    if (params_.output.verbose)
        std::cout << "[AMR] Mesh refinement complete.\n";
}

// Explicit instantiation
template class PhaseFieldProblem<2>;
