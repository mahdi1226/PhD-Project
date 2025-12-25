// ============================================================================
// core/phase_field_amr.cc - Adaptive Mesh Refinement for PhaseFieldProblem
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Section 6.1, Eq. 99 (Kelly-type error indicator)
// ============================================================================

#include "core/phase_field.h"

#include <deal.II/grid/grid_refinement.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/derivative_approximation.h>

#include <iostream>
#include <memory>
#include <limits>
#include <algorithm>

template <int dim>
void PhaseFieldProblem<dim>::refine_mesh()
{
    // =========================================================================
    // Check if AMR is enabled
    // =========================================================================
    // Validate parameters (even if AMR disabled, catch config errors)
    Assert(params_.mesh.interface_coarsen_threshold > 0.0 &&
           params_.mesh.interface_coarsen_threshold < 1.0,
           dealii::ExcMessage("interface_coarsen_threshold must be in (0,1)"));
    if (!params_.mesh.use_amr)
    {
        return;
    }

    if (params_.output.verbose)
        std::cout << "[AMR] Starting mesh refinement...\n";

    // =========================================================================
    // Step 1: Compute refinement indicators using multi-field approach
    // Combines phase field (ψ), velocity (ux, uy), and magnetic potential (φ)
    // to capture interface, flow, and magnetic field gradients
    // =========================================================================
    const unsigned int n_cells = triangulation_.n_active_cells();
    dealii::Vector<float> indicators(n_cells);

    // --- OLD METHOD (single field: ψ only) ---
    // dealii::DerivativeApproximation::approximate_gradient(
    //     psi_dof_handler_,
    //     psi_solution_,
    //     indicators);
    // for (const auto& cell : psi_dof_handler_.active_cell_iterators())
    // {
    //     const auto idx = cell->active_cell_index();
    //     indicators[idx] *= cell->diameter();
    // }
    // --- END OLD METHOD ---

    // --- NEW METHOD: Multi-field indicator (ψ + velocity + magnetic) ---
    {
        // Phase field indicator (ψ - chemical potential, sharp at interface)
        dealii::Vector<float> psi_ind(n_cells);
        dealii::DerivativeApproximation::approximate_gradient(
            psi_dof_handler_,
            psi_solution_,
            psi_ind);

        // Start with phase field as baseline
        for (unsigned int i = 0; i < n_cells; ++i)
            indicators[i] = psi_ind[i];

        // Velocity indicators (if NS enabled)
        if (params_.enable_ns)
        {
            dealii::Vector<float> ux_ind(n_cells);
            dealii::Vector<float> uy_ind(n_cells);

            dealii::DerivativeApproximation::approximate_gradient(
                ux_dof_handler_,
                ux_solution_,
                ux_ind);
            dealii::DerivativeApproximation::approximate_gradient(
                uy_dof_handler_,
                uy_solution_,
                uy_ind);

            for (unsigned int i = 0; i < n_cells; ++i)
                indicators[i] = std::max({indicators[i], ux_ind[i], uy_ind[i]});
        }

        // Magnetic potential indicator (if magnetic enabled)
        if (params_.enable_magnetic)
        {
            dealii::Vector<float> phi_ind(n_cells);
            dealii::DerivativeApproximation::approximate_gradient(
                phi_dof_handler_,
                phi_solution_,
                phi_ind);

            for (unsigned int i = 0; i < n_cells; ++i)
                indicators[i] = std::max(indicators[i], phi_ind[i]);
        }
    }
    // --- END NEW METHOD ---

    // Scale by cell diameter (Kelly-type indicator)
    for (const auto& cell : triangulation_.active_cell_iterators())
    {
        const auto idx = cell->active_cell_index();
        AssertIndexRange(idx, indicators.size());
        indicators[idx] *= cell->diameter();
    }
    // =========================================================================
    // Step 2: Mark cells for refinement and coarsening
    // =========================================================================
    dealii::GridRefinement::refine_and_coarsen_fixed_fraction(
        triangulation_,
        indicators,
        params_.mesh.amr_upper_fraction,   // 0.3
        params_.mesh.amr_lower_fraction);  // 0.1

    // Enforce min/max refinement levels
    for (const auto& cell : triangulation_.active_cell_iterators())
    {
        if (cell->level() >= static_cast<int>(params_.mesh.amr_max_level))
            cell->clear_refine_flag();
        if (cell->level() <= static_cast<int>(params_.mesh.amr_min_level))
            cell->clear_coarsen_flag();
    }

    // Protect interface region from coarsening
    // Note: We refine based on |∇ψ| (sharper gradients) but protect based on θ
    // (clear ±1 bounds for interface detection)
    std::vector<dealii::types::global_dof_index> dof_indices(
        theta_dof_handler_.get_fe().n_dofs_per_cell());

    for (const auto& cell : theta_dof_handler_.active_cell_iterators())
    {
        if (cell->coarsen_flag_set())
        {
            cell->get_dof_indices(dof_indices);

            double min_abs_theta = std::numeric_limits<double>::max();
            for (const auto idx : dof_indices)
            {
                // Clamp to handle overshoot (θ can exceed ±1 slightly)
                const double v = std::min(std::abs(theta_solution_[idx]), 1.0);
                min_abs_theta = std::min(min_abs_theta, v);
            }

            if (min_abs_theta < params_.mesh.interface_coarsen_threshold)
                cell->clear_coarsen_flag();
        }
    }

    // Check if any mesh changes will happen
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
    // Step 3: Create SolutionTransfer objects for all ENABLED solution vectors
    // =========================================================================

    // Cahn-Hilliard (always enabled)
    dealii::SolutionTransfer<dim, dealii::Vector<double>> theta_transfer(theta_dof_handler_);
    dealii::SolutionTransfer<dim, dealii::Vector<double>> theta_old_transfer(theta_dof_handler_);
    dealii::SolutionTransfer<dim, dealii::Vector<double>> psi_transfer(psi_dof_handler_);

    // Poisson (only if magnetic enabled)
    std::unique_ptr<dealii::SolutionTransfer<dim, dealii::Vector<double>>> phi_transfer;
    if (params_.enable_magnetic)
    {
        phi_transfer = std::make_unique<dealii::SolutionTransfer<dim, dealii::Vector<double>>>(phi_dof_handler_);
    }

    // Magnetization DG (only if magnetic + DG transport enabled)
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

    // Navier-Stokes (only if NS enabled)
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

    // CH (always)
    theta_transfer.prepare_for_coarsening_and_refinement(theta_solution_);
    theta_old_transfer.prepare_for_coarsening_and_refinement(theta_old_solution_);
    psi_transfer.prepare_for_coarsening_and_refinement(psi_solution_);

    // Poisson (if magnetic)
    if (params_.enable_magnetic)
    {
        phi_transfer->prepare_for_coarsening_and_refinement(phi_solution_);
    }

    // Magnetization (if magnetic + DG)
    if (params_.enable_magnetic && params_.use_dg_transport)
    {
        mx_transfer->prepare_for_coarsening_and_refinement(mx_solution_);
        my_transfer->prepare_for_coarsening_and_refinement(my_solution_);
        mx_old_transfer->prepare_for_coarsening_and_refinement(mx_old_solution_);
        my_old_transfer->prepare_for_coarsening_and_refinement(my_old_solution_);
    }

    // NS (if enabled)
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

    const unsigned int new_n_cells = triangulation_.n_active_cells();

    if (params_.output.verbose)
        std::cout << "[AMR] Cells: " << old_n_cells << " -> " << new_n_cells << "\n";

    // =========================================================================
    // Step 6: Setup systems on new mesh
    // =========================================================================
    setup_dof_handlers();
    setup_constraints();

    // Reinitialize sparsity patterns and matrices
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

    // θ (always)
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

    // =========================================================================
    // Enforce maximum principle for phase field after interpolation
    // (Interpolation can create Gibbs-like oscillations near interface)
    // =========================================================================
    for (unsigned int i = 0; i < theta_solution_.size(); ++i)
    {
        theta_solution_(i) = std::max(-1.0, std::min(1.0, theta_solution_(i)));
        theta_old_solution_(i) = std::max(-1.0, std::min(1.0, theta_old_solution_(i)));
    }

    // ψ (always)
    {
        dealii::Vector<double> tmp(psi_dof_handler_.n_dofs());
        psi_transfer.interpolate(tmp);
        psi_solution_ = tmp;
    }

    // φ (if magnetic)
    if (params_.enable_magnetic)
    {
        dealii::Vector<double> tmp(phi_dof_handler_.n_dofs());
        phi_transfer->interpolate(tmp);
        phi_solution_ = tmp;
    }

    // M_x, M_y (if magnetic + DG)
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

    // u_x, u_y, p (if NS)
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
    // Step 8: Apply constraints to ensure consistency
    // =========================================================================
    theta_constraints_.distribute(theta_solution_);
    theta_constraints_.distribute(theta_old_solution_);
    psi_constraints_.distribute(psi_solution_);

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

    if (params_.output.verbose)
    {
        std::cout << "[AMR] New DoFs: theta=" << theta_dof_handler_.n_dofs();
        if (params_.enable_ns)
            std::cout << ", ux=" << ux_dof_handler_.n_dofs()
                << ", p=" << p_dof_handler_.n_dofs();
        std::cout << "\n";
    }

    // =========================================================================
    // Step 9: Rebuild ns_solution_ and restore divergence-free velocity
    // =========================================================================
    if (params_.enable_ns)
    {
        const unsigned int n_ns = ux_dof_handler_.n_dofs() +
            uy_dof_handler_.n_dofs() +
            p_dof_handler_.n_dofs();

        // Verify maps were rebuilt correctly
        Assert(ux_to_ns_map_.size() == ux_dof_handler_.n_dofs(),
               dealii::ExcMessage("ux_to_ns_map size mismatch after AMR"));
        Assert(uy_to_ns_map_.size() == uy_dof_handler_.n_dofs(),
               dealii::ExcMessage("uy_to_ns_map size mismatch after AMR"));
        Assert(p_to_ns_map_.size() == p_dof_handler_.n_dofs(),
               dealii::ExcMessage("p_to_ns_map size mismatch after AMR"));

        ns_solution_.reinit(n_ns);

        for (unsigned int i = 0; i < ux_to_ns_map_.size(); ++i)
            ns_solution_[ux_to_ns_map_[i]] = ux_solution_[i];
        for (unsigned int i = 0; i < uy_to_ns_map_.size(); ++i)
            ns_solution_[uy_to_ns_map_[i]] = uy_solution_[i];
        for (unsigned int i = 0; i < p_to_ns_map_.size(); ++i)
            ns_solution_[p_to_ns_map_[i]] = p_solution_[i];

        ns_combined_constraints_.distribute(ns_solution_);

        // =====================================================================
        // Step 10: RESTORE DIVERGENCE-FREE VELOCITY
        //
        // After AMR, interpolated velocity is NOT divergence-free.
        // Fix: One NS solve with U_old = U_current enforces div(U) = 0.
        // =====================================================================
        if (params_.output.verbose)
            std::cout << "[AMR] Restoring divergence-free velocity...\n";

        // DEBUG: Verify sizes before solve
        const unsigned int n_ns_expected = ux_dof_handler_.n_dofs() +
            uy_dof_handler_.n_dofs() + p_dof_handler_.n_dofs();

        Assert(ns_solution_.size() == n_ns_expected, dealii::ExcInternalError());
        Assert(ns_matrix_.m() == n_ns_expected, dealii::ExcInternalError());
        Assert(ns_rhs_.size() == n_ns_expected, dealii::ExcInternalError());
        Assert(ux_to_ns_map_.size() == ux_dof_handler_.n_dofs(), dealii::ExcInternalError());
        Assert(uy_to_ns_map_.size() == uy_dof_handler_.n_dofs(), dealii::ExcInternalError());
        Assert(p_to_ns_map_.size() == p_dof_handler_.n_dofs(), dealii::ExcInternalError());


        // Set old = current -> time derivative term becomes zero
        ux_old_solution_ = ux_solution_;
        uy_old_solution_ = uy_solution_;

        // Paper recommends K ≥ 9 direct solves after AMR for stability.
        // Don't reset if already counting down (frequent AMR case)
        direct_solve_countdown_ = std::max(direct_solve_countdown_, 9);  //Only 3 direct solves

        // Invalidate Schur preconditioner (must rebuild with new mesh)
        schur_preconditioner_.reset();

        // Solve NS to restore consistency with incompressibility constraint
        // (interpolated velocity after AMR is generally not divergence-free)
        solve_ns();

        // Update old solutions with the corrected (div-free) velocity
        ux_old_solution_ = ux_solution_;
        uy_old_solution_ = uy_solution_;

        if (params_.output.verbose)
            std::cout << "[AMR] Divergence-free velocity restored\n";

        // Track first AMR for solver optimization
        if (!first_amr_occurred_)
            first_amr_occurred_ = true;
    }
}

// Explicit instantiation
template class PhaseFieldProblem<2>;