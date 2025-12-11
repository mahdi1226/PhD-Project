// ============================================================================
// core/nsch_problem_amr.cc - Adaptive Mesh Refinement with separate transfers
//
// REFACTORED VERSION: Uses separate SolutionTransfer for each scalar field
// This FIXES the deal.II 9.7 issue with BlockVector + FESystem + hanging nodes
//
// Based on: Nochetto, Salgado & Tomas (2016)
// "A diffuse interface model for two-phase ferrofluid flows"
// ============================================================================
#include "core/nsch_problem.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/derivative_approximation.h>

#include <iostream>
#include <cmath>
#include <limits>

// ============================================================================
// Compute refinement indicators based on phase field gradient
// ============================================================================
template <int dim>
void NSCHProblem<dim>::compute_refinement_indicators(dealii::Vector<float>& indicators) const
{
    indicators.reinit(triangulation_.n_active_cells());

    if (params_.amr_indicator_type == 0)
    {
        // Type 0: Gradient-based refinement (sharp interface tracking)
        // Refine where |grad(c)| is large
        dealii::DerivativeApproximation::approximate_gradient(
            c_dof_handler_,
            c_solution_,
            indicators);
    }
    else if (params_.amr_indicator_type == 1)
    {
        // Type 1: Kelly error estimator
        dealii::KellyErrorEstimator<dim>::estimate(
            c_dof_handler_,
            dealii::QGauss<dim-1>(params_.fe_degree_phase + 1),
            {},  // No Neumann boundary functions
            c_solution_,
            indicators);
    }
    else if (params_.amr_indicator_type == 2)
    {
        // Type 2: Combined gradient of c and velocity magnitude
        dealii::Vector<float> c_indicators(triangulation_.n_active_cells());
        dealii::Vector<float> u_indicators(triangulation_.n_active_cells());

        dealii::DerivativeApproximation::approximate_gradient(
            c_dof_handler_,
            c_solution_,
            c_indicators);

        dealii::DerivativeApproximation::approximate_gradient(
            ux_dof_handler_,
            ux_solution_,
            u_indicators);

        // Combine: prioritize interface (c) but also consider flow
        for (unsigned int i = 0; i < indicators.size(); ++i)
            indicators[i] = c_indicators[i] + 0.1f * u_indicators[i];
    }
}

// ============================================================================
// Refine mesh with proper solution transfer for separate scalar fields
// THIS IS THE KEY FIX for the AMR crash issue
// ============================================================================
template <int dim>
void NSCHProblem<dim>::refine_mesh()
{
    if (params_.verbose)
        std::cout << "[AMR] Refining mesh...\n";

    // ========================================================================
    // Step 1: Compute refinement indicators
    // ========================================================================
    dealii::Vector<float> indicators;
    compute_refinement_indicators(indicators);

    // ========================================================================
    // Step 2: Mark cells for refinement (no coarsening to avoid complications)
    // ========================================================================
    dealii::GridRefinement::refine_and_coarsen_fixed_fraction(
        triangulation_,
        indicators,
        params_.amr_refine_fraction,
        0.0);  // No coarsening for stability

    // Enforce min/max refinement levels
    for (const auto& cell : triangulation_.active_cell_iterators())
    {
        if (cell->level() >= static_cast<int>(params_.amr_max_level))
            cell->clear_refine_flag();
        if (cell->level() <= static_cast<int>(params_.amr_min_level))
            cell->clear_coarsen_flag();
        // Clear all coarsen flags for stability
        cell->clear_coarsen_flag();
    }

    // Check if any refinement will happen
    bool any_refinement = false;
    for (const auto& cell : triangulation_.active_cell_iterators())
    {
        if (cell->refine_flag_set())
        {
            any_refinement = true;
            break;
        }
    }

    if (!any_refinement)
    {
        if (params_.verbose)
            std::cout << "[AMR] No refinement needed\n";
        return;
    }

    // ========================================================================
    // Step 3: Create SolutionTransfer objects for EACH scalar field
    // This is the FIX: using separate transfers instead of BlockVector transfer
    // ========================================================================
    dealii::SolutionTransfer<dim, dealii::Vector<double>> c_transfer(c_dof_handler_);
    dealii::SolutionTransfer<dim, dealii::Vector<double>> mu_transfer(mu_dof_handler_);
    dealii::SolutionTransfer<dim, dealii::Vector<double>> ux_transfer(ux_dof_handler_);
    dealii::SolutionTransfer<dim, dealii::Vector<double>> uy_transfer(uy_dof_handler_);
    dealii::SolutionTransfer<dim, dealii::Vector<double>> p_transfer(p_dof_handler_);
    dealii::SolutionTransfer<dim, dealii::Vector<double>> phi_transfer(phi_dof_handler_);

    // Also transfer old solutions for time stepping
    dealii::SolutionTransfer<dim, dealii::Vector<double>> c_old_transfer(c_dof_handler_);
    dealii::SolutionTransfer<dim, dealii::Vector<double>> ux_old_transfer(ux_dof_handler_);
    dealii::SolutionTransfer<dim, dealii::Vector<double>> uy_old_transfer(uy_dof_handler_);

    // ========================================================================
    // Step 4: Prepare for coarsening and refinement
    // ========================================================================
    triangulation_.prepare_coarsening_and_refinement();

    // Prepare each transfer with its solution vector
    c_transfer.prepare_for_coarsening_and_refinement(c_solution_);
    c_old_transfer.prepare_for_coarsening_and_refinement(c_old_solution_);
    mu_transfer.prepare_for_coarsening_and_refinement(mu_solution_);
    ux_transfer.prepare_for_coarsening_and_refinement(ux_solution_);
    ux_old_transfer.prepare_for_coarsening_and_refinement(ux_old_solution_);
    uy_transfer.prepare_for_coarsening_and_refinement(uy_solution_);
    uy_old_transfer.prepare_for_coarsening_and_refinement(uy_old_solution_);
    p_transfer.prepare_for_coarsening_and_refinement(p_solution_);

    if (params_.enable_magnetic)
        phi_transfer.prepare_for_coarsening_and_refinement(phi_solution_);

    // ========================================================================
    // Step 5: Execute mesh refinement
    // ========================================================================
    triangulation_.execute_coarsening_and_refinement();

    // ========================================================================
    // Step 6: Setup systems on new mesh (redistribute DoFs, constraints, etc.)
    // ========================================================================
    bool old_verbose = params_.verbose;
    const_cast<NSCHParameters&>(params_).verbose = false;  // Suppress verbose during setup

    setup_all_systems();

    const_cast<NSCHParameters&>(params_).verbose = old_verbose;

    // ========================================================================
    // Step 7: Interpolate solutions to new mesh
    // Each scalar field is transferred independently - no conflicts!
    // ========================================================================

    // Concentration
    {
        dealii::Vector<double> tmp(c_dof_handler_.n_dofs());
        c_transfer.interpolate(tmp);
        c_solution_ = tmp;
    }
    {
        dealii::Vector<double> tmp(c_dof_handler_.n_dofs());
        c_old_transfer.interpolate(tmp);
        c_old_solution_ = tmp;
    }

    // Chemical potential
    {
        dealii::Vector<double> tmp(mu_dof_handler_.n_dofs());
        mu_transfer.interpolate(tmp);
        mu_solution_ = tmp;
    }

    // Velocity x
    {
        dealii::Vector<double> tmp(ux_dof_handler_.n_dofs());
        ux_transfer.interpolate(tmp);
        ux_solution_ = tmp;
    }
    {
        dealii::Vector<double> tmp(ux_dof_handler_.n_dofs());
        ux_old_transfer.interpolate(tmp);
        ux_old_solution_ = tmp;
    }

    // Velocity y
    {
        dealii::Vector<double> tmp(uy_dof_handler_.n_dofs());
        uy_transfer.interpolate(tmp);
        uy_solution_ = tmp;
    }
    {
        dealii::Vector<double> tmp(uy_dof_handler_.n_dofs());
        uy_old_transfer.interpolate(tmp);
        uy_old_solution_ = tmp;
    }

    // Pressure
    {
        dealii::Vector<double> tmp(p_dof_handler_.n_dofs());
        p_transfer.interpolate(tmp);
        p_solution_ = tmp;
    }

    // Magnetic potential
    if (params_.enable_magnetic)
    {
        dealii::Vector<double> tmp(phi_dof_handler_.n_dofs());
        phi_transfer.interpolate(tmp);
        phi_solution_ = tmp;
    }

    // ========================================================================
    // Step 8: Apply constraints to ensure consistency with hanging nodes
    // ========================================================================
    c_constraints_.distribute(c_solution_);
    c_constraints_.distribute(c_old_solution_);
    mu_constraints_.distribute(mu_solution_);
    ux_constraints_.distribute(ux_solution_);
    ux_constraints_.distribute(ux_old_solution_);
    uy_constraints_.distribute(uy_solution_);
    uy_constraints_.distribute(uy_old_solution_);
    p_constraints_.distribute(p_solution_);

    if (params_.enable_magnetic)
        phi_constraints_.distribute(phi_solution_);

    // ========================================================================
    // Report
    // ========================================================================
    if (params_.verbose)
    {
        std::cout << "[AMR] New mesh: " << triangulation_.n_active_cells() << " cells\n";
        std::cout << "[AMR] DoFs: c=" << c_dof_handler_.n_dofs()
                  << ", mu=" << mu_dof_handler_.n_dofs()
                  << ", ux=" << ux_dof_handler_.n_dofs()
                  << ", uy=" << uy_dof_handler_.n_dofs()
                  << ", p=" << p_dof_handler_.n_dofs();
        if (params_.enable_magnetic)
            std::cout << ", phi=" << phi_dof_handler_.n_dofs();
        std::cout << "\n";
        std::cout << "[AMR] Total DoFs: " << get_total_dofs() << "\n";
    }
}

// ============================================================================
// Explicit instantiation
// ============================================================================
template class NSCHProblem<2>;