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

template <int dim>
void PhaseFieldProblem<dim>::refine_mesh()
{
    // =========================================================================
    // Check if AMR is enabled
    // =========================================================================
    if (!params_.mesh.use_amr)
    {
        return;
    }

    if (params_.output.verbose)
        std::cout << "[AMR] Starting mesh refinement...\n";

    // =========================================================================
    // Step 1: Compute refinement indicators based on |∇θ|
    // Paper Eq. 99: η_T = h_T * ||∂θ/∂n||_{∂T}
    // We use gradient approximation as a simpler alternative
    // =========================================================================
    dealii::Vector<float> indicators(triangulation_.n_active_cells());

    dealii::DerivativeApproximation::approximate_gradient(
        theta_dof_handler_,
        theta_solution_,
        indicators);

    // Scale by cell diameter (Kelly-type indicator)
    unsigned int cell_idx = 0;
    for (const auto& cell : theta_dof_handler_.active_cell_iterators())
    {
        indicators[cell_idx] *= cell->diameter();
        ++cell_idx;
    }

    // =========================================================================
    // Step 2: Mark cells for refinement (no coarsening for stability)
    // Paper uses Dörfler marking with both refinement and coarsening,
    // but we disable coarsening for simplicity and stability
    // =========================================================================
    dealii::GridRefinement::refine_and_coarsen_fixed_fraction(
        triangulation_,
        indicators,
        params_.mesh.amr_upper_fraction,  // Refine top 30%
        0.0);  // No coarsening (paper allows it, but risky for stability)

    // Enforce min/max refinement levels
    for (const auto& cell : triangulation_.active_cell_iterators())
    {
        if (cell->level() >= static_cast<int>(params_.mesh.amr_max_level))
            cell->clear_refine_flag();
        if (cell->level() <= static_cast<int>(params_.mesh.amr_min_level))
            cell->clear_coarsen_flag();
        cell->clear_coarsen_flag();  // Always clear coarsen for stability
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
        if (params_.output.verbose)
            std::cout << "[AMR] No refinement needed\n";
        return;
    }

    const unsigned int old_n_cells = triangulation_.n_active_cells();

    // =========================================================================
    // Step 3: Create SolutionTransfer objects for all solution vectors
    // =========================================================================
    dealii::SolutionTransfer<dim, dealii::Vector<double>> theta_transfer(theta_dof_handler_);
    dealii::SolutionTransfer<dim, dealii::Vector<double>> theta_old_transfer(theta_dof_handler_);
    dealii::SolutionTransfer<dim, dealii::Vector<double>> psi_transfer(psi_dof_handler_);
    dealii::SolutionTransfer<dim, dealii::Vector<double>> phi_transfer(phi_dof_handler_);

    // Magnetization (DG)
    dealii::SolutionTransfer<dim, dealii::Vector<double>> mx_transfer(mx_dof_handler_);
    dealii::SolutionTransfer<dim, dealii::Vector<double>> my_transfer(my_dof_handler_);
    dealii::SolutionTransfer<dim, dealii::Vector<double>> mx_old_transfer(mx_dof_handler_);
    dealii::SolutionTransfer<dim, dealii::Vector<double>> my_old_transfer(my_dof_handler_);

    // Velocity and pressure
    dealii::SolutionTransfer<dim, dealii::Vector<double>> ux_transfer(ux_dof_handler_);
    dealii::SolutionTransfer<dim, dealii::Vector<double>> uy_transfer(uy_dof_handler_);
    dealii::SolutionTransfer<dim, dealii::Vector<double>> ux_old_transfer(ux_dof_handler_);
    dealii::SolutionTransfer<dim, dealii::Vector<double>> uy_old_transfer(uy_dof_handler_);
    dealii::SolutionTransfer<dim, dealii::Vector<double>> p_transfer(p_dof_handler_);

    // =========================================================================
    // Step 4: Prepare for coarsening and refinement
    // =========================================================================
    triangulation_.prepare_coarsening_and_refinement();

    theta_transfer.prepare_for_coarsening_and_refinement(theta_solution_);
    theta_old_transfer.prepare_for_coarsening_and_refinement(theta_old_solution_);
    psi_transfer.prepare_for_coarsening_and_refinement(psi_solution_);
    phi_transfer.prepare_for_coarsening_and_refinement(phi_solution_);

    mx_transfer.prepare_for_coarsening_and_refinement(mx_solution_);
    my_transfer.prepare_for_coarsening_and_refinement(my_solution_);
    mx_old_transfer.prepare_for_coarsening_and_refinement(mx_old_solution_);
    my_old_transfer.prepare_for_coarsening_and_refinement(my_old_solution_);

    ux_transfer.prepare_for_coarsening_and_refinement(ux_solution_);
    uy_transfer.prepare_for_coarsening_and_refinement(uy_solution_);
    ux_old_transfer.prepare_for_coarsening_and_refinement(ux_old_solution_);
    uy_old_transfer.prepare_for_coarsening_and_refinement(uy_old_solution_);
    p_transfer.prepare_for_coarsening_and_refinement(p_solution_);

    // =========================================================================
    // Step 5: Execute refinement
    // =========================================================================
    triangulation_.execute_coarsening_and_refinement();

    const unsigned int new_n_cells = triangulation_.n_active_cells();

    if (params_.output.verbose)
        std::cout << "[AMR] Cells: " << old_n_cells << " → " << new_n_cells << "\n";

    // =========================================================================
    // Step 6: Setup systems on new mesh
    // =========================================================================
    setup_dof_handlers();
    setup_constraints();

    // Reinitialize sparsity patterns and matrices
    setup_ch_system();
    if (params_.magnetic.enabled)
    {
        setup_poisson_system();
        if (params_.magnetic.use_dg_transport)
            setup_magnetization_system();
    }
    if (params_.ns.enabled)
        setup_ns_system();

    // =========================================================================
    // Step 7: Interpolate solutions to new mesh
    // =========================================================================

    // θ
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

    // ψ
    {
        dealii::Vector<double> tmp(psi_dof_handler_.n_dofs());
        psi_transfer.interpolate(tmp);
        psi_solution_ = tmp;
    }

    // φ
    {
        dealii::Vector<double> tmp(phi_dof_handler_.n_dofs());
        phi_transfer.interpolate(tmp);
        phi_solution_ = tmp;
    }

    // M_x
    {
        dealii::Vector<double> tmp(mx_dof_handler_.n_dofs());
        mx_transfer.interpolate(tmp);
        mx_solution_ = tmp;
    }
    {
        dealii::Vector<double> tmp(mx_dof_handler_.n_dofs());
        mx_old_transfer.interpolate(tmp);
        mx_old_solution_ = tmp;
    }

    // M_y
    {
        dealii::Vector<double> tmp(my_dof_handler_.n_dofs());
        my_transfer.interpolate(tmp);
        my_solution_ = tmp;
    }
    {
        dealii::Vector<double> tmp(my_dof_handler_.n_dofs());
        my_old_transfer.interpolate(tmp);
        my_old_solution_ = tmp;
    }

    // u_x
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

    // u_y
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

    // p
    {
        dealii::Vector<double> tmp(p_dof_handler_.n_dofs());
        p_transfer.interpolate(tmp);
        p_solution_ = tmp;
    }

    // =========================================================================
    // Step 8: Apply constraints to ensure consistency
    // NOTE: DG magnetization (mx, my) has no constraints (discontinuous space)
    // =========================================================================
    theta_constraints_.distribute(theta_solution_);
    theta_constraints_.distribute(theta_old_solution_);
    psi_constraints_.distribute(psi_solution_);
    phi_constraints_.distribute(phi_solution_);
    // mx, my are DG0 - no constraints needed
    ux_constraints_.distribute(ux_solution_);
    ux_constraints_.distribute(ux_old_solution_);
    uy_constraints_.distribute(uy_solution_);
    uy_constraints_.distribute(uy_old_solution_);
    p_constraints_.distribute(p_solution_);

    if (params_.output.verbose)
    {
        std::cout << "[AMR] New DoFs: θ=" << theta_dof_handler_.n_dofs()
                  << ", ux=" << ux_dof_handler_.n_dofs()
                  << ", p=" << p_dof_handler_.n_dofs() << "\n";
    }
}

// Explicit instantiation
template class PhaseFieldProblem<2>;