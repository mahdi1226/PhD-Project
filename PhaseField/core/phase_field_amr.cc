// ============================================================================
// core/phase_field_amr.cc - Adaptive Mesh Refinement for PhaseFieldProblem
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Section 6.2, p.522
// ============================================================================

#include "phase_field.h"
#include "output/logger.h"

#include <deal.II/grid/grid_refinement.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/derivative_approximation.h>

template <int dim>
void PhaseFieldProblem<dim>::refine_mesh()
{
    Logger::info("    refine_mesh() started");

    if (!params_.amr.enabled)
    {
        Logger::info("      AMR disabled, skipping");
        return;
    }

    // =========================================================================
    // Step 1: Compute refinement indicators based on |∇θ|
    // =========================================================================
    Logger::info("      Computing refinement indicators...");
    dealii::Vector<float> indicators(triangulation_.n_active_cells());

    dealii::DerivativeApproximation::approximate_gradient(
        theta_dof_handler_,
        theta_solution_,
        indicators);

    // Scale by cell diameter (Eq. 99)
    unsigned int cell_idx = 0;
    for (const auto& cell : theta_dof_handler_.active_cell_iterators())
    {
        indicators[cell_idx] *= cell->diameter();
        ++cell_idx;
    }

    // =========================================================================
    // Step 2: Mark cells (no coarsening for stability)
    // =========================================================================
    Logger::info("      Marking cells...");
    dealii::GridRefinement::refine_and_coarsen_fixed_fraction(
        triangulation_, indicators, 0.3, 0.0);  // No coarsening

    // Enforce min/max refinement levels
    for (const auto& cell : triangulation_.active_cell_iterators())
    {
        if (cell->level() >= static_cast<int>(params_.amr.max_level))
            cell->clear_refine_flag();
        if (cell->level() <= static_cast<int>(params_.amr.min_level))
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
        Logger::info("      No refinement needed");
        return;
    }

    // =========================================================================
    // Step 3: Create SolutionTransfer for each scalar field
    // =========================================================================
    Logger::info("      Preparing solution transfer...");

    dealii::SolutionTransfer<dim, dealii::Vector<double>> theta_transfer(theta_dof_handler_);
    dealii::SolutionTransfer<dim, dealii::Vector<double>> theta_old_transfer(theta_dof_handler_);
    dealii::SolutionTransfer<dim, dealii::Vector<double>> psi_transfer(psi_dof_handler_);
    dealii::SolutionTransfer<dim, dealii::Vector<double>> mx_transfer(mx_dof_handler_);
    dealii::SolutionTransfer<dim, dealii::Vector<double>> my_transfer(my_dof_handler_);
    dealii::SolutionTransfer<dim, dealii::Vector<double>> mx_old_transfer(mx_dof_handler_);
    dealii::SolutionTransfer<dim, dealii::Vector<double>> my_old_transfer(my_dof_handler_);
    dealii::SolutionTransfer<dim, dealii::Vector<double>> phi_transfer(phi_dof_handler_);
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
    mx_transfer.prepare_for_coarsening_and_refinement(mx_solution_);
    my_transfer.prepare_for_coarsening_and_refinement(my_solution_);
    mx_old_transfer.prepare_for_coarsening_and_refinement(mx_old_solution_);
    my_old_transfer.prepare_for_coarsening_and_refinement(my_old_solution_);
    phi_transfer.prepare_for_coarsening_and_refinement(phi_solution_);
    ux_transfer.prepare_for_coarsening_and_refinement(ux_solution_);
    uy_transfer.prepare_for_coarsening_and_refinement(uy_solution_);
    ux_old_transfer.prepare_for_coarsening_and_refinement(ux_old_solution_);
    uy_old_transfer.prepare_for_coarsening_and_refinement(uy_old_solution_);
    p_transfer.prepare_for_coarsening_and_refinement(p_solution_);

    // =========================================================================
    // Step 5: Execute refinement
    // =========================================================================
    Logger::info("      Executing mesh refinement...");
    triangulation_.execute_coarsening_and_refinement();

    Logger::info("        New cell count: " + std::to_string(triangulation_.n_active_cells()));

    // =========================================================================
    // Step 6: Setup systems on new mesh
    // =========================================================================
    Logger::info("      Redistributing DoFs...");
    setup_dof_handlers();
    setup_constraints();
    setup_sparsity_patterns();

    // =========================================================================
    // Step 7: Interpolate solutions (following working example pattern)
    // =========================================================================
    Logger::info("      Transferring solutions...");

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

    // m_x
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

    // m_y
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

    // φ
    {
        dealii::Vector<double> tmp(phi_dof_handler_.n_dofs());
        phi_transfer.interpolate(tmp);
        phi_solution_ = tmp;
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
    // Step 8: Apply constraints
    // =========================================================================
    theta_constraints_.distribute(theta_solution_);
    theta_constraints_.distribute(theta_old_solution_);
    psi_constraints_.distribute(psi_solution_);
    mx_constraints_.distribute(mx_solution_);
    mx_constraints_.distribute(mx_old_solution_);
    my_constraints_.distribute(my_solution_);
    my_constraints_.distribute(my_old_solution_);
    phi_constraints_.distribute(phi_solution_);
    ux_constraints_.distribute(ux_solution_);
    ux_constraints_.distribute(ux_old_solution_);
    uy_constraints_.distribute(uy_solution_);
    uy_constraints_.distribute(uy_old_solution_);
    p_constraints_.distribute(p_solution_);

    Logger::success("    refine_mesh() completed");
}

template class PhaseFieldProblem<2>;
// template class PhaseFieldProblem<3>;