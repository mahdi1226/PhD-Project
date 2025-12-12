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

    // -------------------------------------------------------------------------
    // Step 1: Compute refinement indicators based on |∇θ|
    // -------------------------------------------------------------------------
    Logger::info("      Computing refinement indicators...");
    dealii::Vector<float> indicators(triangulation_.n_active_cells());

    dealii::DerivativeApproximation::approximate_gradient(
        theta_dof_handler_,
        theta_solution_,
        indicators);

    // Scale by cell diameter
    unsigned int cell_idx = 0;
    for (const auto& cell : theta_dof_handler_.active_cell_iterators())
    {
        indicators[cell_idx] *= cell->diameter();
        ++cell_idx;
    }

    // -------------------------------------------------------------------------
    // Step 2: Mark cells (fixed fraction strategy)
    // -------------------------------------------------------------------------
    Logger::info("      Marking cells...");
    dealii::GridRefinement::refine_and_coarsen_fixed_fraction(
        triangulation_, indicators, 0.3, 0.03);

    // Enforce min/max refinement levels
    for (const auto& cell : triangulation_.active_cell_iterators())
    {
        if (cell->level() >= static_cast<int>(params_.amr.max_level))
            cell->clear_refine_flag();
        if (cell->level() <= static_cast<int>(params_.amr.min_level))
            cell->clear_coarsen_flag();
    }

    // -------------------------------------------------------------------------
    // Step 3: Prepare SolutionTransfer for all fields
    // -------------------------------------------------------------------------
    Logger::info("      Preparing solution transfer...");

    dealii::SolutionTransfer<dim> theta_transfer(theta_dof_handler_);
    dealii::SolutionTransfer<dim> theta_old_transfer(theta_dof_handler_);
    dealii::SolutionTransfer<dim> psi_transfer(psi_dof_handler_);
    dealii::SolutionTransfer<dim> mx_transfer(mx_dof_handler_);
    dealii::SolutionTransfer<dim> my_transfer(my_dof_handler_);
    dealii::SolutionTransfer<dim> mx_old_transfer(mx_dof_handler_);
    dealii::SolutionTransfer<dim> my_old_transfer(my_dof_handler_);
    dealii::SolutionTransfer<dim> phi_transfer(phi_dof_handler_);
    dealii::SolutionTransfer<dim> ux_transfer(ux_dof_handler_);
    dealii::SolutionTransfer<dim> uy_transfer(uy_dof_handler_);
    dealii::SolutionTransfer<dim> ux_old_transfer(ux_dof_handler_);
    dealii::SolutionTransfer<dim> uy_old_transfer(uy_dof_handler_);
    dealii::SolutionTransfer<dim> p_transfer(p_dof_handler_);

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

    // -------------------------------------------------------------------------
    // Step 4: Execute refinement
    // -------------------------------------------------------------------------
    Logger::info("      Executing mesh refinement...");
    triangulation_.execute_coarsening_and_refinement();

    Logger::info("        New cell count: " + std::to_string(triangulation_.n_active_cells()));

    // -------------------------------------------------------------------------
    // Step 5: Redistribute DoFs and rebuild systems
    // -------------------------------------------------------------------------
    Logger::info("      Redistributing DoFs...");
    setup_dof_handlers();
    setup_constraints();
    setup_sparsity_patterns();

    // -------------------------------------------------------------------------
    // Step 6: Transfer solutions to new mesh (new API: single argument)
    // -------------------------------------------------------------------------
    Logger::info("      Transferring solutions...");

    // Resize and interpolate
    theta_solution_.reinit(theta_dof_handler_.n_dofs());
    theta_old_solution_.reinit(theta_dof_handler_.n_dofs());
    psi_solution_.reinit(psi_dof_handler_.n_dofs());
    mx_solution_.reinit(mx_dof_handler_.n_dofs());
    my_solution_.reinit(my_dof_handler_.n_dofs());
    mx_old_solution_.reinit(mx_dof_handler_.n_dofs());
    my_old_solution_.reinit(my_dof_handler_.n_dofs());
    phi_solution_.reinit(phi_dof_handler_.n_dofs());
    ux_solution_.reinit(ux_dof_handler_.n_dofs());
    uy_solution_.reinit(uy_dof_handler_.n_dofs());
    ux_old_solution_.reinit(ux_dof_handler_.n_dofs());
    uy_old_solution_.reinit(uy_dof_handler_.n_dofs());
    p_solution_.reinit(p_dof_handler_.n_dofs());

    theta_transfer.interpolate(theta_solution_);
    theta_old_transfer.interpolate(theta_old_solution_);
    psi_transfer.interpolate(psi_solution_);
    mx_transfer.interpolate(mx_solution_);
    my_transfer.interpolate(my_solution_);
    mx_old_transfer.interpolate(mx_old_solution_);
    my_old_transfer.interpolate(my_old_solution_);
    phi_transfer.interpolate(phi_solution_);
    ux_transfer.interpolate(ux_solution_);
    uy_transfer.interpolate(uy_solution_);
    ux_old_transfer.interpolate(ux_old_solution_);
    uy_old_transfer.interpolate(uy_old_solution_);
    p_transfer.interpolate(p_solution_);

    // Apply constraints
    theta_constraints_.distribute(theta_solution_);
    theta_constraints_.distribute(theta_old_solution_);
    psi_constraints_.distribute(psi_solution_);
    mx_constraints_.distribute(mx_solution_);
    my_constraints_.distribute(my_solution_);
    mx_constraints_.distribute(mx_old_solution_);
    my_constraints_.distribute(my_old_solution_);
    phi_constraints_.distribute(phi_solution_);
    ux_constraints_.distribute(ux_solution_);
    uy_constraints_.distribute(uy_solution_);
    ux_constraints_.distribute(ux_old_solution_);
    uy_constraints_.distribute(uy_old_solution_);
    p_constraints_.distribute(p_solution_);

    Logger::success("    refine_mesh() completed");
}

template class PhaseFieldProblem<2>;
// template class PhaseFieldProblem<3>;