// ============================================================================
// diagnostics/mms_runtime.cc - MMS Runtime Helpers Implementation
// ============================================================================

#include "mms_runtime.h"
#include "ch_mms.h"

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/vector_tools.h>

// ============================================================================
// update_ch_mms_constraints
// ============================================================================
template <int dim>
void update_ch_mms_constraints(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::DoFHandler<dim>& psi_dof_handler,
    dealii::AffineConstraints<double>& theta_constraints,
    dealii::AffineConstraints<double>& psi_constraints,
    double time)
{
    // ========================================================================
    // θ constraints: hanging nodes + MMS Dirichlet BCs
    // ========================================================================
    theta_constraints.clear();
    dealii::DoFTools::make_hanging_node_constraints(theta_dof_handler, theta_constraints);

    CHMMSBoundaryTheta<dim> theta_bc;
    theta_bc.set_time(time);
    for (unsigned int bid = 0; bid < 4; ++bid)
    {
        dealii::VectorTools::interpolate_boundary_values(
            theta_dof_handler, bid, theta_bc, theta_constraints);
    }
    theta_constraints.close();

    // ========================================================================
    // ψ constraints: hanging nodes + MMS Dirichlet BCs
    // ========================================================================
    psi_constraints.clear();
    dealii::DoFTools::make_hanging_node_constraints(psi_dof_handler, psi_constraints);

    CHMMSBoundaryPsi<dim> psi_bc;
    psi_bc.set_time(time);
    for (unsigned int bid = 0; bid < 4; ++bid)
    {
        dealii::VectorTools::interpolate_boundary_values(
            psi_dof_handler, bid, psi_bc, psi_constraints);
    }
    psi_constraints.close();
}

// ============================================================================
// rebuild_ch_combined_constraints
// ============================================================================
template <int dim>
void rebuild_ch_combined_constraints(
    const dealii::AffineConstraints<double>& theta_constraints,
    const dealii::AffineConstraints<double>& psi_constraints,
    const std::vector<dealii::types::global_dof_index>& theta_to_ch_map,
    const std::vector<dealii::types::global_dof_index>& psi_to_ch_map,
    unsigned int n_theta,
    unsigned int n_psi,
    dealii::AffineConstraints<double>& ch_combined_constraints)
{
    ch_combined_constraints.clear();

    // Map θ constraints to coupled system
    for (unsigned int i = 0; i < n_theta; ++i)
    {
        if (theta_constraints.is_constrained(i))
        {
            const auto* entries = theta_constraints.get_constraint_entries(i);
            const double inhom = theta_constraints.get_inhomogeneity(i);
            const auto coupled_i = theta_to_ch_map[i];

            ch_combined_constraints.add_line(coupled_i);

            if (entries && !entries->empty())
            {
                for (const auto& entry : *entries)
                {
                    ch_combined_constraints.add_entry(
                        coupled_i,
                        theta_to_ch_map[entry.first],
                        entry.second);
                }
            }
            ch_combined_constraints.set_inhomogeneity(coupled_i, inhom);
        }
    }

    // Map ψ constraints to coupled system
    for (unsigned int i = 0; i < n_psi; ++i)
    {
        if (psi_constraints.is_constrained(i))
        {
            const auto* entries = psi_constraints.get_constraint_entries(i);
            const double inhom = psi_constraints.get_inhomogeneity(i);
            const auto coupled_i = psi_to_ch_map[i];

            ch_combined_constraints.add_line(coupled_i);

            if (entries && !entries->empty())
            {
                for (const auto& entry : *entries)
                {
                    ch_combined_constraints.add_entry(
                        coupled_i,
                        psi_to_ch_map[entry.first],
                        entry.second);
                }
            }
            ch_combined_constraints.set_inhomogeneity(coupled_i, inhom);
        }
    }

    ch_combined_constraints.close();
}

// ============================================================================
// update_all_ch_mms_constraints - Convenience wrapper
// ============================================================================
template <int dim>
void update_all_ch_mms_constraints(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::DoFHandler<dim>& psi_dof_handler,
    dealii::AffineConstraints<double>& theta_constraints,
    dealii::AffineConstraints<double>& psi_constraints,
    const std::vector<dealii::types::global_dof_index>& theta_to_ch_map,
    const std::vector<dealii::types::global_dof_index>& psi_to_ch_map,
    dealii::AffineConstraints<double>& ch_combined_constraints,
    double time)
{
    // Step 1: Update individual constraints
    update_ch_mms_constraints<dim>(
        theta_dof_handler,
        psi_dof_handler,
        theta_constraints,
        psi_constraints,
        time);

    // Step 2: Rebuild combined constraints
    rebuild_ch_combined_constraints<dim>(
        theta_constraints,
        psi_constraints,
        theta_to_ch_map,
        psi_to_ch_map,
        theta_dof_handler.n_dofs(),
        psi_dof_handler.n_dofs(),
        ch_combined_constraints);
}

// ============================================================================
// Explicit instantiations
// ============================================================================
template void update_ch_mms_constraints<2>(
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    dealii::AffineConstraints<double>&,
    dealii::AffineConstraints<double>&,
    double);

template void rebuild_ch_combined_constraints<2>(
    const dealii::AffineConstraints<double>&,
    const dealii::AffineConstraints<double>&,
    const std::vector<dealii::types::global_dof_index>&,
    const std::vector<dealii::types::global_dof_index>&,
    unsigned int,
    unsigned int,
    dealii::AffineConstraints<double>&);

template void update_all_ch_mms_constraints<2>(
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    dealii::AffineConstraints<double>&,
    dealii::AffineConstraints<double>&,
    const std::vector<dealii::types::global_dof_index>&,
    const std::vector<dealii::types::global_dof_index>&,
    dealii::AffineConstraints<double>&,
    double);