// ============================================================================
// mms/mms_runtime.cc - MMS Runtime Helpers Implementation
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "mms/mms_runtime.h"
#include "mms/ch_mms.h"
#include "mms/poisson_mms.h"
#include "mms/ns_mms.h"

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <iostream>

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
    // θ constraints
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

    // ψ constraints
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

    // Map θ constraints
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

    // Map ψ constraints
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
// update_all_ch_mms_constraints
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
    update_ch_mms_constraints<dim>(
        theta_dof_handler, psi_dof_handler,
        theta_constraints, psi_constraints,
        time);

    rebuild_ch_combined_constraints<dim>(
        theta_constraints, psi_constraints,
        theta_to_ch_map, psi_to_ch_map,
        theta_dof_handler.n_dofs(), psi_dof_handler.n_dofs(),
        ch_combined_constraints);
}

// ============================================================================
// compute_all_mms_errors
// ============================================================================
template <int dim>
void compute_mms_errors(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::DoFHandler<dim>& psi_dof_handler,
    const dealii::Vector<double>& theta_solution,
    const dealii::Vector<double>& psi_solution,
    const dealii::DoFHandler<dim>* phi_dof_handler,
    const dealii::Vector<double>* phi_solution,
    const dealii::DoFHandler<dim>* ux_dof_handler,
    const dealii::DoFHandler<dim>* uy_dof_handler,
    const dealii::DoFHandler<dim>* p_dof_handler,
    const dealii::Vector<double>* ux_solution,
    const dealii::Vector<double>* uy_solution,
    const dealii::Vector<double>* p_solution,
    double time,
    double L_y,
    double h_min,
    unsigned int refinement_level,
    bool enable_magnetic,
    bool enable_ns)
{
    std::cout << "\n--- MMS Error Analysis ---\n";

    // CH errors (always computed)
    auto ch_errors = compute_ch_mms_errors<dim>(
        theta_dof_handler, psi_dof_handler,
        theta_solution, psi_solution,
        time);
    ch_errors.print();

    std::cout << "\nCH Convergence (h, theta_L2, theta_H1, psi_L2):\n";
    ch_errors.print_for_convergence();

    // Poisson errors (if magnetic enabled)
    if (enable_magnetic && phi_dof_handler && phi_solution)
    {
        auto poisson_errors = compute_poisson_mms_error<dim>(
            *phi_dof_handler, *phi_solution, time, L_y);
        poisson_errors.print(refinement_level, h_min);
    }

    // NS errors (if NS enabled)
    if (enable_ns && ux_dof_handler && uy_dof_handler && p_dof_handler &&
        ux_solution && uy_solution && p_solution)
    {
        auto ns_errors = compute_ns_mms_error<dim>(
            *ux_dof_handler, *uy_dof_handler, *p_dof_handler,
            *ux_solution, *uy_solution, *p_solution,
            time, L_y);

        std::cout << "\n--- NS MMS Errors ---\n";
        ns_errors.print(refinement_level, h_min);

        std::cout << "\nNS Convergence (h, ux_L2, ux_H1, uy_L2, uy_H1, p_L2):\n";
        ns_errors.print_for_convergence(h_min);
    }
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

template void compute_mms_errors<2>(
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    const dealii::Vector<double>&,
    const dealii::Vector<double>&,
    const dealii::DoFHandler<2>*,
    const dealii::Vector<double>*,
    const dealii::DoFHandler<2>*,
    const dealii::DoFHandler<2>*,
    const dealii::DoFHandler<2>*,
    const dealii::Vector<double>*,
    const dealii::Vector<double>*,
    const dealii::Vector<double>*,
    double, double, double, unsigned int, bool, bool);