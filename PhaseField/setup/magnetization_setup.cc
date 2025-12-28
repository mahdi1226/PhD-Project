// ============================================================================
// setup/magnetization_setup.cc - Magnetization DG System Setup (PAPER_MATCH v2)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Section 5, Eq. 56: M_h DG space
// Section 5.1, Eq. 41: M⁰ = I_{M_h}(χ(θ⁰) H⁰)
//
// FIX: Uses params.physics.epsilon and params.physics.chi_0 instead of globals.
// ============================================================================

#include "setup/magnetization_setup.h"
#include "physics/material_properties.h"  // For compute_susceptibility()
#include "utilities/parameters.h"          // For Parameters struct

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>

#include <iostream>

// ============================================================================
// setup_magnetization_sparsity
//
// Creates DG flux sparsity pattern (includes face coupling for upwind).
// This was previously hardcoded in phase_field_setup.cc.
// ============================================================================
template <int dim>
void setup_magnetization_sparsity(
    const dealii::DoFHandler<dim>& M_dof_handler,
    dealii::SparsityPattern& M_sparsity,
    bool verbose)
{
    const unsigned int n_M = M_dof_handler.n_dofs();

    // DG sparsity: includes face coupling (for upwind flux)
    dealii::DynamicSparsityPattern dsp(n_M, n_M);
    dealii::DoFTools::make_flux_sparsity_pattern(M_dof_handler, dsp);
    M_sparsity.copy_from(dsp);

    if (verbose)
    {
        std::cout << "[Setup] Magnetization sparsity: "
                  << M_sparsity.n_nonzero_elements() << " nonzeros (DG flux)\n";
    }
}

// ============================================================================
// initialize_magnetization_equilibrium
//
// L² projection of M⁰ = χ(θ⁰) H⁰ onto DG space.
// Cell-local projection (DG mass matrix is block-diagonal).
//
// FIX: Uses params.physics.epsilon and params.physics.chi_0 (not globals!)
// ============================================================================
template <int dim>
void initialize_magnetization_equilibrium(
    const dealii::DoFHandler<dim>& M_dof_handler,
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::Vector<double>& theta_solution,
    const dealii::Vector<double>& phi_solution,
    const Parameters& params,
    dealii::Vector<double>& Mx_solution,
    dealii::Vector<double>& My_solution)
{
    // All DoFHandlers must share the same triangulation
    Assert(&M_dof_handler.get_triangulation() == &theta_dof_handler.get_triangulation(),
           dealii::ExcMessage("M and theta DoFHandlers must share the same triangulation"));
    Assert(&M_dof_handler.get_triangulation() == &phi_dof_handler.get_triangulation(),
           dealii::ExcMessage("M and phi DoFHandlers must share the same triangulation"));

    // Initialize output vectors
    Mx_solution.reinit(M_dof_handler.n_dofs());
    My_solution.reinit(M_dof_handler.n_dofs());

    const dealii::FiniteElement<dim>& fe_M = M_dof_handler.get_fe();
    const dealii::FiniteElement<dim>& fe_theta = theta_dof_handler.get_fe();
    const dealii::FiniteElement<dim>& fe_phi = phi_dof_handler.get_fe();

    const unsigned int dofs_per_cell = fe_M.n_dofs_per_cell();

    // Quadrature for integration (higher order for RHS accuracy)
    dealii::QGauss<dim> quadrature(fe_M.degree + 2);
    const unsigned int n_q_points = quadrature.size();

    // FEValues for M (DG), θ (CG), and φ (CG)
    dealii::FEValues<dim> fe_values_M(fe_M, quadrature,
                                       dealii::update_values | dealii::update_JxW_values);
    dealii::FEValues<dim> fe_values_theta(fe_theta, quadrature,
                                           dealii::update_values);
    dealii::FEValues<dim> fe_values_phi(fe_phi, quadrature,
                                         dealii::update_gradients);

    // Local data structures
    dealii::FullMatrix<double> local_mass(dofs_per_cell, dofs_per_cell);
    dealii::FullMatrix<double> local_mass_inv(dofs_per_cell, dofs_per_cell);
    dealii::Vector<double> local_rhs_x(dofs_per_cell);
    dealii::Vector<double> local_rhs_y(dofs_per_cell);
    dealii::Vector<double> local_sol_x(dofs_per_cell);
    dealii::Vector<double> local_sol_y(dofs_per_cell);

    std::vector<double> theta_values(n_q_points);
    std::vector<dealii::Tensor<1, dim>> grad_phi_values(n_q_points);
    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    // Iterate over cells (all DoFHandlers share the same triangulation)
    auto cell_M = M_dof_handler.begin_active();
    auto cell_theta = theta_dof_handler.begin_active();
    auto cell_phi = phi_dof_handler.begin_active();

    for (; cell_M != M_dof_handler.end(); ++cell_M, ++cell_theta, ++cell_phi)
    {
        fe_values_M.reinit(cell_M);
        fe_values_theta.reinit(cell_theta);
        fe_values_phi.reinit(cell_phi);

        // Get θ and ∇φ values at quadrature points
        fe_values_theta.get_function_values(theta_solution, theta_values);
        fe_values_phi.get_function_gradients(phi_solution, grad_phi_values);

        // Build local mass matrix and RHS
        local_mass = 0;
        local_rhs_x = 0;
        local_rhs_y = 0;

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = fe_values_M.JxW(q);
            const double theta_q = theta_values[q];
            const dealii::Tensor<1, dim>& grad_phi_q = grad_phi_values[q];

            // Compute χ(θ) using params.physics.epsilon and params.physics.chi_0
            // χ(θ) = χ₀ H(θ/ε) where H is smoothed Heaviside
            const double chi = susceptibility(theta_q,
                params.physics.epsilon, params.physics.chi_0);

            // Compute H = ∇φ (paper convention)
            const dealii::Tensor<1, dim>& H = grad_phi_q;

            // Target: M = χ(θ) H
            const double target_Mx = chi * H[0];
            const double target_My = (dim > 1) ? chi * H[1] : 0.0;

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const double phi_i = fe_values_M.shape_value(i, q);

                // RHS: (χ(θ) H, φ_i)_T
                local_rhs_x(i) += target_Mx * phi_i * JxW;
                local_rhs_y(i) += target_My * phi_i * JxW;

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    const double phi_j = fe_values_M.shape_value(j, q);

                    // Mass: (φ_i, φ_j)_T
                    local_mass(i, j) += phi_i * phi_j * JxW;
                }
            }
        }

        // Solve local system: M_T * sol = rhs
        // Invert local mass matrix (small matrix, direct inversion is fine)
        local_mass_inv.invert(local_mass);

        local_mass_inv.vmult(local_sol_x, local_rhs_x);
        local_mass_inv.vmult(local_sol_y, local_rhs_y);

        // Store in global vectors
        cell_M->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            Mx_solution(local_dof_indices[i]) = local_sol_x(i);
            My_solution(local_dof_indices[i]) = local_sol_y(i);
        }
    }
}

// ============================================================================
// Explicit instantiation
// ============================================================================
template void setup_magnetization_sparsity<2>(
    const dealii::DoFHandler<2>&,
    dealii::SparsityPattern&,
    bool);

template void setup_magnetization_sparsity<3>(
    const dealii::DoFHandler<3>&,
    dealii::SparsityPattern&,
    bool);

template void initialize_magnetization_equilibrium<2>(
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    const dealii::Vector<double>&,
    const dealii::Vector<double>&,
    const Parameters&,
    dealii::Vector<double>&,
    dealii::Vector<double>&);

template void initialize_magnetization_equilibrium<3>(
    const dealii::DoFHandler<3>&,
    const dealii::DoFHandler<3>&,
    const dealii::DoFHandler<3>&,
    const dealii::Vector<double>&,
    const dealii::Vector<double>&,
    const Parameters&,
    dealii::Vector<double>&,
    dealii::Vector<double>&);