// ============================================================================
// setup/magnetization_setup.cc - Magnetization DG System Setup (PARALLEL)
//
// PARALLEL VERSION:
//   - Uses Trilinos sparsity pattern with DG flux coupling
//   - Cell-local L² projection for initialization
//
// FIX: Now uses total field H = h_a + h_d for equilibrium initialization
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "setup/magnetization_setup.h"
#include "physics/material_properties.h"
#include "physics/applied_field.h"

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>

// ============================================================================
// setup_magnetization_sparsity (PARALLEL)
// ============================================================================
template <int dim>
void setup_magnetization_sparsity(
    const dealii::DoFHandler<dim>& M_dof_handler,
    const dealii::IndexSet& M_locally_owned,
    const dealii::IndexSet& M_locally_relevant,
    dealii::TrilinosWrappers::SparseMatrix& M_matrix,
    MPI_Comm mpi_communicator,
    dealii::ConditionalOStream& pcout)
{
    // DG sparsity: includes face coupling for upwind flux
    dealii::TrilinosWrappers::SparsityPattern trilinos_sp(
        M_locally_owned, M_locally_owned, M_locally_relevant,
        mpi_communicator);

    // DG flux sparsity pattern - includes neighbor coupling across faces
    dealii::DoFTools::make_flux_sparsity_pattern(M_dof_handler, trilinos_sp);

    trilinos_sp.compress();

    // Initialize matrix with sparsity pattern
    M_matrix.reinit(trilinos_sp);

    pcout << "[Magnetization Setup] n_dofs = " << M_dof_handler.n_dofs()
          << ", locally_owned = " << M_locally_owned.n_elements()
          << ", nnz = " << trilinos_sp.n_nonzero_elements() << " (DG flux)\n";
}

// ============================================================================
// initialize_magnetization_equilibrium (PARALLEL)
//
// Cell-local L² projection: M⁰ = χ(θ⁰) H⁰
// DG mass matrix is block-diagonal, so each cell solves independently.
//
// FIX: H = h_a + h_d (total field), not just h_d = ∇φ
// ============================================================================
template <int dim>
void initialize_magnetization_equilibrium(
    const dealii::DoFHandler<dim>& M_dof_handler,
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::TrilinosWrappers::MPI::Vector& theta_solution,
    const dealii::TrilinosWrappers::MPI::Vector& phi_solution,
    const Parameters& params,
    dealii::TrilinosWrappers::MPI::Vector& Mx_solution,
    dealii::TrilinosWrappers::MPI::Vector& My_solution)
{
    const dealii::FiniteElement<dim>& fe_M = M_dof_handler.get_fe();
    const dealii::FiniteElement<dim>& fe_theta = theta_dof_handler.get_fe();
    const dealii::FiniteElement<dim>& fe_phi = phi_dof_handler.get_fe();

    const unsigned int dofs_per_cell = fe_M.n_dofs_per_cell();

    // Quadrature for integration
    dealii::QGauss<dim> quadrature(fe_M.degree + 2);
    const unsigned int n_q_points = quadrature.size();

    // FEValues - need quadrature_points for h_a computation
    dealii::FEValues<dim> fe_values_M(fe_M, quadrature,
        dealii::update_values | dealii::update_quadrature_points | dealii::update_JxW_values);
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

    // Initial time for field ramp (use params.current_time, typically 0 at init)
    const double current_time = params.current_time;

    // Iterate over cells
    auto cell_M = M_dof_handler.begin_active();
    auto cell_theta = theta_dof_handler.begin_active();
    auto cell_phi = phi_dof_handler.begin_active();

    for (; cell_M != M_dof_handler.end(); ++cell_M, ++cell_theta, ++cell_phi)
    {
        // PARALLEL: Only process locally owned cells
        if (!cell_M->is_locally_owned())
            continue;

        fe_values_M.reinit(cell_M);
        fe_values_theta.reinit(cell_theta);
        fe_values_phi.reinit(cell_phi);

        // Get field values at quadrature points
        fe_values_theta.get_function_values(theta_solution, theta_values);
        fe_values_phi.get_function_gradients(phi_solution, grad_phi_values);

        // Build local mass matrix and RHS
        local_mass = 0;
        local_rhs_x = 0;
        local_rhs_y = 0;

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = fe_values_M.JxW(q);
            const dealii::Point<dim>& x_q = fe_values_M.quadrature_point(q);
            const double theta_q = theta_values[q];

            // FIX: Compute total field H = h_a + h_d
            // h_a = applied field from dipoles
            // h_d = ∇φ (demagnetizing field from Poisson solve)
            dealii::Tensor<1, dim> h_a = compute_applied_field<dim>(x_q, params, current_time);
            dealii::Tensor<1, dim> H;
            if (params.use_reduced_magnetic_field)
            {
                // Dome mode: H = h_a only
                H = h_a;
            }
            else
            {
                // Full physics: H = h_a + h_d
                H[0] = h_a[0] + grad_phi_values[q][0];
                H[1] = h_a[1] + grad_phi_values[q][1];
            }

            // Susceptibility χ(θ)
            const double chi = susceptibility(theta_q,
                params.physics.epsilon, params.physics.chi_0);

            // Target: M = χ(θ) H
            const double target_Mx = chi * H[0];
            const double target_My = (dim > 1) ? chi * H[1] : 0.0;

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const double phi_i = fe_values_M.shape_value(i, q);

                local_rhs_x(i) += target_Mx * phi_i * JxW;
                local_rhs_y(i) += target_My * phi_i * JxW;

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    const double phi_j = fe_values_M.shape_value(j, q);
                    local_mass(i, j) += phi_i * phi_j * JxW;
                }
            }
        }

        // Solve local system: M_local * sol = rhs
        local_mass_inv.invert(local_mass);
        local_mass_inv.vmult(local_sol_x, local_rhs_x);
        local_mass_inv.vmult(local_sol_y, local_rhs_y);

        // Store in global vectors
        cell_M->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            Mx_solution[local_dof_indices[i]] = local_sol_x(i);
            My_solution[local_dof_indices[i]] = local_sol_y(i);
        }
    }

    // Compress vectors after direct writes
    Mx_solution.compress(dealii::VectorOperation::insert);
    My_solution.compress(dealii::VectorOperation::insert);
}

// ============================================================================
// Explicit instantiations
// ============================================================================
template void setup_magnetization_sparsity<2>(
    const dealii::DoFHandler<2>&,
    const dealii::IndexSet&,
    const dealii::IndexSet&,
    dealii::TrilinosWrappers::SparseMatrix&,
    MPI_Comm,
    dealii::ConditionalOStream&);

template void initialize_magnetization_equilibrium<2>(
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const dealii::TrilinosWrappers::MPI::Vector&,
    const Parameters&,
    dealii::TrilinosWrappers::MPI::Vector&,
    dealii::TrilinosWrappers::MPI::Vector&);