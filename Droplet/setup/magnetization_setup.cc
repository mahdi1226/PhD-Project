// ============================================================================
// setup/magnetization_setup.cc - Magnetization Initialization Implementation
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Section 5.1, Eq. 41: M⁰ = I_{M_h}(χ(θ⁰) H⁰)
//
// L² projection of χ(θ)H onto DG space, computed cell-by-cell.
// ============================================================================

#include "setup/magnetization_setup.h"

#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/full_matrix.h>

// ============================================================================
// initialize_magnetization_dg
// ============================================================================
template <int dim>
void initialize_magnetization_dg(
    const dealii::DoFHandler<dim>& M_dof_handler,
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::Vector<double>& theta_solution,
    const dealii::Vector<double>& phi_solution,
    double chi_0,
    dealii::Vector<double>& Mx_solution,
    dealii::Vector<double>& My_solution)
{

    // All DoFHandlers must share the same triangulation
    Assert(&M_dof_handler.get_triangulation() == &theta_dof_handler.get_triangulation(),
           ExcMessage("M and theta DoFHandlers must share the same triangulation"));
    Assert(&M_dof_handler.get_triangulation() == &phi_dof_handler.get_triangulation(),
           ExcMessage("M and phi DoFHandlers must share the same triangulation"));

    // Initialize output vectors
    Mx_solution.reinit(M_dof_handler.n_dofs());
    My_solution.reinit(M_dof_handler.n_dofs());

    const FiniteElement<dim>& fe_M = M_dof_handler.get_fe();
    const FiniteElement<dim>& fe_theta = theta_dof_handler.get_fe();
    const FiniteElement<dim>& fe_phi = phi_dof_handler.get_fe();

    const unsigned int dofs_per_cell = fe_M.dofs_per_cell;

    // Quadrature for integration (higher order for RHS accuracy)
    QGauss<dim> quadrature(fe_M.degree + 2);
    const unsigned int n_q_points = quadrature.size();

    // FEValues for M (DG), θ (CG), and φ (CG)
    FEValues<dim> fe_values_M(fe_M, quadrature,
                               update_values | update_JxW_values);
    FEValues<dim> fe_values_theta(fe_theta, quadrature,
                                   update_values);
    FEValues<dim> fe_values_phi(fe_phi, quadrature,
                                 update_gradients);

    // Local data structures
    FullMatrix<double> local_mass(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_mass_inv(dofs_per_cell, dofs_per_cell);
    Vector<double> local_rhs_x(dofs_per_cell);
    Vector<double> local_rhs_y(dofs_per_cell);
    Vector<double> local_sol_x(dofs_per_cell);
    Vector<double> local_sol_y(dofs_per_cell);

    std::vector<double> theta_values(n_q_points);
    std::vector<Tensor<1, dim>> grad_phi_values(n_q_points);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

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
        dealii::local_mass = 0;
        dealii::local_rhs_x = 0;
        dealii::local_rhs_y = 0;

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = dealii::fe_values_M.JxW(q);
            const double theta_q = theta_values[q];
            const dealii::Tensor<1, dim>& grad_phi_q = grad_phi_values[q];

            // Compute χ(θ) = χ₀(1+θ)/2
            const double chi = susceptibility(theta_q, chi_0);

            // Compute H = ∇φ
            dealii::Tensor<1, dim> H;
            for (unsigned int d = 0; d < dim; ++d)
                H[d] = grad_phi_q[d];

            // Target: M = χ(θ) H
            const double target_Mx = chi * H[0];
            const double target_My = (dim > 1) ? chi * H[1] : 0.0;

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const double phi_i = fe_values_M.shape_value(i, q);

                // RHS: (χ(θ) H, φ_i)_T
                dealii::local_rhs_x(i) += target_Mx * phi_i * JxW;
                dealii::local_rhs_y(i) += target_My * phi_i * JxW;

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    const double phi_j = fe_values_M.shape_value(j, q);

                    // Mass: (φ_i, φ_j)_T
                    dealii::local_mass(i, j) += phi_i * phi_j * JxW;
                }
            }
        }

        // Solve local system: M_T * sol = rhs
        // Invert local mass matrix (small matrix, direct inversion is fine)
        dealii::local_mass_inv.invert(local_mass);

        dealii::local_mass_inv.vmult(local_sol_x, local_rhs_x);
        dealii::local_mass_inv.vmult(local_sol_y, local_rhs_y);

        // Store in global vectors
        cell_M->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            Mx_solution(local_dof_indices[i]) = dealii::local_sol_x(i);
            My_solution(local_dof_indices[i]) = dealii::local_sol_y(i);
        }
    }
}

// ============================================================================
// Explicit instantiation
// ============================================================================
template void initialize_magnetization_dg<2>(
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    const dealii::DoFHandler<2>&,
    const dealii::Vector<double>&,
    const dealii::Vector<double>&,
    double,
    dealii::Vector<double>&,
    dealii::Vector<double>&);

template void initialize_magnetization_dg<3>(
    const dealii::DoFHandler<3>&,
    const dealii::DoFHandler<3>&,
    const dealii::DoFHandler<3>&,
    const dealii::Vector<double>&,
    const dealii::Vector<double>&,
    double,
    dealii::Vector<double>&,
    dealii::Vector<double>&);