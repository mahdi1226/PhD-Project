// ============================================================================
// assembly/magnetic_assembler.cc - Monolithic Magnetics Assembler (PARALLEL)
//
// Equilibrium-limit assembler (tau_M -> 0): no transport PDE for M.
// The M block enforces M = chi * grad(phi) algebraically.
//
// Cell assembly uses FEValuesExtractors for clean block contributions.
//
// Sign derivation for C_phi_M:
//   Paper Eq 42d: (grad phi, grad X) = (h_a - M, grad X)
//   Rearranged:   (grad phi, grad X) + (M, grad X) = (h_a, grad X)
//   So C_phi_M = +(M, grad X) on LHS (POSITIVE sign)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "assembly/magnetic_assembler.h"
#include "physics/material_properties.h"
#include "physics/applied_field.h"
#include "mms/poisson/poisson_mms.h"
#include "mms/magnetization/magnetization_mms.h"

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/full_matrix.h>

// ============================================================================
// Constructor
// ============================================================================
template <int dim>
MagneticAssembler<dim>::MagneticAssembler(
    const Parameters& params,
    const dealii::DoFHandler<dim>& mag_dof,
    const dealii::DoFHandler<dim>& U_dof,
    const dealii::DoFHandler<dim>& theta_dof,
    const dealii::AffineConstraints<double>& mag_constraints,
    MPI_Comm mpi_communicator)
    : params_(params)
    , mag_dof_handler_(mag_dof)
    , U_dof_handler_(U_dof)
    , theta_dof_handler_(theta_dof)
    , mag_constraints_(mag_constraints)
    , mpi_communicator_(mpi_communicator)
{
}

// ============================================================================
// Main assembly (PARALLEL)
// ============================================================================
template <int dim>
void MagneticAssembler<dim>::assemble(
    dealii::TrilinosWrappers::SparseMatrix& system_matrix,
    dealii::TrilinosWrappers::MPI::Vector& system_rhs,
    const dealii::TrilinosWrappers::MPI::Vector& Ux,
    const dealii::TrilinosWrappers::MPI::Vector& Uy,
    const dealii::TrilinosWrappers::MPI::Vector& theta,
    const dealii::TrilinosWrappers::MPI::Vector& mag_old,
    double dt,
    double current_time) const
{
    (void)Ux;       // Velocity no longer needed (no transport)
    (void)Uy;
    (void)mag_old;  // No time derivative term
    (void)dt;       // No 1/dt term

    const bool mms_mode = params_.enable_mms;
    const double L_y = params_.domain.y_max - params_.domain.y_min;

    const auto& fe_mag = mag_dof_handler_.get_fe();
    const auto& fe_theta = theta_dof_handler_.get_fe();

    const unsigned int dofs_per_cell = fe_mag.n_dofs_per_cell();

    // Extractors for the FESystem components
    const dealii::FEValuesExtractors::Vector M(0);     // components 0, 1
    const dealii::FEValuesExtractors::Scalar phi(dim); // component dim

    // Quadrature
    const unsigned int quad_degree = std::max(fe_mag.degree, 2u) + 2;
    dealii::QGauss<dim> quadrature_cell(quad_degree);
    const unsigned int n_q_cell = quadrature_cell.size();

    // FEValues for cells
    dealii::FEValues<dim> fe_values_mag(fe_mag, quadrature_cell,
        dealii::update_values | dealii::update_gradients |
        dealii::update_quadrature_points | dealii::update_JxW_values);
    dealii::FEValues<dim> fe_values_theta(fe_theta, quadrature_cell,
        dealii::update_values);

    // Pre-allocated storage for field values
    std::vector<double> theta_vals(n_q_cell);

    // Local contributions
    dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    dealii::Vector<double> cell_rhs(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dofs(dofs_per_cell);

    // Physical parameters
    const double tau_M_val = params_.physics.tau_M;
    const double mass_coeff = (tau_M_val > 0.0) ? 1.0 / tau_M_val : 1.0;
    const double relax_coeff = (tau_M_val > 0.0) ? 1.0 / tau_M_val : 0.0;

    // Initialize
    system_matrix = 0;
    system_rhs = 0;

    // ========================================================================
    // CELL LOOP
    // ========================================================================
    auto cell_mag = mag_dof_handler_.begin_active();
    auto cell_theta = theta_dof_handler_.begin_active();

    for (; cell_mag != mag_dof_handler_.end();
         ++cell_mag, ++cell_theta)
    {
        if (!cell_mag->is_locally_owned())
            continue;

        fe_values_mag.reinit(cell_mag);
        fe_values_theta.reinit(cell_theta);

        cell_matrix = 0;
        cell_rhs = 0;

        // Get field values at quadrature points
        fe_values_theta.get_function_values(theta, theta_vals);

        // ====================================================================
        // Cell integrals: all 4 blocks
        // ====================================================================
        for (unsigned int q = 0; q < n_q_cell; ++q)
        {
            const double JxW = fe_values_mag.JxW(q);
            const dealii::Point<dim>& x_q = fe_values_mag.quadrature_point(q);

            // Susceptibility chi(theta)
            const double chi_theta = susceptibility(
                theta_vals[q], params_.physics.epsilon, params_.physics.chi_0);

            // Applied field (zero in MMS mode)
            dealii::Tensor<1, dim> h_a = compute_applied_field<dim>(
                x_q, params_, current_time);

            // MMS sources
            double f_mms_phi = 0.0;
            dealii::Tensor<1, dim> f_mms_M;
            if (mms_mode)
            {
                // phi source: coupled (-Delta phi* - div M*)
                f_mms_phi = compute_poisson_mms_source_coupled<dim>(
                    x_q, current_time, L_y);

                // M source: (1/tau_M)(M* - chi*grad(phi*))
                // Nonzero because M* and grad(phi*) are independently chosen
                f_mms_M = compute_mag_mms_source_equilibrium<dim>(
                    x_q, current_time, tau_M_val, chi_theta, L_y);
            }

            // ================================================================
            // Matrix entries: loop over test (i) and trial (j) DoFs
            // ================================================================
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                // Test function values via extractors
                const dealii::Tensor<1, dim> Z_i = fe_values_mag[M].value(i, q);
                const double X_i = fe_values_mag[phi].value(i, q);
                const dealii::Tensor<1, dim> grad_X_i = fe_values_mag[phi].gradient(i, q);

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    // Trial function values via extractors
                    const dealii::Tensor<1, dim> M_j = fe_values_mag[M].value(j, q);
                    const dealii::Tensor<1, dim> grad_phi_j = fe_values_mag[phi].gradient(j, q);

                    double val = 0.0;

                    // A_M: relaxation mass (equilibrium limit)
                    // (1/tau_M)(M_j, Z_i)
                    val += mass_coeff * (M_j * Z_i);

                    // C_M_phi: -(1/tau_M) chi (grad phi_j, Z_i)
                    // From Eq 52d RHS: (kappa_0/T)(H^k, Z) with H^k = grad Phi^k
                    // H = grad phi is the TOTAL field (Eq 52e encodes h_a in phi)
                    val += -relax_coeff * chi_theta * (grad_phi_j * Z_i);

                    // C_phi_M: +(M_j, grad X_i)
                    // From Eq 42d: (grad phi, grad X) + (M, grad X) = (h_a, grad X)
                    val += M_j * grad_X_i;

                    // A_phi: (grad phi_j, grad X_i)
                    val += grad_phi_j * grad_X_i;

                    cell_matrix(i, j) += val * JxW;
                }

                // ============================================================
                // RHS
                // ============================================================
                double rhs_val = 0.0;

                // M block RHS: 0 in physics (no time derivative, no old M)

                // phi block: (h_a, grad X_i)
                // NOTE: No h_a term in M block! H = grad phi already includes h_a
                // via the Poisson equation (Eq 52e). See paper Eq 52d.
                rhs_val += h_a * grad_X_i;

                // MMS sources
                if (mms_mode)
                {
                    // M block: (f_M, Z_i) where f_M = (1/tau_M)(M* - chi*H*)
                    rhs_val += f_mms_M * Z_i;
                    // phi block: (f_phi, X_i)
                    rhs_val += f_mms_phi * X_i;
                }

                cell_rhs(i) += rhs_val * JxW;
            }
        }

        // Distribute cell contributions (handles phi constraints)
        cell_mag->get_dof_indices(local_dofs);
        mag_constraints_.distribute_local_to_global(
            cell_matrix, cell_rhs, local_dofs,
            system_matrix, system_rhs);
    } // end cell loop

    // Synchronize parallel contributions
    system_matrix.compress(dealii::VectorOperation::add);
    system_rhs.compress(dealii::VectorOperation::add);
}

// ============================================================================
// Explicit instantiation
// ============================================================================
template class MagneticAssembler<2>;
