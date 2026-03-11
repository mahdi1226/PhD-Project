// ============================================================================
// assembly/magnetic_assembler.h - Monolithic Magnetics Assembler (PARALLEL)
//
// Equilibrium-limit assembler (tau_M -> 0): no transport PDE for M.
// The M block enforces M = chi * grad(phi) algebraically.
//
// Assembles the 2x2 block system for combined M + phi:
//
//   | A_M        C_M_phi | | M^k   |   | f_M   |
//   |                     | |       | = |       |
//   | C_phi_M    A_phi   | | phi^k |   | f_phi |
//
// Block terms:
//   A_M:       (1/tau_M)(M^k, Z)
//   C_M_phi:   -(1/tau_M) chi(theta) (grad phi^k, Z)
//   C_phi_M:   +(M^k, grad X)
//   A_phi:     (grad phi^k, grad X)
//
// RHS:
//   f_M:   0
//   f_phi: (h_a, grad X)
//
// No transport terms, no face terms.
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef MAGNETIC_ASSEMBLER_H
#define MAGNETIC_ASSEMBLER_H

#include "utilities/parameters.h"

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <mpi.h>

/**
 * @brief Monolithic Magnetics Assembler (PARALLEL)
 *
 * Equilibrium-limit assembler: M = chi * grad(phi) algebraically.
 * No transport PDE, no face terms. Cell assembly only using FEValuesExtractors.
 */
template <int dim>
class MagneticAssembler
{
public:
    /**
     * @brief Constructor
     *
     * @param params            Simulation parameters
     * @param mag_dof           DoFHandler for combined M+phi FESystem
     * @param U_dof             DoFHandler for velocity (CG)
     * @param theta_dof         DoFHandler for phase field (CG)
     * @param mag_constraints   AffineConstraints for M+phi system
     * @param mpi_communicator  MPI communicator
     */
    MagneticAssembler(
        const Parameters& params,
        const dealii::DoFHandler<dim>& mag_dof,
        const dealii::DoFHandler<dim>& U_dof,
        const dealii::DoFHandler<dim>& theta_dof,
        const dealii::AffineConstraints<double>& mag_constraints,
        MPI_Comm mpi_communicator);

    /**
     * @brief Assemble the monolithic magnetics system (PARALLEL)
     *
     * @param system_matrix   [OUT] System matrix
     * @param system_rhs      [OUT] RHS vector
     * @param Ux              x-velocity (ghosted)
     * @param Uy              y-velocity (ghosted)
     * @param theta           Phase field (ghosted)
     * @param mag_old         Previous M+phi solution (ghosted)
     * @param dt              Time step
     * @param current_time    Current time
     */
    void assemble(
        dealii::TrilinosWrappers::SparseMatrix& system_matrix,
        dealii::TrilinosWrappers::MPI::Vector& system_rhs,
        const dealii::TrilinosWrappers::MPI::Vector& Ux,
        const dealii::TrilinosWrappers::MPI::Vector& Uy,
        const dealii::TrilinosWrappers::MPI::Vector& theta,
        const dealii::TrilinosWrappers::MPI::Vector& mag_old,
        double dt,
        double current_time) const;

private:
    const Parameters params_;
    const dealii::DoFHandler<dim>& mag_dof_handler_;
    const dealii::DoFHandler<dim>& U_dof_handler_;
    const dealii::DoFHandler<dim>& theta_dof_handler_;
    const dealii::AffineConstraints<double>& mag_constraints_;
    MPI_Comm mpi_communicator_;
};

#endif // MAGNETIC_ASSEMBLER_H
