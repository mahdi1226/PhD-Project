// ============================================================================
// assembly/magnetic_assembler.h - Monolithic Magnetics Assembler (PARALLEL)
//
// Full M transport + Poisson assembler (Paper Eq 42c-42d):
//
//   Eq 42c: (δM^k/τ, Z) - B_h^m(U^k, Z, M^k) + (1/T)(M^k, Z) = (1/T)(χ H^k, Z)
//   Eq 42d: (∇Φ^k, ∇X) + (M^k, ∇X) = (h_a, ∇X)
//
// Block terms:
//   A_M:       (1/dt + 1/T)(M^k, Z) - B_h^m(U, Z, M) [cell + face]
//   C_M_phi:   -(1/T) chi(theta) (∇Φ^k, Z)
//   C_phi_M:   +(M^k, ∇X)
//   A_phi:     (∇Φ^k, ∇X)
//
// RHS:
//   f_M:   (1/dt)(M^{k-1}, Z)
//   f_phi: (h_a, ∇X)
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
 * Full M transport + Poisson (Paper Eq 42c-42d).
 * Cell + DG face assembly for M convection using FEValuesExtractors.
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
