// ============================================================================
// assembly/magnetization_assembler.h - DG Magnetization Assembler (PARALLEL)
//
// PARALLEL VERSION:
//   - Uses Trilinos matrix/vectors
//   - Only assembles locally owned cells
//   - Face integrals handle ghost cell coupling
//
// FIX: assemble_rhs_only now takes current_time for H = h_a + h_d computation
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Eq. 42c, Eq. 57
//
// EQUATION 42c (rearranged):
//   (1/τ + 1/T)(M^k, Z) - B_h^m(U^{k-1}, Z, M^k) = (1/T)(χ_θ H^k, Z) + (1/τ)(M^{k-1}, Z)
//
// ============================================================================
#ifndef MAGNETIZATION_ASSEMBLER_H
#define MAGNETIZATION_ASSEMBLER_H

#include "utilities/parameters.h"

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <mpi.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * @brief DG Magnetization Assembler (PARALLEL)
 *
 * Assembles the scalar DG magnetization system using skew-symmetric forms.
 */
template <int dim>
class MagnetizationAssembler
{
public:
    /**
     * @brief Constructor
     *
     * @param params          Simulation parameters
     * @param M_dof           DoFHandler for M (DG scalar)
     * @param U_dof           DoFHandler for U (CG velocity)
     * @param phi_dof         DoFHandler for φ (CG potential)
     * @param theta_dof       DoFHandler for θ (CG phase field)
     * @param mpi_communicator MPI communicator
     */
    MagnetizationAssembler(
        const Parameters& params,
        const dealii::DoFHandler<dim>& M_dof,
        const dealii::DoFHandler<dim>& U_dof,
        const dealii::DoFHandler<dim>& phi_dof,
        const dealii::DoFHandler<dim>& theta_dof,
        MPI_Comm mpi_communicator);

    /**
     * @brief Assemble the DG magnetization system (PARALLEL)
     *
     * @param system_matrix   [OUT] System matrix (Trilinos)
     * @param rhs_x           [OUT] RHS for Mx (Trilinos, owned)
     * @param rhs_y           [OUT] RHS for My (Trilinos, owned)
     * @param Ux              x-velocity (ghosted)
     * @param Uy              y-velocity (ghosted)
     * @param phi             Magnetic potential (ghosted)
     * @param theta           Phase field (ghosted)
     * @param Mx_old          Previous Mx (ghosted)
     * @param My_old          Previous My (ghosted)
     * @param dt              Time step
     * @param current_time    Current time (for h_a computation and MMS)
     */
    void assemble(
        dealii::TrilinosWrappers::SparseMatrix& system_matrix,
        dealii::TrilinosWrappers::MPI::Vector& rhs_x,
        dealii::TrilinosWrappers::MPI::Vector& rhs_y,
        const dealii::TrilinosWrappers::MPI::Vector& Ux,
        const dealii::TrilinosWrappers::MPI::Vector& Uy,
        const dealii::TrilinosWrappers::MPI::Vector& phi,
        const dealii::TrilinosWrappers::MPI::Vector& theta,
        const dealii::TrilinosWrappers::MPI::Vector& Mx_old,
        const dealii::TrilinosWrappers::MPI::Vector& My_old,
        double dt,
        double current_time = 0.0) const;

    /**
     * @brief Assemble only the RHS (for fixed matrix reuse)
     *
     * FIX: Now requires current_time for H = h_a + h_d computation
     *
     * @param rhs_x           [OUT] RHS for Mx (Trilinos, owned)
     * @param rhs_y           [OUT] RHS for My (Trilinos, owned)
     * @param phi             Magnetic potential (ghosted)
     * @param theta           Phase field (ghosted)
     * @param Mx_old          Previous Mx (ghosted)
     * @param My_old          Previous My (ghosted)
     * @param dt              Time step
     * @param current_time    Current time (for h_a computation)
     */
    void assemble_rhs_only(
        dealii::TrilinosWrappers::MPI::Vector& rhs_x,
        dealii::TrilinosWrappers::MPI::Vector& rhs_y,
        const dealii::TrilinosWrappers::MPI::Vector& phi,
        const dealii::TrilinosWrappers::MPI::Vector& theta,
        const dealii::TrilinosWrappers::MPI::Vector& Mx_old,
        const dealii::TrilinosWrappers::MPI::Vector& My_old,
        double dt,
        double current_time) const;  // NEW: added current_time parameter

private:
    const Parameters& params_;
    const dealii::DoFHandler<dim>& M_dof_handler_;
    const dealii::DoFHandler<dim>& U_dof_handler_;
    const dealii::DoFHandler<dim>& phi_dof_handler_;
    const dealii::DoFHandler<dim>& theta_dof_handler_;
    MPI_Comm mpi_communicator_;

    /// Susceptibility function: χ(θ)
    double chi(double theta_val) const;
};

#endif // MAGNETIZATION_ASSEMBLER_H