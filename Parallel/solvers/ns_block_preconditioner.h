// ============================================================================
// solvers/ns_block_preconditioner.h - Parallel Block Schur Preconditioner
//
// Block preconditioner for saddle-point NS system:
//   [A   B^T] [u]   [f]
//   [B   0  ] [p] = [g]
//
// Preconditioner approximates:
//   P^{-1} = [A^{-1}  -A^{-1}B^T S^{-1}]
//            [0       -S^{-1}          ]
//
// where S = B A^{-1} B^T ≈ (1/ν) M_p (pressure mass matrix)
// ============================================================================
#ifndef NS_BLOCK_PRECONDITIONER_H
#define NS_BLOCK_PRECONDITIONER_H

#include <deal.II/base/subscriptor.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_control.h>

#include <vector>
#include <memory>

/**
 * @brief Parallel Block Schur preconditioner for Navier-Stokes
 *
 * Usage:
 *   BlockSchurPreconditionerParallel preconditioner(
 *       system_matrix, pressure_mass,
 *       ux_map, uy_map, p_map,
 *       ns_owned, vel_owned, p_owned,
 *       mpi_comm, viscosity);
 *
 *   preconditioner.inner_tolerance = 1e-3;
 *   preconditioner.max_inner_iterations = 500;
 *
 *   SolverFGMRES<...> solver(control);
 *   solver.solve(system_matrix, solution, rhs, preconditioner);
 */
class BlockSchurPreconditionerParallel : public dealii::EnableObserverPointer
{
public:
    /**
     * @brief Constructor - sets up the preconditioner
     *
     * @param system_matrix     Full NS system [A, B^T; B, 0]
     * @param pressure_mass     Pressure mass matrix (for Schur approximation)
     * @param ux_to_ns_map      Mapping ux DoF -> coupled index
     * @param uy_to_ns_map      Mapping uy DoF -> coupled index
     * @param p_to_ns_map       Mapping p DoF -> coupled index
     * @param ns_owned          Locally owned DoFs in coupled system
     * @param vel_owned         Locally owned velocity DoFs
     * @param p_owned           Locally owned pressure DoFs
     * @param mpi_comm          MPI communicator
     * @param viscosity         Viscosity for Schur scaling
     */
    BlockSchurPreconditionerParallel(
        const dealii::TrilinosWrappers::SparseMatrix& system_matrix,
        const dealii::TrilinosWrappers::SparseMatrix& pressure_mass,
        const std::vector<dealii::types::global_dof_index>& ux_to_ns_map,
        const std::vector<dealii::types::global_dof_index>& uy_to_ns_map,
        const std::vector<dealii::types::global_dof_index>& p_to_ns_map,
        const dealii::IndexSet& ns_owned,
        const dealii::IndexSet& vel_owned,
        const dealii::IndexSet& p_owned,
        MPI_Comm mpi_comm,
        double viscosity = 1.0);

    /**
     * @brief Apply preconditioner: dst = P^{-1} * src
     */
    void vmult(dealii::TrilinosWrappers::MPI::Vector& dst,
               const dealii::TrilinosWrappers::MPI::Vector& src) const;

    // Statistics (reset each solve)
    mutable unsigned int n_iterations_A;
    mutable unsigned int n_iterations_S;

    // Tuning parameters
    double inner_tolerance;
    unsigned int max_inner_iterations;

private:
    // Helper methods
    void extract_velocity(const dealii::TrilinosWrappers::MPI::Vector& src,
                          dealii::TrilinosWrappers::MPI::Vector& vel) const;
    void extract_pressure(const dealii::TrilinosWrappers::MPI::Vector& src,
                          dealii::TrilinosWrappers::MPI::Vector& p) const;
    void insert_velocity(const dealii::TrilinosWrappers::MPI::Vector& vel,
                         dealii::TrilinosWrappers::MPI::Vector& dst) const;
    void insert_pressure(const dealii::TrilinosWrappers::MPI::Vector& p,
                         dealii::TrilinosWrappers::MPI::Vector& dst) const;
    void apply_BT(const dealii::TrilinosWrappers::MPI::Vector& p,
                  dealii::TrilinosWrappers::MPI::Vector& vel) const;

    // Matrix pointers
    const dealii::TrilinosWrappers::SparseMatrix* system_matrix_ptr_;
    const dealii::TrilinosWrappers::SparseMatrix* pressure_mass_ptr_;

    // Index mappings
    std::vector<dealii::types::global_dof_index> ux_map_;
    std::vector<dealii::types::global_dof_index> uy_map_;
    std::vector<dealii::types::global_dof_index> p_map_;
    std::vector<int> global_to_vel_;
    std::vector<int> global_to_p_;

    dealii::types::global_dof_index n_ux_, n_uy_, n_p_, n_vel_, n_total_;

    // Index sets
    dealii::IndexSet ns_owned_;
    dealii::IndexSet vel_owned_;
    dealii::IndexSet p_owned_;
    dealii::IndexSet pressure_indices_for_BT_;

    MPI_Comm mpi_comm_;
    int rank_;

    // Velocity block matrix
    dealii::TrilinosWrappers::SparseMatrix velocity_block_;

    // Preconditioners
    dealii::TrilinosWrappers::PreconditionAMG A_preconditioner_;
    dealii::TrilinosWrappers::PreconditionAMG S_preconditioner_;

    double viscosity_;
};

#endif // NS_BLOCK_PRECONDITIONER_H