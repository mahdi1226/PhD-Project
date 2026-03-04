// ============================================================================
// solvers/ns_block_preconditioner.h - Parallel Block Schur Preconditioner
//
// Block preconditioner for saddle-point NS system:
//   [A   B^T] [u]   [f]
//   [B   0  ] [p] = [g]
//
// Preconditioner approximates (time-consistent):
//   P^{-1} = [ A^{-1}        -A^{-1} B^T S^{-1} ]
//            [ 0             -S^{-1}           ]
//
// where:
//   S ≈ (ν + 1/Δt) M_p     (unsteady)
//   S ≈ ν M_p              (steady)
//
// Pressure pinning is enforced consistently in p-space.
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

// ============================================================================
// Parallel Block Schur Preconditioner
// ============================================================================
class BlockSchurPreconditionerParallel : public dealii::EnableObserverPointer
{
public:
    /**
     * @brief Constructor
     *
     * @param system_matrix     Full coupled NS matrix [A B^T; B 0]
     * @param pressure_mass     Pressure mass matrix M_p
     * @param ux_to_ns_map      ux DoF → coupled index
     * @param uy_to_ns_map      uy DoF → coupled index
     * @param p_to_ns_map       p  DoF → coupled index
     * @param ns_owned          Locally owned coupled DoFs
     * @param vel_owned         (unused – computed internally)
     * @param p_owned           Locally owned pressure DoFs
     * @param mpi_comm          MPI communicator
     * @param viscosity         Kinematic viscosity ν
     * @param dt                Time step Δt (≤0 ⇒ steady)
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
        double viscosity,
        double dt);

    /**
     * @brief Apply preconditioner: dst = P^{-1} src
     */
    void vmult(dealii::TrilinosWrappers::MPI::Vector& dst,
               const dealii::TrilinosWrappers::MPI::Vector& src) const;

    // ------------------------------------------------------------------------
    // Diagnostics (accumulated per outer solve)
    // ------------------------------------------------------------------------
    mutable unsigned int n_iterations_A;
    mutable unsigned int n_iterations_S;

    // ------------------------------------------------------------------------
    // Tuning parameters (public by design)
    // ------------------------------------------------------------------------
    double inner_tolerance;
    unsigned int max_inner_iterations;

private:
    // ------------------------------------------------------------------------
    // Helper methods
    // ------------------------------------------------------------------------
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

    // ------------------------------------------------------------------------
    // Matrix pointers
    // ------------------------------------------------------------------------
    const dealii::TrilinosWrappers::SparseMatrix* system_matrix_ptr_;
    const dealii::TrilinosWrappers::SparseMatrix* pressure_mass_ptr_;

    // ------------------------------------------------------------------------
    // Index mappings
    // ------------------------------------------------------------------------
    std::vector<dealii::types::global_dof_index> ux_map_;
    std::vector<dealii::types::global_dof_index> uy_map_;
    std::vector<dealii::types::global_dof_index> p_map_;

    std::vector<int> global_to_vel_;
    std::vector<int> global_to_p_;

    dealii::types::global_dof_index n_ux_;
    dealii::types::global_dof_index n_uy_;
    dealii::types::global_dof_index n_p_;
    dealii::types::global_dof_index n_vel_;
    dealii::types::global_dof_index n_total_;

    // ------------------------------------------------------------------------
    // Index sets
    // ------------------------------------------------------------------------
    dealii::IndexSet ns_owned_;
    dealii::IndexSet vel_owned_;
    dealii::IndexSet p_owned_;
    dealii::IndexSet pressure_indices_for_BT_;

    // ------------------------------------------------------------------------
    // MPI
    // ------------------------------------------------------------------------
    MPI_Comm mpi_comm_;
    int rank_;

    // ------------------------------------------------------------------------
    // Velocity block and preconditioners
    // ------------------------------------------------------------------------
    dealii::TrilinosWrappers::SparseMatrix velocity_block_;
    dealii::TrilinosWrappers::PreconditionAMG A_preconditioner_;
    dealii::TrilinosWrappers::PreconditionAMG S_preconditioner_;

    // ------------------------------------------------------------------------
    // Physical / scaling parameters
    // ------------------------------------------------------------------------
    double viscosity_;
    double dt_;
    double schur_alpha_;        // ν + 1/Δt (or ν if steady)

    // Pressure pin consistency (p-space index, -1 if inactive)
    int pinned_p_local_;
};

#endif // NS_BLOCK_PRECONDITIONER_H