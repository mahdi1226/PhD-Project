// ============================================================================
// mms/mms_core/mms_context.h - MMS Test Context (Parallel Version)
//
// CH-ONLY VERSION: Other subsystems not yet converted to parallel.
//
// PARALLEL VERSION:
//   - Uses parallel::distributed::Triangulation
//   - Uses TrilinosWrappers::MPI::Vector
//   - Uses TrilinosWrappers::SparseMatrix
//   - Tracks IndexSets for owned/relevant DoFs
//
// Usage:
//   MMSContext<2> ctx(MPI_COMM_WORLD);
//   ctx.setup_mesh(params, refinement);
//   ctx.setup_ch(params, mms_time);
//   ctx.apply_ch_initial_conditions(params, t_init);
//   // ... use ctx.theta_dof_handler, ctx.ch_matrix, etc.
//
// ============================================================================
#ifndef MMS_CONTEXT_H
#define MMS_CONTEXT_H

#include "utilities/parameters.h"

#include <deal.II/base/mpi.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <memory>
#include <vector>

/**
 * @brief Unified data structure for MMS verification tests (Parallel Version)
 *
 * CH-ONLY VERSION: Only Cahn-Hilliard subsystem is available.
 * Other subsystems (NS, Poisson, Magnetization) will be added incrementally.
 *
 * @tparam dim Spatial dimension (typically 2)
 */
template <int dim>
struct MMSContext
{
    // ========================================================================
    // MPI
    // ========================================================================
    MPI_Comm mpi_communicator;
    dealii::ConditionalOStream pcout;

    // ========================================================================
    // Mesh (Distributed)
    // ========================================================================
    dealii::parallel::distributed::Triangulation<dim> triangulation;

    // ========================================================================
    // Finite Elements
    // ========================================================================
    std::unique_ptr<dealii::FE_Q<dim>> fe_phase;  ///< Q1/Q2 for θ, ψ

    // ========================================================================
    // Cahn-Hilliard System (θ, ψ)
    // ========================================================================
    dealii::DoFHandler<dim> theta_dof_handler;
    dealii::DoFHandler<dim> psi_dof_handler;

    // Ownership tracking
    dealii::IndexSet theta_locally_owned;
    dealii::IndexSet theta_locally_relevant;
    dealii::IndexSet psi_locally_owned;
    dealii::IndexSet psi_locally_relevant;
    dealii::IndexSet ch_locally_owned;      ///< Combined for coupled system
    dealii::IndexSet ch_locally_relevant;

    dealii::AffineConstraints<double> theta_constraints;
    dealii::AffineConstraints<double> psi_constraints;
    dealii::AffineConstraints<double> ch_constraints;  ///< Combined for coupled solve

    std::vector<dealii::types::global_dof_index> theta_to_ch_map;
    std::vector<dealii::types::global_dof_index> psi_to_ch_map;

    // Trilinos matrix and vectors
    dealii::TrilinosWrappers::SparseMatrix ch_matrix;
    dealii::TrilinosWrappers::MPI::Vector ch_rhs;

    // Solution vectors: owned (for solving) and relevant (for assembly/reading)
    dealii::TrilinosWrappers::MPI::Vector theta_owned;
    dealii::TrilinosWrappers::MPI::Vector theta_relevant;
    dealii::TrilinosWrappers::MPI::Vector theta_old_owned;
    dealii::TrilinosWrappers::MPI::Vector theta_old_relevant;
    dealii::TrilinosWrappers::MPI::Vector psi_owned;
    dealii::TrilinosWrappers::MPI::Vector psi_relevant;

    // ========================================================================
    // Constructor
    // ========================================================================
    explicit MMSContext(MPI_Comm mpi_comm = MPI_COMM_WORLD);

    // ========================================================================
    // Setup Methods
    // ========================================================================

    /**
     * @brief Create distributed mesh from parameters and refine globally
     *
     * - Domain from params.domain.*
     * - Subdivisions from params.domain.initial_cells_*
     * - Boundary IDs: 0=bottom, 1=right, 2=top, 3=left
     *
     * @param params Simulation parameters
     * @param refinement Number of global refinements
     */
    void setup_mesh(const Parameters& params, unsigned int refinement);

    /**
     * @brief Setup Cahn-Hilliard subsystem
     *
     * Calls PRODUCTION setup_ch_coupled_system() from setup/ch_setup.h.
     * For MMS, applies exact Dirichlet BCs.
     *
     * @param params Simulation parameters
     * @param mms_time Time for MMS boundary conditions
     */
    void setup_ch(const Parameters& params, double mms_time);

    // ========================================================================
    // Initial Condition Methods
    // ========================================================================

    /**
     * @brief Apply CH MMS initial conditions at t_init
     * @param params Simulation parameters
     * @param t_init Initial time
     */
    void apply_ch_initial_conditions(const Parameters& params, double t_init);

    // ========================================================================
    // Convenience Methods
    // ========================================================================

    /// Get minimum cell diameter (for h in convergence tables) - uses MPI_MIN reduction
    double get_min_h() const;

    /// Get total CH DoFs (global)
    unsigned int n_ch_dofs() const;

    /// Get MPI rank
    unsigned int this_mpi_process() const {
        return dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
    }

    /// Get number of MPI ranks
    unsigned int n_mpi_processes() const {
        return dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);
    }

    // ========================================================================
    // Update Constraints (for time-dependent MMS BCs)
    // ========================================================================

    /**
     * @brief Update CH constraints for new time (MMS)
     * @param params Simulation parameters
     * @param time Current time
     */
    void update_ch_constraints(const Parameters& params, double time);

    // ========================================================================
    // Ghost Update Helpers
    // ========================================================================

    /// Update theta_relevant from theta_owned (import ghosts)
    void update_theta_ghosts();

    /// Update theta_old_relevant from theta_old_owned
    void update_theta_old_ghosts();

    /// Update psi_relevant from psi_owned
    void update_psi_ghosts();

    /// Update all CH ghosts
    void update_ch_ghosts();
};

#endif // MMS_CONTEXT_H