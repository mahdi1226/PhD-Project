// ============================================================================
// core/phase_field.h - Phase Field Problem (PARALLEL)
//
// Full ferrofluid solver implementing Nochetto et al. CMAME 309 (2016)
//
// PARALLEL VERSION:
//   - Uses parallel::distributed::Triangulation (p4est backend)
//   - TrilinosWrappers for matrices/vectors
//   - MPI-aware output and diagnostics
//
// OPTIMIZED VERSION:
//   - Cached assemblers and solvers (avoid recreation each timestep)
//   - Picard iteration for Poisson ↔ Magnetization coupling (Paper Algorithm 1)
//   - RHS-only assembly for Picard iterations 2+ (matrix reuse)
//
// Subsystems:
//   - Cahn-Hilliard (θ, ψ): phase separation with convection
//   - Poisson (φ): magnetostatic potential, H = ∇φ
//   - Magnetization (Mx, My): DG transport of M (Eq. 42c)
//   - Navier-Stokes (ux, uy, p): fluid flow with Kelvin force B_h^m
//
// Time stepping (Paper Algorithm 1):
//   1. Solve CH → θ^n, ψ^n (uses U^{n-1})
//   2. Picard loop for k = 1, ..., N:
//      a. Solve Poisson → φ^{n,k} (uses M^{n,k-1})
//      b. Solve Magnetization → M^{n,k} (uses φ^{n,k})
//      c. Check convergence: ||M^{n,k} - M^{n,k-1}|| < tol
//   3. Solve NS → u^n, p^n (uses θ^{n-1}, converged H^n, M^n)
//
// CRITICAL: θ is LAGGED in NS (θ^{n-1}) for energy stability!
//
// ============================================================================
#ifndef PHASE_FIELD_H
#define PHASE_FIELD_H

#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>

#include "utilities/parameters.h"
#include "solvers/solver_info.h"
#include "solvers/poisson_solver.h"
#include "solvers/magnetization_solver.h"
#include "assembly/magnetization_assembler.h"

#include <vector>
#include <memory>
#include <string>

/**
 * @brief Full ferrofluid phase field solver (PARALLEL)
 *
 * Implements all subsystems from the paper:
 *   - Cahn-Hilliard (θ, ψ): phase separation
 *   - Poisson (φ): magnetostatic potential
 *   - Magnetization (Mx, My): DG transport
 *   - Navier-Stokes (ux, uy, p): fluid flow with full Kelvin B_h^m
 *
 * OPTIMIZATION: Assemblers and solvers are cached as class members
 * to avoid recreation overhead each timestep.
 */
template <int dim>
class PhaseFieldProblem
{
public:
    explicit PhaseFieldProblem(const Parameters& params);

    /// Main entry point
    void run();

private:
    // ========================================================================
    // Setup methods
    // ========================================================================
    void setup_mesh();
    void setup_dof_handlers();
    void setup_ch_system();
    void setup_poisson_system();
    void setup_magnetization_system();
    void setup_ns_system();
    void initialize_solutions();

    // ========================================================================
    // Solve methods
    // ========================================================================
    void time_step(double dt);
    void solve_ch(double dt);
    void solve_poisson();
    void solve_magnetization(double dt, bool matrix_changed);
    void solve_magnetization_rhs_only(double dt);
    void solve_ns(double dt);

    // ========================================================================
    // Picard iteration for Poisson ↔ Magnetization coupling
    // Returns number of iterations used
    // ========================================================================
    unsigned int solve_poisson_magnetization_picard(double dt);

    // ========================================================================
    // Output
    // ========================================================================
    void output_results(const std::string& output_dir);

    // ========================================================================
    // Diagnostics helper
    // ========================================================================
    double get_min_h() const;

    // ========================================================================
    // MPI and parallel infrastructure
    // ========================================================================
    MPI_Comm mpi_communicator_;
    dealii::ConditionalOStream pcout_;  // Only rank 0 prints

    // Parameters
    const Parameters& params_;

    // ========================================================================
    // Mesh (PARALLEL: p4est-based distributed triangulation)
    // ========================================================================
    dealii::parallel::distributed::Triangulation<dim> triangulation_;

    // ========================================================================
    // Finite elements
    // ========================================================================
    dealii::FE_Q<dim> fe_Q2_;    // Q2 for velocity, θ, ψ, φ
    dealii::FE_Q<dim> fe_Q1_;    // Q1 for pressure
    dealii::FE_DGQ<dim> fe_DG_;  // DG for magnetization M

    // ========================================================================
    // Cahn-Hilliard system (θ, ψ)
    // ========================================================================
    dealii::DoFHandler<dim> theta_dof_handler_;
    dealii::DoFHandler<dim> psi_dof_handler_;

    dealii::IndexSet theta_locally_owned_;
    dealii::IndexSet theta_locally_relevant_;
    dealii::IndexSet psi_locally_owned_;
    dealii::IndexSet psi_locally_relevant_;
    dealii::IndexSet ch_locally_owned_;
    dealii::IndexSet ch_locally_relevant_;

    dealii::AffineConstraints<double> theta_constraints_;
    dealii::AffineConstraints<double> psi_constraints_;
    dealii::AffineConstraints<double> ch_constraints_;

    std::vector<dealii::types::global_dof_index> theta_to_ch_map_;
    std::vector<dealii::types::global_dof_index> psi_to_ch_map_;

    dealii::TrilinosWrappers::SparseMatrix ch_matrix_;
    dealii::TrilinosWrappers::MPI::Vector ch_rhs_;
    dealii::TrilinosWrappers::MPI::Vector ch_solution_;

    dealii::TrilinosWrappers::MPI::Vector theta_solution_;
    dealii::TrilinosWrappers::MPI::Vector theta_old_solution_;  // θ^{k-1} for lagging
    dealii::TrilinosWrappers::MPI::Vector psi_solution_;

    // Ghosted vectors for assembly (read access to ghost values)
    dealii::TrilinosWrappers::MPI::Vector theta_relevant_;
    dealii::TrilinosWrappers::MPI::Vector theta_old_relevant_;
    dealii::TrilinosWrappers::MPI::Vector psi_relevant_;

    // ========================================================================
    // Poisson system (φ)
    // ========================================================================
    dealii::DoFHandler<dim> phi_dof_handler_;

    dealii::IndexSet phi_locally_owned_;
    dealii::IndexSet phi_locally_relevant_;

    dealii::AffineConstraints<double> phi_constraints_;

    dealii::TrilinosWrappers::SparseMatrix phi_matrix_;
    dealii::TrilinosWrappers::MPI::Vector phi_rhs_;
    dealii::TrilinosWrappers::MPI::Vector phi_solution_;
    dealii::TrilinosWrappers::MPI::Vector phi_relevant_;

    // Cached Poisson solver (AMG preconditioner reused across timesteps)
    std::unique_ptr<PoissonSolver> poisson_solver_;

    // ========================================================================
    // Magnetization system (Mx, My) - DG transport
    //
    // Paper Eq. 42c: ∂M/∂t + (u·∇)M = (1/τ_M)(χ(θ)H - M)
    // ========================================================================
    dealii::DoFHandler<dim> M_dof_handler_;  // Single handler for both Mx, My

    dealii::IndexSet M_locally_owned_;
    dealii::IndexSet M_locally_relevant_;

    // DG has no constraints (no hanging nodes for DG)
    dealii::AffineConstraints<double> M_constraints_;

    dealii::TrilinosWrappers::SparseMatrix M_matrix_;
    dealii::TrilinosWrappers::MPI::Vector Mx_rhs_;
    dealii::TrilinosWrappers::MPI::Vector My_rhs_;

    dealii::TrilinosWrappers::MPI::Vector Mx_solution_;
    dealii::TrilinosWrappers::MPI::Vector My_solution_;
    dealii::TrilinosWrappers::MPI::Vector Mx_old_solution_;
    dealii::TrilinosWrappers::MPI::Vector My_old_solution_;

    // Ghosted for assembly
    dealii::TrilinosWrappers::MPI::Vector Mx_relevant_;
    dealii::TrilinosWrappers::MPI::Vector My_relevant_;

    // OPTIMIZATION: Cached magnetization assembler and solver
    std::unique_ptr<MagnetizationAssembler<dim>> magnetization_assembler_;
    std::unique_ptr<MagnetizationSolver<dim>> magnetization_solver_;

    // ========================================================================
    // Navier-Stokes system (ux, uy, p)
    // ========================================================================
    dealii::DoFHandler<dim> ux_dof_handler_;
    dealii::DoFHandler<dim> uy_dof_handler_;
    dealii::DoFHandler<dim> p_dof_handler_;

    dealii::IndexSet ux_locally_owned_;
    dealii::IndexSet ux_locally_relevant_;
    dealii::IndexSet uy_locally_owned_;
    dealii::IndexSet uy_locally_relevant_;
    dealii::IndexSet p_locally_owned_;
    dealii::IndexSet p_locally_relevant_;
    dealii::IndexSet ns_locally_owned_;
    dealii::IndexSet ns_locally_relevant_;

    dealii::AffineConstraints<double> ux_constraints_;
    dealii::AffineConstraints<double> uy_constraints_;
    dealii::AffineConstraints<double> p_constraints_;
    dealii::AffineConstraints<double> ns_constraints_;

    std::vector<dealii::types::global_dof_index> ux_to_ns_map_;
    std::vector<dealii::types::global_dof_index> uy_to_ns_map_;
    std::vector<dealii::types::global_dof_index> p_to_ns_map_;

    dealii::TrilinosWrappers::SparseMatrix ns_matrix_;
    dealii::TrilinosWrappers::MPI::Vector ns_rhs_;
    dealii::TrilinosWrappers::MPI::Vector ns_solution_;

    dealii::TrilinosWrappers::MPI::Vector ux_solution_;
    dealii::TrilinosWrappers::MPI::Vector ux_old_solution_;
    dealii::TrilinosWrappers::MPI::Vector uy_solution_;
    dealii::TrilinosWrappers::MPI::Vector uy_old_solution_;
    dealii::TrilinosWrappers::MPI::Vector p_solution_;

    // Ghosted for assembly
    dealii::TrilinosWrappers::MPI::Vector ux_relevant_;
    dealii::TrilinosWrappers::MPI::Vector uy_relevant_;
    dealii::TrilinosWrappers::MPI::Vector p_relevant_;

    // Pressure mass matrix (for Schur preconditioner)
    dealii::TrilinosWrappers::SparseMatrix pressure_mass_matrix_;

    // ========================================================================
    // Time state
    // ========================================================================
    double time_;
    unsigned int timestep_number_;

    // Solver diagnostics
    SolverInfo last_ch_info_;
    SolverInfo last_poisson_info_;
    SolverInfo last_M_info_;
    SolverInfo last_ns_info_;

    // Picard iteration diagnostics
    unsigned int last_picard_iterations_;
    double last_picard_residual_;
};

#endif // PHASE_FIELD_H