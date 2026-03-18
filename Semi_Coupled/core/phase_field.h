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
// Subsystems:
//   - Cahn-Hilliard (θ, ψ): phase separation with convection
//   - Monolithic Magnetics (M, φ): combined DG M + CG φ block system
//   - Navier-Stokes (ux, uy, p): fluid flow with Kelvin force B_h^m
//
// Time stepping:
//   1. Solve CH → θ^n, ψ^n (uses U^{n-1})
//   2. Solve Magnetics → M^n, φ^n (monolithic block system, no Picard)
//   3. Solve NS → u^n, p^n (uses θ^{n-1}, H^n, M^n)
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
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>

#include "utilities/parameters.h"
#include "solvers/solver_info.h"
#include "assembly/magnetic_assembler.h"
#include "solvers/magnetic_solver.h"

#include <vector>
#include <memory>
#include <string>

/**
 * @brief Full ferrofluid phase field solver (PARALLEL)
 *
 * Implements all subsystems from the paper:
 *   - Cahn-Hilliard (θ, ψ): phase separation
 *   - Monolithic Magnetics (M, φ): combined DG M + CG φ block system
 *   - Navier-Stokes (ux, uy, p): fluid flow with full Kelvin B_h^m
 *
 * Assemblers and solvers are cached as class members.
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
    void setup_magnetic_system();
    void setup_ns_system();
    void initialize_solutions();

    // ========================================================================
    // Solve methods
    // ========================================================================
    void solve_ch(double dt);
    void solve_magnetics(double dt);
    void solve_ns(double dt);

    // ========================================================================
    // Output
    // ========================================================================
    void output_results(const std::string& output_dir);

    // ========================================================================
    // Adaptive Mesh Refinement
    // ========================================================================
    void refine_mesh();

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
    // Velocity/θ/ψ: CG Q2.  Pressure: DG Q1 (paper A1).
    // Magnetization: DG Q1 (paper requirement for upwind transport).
    // ========================================================================
    dealii::FE_Q<dim> fe_Q2_;    // Q2 for velocity, θ, ψ
    dealii::FE_DGQ<dim> fe_DG_;  // DG Q1 for pressure (A1) and magnetization
    // FESystem for combined M+phi: FE_DGQ(deg_M)^dim + FE_Q(deg_phi)
    std::unique_ptr<dealii::FESystem<dim>> fe_mag_;

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
    // Monolithic Magnetics system (M + φ)
    //
    // Combined FESystem: FE_DGQ(deg_M)^dim + FE_Q(deg_phi)
    // Single DoFHandler, single matrix, single solution vector.
    // Replaces separate Poisson + Magnetization + Picard iteration.
    //
    // Paper Eq. 42c-42d solved as monolithic block system.
    // ========================================================================
    dealii::DoFHandler<dim> mag_dof_handler_;

    dealii::IndexSet mag_locally_owned_;
    dealii::IndexSet mag_locally_relevant_;

    dealii::AffineConstraints<double> mag_constraints_;

    dealii::TrilinosWrappers::SparseMatrix mag_matrix_;
    dealii::TrilinosWrappers::MPI::Vector mag_rhs_;
    dealii::TrilinosWrappers::MPI::Vector mag_solution_;

    // Ghosted vectors for assembly (read access to ghost values)
    dealii::TrilinosWrappers::MPI::Vector mag_relevant_;

    // Previous timestep M+φ for time derivative in Eq. 42c
    dealii::TrilinosWrappers::MPI::Vector mag_old_solution_;
    dealii::TrilinosWrappers::MPI::Vector mag_old_relevant_;  // ghosted

    // Separate ghost vectors for Mx, My, phi (for NS assembly and diagnostics)
    // These are VIEWS extracted from mag_solution_ after solve
    dealii::TrilinosWrappers::MPI::Vector Mx_relevant_;
    dealii::TrilinosWrappers::MPI::Vector My_relevant_;
    dealii::TrilinosWrappers::MPI::Vector phi_relevant_;

    // Separate DoFHandlers for output and NS assembly access
    // These share the same triangulation and use individual FE elements
    dealii::DoFHandler<dim> phi_dof_handler_;  // CG Q2, for diagnostics/output
    dealii::DoFHandler<dim> M_dof_handler_;    // DG Q1, for diagnostics/output

    dealii::IndexSet phi_locally_owned_;
    dealii::IndexSet phi_locally_relevant_;
    dealii::IndexSet M_locally_owned_;
    dealii::IndexSet M_locally_relevant_;

    // Individual solution vectors (extracted from mag_solution_ after solve)
    dealii::TrilinosWrappers::MPI::Vector phi_solution_;
    dealii::TrilinosWrappers::MPI::Vector Mx_solution_;
    dealii::TrilinosWrappers::MPI::Vector My_solution_;

    // Cached assembler and solver
    std::unique_ptr<MagneticAssembler<dim>> magnetic_assembler_;
    std::unique_ptr<MagneticSolver<dim>> magnetic_solver_;

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
    SolverInfo last_mag_info_;
    SolverInfo last_ns_info_;

    // Block-Gauss-Seidel diagnostics
    unsigned int last_bgs_iterations_;
    double last_bgs_residual_;

    // Cached h_min (recomputed only after AMR or first call)
    mutable double cached_h_min_ = -1.0;

    // Persistent BGS convergence vectors (avoid per-step reallocation)
    dealii::TrilinosWrappers::MPI::Vector theta_bgs_prev_;
    dealii::TrilinosWrappers::MPI::Vector ux_bgs_prev_;
    dealii::TrilinosWrappers::MPI::Vector uy_bgs_prev_;
    bool bgs_vectors_initialized_ = false;

    // ========================================================================
    // Parallel diagnostics: assembly vs solve timing (set inside solve methods)
    // ========================================================================
    double last_ch_assemble_time_ = 0.0;
    double last_ch_solve_time_ = 0.0;
    double last_mag_assemble_time_ = 0.0;
    double last_mag_solve_time_ = 0.0;
    double last_ns_assemble_time_ = 0.0;
    double last_ns_solve_time_ = 0.0;

    // ========================================================================
    // Helper: extract individual M, phi from combined mag_solution_
    // ========================================================================
    void extract_magnetic_components();
    void build_mag_extraction_maps();

    // Precomputed index maps: mag_solution_[mag_idx] → Mx/My/phi_solution_[scalar_idx]
    // Built once in setup, used every step for O(n) vector copy instead of cell loop
    std::vector<std::pair<dealii::types::global_dof_index, dealii::types::global_dof_index>> mag_to_Mx_map_;
    std::vector<std::pair<dealii::types::global_dof_index, dealii::types::global_dof_index>> mag_to_My_map_;
    std::vector<std::pair<dealii::types::global_dof_index, dealii::types::global_dof_index>> mag_to_phi_map_;
    bool mag_extraction_maps_built_ = false;
};

#endif // PHASE_FIELD_H