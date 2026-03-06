// ============================================================================
// poisson/poisson.h - Magnetostatic Poisson Subsystem (Public Facade)
//
// PAPER EQUATION 42d (Nochetto, Salgado & Tomas, arXiv:1511.04381):
//
//   (∇φ^k, ∇X) = (h_a^k − M^k, ∇X)    ∀X ∈ X_h
//
//   BCs:       ∇φ·n = 0 on ∂Ω        (pure Neumann)
//   Null-space: pin DoF 0 = 0
//   FE space:   X_h = CG Q_ℓ          (ℓ = degree_potential, default 2)
//
// Properties:
//   - Constant-coefficient Laplacian: matrix assembled ONCE, AMG built ONCE
//   - RHS changes each Picard iteration (M^k) and timestep (h_a ramp)
//   - H = ∇φ is the demagnetizing field
//   - Total field h = h_a + ∇φ
//
// Constraint (Nochetto Section 4.3):
//   ∇X_h ⊂ M_h required for energy stability.
//   With X_h = CG Q_ℓ, M_h = DG [Q_ℓ]^d, the inclusion ∇X ⊂ M holds.
//
// Parallel:
//   - Trilinos vectors/matrices
//   - p4est distributed triangulation (passed by reference, not owned)
//   - MPI-aware assembly, solve, diagnostics
//
// Usage:
//   PoissonSubsystem<dim> poisson(params, mpi_comm, triangulation);
//   poisson.setup();                   // DoFs, constraints, matrix, AMG
//   poisson.assemble_rhs(...);         // each Picard / timestep
//   poisson.solve();                   // CG + cached AMG
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015)
// ============================================================================
#ifndef FHD_POISSON_H
#define FHD_POISSON_H

#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>

#include "utilities/parameters.h"
#include "utilities/solver_info.h"

#include <mpi.h>
#include <functional>
#include <string>

template <int dim>
class PoissonSubsystem
{
public:
    // ========================================================================
    // Construction — does not own triangulation or parameters
    // ========================================================================
    PoissonSubsystem(const Parameters& params,
                     MPI_Comm mpi_comm,
                     dealii::parallel::distributed::Triangulation<dim>& triangulation);

    // ========================================================================
    // Setup — call once after mesh is ready
    //
    //   1. Distribute DoFs (FE_Q<dim>(degree_potential))
    //   2. Build constraints (hanging nodes + pin DoF 0)
    //   3. Build Trilinos sparsity pattern
    //   4. Assemble Laplacian matrix (∇φ, ∇X) — ONCE
    //   5. Initialize AMG preconditioner — ONCE
    //   6. Allocate vectors (RHS, solution, ghosted)
    // ========================================================================
    void setup();

    // ========================================================================
    // Assemble RHS — call every Picard iteration / timestep
    //
    // Eq. 42d RHS: (h_a^k − M^k, ∇X) + MMS source (if enabled)
    //
    // Inputs from magnetization subsystem:
    //   M_x, M_y:     magnetization components (DG, ghosted/relevant)
    //   M_dof_handler: DoFHandler for M (DG elements)
    //   current_time:  for h_a ramp and MMS time dependence
    //
    // If M has size 0, assembles with M = 0 (standalone Poisson test).
    // ========================================================================
    void assemble_rhs(
        const dealii::TrilinosWrappers::MPI::Vector& M_x_relevant,
        const dealii::TrilinosWrappers::MPI::Vector& M_y_relevant,
        const dealii::DoFHandler<dim>& M_dof_handler,
        double current_time);

    // ========================================================================
    // Solve — call after assemble_rhs
    //
    // Uses CG + cached AMG preconditioner.
    // Returns solver statistics (iterations, residual, timing).
    // ========================================================================
    SolverInfo solve();

    // ========================================================================
    // VTK Output
    // ========================================================================
    void write_vtu(const std::string& output_dir,
                   unsigned int step,
                   double time) const;

    // ========================================================================
    // MMS source injection
    //
    // Set f_mms(x, t) for standalone MMS testing.
    // When set and enable_mms = true, assemble_rhs adds (f_mms, X) to RHS.
    // ========================================================================
    using MMSSourceFunction = std::function<double(const dealii::Point<dim>&, double)>;
    void set_mms_source(MMSSourceFunction source);

    // ========================================================================
    // Accessors — for other subsystems
    //
    // Consumed by:
    //   Magnetization: H = ∇φ for relaxation term (1/𝒯)(κ₀·h, z)
    //   NS:            H = ∇φ for Kelvin force μ₀·B_h^m(v, h, m)
    //   Angular Mom:   h for magnetic torque μ₀(m × h, ξ)
    // ========================================================================
    const dealii::DoFHandler<dim>& get_dof_handler() const;
    const dealii::TrilinosWrappers::MPI::Vector& get_solution() const;
    const dealii::TrilinosWrappers::MPI::Vector& get_solution_relevant() const;

    // Mutable accessor (for AMR solution transfer)
    dealii::TrilinosWrappers::MPI::Vector& get_solution_mutable();

    // ========================================================================
    // Ghost management
    // ========================================================================
    void update_ghosts();
    void invalidate_ghosts();

    // ========================================================================
    // Diagnostics
    // ========================================================================
    struct Diagnostics
    {
        double phi_min = 0.0;
        double phi_max = 0.0;
        double H_max = 0.0;        // max|∇φ|
        double H_L2 = 0.0;         // ‖∇φ‖_L2
        int iterations = 0;
        double residual = 0.0;
        double solve_time = 0.0;
        double assemble_time = 0.0;
    };

    Diagnostics compute_diagnostics() const;

private:
    // ========================================================================
    // Internal methods — implementations in separate .cc files
    // ========================================================================

    // poisson_setup.cc
    void distribute_dofs();
    void build_constraints();
    void build_sparsity_pattern();
    void allocate_vectors();

    // poisson_assemble.cc
    void assemble_matrix();
    void initialize_preconditioner();

    // ========================================================================
    // References (not owned)
    // ========================================================================
    const Parameters& params_;
    MPI_Comm mpi_comm_;
    dealii::parallel::distributed::Triangulation<dim>& triangulation_;

    // Console output (rank 0 only)
    dealii::ConditionalOStream pcout_;

    // ========================================================================
    // Finite element and DoFHandler
    //
    // X_h = CG Q_ℓ (degree_potential from params, default ℓ=2)
    // ========================================================================
    dealii::FE_Q<dim> fe_;
    dealii::DoFHandler<dim> dof_handler_;

    // Parallel index sets
    dealii::IndexSet locally_owned_dofs_;
    dealii::IndexSet locally_relevant_dofs_;

    // Constraints: hanging nodes + pin DoF 0 = 0
    dealii::AffineConstraints<double> constraints_;

    // ========================================================================
    // Linear system
    //
    // Matrix: (∇φ, ∇X) — CONSTANT, assembled once
    // RHS:    (h_a − M, ∇X) — changes each iteration
    // ========================================================================
    dealii::TrilinosWrappers::SparseMatrix system_matrix_;
    dealii::TrilinosWrappers::MPI::Vector system_rhs_;

    // Solution vectors
    dealii::TrilinosWrappers::MPI::Vector solution_;
    dealii::TrilinosWrappers::MPI::Vector solution_relevant_;

    // Ghost tracking
    bool ghosts_valid_ = false;

    // AMG preconditioner — built ONCE, reused for all solves
    dealii::TrilinosWrappers::PreconditionAMG amg_preconditioner_;
    bool amg_initialized_ = false;

    // MMS source callback
    MMSSourceFunction mms_source_;

    // Cached solver diagnostics
    SolverInfo last_solve_info_;
    double last_assemble_time_ = 0.0;
};

// Explicit instantiations (in poisson.cc)
extern template class PoissonSubsystem<2>;
extern template class PoissonSubsystem<3>;

#endif // FHD_POISSON_H
