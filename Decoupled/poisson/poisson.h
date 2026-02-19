// ============================================================================
// poisson/poisson.h - Magnetostatic Poisson Subsystem (Public Facade)
//
// PAPER EQUATION 42d (Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531):
//
//   (∇φ^k, ∇X) = (h_a^k − M^k, ∇X)    ∀X ∈ X_h
//
//   BCs:       ∇φ·n = 0 on ∂Ω        (pure Neumann)
//   Null-space: pin DoF 0 = 0
//   FE space:   X_h = CG Q1
//
// Properties:
//   - Constant-coefficient Laplacian: matrix assembled ONCE, AMG built ONCE
//   - RHS changes each Picard iteration (M^k) and timestep (h_a ramp)
//   - H = ∇φ is the demagnetizing field
//   - Total field H_total = h_a + ∇φ computed by other subsystems
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
//   auto diag = poisson.diagnostics(); // post-solve quantities
//
// ============================================================================
#ifndef POISSON_H
#define POISSON_H

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
    // Construction
    //
    // Takes references only — does not own triangulation or parameters.
    // ========================================================================
    PoissonSubsystem(const Parameters& params,
                     MPI_Comm mpi_comm,
                     dealii::parallel::distributed::Triangulation<dim>& triangulation);

    // ========================================================================
    // Setup — call once after mesh is ready
    //
    // Performs:
    //   1. Distribute DoFs (FE_Q<dim>(1))
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
    // Inputs from other subsystems:
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
    // VTK Output — write parallel VTU/PVTU files
    //
    // Writes phi, H_x, H_y, H_mag, subdomain.
    // Derived fields (H = grad(phi)) computed via cell-averaged DG0 projection.
    // Directory is created if it does not exist.
    // Ghosts must be up to date (call update_ghosts() first).
    //
    // @param output_dir   Timestamped directory for this run's VTU files
    // @param step         Time step number (used in filename)
    // @param time         Physical time (stored in VTU metadata)
    // ========================================================================
    void write_vtu(const std::string& output_dir,
                   unsigned int step,
                   double time) const;

    // ========================================================================
    // MMS source injection
    //
    // Set a volumetric source f_mms(x, t) for MMS testing.
    // When set and enable_mms = true, assemble_rhs adds (f_mms, X) to RHS.
    // Production code never calls this — only MMS tests.
    // ========================================================================
    using MMSSourceFunction = std::function<double(const dealii::Point<dim>&, double)>;
    void set_mms_source(MMSSourceFunction source);

    // ========================================================================
    // Accessors — for other subsystems to read results
    //
    // Consumed by:
    //   Magnetization: H = ∇φ for relaxation term (1/τ_M)(χ(θ)H, Z)
    //   NS:            H = ∇φ for Kelvin force μ₀(M·∇)H
    // ========================================================================
    const dealii::DoFHandler<dim>& get_dof_handler() const;
    const dealii::TrilinosWrappers::MPI::Vector& get_solution() const;
    const dealii::TrilinosWrappers::MPI::Vector& get_solution_relevant() const;

    // ========================================================================
    // Ghost management
    //
    // After solve(), call update_ghosts() before other subsystems read φ.
    // Call invalidate_ghosts() if solution is modified externally (e.g., AMR).
    // ========================================================================
    void update_ghosts();
    void invalidate_ghosts();

    // ========================================================================
    // Diagnostics — computed post-solve
    //
    // θ is needed ONLY here (for μ(θ) in E_mag), NOT in assembly.
    // If θ is not available (standalone test), pass empty vector.
    // ========================================================================
    struct Diagnostics
    {
        // Potential bounds
        double phi_min = 0.0;
        double phi_max = 0.0;

        // Demagnetizing field H = ∇φ
        double H_max = 0.0;       // max|∇φ|
        double H_L2 = 0.0;        // ‖∇φ‖_L2

        // Magnetic energy: E_mag = ½∫μ(θ)|H|² dΩ
        // Requires θ for μ(θ) = 1 + χ₀H(θ/ε)
        double E_mag = 0.0;

        // Permeability range (from θ)
        double mu_min = 0.0;
        double mu_max = 0.0;

        // Solver performance
        int iterations = 0;
        double residual = 0.0;
        double solve_time = 0.0;
        double assemble_time = 0.0;

        // Verification quantities
        double gauss_law_residual = 0.0;  // ‖∇·(∇φ) − RHS‖ post-solve
        double phi_pinned_value = 0.0;    // |φ(DoF 0)| — should be ~0
    };

    Diagnostics compute_diagnostics(
        const dealii::TrilinosWrappers::MPI::Vector& theta_relevant,
        const dealii::DoFHandler<dim>& theta_dof_handler,
        double current_time) const;

    // Lightweight diagnostics without θ (standalone MMS test)
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

    // poisson_solve.cc — solve logic is in public solve()

    // ========================================================================
    // References (not owned)
    // ========================================================================
    const Parameters& params_;
    MPI_Comm mpi_comm_;
    dealii::parallel::distributed::Triangulation<dim>& triangulation_;

    // ========================================================================
    // Console output (rank 0 only)
    // ========================================================================
    dealii::ConditionalOStream pcout_;

    // ========================================================================
    // Finite element and DoFHandler
    //
    // Paper: X_h = CG Q1 (piecewise linear, continuous)
    // ========================================================================
    dealii::FE_Q<dim> fe_;
    dealii::DoFHandler<dim> dof_handler_;

    // ========================================================================
    // Parallel index sets
    // ========================================================================
    dealii::IndexSet locally_owned_dofs_;
    dealii::IndexSet locally_relevant_dofs_;

    // ========================================================================
    // Constraints: hanging nodes (AMR) + pin DoF 0 = 0 (Neumann null-space)
    // ========================================================================
    dealii::AffineConstraints<double> constraints_;

    // ========================================================================
    // Linear system
    //
    // Matrix: (∇φ, ∇X) — CONSTANT, assembled once in setup()
    // RHS:    (h_a − M, ∇X) — changes each Picard iteration / timestep
    // ========================================================================
    dealii::TrilinosWrappers::SparseMatrix system_matrix_;
    dealii::TrilinosWrappers::MPI::Vector system_rhs_;

    // ========================================================================
    // Solution vectors
    //
    // solution_:          locally owned (for solver output)
    // solution_relevant_: ghosted (for other subsystems to read ∇φ)
    // ========================================================================
    dealii::TrilinosWrappers::MPI::Vector solution_;
    dealii::TrilinosWrappers::MPI::Vector solution_relevant_;

    // ========================================================================
    // Ghost tracking
    // ========================================================================
    bool ghosts_valid_ = false;

    // ========================================================================
    // AMG preconditioner — built ONCE in setup(), reused for all solves
    //
    // Poisson matrix is constant ⟹ AMG never needs rebuilding
    // (unless mesh changes via AMR, which triggers full setup() again)
    // ========================================================================
    dealii::TrilinosWrappers::PreconditionAMG amg_preconditioner_;
    bool amg_initialized_ = false;

    // ========================================================================
    // MMS source callback (set via set_mms_source, used by assemble_rhs)
    // ========================================================================
    MMSSourceFunction mms_source_;

    // ========================================================================
    // Cached solver diagnostics (from last solve)
    // ========================================================================
    SolverInfo last_solve_info_;
    double last_assemble_time_ = 0.0;
};

// ============================================================================
// Explicit instantiations (in poisson.cc)
// ============================================================================
extern template class PoissonSubsystem<2>;
extern template class PoissonSubsystem<3>;

#endif // POISSON_H