// ============================================================================
// navier_stokes/navier_stokes.h - Navier-Stokes Subsystem (Public Facade)
//
// PRESSURE-CORRECTION PROJECTION METHOD (Zhang Algorithm 3.1, Steps 2-4):
//
//   Step 2 — Velocity predictor ū^n:
//     (1/δt)(ū^n − u^{n-1}, v) + ν(D(ū^n), D(v)) + b(u^{n-1}, ū^n, v)
//         + b_stab(m^{n-1}, ū^n, v) − (p^{n-1}, ∇·v) = (f, v)
//
//   Step 3 — Pressure Poisson:
//     (∇p^n, ∇q) = −(1/δt)(∇·ū^n, q) + (∇p^{n-1}, ∇q)
//
//   Step 4 — Velocity correction (algebraic):
//     u^n = ū^n − δt ∇(p^n − p^{n-1})
//
// FE spaces:
//   - Velocity: FE_Q<dim>(2) — CG Q2 continuous
//   - Pressure: FE_Q<dim>(1) — CG Q1 continuous (for pressure Poisson)
//
// System structure (3 SEPARATE scalar systems):
//   [A_ux] [ūx^n] = [F_x]     velocity predictor x
//   [A_uy] [ūy^n] = [F_y]     velocity predictor y
//   [L_p ] [p^n ] = [G  ]     pressure Poisson
//
// The viscous cross-terms (ux-uy coupling from D:D) and b_stab cross-terms
// are treated explicitly on the RHS using old velocity.
//
// Reference: Zhang, He & Yang, SIAM J. Sci. Comput. 43(1), 2021
// ============================================================================
#ifndef NAVIER_STOKES_H
#define NAVIER_STOKES_H

#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/base/conditional_ostream.h>

#include "utilities/parameters.h"
#include "utilities/solver_info.h"

#include <mpi.h>
#include <vector>
#include <string>
#include <functional>

template <int dim>
class NSSubsystem
{
public:
    // ========================================================================
    // Construction
    // ========================================================================
    NSSubsystem(const Parameters& params,
                MPI_Comm mpi_comm,
                dealii::parallel::distributed::Triangulation<dim>& triangulation);

    // ========================================================================
    // Setup — call once after mesh is ready
    //
    // 1. Distribute DoFs for ux (Q2), uy (Q2), p (Q1)
    // 2. Build constraints (hanging nodes + Dirichlet for velocity)
    // 3. Build 3 separate sparsity patterns
    // 4. Allocate matrices, vectors, lumped mass diagonal
    // ========================================================================
    void setup();

    // ========================================================================
    // Assembly — velocity predictor (Zhang Step 2)
    //
    // Builds ux_matrix_ and uy_matrix_ separately.
    // Viscous cross-terms and b_stab cross-terms go to RHS using old velocity.
    // Old pressure gradient on RHS.
    // ========================================================================

    // Standalone NS (for testing): constant viscosity, no coupling
    void assemble_stokes(
        double dt, double nu,
        bool include_time_derivative = true,
        bool include_convection = false,
        const std::function<dealii::Tensor<1, dim>(const dealii::Point<dim>&, double)>* body_force = nullptr,
        double body_force_time = 0.0);

    // Coupled assembly — for production ferrofluid driver
    void assemble_coupled(
        double dt,
        const dealii::TrilinosWrappers::MPI::Vector& theta_relevant,
        const dealii::TrilinosWrappers::MPI::Vector& theta_old_relevant,
        const dealii::DoFHandler<dim>&               theta_dof_handler,
        const dealii::TrilinosWrappers::MPI::Vector& psi_relevant,
        const dealii::DoFHandler<dim>&               psi_dof_handler,
        const dealii::TrilinosWrappers::MPI::Vector& phi_relevant,
        const dealii::DoFHandler<dim>&               phi_dof_handler,
        const dealii::TrilinosWrappers::MPI::Vector& Mx_relevant,
        const dealii::TrilinosWrappers::MPI::Vector& My_relevant,
        const dealii::DoFHandler<dim>&               M_dof_handler,
        double current_time,
        bool include_convection = true);

    // Coupled assembly with algebraic magnetization
    void assemble_coupled_algebraic_M(
        double dt,
        const dealii::TrilinosWrappers::MPI::Vector& theta_relevant,
        const dealii::TrilinosWrappers::MPI::Vector& theta_old_relevant,
        const dealii::DoFHandler<dim>&               theta_dof_handler,
        const dealii::TrilinosWrappers::MPI::Vector& psi_relevant,
        const dealii::DoFHandler<dim>&               psi_dof_handler,
        const dealii::TrilinosWrappers::MPI::Vector& phi_relevant,
        const dealii::DoFHandler<dim>&               phi_dof_handler,
        double current_time,
        bool include_convection = true);

    // ========================================================================
    // Assembly — pressure Poisson (Zhang Step 3)
    //
    // (∇p^n, ∇q) = −(1/δt)(∇·ū^n, q) + (∇p^{n-1}, ∇q)
    // ========================================================================
    void assemble_pressure_poisson(double dt);

    // ========================================================================
    // Velocity correction (Zhang Step 4) — algebraic
    //
    // u^n = ū^n − δt ∇(p^n − p^{n-1})
    // Applied weakly via lumped mass matrix.
    // ========================================================================
    void velocity_correction(double dt);

    // ========================================================================
    // Solve — call after assembly
    //
    // solve_velocity(): CG+AMG for ux_matrix_ and uy_matrix_
    // solve_pressure(): CG+AMG for p_matrix_
    // solve():          backward-compat wrapper (velocity only)
    // ========================================================================
    SolverInfo solve_velocity();
    SolverInfo solve_pressure();
    SolverInfo solve();  // calls solve_velocity() for backward compat

    // ========================================================================
    // Time advancement — call AFTER solve/correction
    //
    // Swaps U^{n-1} ← U^n, P^{n-1} ← P^n for the next timestep.
    // ========================================================================
    void advance_time();

    // ========================================================================
    // VTK Output
    // ========================================================================
    void write_vtu(const std::string& output_dir,
                   unsigned int step,
                   double time) const;

    // ========================================================================
    // Initialize solutions
    // ========================================================================
    void initialize_zero();
    void initialize_velocity(const dealii::Function<dim>& ux_init,
                             const dealii::Function<dim>& uy_init);
    void set_old_velocity(const dealii::Function<dim>& ux_exact,
                          const dealii::Function<dim>& uy_exact);

    // ========================================================================
    // Accessors — for other subsystems to read results
    // ========================================================================
    const dealii::DoFHandler<dim>& get_ux_dof_handler() const;
    const dealii::DoFHandler<dim>& get_uy_dof_handler() const;
    const dealii::DoFHandler<dim>& get_p_dof_handler() const;

    // Owned vectors (current timestep)
    const dealii::TrilinosWrappers::MPI::Vector& get_ux_solution() const;
    const dealii::TrilinosWrappers::MPI::Vector& get_uy_solution() const;
    const dealii::TrilinosWrappers::MPI::Vector& get_p_solution() const;

    // Ghosted vectors (current timestep)
    const dealii::TrilinosWrappers::MPI::Vector& get_ux_relevant() const;
    const dealii::TrilinosWrappers::MPI::Vector& get_uy_relevant() const;
    const dealii::TrilinosWrappers::MPI::Vector& get_p_relevant() const;

    // Old timestep (for convection linearization + pressure correction)
    const dealii::TrilinosWrappers::MPI::Vector& get_ux_old_relevant() const;
    const dealii::TrilinosWrappers::MPI::Vector& get_uy_old_relevant() const;
    const dealii::TrilinosWrappers::MPI::Vector& get_p_old_relevant() const;

    // ========================================================================
    // Mutable accessors — for AMR SolutionTransfer
    // ========================================================================
    dealii::DoFHandler<dim>& get_ux_dof_handler_mutable();
    dealii::DoFHandler<dim>& get_uy_dof_handler_mutable();
    dealii::DoFHandler<dim>& get_p_dof_handler_mutable();

    dealii::TrilinosWrappers::MPI::Vector& get_ux_solution_mutable();
    dealii::TrilinosWrappers::MPI::Vector& get_uy_solution_mutable();
    dealii::TrilinosWrappers::MPI::Vector& get_p_solution_mutable();

    dealii::TrilinosWrappers::MPI::Vector& get_ux_old_solution_mutable();
    dealii::TrilinosWrappers::MPI::Vector& get_uy_old_solution_mutable();
    dealii::TrilinosWrappers::MPI::Vector& get_p_old_solution_mutable();

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
        double ux_min = 0, ux_max = 0;
        double uy_min = 0, uy_max = 0;
        double U_max  = 0;
        double E_kin  = 0;
        double CFL    = 0;
        double divU_L2   = 0;
        double divU_Linf = 0;
        double p_min  = 0, p_max = 0;
        double p_mean = 0;
        double omega_L2   = 0;
        double omega_Linf = 0;
        double enstrophy  = 0;
        int    iterations  = 0;
        double residual    = 0;
        double solve_time  = 0;
        double assemble_time = 0;
        double Re_max = 0;
    };

    Diagnostics compute_diagnostics(double dt) const;

    // ========================================================================
    // MMS source injection
    // ========================================================================
    using MmsSourceFunction = std::function<
        dealii::Tensor<1, dim>(const dealii::Point<dim>&, double)>;
    void set_mms_source(MmsSourceFunction source);

    // ========================================================================
    // Direct access to internals (for MMS tests / coupled testing)
    // ========================================================================
    const dealii::IndexSet& get_ux_locally_owned() const   { return ux_locally_owned_; }
    const dealii::IndexSet& get_uy_locally_owned() const   { return uy_locally_owned_; }
    const dealii::IndexSet& get_p_locally_owned() const    { return p_locally_owned_; }
    const dealii::IndexSet& get_ux_locally_relevant() const { return ux_locally_relevant_; }
    const dealii::IndexSet& get_uy_locally_relevant() const { return uy_locally_relevant_; }
    const dealii::IndexSet& get_p_locally_relevant() const  { return p_locally_relevant_; }

    const dealii::AffineConstraints<double>& get_ux_constraints() const { return ux_constraints_; }
    const dealii::AffineConstraints<double>& get_uy_constraints() const { return uy_constraints_; }
    const dealii::AffineConstraints<double>& get_p_constraints() const  { return p_constraints_; }

    const dealii::TrilinosWrappers::SparseMatrix& get_ux_matrix() const { return ux_matrix_; }
    const dealii::TrilinosWrappers::SparseMatrix& get_uy_matrix() const { return uy_matrix_; }
    const dealii::TrilinosWrappers::SparseMatrix& get_p_matrix() const  { return p_matrix_; }

    const dealii::parallel::distributed::Triangulation<dim>& get_triangulation() const { return triangulation_; }
    const dealii::FE_Q<dim>& get_fe_velocity() const { return fe_velocity_; }
    const dealii::FE_Q<dim>& get_fe_pressure() const { return fe_pressure_; }

private:
    // ========================================================================
    // References (not owned)
    // ========================================================================
    const Parameters& params_;
    MPI_Comm          mpi_comm_;
    dealii::parallel::distributed::Triangulation<dim>& triangulation_;
    dealii::ConditionalOStream pcout_;

    // ========================================================================
    // Finite elements
    //   Velocity: Q2 continuous
    //   Pressure: Q1 continuous (for pressure Poisson in projection method)
    // ========================================================================
    dealii::FE_Q<dim> fe_velocity_;
    dealii::FE_Q<dim> fe_pressure_;

    // ========================================================================
    // DoF handlers — three separate scalar handlers
    // ========================================================================
    dealii::DoFHandler<dim> ux_dof_handler_;
    dealii::DoFHandler<dim> uy_dof_handler_;
    dealii::DoFHandler<dim> p_dof_handler_;

    // ========================================================================
    // Index sets — per-component
    // ========================================================================
    dealii::IndexSet ux_locally_owned_, ux_locally_relevant_;
    dealii::IndexSet uy_locally_owned_, uy_locally_relevant_;
    dealii::IndexSet p_locally_owned_,  p_locally_relevant_;

    // ========================================================================
    // Constraints
    //
    // ux, uy: hanging nodes + homogeneous Dirichlet on all boundaries
    // p:      hanging nodes + mean value constraint (CG Q1, no Dirichlet)
    // ========================================================================
    dealii::AffineConstraints<double> ux_constraints_;
    dealii::AffineConstraints<double> uy_constraints_;
    dealii::AffineConstraints<double> p_constraints_;

    // ========================================================================
    // 3 separate scalar systems (velocity predictor + pressure Poisson)
    // ========================================================================
    dealii::TrilinosWrappers::SparseMatrix ux_matrix_;
    dealii::TrilinosWrappers::SparseMatrix uy_matrix_;
    dealii::TrilinosWrappers::SparseMatrix p_matrix_;

    // Cached AMG for pressure Poisson (matrix is constant — pure Laplacian)
    dealii::TrilinosWrappers::PreconditionAMG p_amg_;
    bool p_amg_initialized_ = false;
    bool p_matrix_assembled_ = false;  // pressure Laplacian assembled (skip if mesh unchanged)

    dealii::TrilinosWrappers::MPI::Vector ux_rhs_;
    dealii::TrilinosWrappers::MPI::Vector uy_rhs_;
    dealii::TrilinosWrappers::MPI::Vector p_rhs_;

    // ========================================================================
    // Component solutions — owned vectors
    // ========================================================================
    dealii::TrilinosWrappers::MPI::Vector ux_solution_;
    dealii::TrilinosWrappers::MPI::Vector ux_old_solution_;   // U_x^{n-1}
    dealii::TrilinosWrappers::MPI::Vector uy_solution_;
    dealii::TrilinosWrappers::MPI::Vector uy_old_solution_;   // U_y^{n-1}
    dealii::TrilinosWrappers::MPI::Vector p_solution_;
    dealii::TrilinosWrappers::MPI::Vector p_old_solution_;    // P^{n-1} for correction

    // ========================================================================
    // Ghosted vectors — for assembly and inter-subsystem reads
    // ========================================================================
    dealii::TrilinosWrappers::MPI::Vector ux_relevant_;
    dealii::TrilinosWrappers::MPI::Vector uy_relevant_;
    dealii::TrilinosWrappers::MPI::Vector p_relevant_;
    dealii::TrilinosWrappers::MPI::Vector ux_old_relevant_;   // ghosted U_x^{n-1}
    dealii::TrilinosWrappers::MPI::Vector uy_old_relevant_;   // ghosted U_y^{n-1}
    dealii::TrilinosWrappers::MPI::Vector p_old_relevant_;    // ghosted P^{n-1}

    // ========================================================================
    // Velocity mass matrix (for velocity correction Step 4)
    //
    // Consistent mass M(i,j) = ∫ φ_i φ_j dx  — used in CG solve for
    // accurate L2 projection in the velocity correction step.
    // Lumped mass M_L kept as Jacobi preconditioner for the CG solve.
    // ========================================================================
    dealii::TrilinosWrappers::SparseMatrix vel_mass_matrix_;
    dealii::TrilinosWrappers::MPI::Vector  vel_mass_lumped_;

    // ========================================================================
    // Cached state
    // ========================================================================
    double last_assemble_time_       = 0.0;
    SolverInfo last_solve_info_;

    // ========================================================================
    // MMS source callback
    // ========================================================================
    MmsSourceFunction mms_source_;

    // ========================================================================
    // Private helpers
    // ========================================================================

    /** Unified coupled assembly — called by assemble_coupled() and
     *  assemble_coupled_algebraic_M(). When Mx_relevant is nullptr,
     *  M is computed algebraically as χ(θ_old)·∇φ. */
    void assemble_coupled_internal(
        double dt,
        const dealii::TrilinosWrappers::MPI::Vector& theta_relevant,
        const dealii::TrilinosWrappers::MPI::Vector& theta_old_relevant,
        const dealii::DoFHandler<dim>&               theta_dof_handler,
        const dealii::TrilinosWrappers::MPI::Vector& psi_relevant,
        const dealii::DoFHandler<dim>&               psi_dof_handler,
        const dealii::TrilinosWrappers::MPI::Vector& phi_relevant,
        const dealii::DoFHandler<dim>&               phi_dof_handler,
        const dealii::TrilinosWrappers::MPI::Vector* Mx_relevant,
        const dealii::TrilinosWrappers::MPI::Vector* My_relevant,
        const dealii::DoFHandler<dim>*               M_dof_handler,
        double current_time,
        bool include_convection);

    void assemble_lumped_mass();
    void subtract_mean_pressure();
};

#endif // NAVIER_STOKES_H
