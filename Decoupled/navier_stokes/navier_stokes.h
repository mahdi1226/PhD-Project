// ============================================================================
// navier_stokes/navier_stokes.h - Navier-Stokes Subsystem (Public Facade)
//
// PAPER EQUATION 42e-42f (Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531):
//
//   (1/Ï„)(U^n âˆ’ U^{n-1}, V) + Î½(T(U^n), T(V))/2
//       + B_h(U^{n-1}; U^n, V)
//       âˆ’ (P^n, âˆ‡Â·V) = (f, V)
//
//   (âˆ‡Â·U^n, Q) = 0                            (incompressibility)
//
// FE spaces:
//   - Velocity: FE_Q<dim>(degree_velocity) â€” Q2 continuous (Taylor-Hood)
//   - Pressure: FE_DGP<dim>(degree_pressure) â€" DG P1 discontinuous
//     (Paper requirement A1: P_{k-1}^{dc} total-degree polynomials)
//
// System structure (3 separate scalar DoFHandlers â†’ monolithic saddle-point):
//   [A_uu   0      B_x^T] [ux^n]   [F_x]
//   [0      A_vv   B_y^T] [uy^n] = [F_y]
//   [B_x    B_y    0    ] [p^n ]   [0  ]
//
// Parallel:
//   - Trilinos vectors/matrices
//   - p4est distributed triangulation (passed by reference, not owned)
//   - MPI-aware assembly, solve, diagnostics
//
// Usage:
//   NSSubsystem<dim> ns(params, mpi_comm, triangulation);
//   ns.setup();
//   ns.initialize_zero();
//   ns.assemble_stokes(dt, nu);
//   auto info = ns.solve();
//   ns.advance_time();
//
// ============================================================================
#ifndef NAVIER_STOKES_H
#define NAVIER_STOKES_H

#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>

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
    //
    // Takes references only â€” does not own triangulation or parameters.
    // ========================================================================
    NSSubsystem(const Parameters& params,
                MPI_Comm mpi_comm,
                dealii::parallel::distributed::Triangulation<dim>& triangulation);

    // ========================================================================
    // Setup â€” call once after mesh is ready
    //
    // 1. Distribute DoFs for ux (Q2), uy (Q2), p (DG Q1)
    // 2. Build velocity constraints (hanging nodes + homogeneous Dirichlet)
    // 3. Build pressure constraints (pin DoF 0 for uniqueness)
    // 4. Build coupled saddle-point sparsity + index maps
    // 5. Allocate matrices and vectors
    // 6. Assemble pressure mass matrix (for Schur preconditioner)
    // ========================================================================
    void setup();

    // ========================================================================
    // Assembly â€” core NS for standalone testing
    //
    // Basic NS: (1/Ï„)(U^n âˆ’ U^{n-1}, V) + Î½(T(U^n), T(V)) âˆ’ (P^n, âˆ‡Â·V) = (f, V)
    //
    // Constant viscosity.
    // Uses internally owned U^{n-1}.
    //
    // @param dt                     Time step size
    // @param nu                     Constant kinematic viscosity
    // @param include_time_derivative Include (1/Ï„)(U^n âˆ’ U^{n-1}, V) mass term
    // @param include_convection     Add B_h(U^{n-1}; U^n, V) skew term
    // @param body_force             Optional RHS source f(x, t)
    // @param body_force_time        Time argument for body_force evaluation
    // ========================================================================
    void assemble_stokes(
        double dt, double nu,
        bool include_time_derivative = true,
        bool include_convection = false,
        const std::function<dealii::Tensor<1, dim>(const dealii::Point<dim>&, double)>* body_force = nullptr,
        double body_force_time = 0.0);

    // ========================================================================
    // Solve â€” call after assemble
    //
    // Direct solver (Amesos) with pressure pinning.
    // Block Schur iterative solver will be added separately.
    // ========================================================================
    SolverInfo solve();

    // ========================================================================
    // Time advancement â€” call AFTER solve()
    //
    // Swaps U^{n-1} â† U^n for the next timestep.
    // Also updates old ghosted vectors.
    // ========================================================================
    void advance_time();

    // ========================================================================
    // VTK Output — write parallel VTU/PVTU files
    //
    // Writes ux, uy, p, subdomain to parallel VTU files.
    // Directory is created if it does not exist.
    // Ghosts must be up to date (call update_ghosts() first).
    //
    // @param output_dir   Timestamped directory for this run's VTU files
    // @param step         Time step / refinement number (used in filename)
    // @param time         Physical time (stored in VTU metadata)
    // ========================================================================
    void write_vtu(const std::string& output_dir,
                   unsigned int step,
                   double time) const;

    // ========================================================================
    // Initialize all solutions to zero â€” call after setup()
    // ========================================================================
    void initialize_zero();

    // ========================================================================
    // Initialize velocity from functions â€” for MMS / restart
    //
    // Interpolates ux_init, uy_init into both current and old solutions.
    // Call after setup().
    // ========================================================================
    void initialize_velocity(const dealii::Function<dim>& ux_init,
                             const dealii::Function<dim>& uy_init);

    // ========================================================================
    // Set old velocity from exact solution functions â€” for MMS testing
    //
    // Interpolates ux_exact, uy_exact onto FE space, stores as U^{n-1},
    // and updates ghosted copies.
    // ========================================================================
    void set_old_velocity(const dealii::Function<dim>& ux_exact,
                          const dealii::Function<dim>& uy_exact);

    // ========================================================================
    // Accessors â€” for other subsystems to read results
    // ========================================================================
    const dealii::DoFHandler<dim>& get_ux_dof_handler() const;
    const dealii::DoFHandler<dim>& get_uy_dof_handler() const;
    const dealii::DoFHandler<dim>& get_p_dof_handler() const;

    // Owned vectors (current timestep)
    const dealii::TrilinosWrappers::MPI::Vector& get_ux_solution() const;
    const dealii::TrilinosWrappers::MPI::Vector& get_uy_solution() const;
    const dealii::TrilinosWrappers::MPI::Vector& get_p_solution() const;

    // Ghosted vectors (current timestep â€” for cross-DoFHandler evaluation)
    const dealii::TrilinosWrappers::MPI::Vector& get_ux_relevant() const;
    const dealii::TrilinosWrappers::MPI::Vector& get_uy_relevant() const;
    const dealii::TrilinosWrappers::MPI::Vector& get_p_relevant() const;

    // Old velocity (for convection linearization)
    const dealii::TrilinosWrappers::MPI::Vector& get_ux_old_relevant() const;
    const dealii::TrilinosWrappers::MPI::Vector& get_uy_old_relevant() const;

    // ========================================================================
    // Ghost management
    //
    // update_ghosts():     copy owned â†’ ghosted for all component vectors
    // invalidate_ghosts(): mark ghosted vectors as stale (after solve)
    // ========================================================================
    void update_ghosts();
    void invalidate_ghosts();

    // ========================================================================
    // Diagnostics â€” computed after solve
    // ========================================================================
    struct Diagnostics
    {
        // Velocity
        double ux_min = 0, ux_max = 0;
        double uy_min = 0, uy_max = 0;
        double U_max  = 0;         // max|U|
        double E_kin  = 0;         // Â½âˆ«|U|Â² dÎ©
        double CFL    = 0;         // U_max Â· dt / h_min

        // Incompressibility
        double divU_L2   = 0;      // ||âˆ‡Â·U||_L2
        double divU_Linf = 0;      // max|âˆ‡Â·U| per element

        // Pressure
        double p_min  = 0, p_max = 0;
        double p_mean = 0;         // âˆ«p dÎ© / |Î©|

        // Vorticity
        double omega_L2   = 0;     // ||âˆ‡Ã—U||_L2
        double omega_Linf = 0;     // max|âˆ‡Ã—U|
        double enstrophy  = 0;     // Â½âˆ«|Ï‰|Â² dÎ©

        // Solver
        int    iterations  = 0;
        double residual    = 0;
        double solve_time  = 0;
        double assemble_time = 0;

        // Non-dimensional
        double Re_max = 0;         // U_max Â· L / Î½
    };

    Diagnostics compute_diagnostics(double dt) const;

    // ========================================================================
    // Direct access to internals (for MMS tests / coupled testing)
    // ========================================================================
    const dealii::IndexSet& get_ux_locally_owned() const   { return ux_locally_owned_; }
    const dealii::IndexSet& get_uy_locally_owned() const   { return uy_locally_owned_; }
    const dealii::IndexSet& get_p_locally_owned() const    { return p_locally_owned_; }
    const dealii::IndexSet& get_ux_locally_relevant() const { return ux_locally_relevant_; }
    const dealii::IndexSet& get_uy_locally_relevant() const { return uy_locally_relevant_; }
    const dealii::IndexSet& get_p_locally_relevant() const  { return p_locally_relevant_; }
    const dealii::IndexSet& get_ns_locally_owned() const   { return ns_locally_owned_; }
    const dealii::IndexSet& get_ns_locally_relevant() const { return ns_locally_relevant_; }

    const std::vector<dealii::types::global_dof_index>& get_ux_to_ns_map() const { return ux_to_ns_map_; }
    const std::vector<dealii::types::global_dof_index>& get_uy_to_ns_map() const { return uy_to_ns_map_; }
    const std::vector<dealii::types::global_dof_index>& get_p_to_ns_map() const  { return p_to_ns_map_; }

    const dealii::AffineConstraints<double>& get_ns_constraints() const { return ns_constraints_; }
    const dealii::TrilinosWrappers::SparseMatrix& get_ns_matrix() const { return ns_matrix_; }
    const dealii::TrilinosWrappers::MPI::Vector&  get_ns_rhs() const    { return ns_rhs_; }
    const dealii::TrilinosWrappers::MPI::Vector&  get_ns_solution() const { return ns_solution_; }
    const dealii::TrilinosWrappers::SparseMatrix& get_pressure_mass_matrix() const { return pressure_mass_matrix_; }

    const dealii::parallel::distributed::Triangulation<dim>& get_triangulation() const { return triangulation_; }
    const dealii::FE_Q<dim>&   get_fe_velocity() const { return fe_velocity_; }
    const dealii::FE_DGP<dim>& get_fe_pressure() const { return fe_pressure_; }

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
    //   Velocity: Q2 continuous (Taylor-Hood)
    //   Pressure: DG P1 discontinuous (Paper requirement A1: P_{k-1}^{dc})
    // ========================================================================
    dealii::FE_Q<dim>   fe_velocity_;
    dealii::FE_DGP<dim> fe_pressure_;

    // ========================================================================
    // DoF handlers â€” three separate scalar handlers
    //
    // Mapped into monolithic saddle-point system via index maps.
    // This matches the existing ns_setup.h pattern.
    // ========================================================================
    dealii::DoFHandler<dim> ux_dof_handler_;
    dealii::DoFHandler<dim> uy_dof_handler_;
    dealii::DoFHandler<dim> p_dof_handler_;

    // ========================================================================
    // Index sets â€” per-component and coupled
    // ========================================================================
    dealii::IndexSet ux_locally_owned_, ux_locally_relevant_;
    dealii::IndexSet uy_locally_owned_, uy_locally_relevant_;
    dealii::IndexSet p_locally_owned_,  p_locally_relevant_;
    dealii::IndexSet ns_locally_owned_, ns_locally_relevant_;

    // ========================================================================
    // Index maps (scalar DoF â†’ coupled system DoF)
    //
    // Built by setup_ns_coupled_system_parallel() in ns_setup.h.
    // ========================================================================
    std::vector<dealii::types::global_dof_index> ux_to_ns_map_;
    std::vector<dealii::types::global_dof_index> uy_to_ns_map_;
    std::vector<dealii::types::global_dof_index> p_to_ns_map_;

    // ========================================================================
    // Constraints
    //
    // ux, uy: hanging nodes + homogeneous Dirichlet on all boundaries
    // p:      pin DoF 0 for uniqueness (no hanging nodes for DG)
    // ns:     coupled constraint set for saddle-point system
    // ========================================================================
    dealii::AffineConstraints<double> ux_constraints_;
    dealii::AffineConstraints<double> uy_constraints_;
    dealii::AffineConstraints<double> p_constraints_;
    dealii::AffineConstraints<double> ns_constraints_;

    // ========================================================================
    // Coupled saddle-point system
    // ========================================================================
    dealii::TrilinosWrappers::SparseMatrix ns_matrix_;
    dealii::TrilinosWrappers::MPI::Vector  ns_rhs_;
    dealii::TrilinosWrappers::MPI::Vector  ns_solution_;

    // ========================================================================
    // Component solutions â€” owned vectors
    // ========================================================================
    dealii::TrilinosWrappers::MPI::Vector ux_solution_;
    dealii::TrilinosWrappers::MPI::Vector ux_old_solution_;   // U_x^{n-1}
    dealii::TrilinosWrappers::MPI::Vector uy_solution_;
    dealii::TrilinosWrappers::MPI::Vector uy_old_solution_;   // U_y^{n-1}
    dealii::TrilinosWrappers::MPI::Vector p_solution_;

    // ========================================================================
    // Ghosted vectors â€” for assembly and inter-subsystem reads
    // ========================================================================
    dealii::TrilinosWrappers::MPI::Vector ux_relevant_;
    dealii::TrilinosWrappers::MPI::Vector uy_relevant_;
    dealii::TrilinosWrappers::MPI::Vector p_relevant_;
    dealii::TrilinosWrappers::MPI::Vector ux_old_relevant_;   // ghosted U_x^{n-1}
    dealii::TrilinosWrappers::MPI::Vector uy_old_relevant_;   // ghosted U_y^{n-1}

    // ========================================================================
    // Pressure mass matrix (for Schur complement preconditioner)
    //
    // S â‰ˆ Î± M_p,  Î± = Î½ + 1/Î”t (unsteady) or Î± = Î½ (steady)
    // Assembled once in setup(), reused every solve.
    // ========================================================================
    dealii::TrilinosWrappers::SparseMatrix pressure_mass_matrix_;

    // ========================================================================
    // Cached state from last assembly (used for Schur scaling in solve)
    // ========================================================================
    double last_assembled_viscosity_ = 0.0;
    double last_assembled_dt_        = -1.0;

    // ========================================================================
    // Last solver result
    // ========================================================================
    SolverInfo last_solve_info_;

    // ========================================================================
    // Private helpers (implemented in navier_stokes_setup.cc)
    // ========================================================================
    void build_coupled_system();
    void assemble_pressure_mass_matrix();

    // ========================================================================
    // Private helpers (implemented in navier_stokes_solve.cc)
    // ========================================================================
    SolverInfo solve_direct(bool verbose);
    void extract_solutions();
};

#endif // NAVIER_STOKES_H