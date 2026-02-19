// ============================================================================
// magnetization/magnetization.h - DG Magnetization Transport Subsystem (Facade)
//
// PAPER EQUATION 42c (Nochetto, Salgado & Tomas, CMAME 309 (2016)):
//
//   ∂M/∂t + B_h^m(U, M, Z) + (1/τ_M)(M - χ(θ)H, Z) = 0   ∀Z ∈ M_h
//
// DISCRETE SCHEME (semi-implicit, Eq. 56-57):
//   (1/τ + 1/τ_M)(M^k, Z) - B_h^m(U^{n-1}, Z, M^k)
//       = (1/τ_M)(χ(θ^{n-1}) H^k, Z) + (1/τ)(M^{n-1}, Z)
//
// where τ = dt and:
//   B_h^m(U,V,W) = Σ_T ∫_T [(U·∇)V·W + ½(∇·U)(V·W)] dx
//                 - Σ_F ∫_F (U·n⁻)[[V]]·{W} dS          (Eq. 57)
//
// KEY PROPERTIES:
//   1. DG-Q1 discontinuous Galerkin (no continuity constraints)
//   2. Face-coupled sparsity (upwind flux integrals)
//   3. Two scalar systems Mx, My sharing one DoFHandler and matrix
//   4. B_h^m(U, M, M) = 0 globally (energy neutrality)
//   5. Matrix depends on U → reassembled each timestep
//   6. RHS depends on H^k → reassembled each Picard iteration
//   7. Preconditioner reused for both Mx and My solves
//
// FILE LAYOUT:
//   magnetization.h              ← this file (public facade)
//   magnetization.cc             ← constructor, setup(), accessors, diagnostics
//   magnetization_setup.cc       ← DoFs, face-coupled sparsity, vectors
//   magnetization_assemble.cc    ← DG cell+face assembly, β-term, MMS source
//   magnetization_solve.cc       ← GMRES+ILU or Direct, solves Mx then My
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef MAGNETIZATION_SUBSYSTEM_H
#define MAGNETIZATION_SUBSYSTEM_H

#include "utilities/parameters.h"
#include "utilities/solver_info.h"

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <functional>
#include <string>

// ============================================================================
// MagnetizationSubsystem<dim> — Public Facade
// ============================================================================
template <int dim>
class MagnetizationSubsystem
{
public:
    // ========================================================================
    // Diagnostics (returned by compute_diagnostics)
    // ========================================================================
    struct Diagnostics
    {
        // -- Field statistics --
        double M_magnitude_mean = 0.0;   // mean |M| over domain
        double M_magnitude_min  = 0.0;   // min |M|
        double M_magnitude_max  = 0.0;   // max |M|
        double Mx_mean = 0.0;            // mean Mx
        double My_mean = 0.0;            // mean My

        // -- Equilibrium departure --
        double M_equilibrium_departure_L2 = 0.0;  // ||M - χH||_L2

        // -- Air-phase confinement --
        double M_air_phase_max = 0.0;    // max|M| where θ < -0.5

        // -- DG conservation --
        double Mx_integral = 0.0;        // ∫Mx dΩ
        double My_integral = 0.0;        // ∫My dΩ

        // -- M×H alignment --
        double M_H_alignment_mean = 0.0; // mean(M·H / |M||H|)
        double M_cross_H_L2 = 0.0;      // ||M×H||_L2

        // -- Solver (combined Mx + My) --
        unsigned int Mx_iterations = 0;
        unsigned int My_iterations = 0;
        double Mx_residual = 0.0;
        double My_residual = 0.0;
        double solve_time    = 0.0;      // wall clock for both solves
        double assemble_time = 0.0;      // wall clock for assembly
    };

    // ========================================================================
    // MMS source callback
    //
    // Signature matches compute_mag_mms_source_with_transport():
    //   (point, t_new, t_old, tau_M, chi_val, H, U, div_U) → F_mms vector
    //
    // The assembler passes locally-computed values from the quadrature loop.
    // The test provides the analytical MMS source via a lambda.
    // Production code never sets this → F_mms = 0.
    // ========================================================================
    using MmsSourceFunction = std::function<
        dealii::Tensor<1, dim>(
            const dealii::Point<dim>& point,
            double time_new,
            double time_old,
            double tau_M,
            double chi_val,
            const dealii::Tensor<1, dim>& H,
            const dealii::Tensor<1, dim>& U,
            double div_U)>;

    // ========================================================================
    // Constructor — takes references, no ownership
    // ========================================================================
    MagnetizationSubsystem(const Parameters& params,
                           MPI_Comm mpi_comm,
                           dealii::parallel::distributed::Triangulation<dim>& triangulation);

    // ========================================================================
    // Lifecycle — called during simulation
    // ========================================================================

    /**
     * @brief Full initialization — call once after mesh is ready.
     *
     * Distributes DG DoFs, builds face-coupled sparsity pattern,
     * allocates all vectors (owned + ghosted) for Mx and My.
     * Does NOT assemble matrix (matrix depends on velocity U).
     */
    void setup();

    /**
     * @brief Assemble matrix + RHS for both Mx and My.
     *
     * Builds the full system: mass + DG transport (B_h^m) matrix,
     * plus RHS with relaxation, old-time, β-term, and MMS source.
     * Call once per timestep (matrix depends on U^{n-1}).
     *
     * @param Mx_old_relevant  Previous Mx (ghosted, from M DoFHandler)
     * @param My_old_relevant  Previous My (ghosted, from M DoFHandler)
     * @param phi_relevant     Magnetic potential (ghosted, from Poisson DoFHandler)
     * @param phi_dof_handler  Poisson DoFHandler (for evaluating ∇φ at DG quads)
     * @param theta_relevant   Phase field (ghosted, from CH DoFHandler)
     * @param theta_dof_handler CH DoFHandler (for evaluating χ(θ) at DG quads)
     * @param ux_relevant      x-velocity (ghosted, from NS DoFHandler)
     * @param uy_relevant      y-velocity (ghosted, from NS DoFHandler)
     * @param u_dof_handler    NS velocity DoFHandler
     * @param dt               Time step size
     * @param current_time     Current time (for h_a and MMS)
     */
    void assemble(
        const dealii::TrilinosWrappers::MPI::Vector& Mx_old_relevant,
        const dealii::TrilinosWrappers::MPI::Vector& My_old_relevant,
        const dealii::TrilinosWrappers::MPI::Vector& phi_relevant,
        const dealii::DoFHandler<dim>&               phi_dof_handler,
        const dealii::TrilinosWrappers::MPI::Vector& theta_relevant,
        const dealii::DoFHandler<dim>&               theta_dof_handler,
        const dealii::TrilinosWrappers::MPI::Vector& ux_relevant,
        const dealii::TrilinosWrappers::MPI::Vector& uy_relevant,
        const dealii::DoFHandler<dim>&               u_dof_handler,
        double dt,
        double current_time);

    /**
     * @brief Assemble RHS only (reuse matrix from previous assemble()).
     *
     * Within a Picard iteration, the matrix is fixed (U^{n-1} doesn't change).
     * Only the RHS changes because H^k = ∇φ^k updates each iteration.
     *
     * @param phi_relevant     Updated φ^k (ghosted, from Poisson)
     * @param phi_dof_handler  Poisson DoFHandler
     * @param theta_relevant   Phase field θ^{n-1} (unchanged within Picard)
     * @param theta_dof_handler CH DoFHandler
     * @param Mx_old_relevant  Previous Mx^{n-1} (unchanged within Picard)
     * @param My_old_relevant  Previous My^{n-1} (unchanged within Picard)
     * @param dt               Time step
     * @param current_time     Current time
     */
    void assemble_rhs_only(
        const dealii::TrilinosWrappers::MPI::Vector& phi_relevant,
        const dealii::DoFHandler<dim>&               phi_dof_handler,
        const dealii::TrilinosWrappers::MPI::Vector& theta_relevant,
        const dealii::DoFHandler<dim>&               theta_dof_handler,
        const dealii::TrilinosWrappers::MPI::Vector& Mx_old_relevant,
        const dealii::TrilinosWrappers::MPI::Vector& My_old_relevant,
        double dt,
        double current_time);

    /**
     * @brief Solve both Mx and My systems using the shared matrix.
     *
     * Solves Mx first, then My, reusing the same preconditioner.
     * Returns combined SolverInfo (total iterations = Mx + My).
     */
    SolverInfo solve();

    // ========================================================================
    // VTK Output — write parallel VTU/PVTU files
    //
    // Writes Mx, My, M_magnitude, subdomain.
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
    // ========================================================================

    /**
     * @brief Set MMS source callback for verification testing.
     *
     * The callback receives assembly-local values (chi, H, U, div_U)
     * from the quadrature loop and returns the manufactured source vector.
     * When not set, F_mms = 0 (production mode).
     */
    void set_mms_source(MmsSourceFunction func);

    // ========================================================================
    // Accessors — for other subsystems to read results
    // ========================================================================

    /** DG DoFHandler (shared by Mx and My) */
    const dealii::DoFHandler<dim>& get_dof_handler() const;

    /** Locally-owned solution vectors (solver output) */
    const dealii::TrilinosWrappers::MPI::Vector& get_Mx_solution() const;
    const dealii::TrilinosWrappers::MPI::Vector& get_My_solution() const;

    /** Ghosted solution vectors (for cross-subsystem reads) */
    const dealii::TrilinosWrappers::MPI::Vector& get_Mx_relevant() const;
    const dealii::TrilinosWrappers::MPI::Vector& get_My_relevant() const;

    /**
     * @brief Copy owned → ghosted (lazy: skips if already valid).
     *
     * Must be called before other subsystems read M via get_M*_relevant().
     * Automatically invalidated by solve().
     */
    void update_ghosts();

    /** Mark ghosts stale (called internally after solve()). */
    void invalidate_ghosts();

    // ========================================================================
    // Diagnostics
    // ========================================================================

    /**
     * @brief Compute full diagnostics (with θ for equilibrium/confinement).
     *
     * Requires ghosted Mx, My, φ, θ vectors and their DoFHandlers.
     */
    Diagnostics compute_diagnostics(
        const dealii::TrilinosWrappers::MPI::Vector& phi_relevant,
        const dealii::DoFHandler<dim>&               phi_dof_handler,
        const dealii::TrilinosWrappers::MPI::Vector& theta_relevant,
        const dealii::DoFHandler<dim>&               theta_dof_handler,
        double current_time) const;

    /**
     * @brief Compute standalone diagnostics (no θ, no equilibrium check).
     *
     * For MMS tests: uses χ = χ₀ everywhere, skips air-phase confinement.
     */
    Diagnostics compute_diagnostics_standalone(
        const dealii::TrilinosWrappers::MPI::Vector& phi_relevant,
        const dealii::DoFHandler<dim>&               phi_dof_handler,
        double current_time) const;

    // ========================================================================
    // Equilibrium initialization
    // ========================================================================

    /**
     * @brief Initialize M⁰ = χ(θ⁰)H⁰ via cell-local L² projection.
     *
     * For DG elements, L² projection is cell-local (no global solve).
     * Each cell inverts its local mass matrix independently.
     *
     * Call after setup() and after Poisson has been solved for the initial φ.
     */
    void initialize_equilibrium(
        const dealii::TrilinosWrappers::MPI::Vector& phi_relevant,
        const dealii::DoFHandler<dim>&               phi_dof_handler,
        const dealii::TrilinosWrappers::MPI::Vector& theta_relevant,
        const dealii::DoFHandler<dim>&               theta_dof_handler,
        double current_time);

    /**
     * @brief L² project arbitrary scalar Functions onto Mx, My.
     *
     * For MMS tests: project exact initial conditions onto DG space.
     * Cell-local (no global solve). Invalidates ghosts.
     */
    void project_initial_condition(
        const dealii::Function<dim>& Mx_exact,
        const dealii::Function<dim>& My_exact);

private:
    // ========================================================================
    // Setup internals (magnetization_setup.cc)
    // ========================================================================
    void distribute_dofs();
    void build_sparsity_pattern();
    void allocate_vectors();

    // ========================================================================
    // Assembly internals (magnetization_assemble.cc)
    // ========================================================================

    /**
     * @brief Assemble DG cell + face contributions for matrix and RHS.
     *
     * Cell integrals:
     *   LHS: (1/τ + 1/τ_M)(M, Z) + (U·∇)M·Z + ½(∇·U)(M·Z)
     *   RHS: (1/τ_M)(χH, Z) + (1/τ)(M^old, Z) + β[M(M·H)-H|M|²]·Z
     *
     * Face integrals (interior faces only, Eq. 57):
     *   LHS: -∫_F (U·n⁻)[[Z]]·{M} dS   (upwind flux)
     */
    void assemble_system_internal(
        const dealii::TrilinosWrappers::MPI::Vector& Mx_old_relevant,
        const dealii::TrilinosWrappers::MPI::Vector& My_old_relevant,
        const dealii::TrilinosWrappers::MPI::Vector& phi_relevant,
        const dealii::DoFHandler<dim>&               phi_dof_handler,
        const dealii::TrilinosWrappers::MPI::Vector& theta_relevant,
        const dealii::DoFHandler<dim>&               theta_dof_handler,
        const dealii::TrilinosWrappers::MPI::Vector& ux_relevant,
        const dealii::TrilinosWrappers::MPI::Vector& uy_relevant,
        const dealii::DoFHandler<dim>&               u_dof_handler,
        double dt,
        double current_time,
        bool matrix_and_rhs);  // true = both, false = RHS only

    /** @brief Initialize ILU preconditioner (after matrix assembly). */
    void initialize_preconditioner();

    // ========================================================================
    // Solve internals (magnetization_solve.cc)
    // ========================================================================

    /**
     * @brief Solve a single scalar DG system: matrix * solution = rhs.
     *
     * Uses GMRES + cached ILU, or direct solver (MUMPS).
     */
    SolverInfo solve_component(
        dealii::TrilinosWrappers::MPI::Vector& solution,
        const dealii::TrilinosWrappers::MPI::Vector& rhs,
        const std::string& component_name);

    // ========================================================================
    // References (not owned)
    // ========================================================================
    const Parameters&                                      params_;
    MPI_Comm                                               mpi_comm_;
    dealii::parallel::distributed::Triangulation<dim>&     triangulation_;

    // ========================================================================
    // Finite element (DG-Q1)
    // ========================================================================
    dealii::FE_DGQ<dim>         fe_;
    dealii::DoFHandler<dim>     dof_handler_;

    // ========================================================================
    // Parallel index sets
    // ========================================================================
    dealii::IndexSet  locally_owned_dofs_;
    dealii::IndexSet  locally_relevant_dofs_;

    // ========================================================================
    // Linear system (shared by Mx and My)
    // ========================================================================
    dealii::TrilinosWrappers::SparseMatrix    system_matrix_;
    dealii::TrilinosWrappers::MPI::Vector     Mx_rhs_;
    dealii::TrilinosWrappers::MPI::Vector     My_rhs_;

    // ========================================================================
    // Solutions
    // ========================================================================
    dealii::TrilinosWrappers::MPI::Vector     Mx_solution_;       // locally owned
    dealii::TrilinosWrappers::MPI::Vector     My_solution_;       // locally owned
    dealii::TrilinosWrappers::MPI::Vector     Mx_relevant_;       // ghosted
    dealii::TrilinosWrappers::MPI::Vector     My_relevant_;       // ghosted

    // ========================================================================
    // Preconditioner (ILU, initialized once per assemble(), reused for Mx+My)
    // ========================================================================
    dealii::TrilinosWrappers::PreconditionILU ilu_preconditioner_;
    bool                                      preconditioner_initialized_;

    // ========================================================================
    // State tracking
    // ========================================================================
    bool         ghosts_valid_;
    SolverInfo   last_Mx_info_;
    SolverInfo   last_My_info_;

    // ========================================================================
    // MMS
    // ========================================================================
    MmsSourceFunction  mms_source_;

    // ========================================================================
    // Output
    // ========================================================================
    dealii::ConditionalOStream  pcout_;
};

// ============================================================================
// Explicit instantiations (defined in .cc files)
// ============================================================================
extern template class MagnetizationSubsystem<2>;
extern template class MagnetizationSubsystem<3>;

#endif // MAGNETIZATION_SUBSYSTEM_H