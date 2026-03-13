// ============================================================================
// magnetization/magnetization.h - CG Magnetization Transport Subsystem (Facade)
//
// ZHANG SCHEME (Algorithm 3.1, Eq. 3.14/3.17):
//
//   Step 5 (Eq 3.14): mass-only matrix, explicit transport on RHS
//     (1/τ + 1/τ_M)(m̃, n) = (1/τ_M)(χ(θ)H, n) + (1/τ)(m^{n-1}, n)
//         - [(U·∇)m^{n-1} + (∇·U)m^{n-1}]·n  + ½(∇×U × m^{n-1}, n)
//
//   Step 6 (Eq 3.17): implicit CG skew transport
//     (1/τ + 1/τ_M)(m^n, n) + b(U, m^n, n) = (1/τ_M)(χ(θ)H^n, n) + (1/τ)(m̃, n)
//
//   where b(U,V,W) = ((U·∇)V, W) + ½((∇·U)V, W)   (CG skew form, no faces)
//
// KEY PROPERTIES:
//   1. CG Q1 continuous Galerkin (Zhang Eq 3.6: N_h ∈ C⁰(Ω))
//   2. Standard sparsity (no face coupling needed for CG)
//   3. Two scalar systems Mx, My sharing one DoFHandler and matrix
//   4. b(U, M, M) = 0 globally (energy neutrality via skew form)
//   5. Matrix depends on U → reassembled each timestep
//   6. RHS depends on H^k → reassembled each iteration
//   7. Preconditioner reused for both Mx and My solves
//
// FILE LAYOUT:
//   magnetization.h              ← this file (public facade)
//   magnetization.cc             ← constructor, setup(), accessors, diagnostics
//   magnetization_setup.cc       ← DoFs, constraints, sparsity, vectors
//   magnetization_assemble.cc    ← CG cell assembly, β-term, MMS source
//   magnetization_solve.cc       ← GMRES+ILU or Direct, solves Mx then My
//
// Reference: Zhang, He & Yang, SIAM J. Sci. Comput. 43(1) (2021) B167-B193
// ============================================================================
#ifndef MAGNETIZATION_SUBSYSTEM_H
#define MAGNETIZATION_SUBSYSTEM_H

#include "utilities/parameters.h"
#include "utilities/solver_info.h"

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/affine_constraints.h>
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

        // -- Conservation --
        double Mx_integral = 0.0;        // ∫Mx dΩ
        double My_integral = 0.0;        // ∫My dΩ

        // -- Energy (for Zhang Theorem 3.1 discrete energy) --
        double M_L2_norm_sq = 0.0;       // ||M||²_L2 = ∫(Mx²+My²) dΩ
        double M_dot_H = 0.0;            // (M,H) = ∫(Mx·Hx + My·Hy) dΩ

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
     * Distributes CG DoFs, builds constraints (hanging nodes),
     * builds sparsity pattern, allocates all vectors for Mx and My.
     * Does NOT assemble matrix (matrix depends on velocity U).
     */
    void setup();

    /**
     * @brief Assemble matrix + RHS for both Mx and My.
     *
     * Builds the full system: mass + CG transport (skew form) matrix,
     * plus RHS with relaxation, old-time, β-term, and MMS source.
     * Call once per timestep (matrix depends on U^{n-1}).
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
        double current_time,
        bool explicit_transport = false);

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
        double current_time,
        bool explicit_transport = false);

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

    /** CG DoFHandler (shared by Mx and My) */
    const dealii::DoFHandler<dim>& get_dof_handler() const;

    /** Locally-owned solution vectors (solver output) */
    const dealii::TrilinosWrappers::MPI::Vector& get_Mx_solution() const;
    const dealii::TrilinosWrappers::MPI::Vector& get_My_solution() const;

    /** Ghosted solution vectors (for cross-subsystem reads) */
    const dealii::TrilinosWrappers::MPI::Vector& get_Mx_relevant() const;
    const dealii::TrilinosWrappers::MPI::Vector& get_My_relevant() const;

    /** Old-time ghosted vectors M^{n-1} (for Picard sub-iteration) */
    const dealii::TrilinosWrappers::MPI::Vector& get_Mx_old_relevant() const;
    const dealii::TrilinosWrappers::MPI::Vector& get_My_old_relevant() const;

    // Mutable accessors — for AMR SolutionTransfer
    dealii::DoFHandler<dim>& get_dof_handler_mutable();
    dealii::TrilinosWrappers::MPI::Vector& get_Mx_solution_mutable();
    dealii::TrilinosWrappers::MPI::Vector& get_My_solution_mutable();
    dealii::TrilinosWrappers::MPI::Vector& get_Mx_old_relevant_mutable();
    dealii::TrilinosWrappers::MPI::Vector& get_My_old_relevant_mutable();

    /**
     * @brief Save current M as M^{n-1} snapshot for Picard sub-iteration.
     *
     * Call at the start of each timestep (before any Picard iterations).
     * Copies current Mx/My_relevant_ → Mx/My_old_relevant_.
     * Ghosts must be up to date before calling (call update_ghosts() first).
     */
    void save_old_solution();

    /**
     * @brief Apply under-relaxation to the solution vectors.
     *
     * Blends the newly-solved M with the previous iterate M^k:
     *   M^{k+1} = ω * M_solve + (1-ω) * M^k
     * where M^k is stored in Mx/My_relevant_ (ghosted, from before solve())
     * and M_solve is the current Mx/My_solution_ (from solve()).
     *
     * Must be called AFTER solve() and BEFORE update_ghosts().
     * The M^k values are read from the ghosted vectors (still valid from
     * the previous update_ghosts() call; solve() only overwrites owned vectors).
     *
     * @param omega  Under-relaxation factor (0 < ω ≤ 1)
     */
    void apply_under_relaxation(double omega);

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
    // Diagnostic accessors (for MMS residual computation)
    // ========================================================================
    const dealii::TrilinosWrappers::SparseMatrix& get_system_matrix() const
    { return system_matrix_; }
    const dealii::TrilinosWrappers::MPI::Vector& get_Mx_rhs() const
    { return Mx_rhs_; }
    const dealii::TrilinosWrappers::MPI::Vector& get_My_rhs() const
    { return My_rhs_; }

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
     * @brief Initialize M⁰ = χ(θ⁰)H⁰ via global L² projection.
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
     * For MMS tests: project exact initial conditions onto CG space.
     * Uses VectorTools::project (global solve). Invalidates ghosts.
     */
    void project_initial_condition(
        const dealii::Function<dim>& Mx_exact,
        const dealii::Function<dim>& My_exact);

private:
    // ========================================================================
    // Setup internals (magnetization_setup.cc)
    // ========================================================================
    void distribute_dofs();
    void build_constraints();
    void build_sparsity_pattern();
    void allocate_vectors();

    // ========================================================================
    // Assembly internals (magnetization_assemble.cc)
    // ========================================================================

    /**
     * @brief Assemble CG cell contributions for matrix and RHS.
     *
     * Cell integrals:
     *   LHS: (1/τ + 1/τ_M)(M, Z) + (U·∇)M·Z + ½(∇·U)(M·Z)
     *   RHS: (1/τ_M)(χH, Z) + (1/τ)(M^old, Z) + β[M(M·H)-H|M|²]·Z
     *
     * No face integrals (CG: continuity enforced by the FE space).
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
        bool matrix_and_rhs,   // true = both, false = RHS only
        bool explicit_transport = false);  // true = Step 5 mass-only matrix

    /** @brief Initialize ILU preconditioner (after matrix assembly). */
    void initialize_preconditioner();

    // ========================================================================
    // Solve internals (magnetization_solve.cc)
    // ========================================================================

    /**
     * @brief Solve a single scalar CG system: matrix * solution = rhs.
     *
     * Uses GMRES + cached ILU, or direct solver (MUMPS).
     * Applies constraints.distribute() after solve for hanging nodes.
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
    // Finite element (CG Q1) — Zhang Eq 3.6: N_h ∈ C⁰(Ω)
    // ========================================================================
    dealii::FE_Q<dim>           fe_;
    dealii::DoFHandler<dim>     dof_handler_;

    // ========================================================================
    // Parallel index sets
    // ========================================================================
    dealii::IndexSet  locally_owned_dofs_;
    dealii::IndexSet  locally_relevant_dofs_;

    // ========================================================================
    // Constraints (hanging nodes for AMR)
    // ========================================================================
    dealii::AffineConstraints<double>  constraints_;

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

    // Old-time snapshots (for Picard sub-iteration: M^{n-1} saved at timestep start)
    dealii::TrilinosWrappers::MPI::Vector     Mx_old_relevant_;   // ghosted
    dealii::TrilinosWrappers::MPI::Vector     My_old_relevant_;   // ghosted

    // ========================================================================
    // Preconditioner (ILU, initialized once per assemble(), reused for Mx+My)
    // ========================================================================
    dealii::TrilinosWrappers::PreconditionILU ilu_preconditioner_;
    bool                                      preconditioner_initialized_;

    // ========================================================================
    // State tracking
    // ========================================================================
    bool         ghosts_valid_;
    double       last_assemble_time_ = 0.0;
    SolverInfo   last_Mx_info_;
    SolverInfo   last_My_info_;

    // ========================================================================
    // Spin-vorticity RHS cache
    //
    // Zhang Eq 3.14: −½(∇×ũ^n × m^{n-1}, Z) on LHS → +½(ω_z)(-My,Mx)·Z on RHS
    // In 2D: ω_z = ∂uy/∂x − ∂ux/∂y, (∇×u)×m = ω_z(-m_y, m_x)
    //
    // Computed during full assembly (matrix_and_rhs=true), cached for
    // RHS-only Picard iterations (where U data is not available).
    // ========================================================================
    dealii::TrilinosWrappers::MPI::Vector  spin_vort_rhs_x_;
    dealii::TrilinosWrappers::MPI::Vector  spin_vort_rhs_y_;

    // ========================================================================
    // Explicit transport RHS cache (Zhang Eq 3.14, Step 5)
    //
    // -[(U·∇)M^{n-1} + (∇·U)M^{n-1}] · Z
    //
    // Note: coefficient of ∇·U is 1 (NOT ½ as in skew form). This is the
    // paper's specific choice for energy stability (Remark 3.2): the extra
    // consistent term ((∇·u)m, n) plays a key role in the energy proof.
    //
    // Computed once per timestep during full assembly (explicit_transport=true),
    // cached for Picard RHS-only iterations.
    // ========================================================================
    dealii::TrilinosWrappers::MPI::Vector  explicit_transport_rhs_x_;
    dealii::TrilinosWrappers::MPI::Vector  explicit_transport_rhs_y_;

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
// extern template class MagnetizationSubsystem<3>;  // 2D only

#endif // MAGNETIZATION_SUBSYSTEM_H