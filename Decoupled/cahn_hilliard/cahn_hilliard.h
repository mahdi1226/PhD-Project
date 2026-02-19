// ============================================================================
// cahn_hilliard/cahn_hilliard.h - Cahn-Hilliard Subsystem Facade
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
//            Equations 42a-42b (discrete coupled θ-ψ system)
//
// Solves the coupled Cahn-Hilliard phase-field equations:
//
//   Eq 42a: (δθ/τ, Λ) + B_h(U; θ^{n-1}, Λ) + γ(∇ψ, ∇Λ) = (1/τ)(θ^{n-1}, Λ)
//   Eq 42b: (ψ, Υ) + ε(∇θ, ∇Υ) + (1/η)(θ, Υ) = -(1/ε)f(θ^{n-1}), Υ) + (1/η)(θ^{n-1}, Υ)
//
// where:
//   θ = phase field (CG Q2): θ=+1 ferrofluid, θ=-1 air
//   ψ = chemical potential (CG Q2)
//   f(θ) = θ³ - θ   (double-well derivative, evaluated at θ^{n-1})
//   η = ε            (stabilization parameter)
//   γ = mobility
//   U = velocity (from NS, LAGGED: U^{n-1})
//
// Architecture:
//   Two separate CG Q2 DoFHandlers for θ and ψ on a shared triangulation.
//   The system is assembled as a monolithic coupled system using index maps
//   (θ DoF → coupled index, ψ DoF → coupled index) with layout:
//     [0, n_theta)        = θ block
//     [n_theta, n_total)  = ψ block
//
// Implementation split:
//   cahn_hilliard.cc         — constructor, setup orchestration, accessors
//   cahn_hilliard_setup.cc   — DoFs, constraints, index maps, sparsity, vectors
//   cahn_hilliard_assemble.cc — coupled θ-ψ assembly (Eq. 42a-42b)
//   cahn_hilliard_solve.cc   — direct solver (MUMPS fallback chain), extraction
//
// MMS verification:
//   Source terms injected via set_mms_source() callback — keeps test code
//   out of production assembly. Dirichlet BCs for MMS via apply_dirichlet_boundary().
// ============================================================================
#ifndef CAHN_HILLIARD_H
#define CAHN_HILLIARD_H

#include "utilities/parameters.h"
#include "utilities/solver_info.h"

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/function.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <functional>
#include <string>
#include <vector>

template <int dim>
class CahnHilliardSubsystem
{
public:
    // ========================================================================
    // Construction
    // ========================================================================

    /**
     * @brief Construct the CH subsystem
     *
     * Takes references to shared Parameters and Triangulation (not owned).
     * Initializes FE_Q with degree_phase (default Q2), attaches DoFHandlers,
     * and sets up rank-0 console output.
     *
     * @param params         Shared runtime parameters (not owned)
     * @param mpi_comm       MPI communicator
     * @param triangulation  Distributed mesh (not owned)
     */
    CahnHilliardSubsystem(
        const Parameters& params,
        MPI_Comm mpi_comm,
        dealii::parallel::distributed::Triangulation<dim>& triangulation);

    // ========================================================================
    // Public API — Lifecycle Methods
    // ========================================================================

    /**
     * @brief Full initialization (call once after mesh is ready)
     *
     * Orchestrates: distribute_dofs → build_constraints → build_index_maps
     * → build_coupled_sparsity → allocate_vectors
     *
     * The system matrix is NOT assembled here (it depends on U through
     * the convection term). Assembly happens in assemble().
     */
    void setup();

    /**
     * @brief Assemble coupled θ-ψ system (matrix + RHS)
     *
     * Assembles the full monolithic system at each time step.
     * The matrix depends on U through the convection term B_h, so
     * it must be rebuilt each time step (unlike Poisson).
     *
     * Within Picard iterations at fixed U, a full reassembly is still
     * needed because the nonlinear term f(θ^{n-1}) changes.
     *
     * @param theta_old_relevant    θ^{n-1} ghosted (from previous time step)
     * @param velocity_components   U^{n-1} ghosted, one vector per spatial dim
     *                              (size == dim: [ux, uy] in 2D, [ux, uy, uz] in 3D)
     * @param u_dof_handler         DoFHandler for velocity (CG Q2)
     * @param dt                    Time step size τ
     * @param current_time          Current simulation time t^n
     */
    void assemble(
        const dealii::TrilinosWrappers::MPI::Vector& theta_old_relevant,
        const std::vector<const dealii::TrilinosWrappers::MPI::Vector*>& velocity_components,
        const dealii::DoFHandler<dim>& u_dof_handler,
        double dt,
        double current_time);

    /**
     * @brief Solve the coupled system and extract θ, ψ
     *
     * Uses direct solver (MUMPS → SuperLU_DIST → KLU fallback chain).
     * After solve, distributes constraints and extracts θ, ψ from the
     * coupled solution vector using the index maps.
     *
     * @return SolverInfo with iterations, residual, timing
     */
    SolverInfo solve();

    // ========================================================================
    // Public API — MMS Verification
    // ========================================================================

    /**
     * @brief Inject MMS source terms for verification
     *
     * The callbacks are evaluated at each quadrature point during assembly.
     * Set to nullptr to disable MMS source injection.
     *
     * @param theta_source  f_θ(x, t) source for θ equation
     * @param psi_source    f_ψ(x, t) source for ψ equation
     */
    using MmsSourceFunction = std::function<double(const dealii::Point<dim>& point,
                                                   double time)>;
    void set_mms_source(MmsSourceFunction theta_source,
                        MmsSourceFunction psi_source);

    /**
     * @brief Apply Dirichlet boundary conditions (for MMS testing)
     *
     * Rebuilds internal constraints with the given time-dependent BCs
     * on all boundary faces, then rebuilds the coupled constraint object.
     * Call before assemble() at each time step in MMS mode.
     *
     * For production (natural Neumann BCs), this is never called.
     *
     * @param theta_bc  Dirichlet BC function for θ (must have set_time() called)
     * @param psi_bc    Dirichlet BC function for ψ (must have set_time() called)
     */
    void apply_dirichlet_boundary(
        const dealii::Function<dim>& theta_bc,
        const dealii::Function<dim>& psi_bc);

    // ========================================================================
    // Public API — Initialization
    // ========================================================================

    /**
     * @brief Project arbitrary initial conditions onto CG Q2 space
     *
     * Uses VectorTools::project for global L² projection.
     * For MMS tests with known exact IC.
     *
     * @param f_theta  Initial condition for θ
     * @param f_psi    Initial condition for ψ
     */
    void project_initial_condition(
        const dealii::Function<dim>& f_theta,
        const dealii::Function<dim>& f_psi);

    // ========================================================================
    // VTK Output — write parallel VTU/PVTU files
    //
    // Writes theta, psi, |grad_theta|, energy_density, subdomain.
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

    /**
     * @brief Set θ = constant everywhere (e.g., spinodal decomposition IC)
     *
     * Sets θ to the given constant plus optional random perturbation.
     * ψ is initialized to f(θ)/ε = 0 for θ = ±1.
     *
     * @param theta_value  Constant θ value (e.g., 0.0 for symmetric quench)
     */
    void initialize_constant(double theta_value);

    // ========================================================================
    // Public API — Accessors
    // ========================================================================

    const dealii::DoFHandler<dim>& get_theta_dof_handler() const;
    const dealii::DoFHandler<dim>& get_psi_dof_handler() const;

    /// Locally-owned θ solution (solver output)
    dealii::TrilinosWrappers::MPI::Vector& get_theta_solution();
    const dealii::TrilinosWrappers::MPI::Vector& get_theta_solution() const;

    /// Locally-owned ψ solution (solver output)
    dealii::TrilinosWrappers::MPI::Vector& get_psi_solution();
    const dealii::TrilinosWrappers::MPI::Vector& get_psi_solution() const;

    /// Ghosted θ (for cross-subsystem reads)
    const dealii::TrilinosWrappers::MPI::Vector& get_theta_relevant() const;

    /// Ghosted ψ (for cross-subsystem reads)
    const dealii::TrilinosWrappers::MPI::Vector& get_psi_relevant() const;

    /// Copy owned → ghosted (lazy: skips if already valid)
    void update_ghosts();

    /// Mark ghosts stale (called internally after solve())
    void invalidate_ghosts();

    // ========================================================================
    // Public API — Diagnostics
    // ========================================================================

    struct Diagnostics
    {
        // Phase field bounds
        double theta_min = 0.0;
        double theta_max = 0.0;
        double theta_mean = 0.0;

        // Mass conservation: ∫θ dΩ (should be conserved for Neumann BCs)
        double mass_integral = 0.0;

        // Cahn-Hilliard energy: E_CH = λ ∫ [ε/2 |∇θ|² + (1/ε)F(θ)] dΩ
        double E_gradient = 0.0;    // λ ε/2 ∫|∇θ|² dΩ
        double E_bulk = 0.0;        // λ/ε ∫F(θ) dΩ
        double E_total = 0.0;       // E_gradient + E_bulk

        // Chemical potential
        double psi_min = 0.0;
        double psi_max = 0.0;
        double psi_L2 = 0.0;       // ||ψ||_L2

        // Interface diagnostics
        double interface_length = 0.0;  // ∫|∇θ| dΩ (approximate)

        // Solver
        unsigned int iterations = 0;
        double residual = 0.0;
        double solve_time = 0.0;
        double assemble_time = 0.0;
    };

    /**
     * @brief Compute diagnostics from current solution
     *
     * Evaluates at quadrature points over locally owned cells,
     * then reduces across all MPI ranks.
     */
    Diagnostics compute_diagnostics() const;

    // ========================================================================
    // Internal — Setup (cahn_hilliard_setup.cc)
    // ========================================================================
private:
    void distribute_dofs();
    void build_constraints();
    void build_index_maps();
    void build_coupled_sparsity();
    void allocate_vectors();

    /// Rebuild coupled constraints from individual θ, ψ constraints
    void rebuild_coupled_constraints();

    // ========================================================================
    // Internal — Assembly (cahn_hilliard_assemble.cc)
    // ========================================================================
private:
    void assemble_system(
        const dealii::TrilinosWrappers::MPI::Vector& theta_old_relevant,
        const std::vector<const dealii::TrilinosWrappers::MPI::Vector*>& velocity_components,
        const dealii::DoFHandler<dim>& u_dof_handler,
        double dt,
        double current_time);

    // ========================================================================
    // Internal — Solve (cahn_hilliard_solve.cc)
    // ========================================================================
private:
    SolverInfo solve_coupled_system();

    // ========================================================================
    // Private Members
    // ========================================================================
private:
    // -- References (not owned) --
    const Parameters& params_;
    MPI_Comm mpi_comm_;
    dealii::parallel::distributed::Triangulation<dim>& triangulation_;

    // -- Finite elements --
    dealii::FE_Q<dim> fe_;    // CG Q2 (shared FE type for θ and ψ)

    dealii::DoFHandler<dim> theta_dof_handler_;
    dealii::DoFHandler<dim> psi_dof_handler_;

    // -- Parallel index sets (individual fields) --
    dealii::IndexSet theta_locally_owned_;
    dealii::IndexSet theta_locally_relevant_;
    dealii::IndexSet psi_locally_owned_;
    dealii::IndexSet psi_locally_relevant_;

    // -- Parallel index sets (coupled system) --
    dealii::IndexSet ch_locally_owned_;
    dealii::IndexSet ch_locally_relevant_;

    // -- Index maps: field DoF → coupled system index --
    //   θ occupies [0, n_theta)
    //   ψ occupies [n_theta, n_total)
    std::vector<dealii::types::global_dof_index> theta_to_ch_map_;
    std::vector<dealii::types::global_dof_index> psi_to_ch_map_;

    // -- Constraints --
    dealii::AffineConstraints<double> theta_constraints_;  // hanging nodes + BCs
    dealii::AffineConstraints<double> psi_constraints_;    // hanging nodes + BCs
    dealii::AffineConstraints<double> ch_constraints_;     // combined for coupled system

    // -- Linear system (coupled) --
    dealii::TrilinosWrappers::SparseMatrix system_matrix_;
    dealii::TrilinosWrappers::MPI::Vector system_rhs_;

    // -- Solution vectors (individual fields) --
    dealii::TrilinosWrappers::MPI::Vector theta_solution_;   // owned
    dealii::TrilinosWrappers::MPI::Vector psi_solution_;     // owned
    dealii::TrilinosWrappers::MPI::Vector theta_relevant_;   // ghosted
    dealii::TrilinosWrappers::MPI::Vector psi_relevant_;     // ghosted

    // -- State tracking --
    bool ghosts_valid_ = false;

    // -- MMS source injection --
    MmsSourceFunction mms_source_theta_;
    MmsSourceFunction mms_source_psi_;

    // -- Cached solver/assembly info --
    SolverInfo last_solve_info_;
    double last_assemble_time_ = 0.0;

    // -- Console output --
    dealii::ConditionalOStream pcout_;
};

#endif // CAHN_HILLIARD_H
