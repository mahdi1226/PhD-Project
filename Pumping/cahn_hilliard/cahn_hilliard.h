// ============================================================================
// cahn_hilliard/cahn_hilliard.h - Cahn-Hilliard Subsystem (Facade)
//
// SPLIT FORMULATION (Nochetto, Salgado & Tomas, CMAME 2016, arXiv:1601.06824):
//
//   (phi^k - phi^{k-1})/dt + u · grad(phi^k) = gamma * Delta(mu^k)
//   mu^k = Psi'(phi^{k-1}) + S*(phi^k - phi^{k-1}) - eps^2 * Delta(phi^k)
//
// Where:
//   phi: phase field in [-1, +1] (+1 = ferrofluid, -1 = carrier)
//   mu: chemical potential
//   Psi(phi) = (1/16)(phi^2 - 1)^2 (double-well potential)
//   S = 1/eps^2 (stabilization for Eyre's convex-concave splitting)
//   gamma: mobility coefficient
//   eps: interface thickness
//
// FE space:
//   FESystem(FE_Q(degree), 2) — monolithic (phi, mu) with CG Q_l
//   Component 0: phi (phase field)
//   Component 1: mu  (chemical potential)
//   Homogeneous Neumann BCs for both (no-flux walls)
//
// Weak form (monolithic 2x2 block):
//   Block(1,1): (1/dt)(phi, v) + b_h(u; phi, v)
//   Block(1,2): gamma * (grad mu, grad v)
//   Block(2,1): -S*(phi, w) - eps^2*(grad phi, grad w)
//   Block(2,2): (mu, w)
//
//   RHS_1: (1/dt)(phi_old, v)
//   RHS_2: (Psi'(phi_old), w) - S*(phi_old, w)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 2016, arXiv:1601.06824
// ============================================================================
#ifndef FHD_CAHN_HILLIARD_H
#define FHD_CAHN_HILLIARD_H

#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/function.h>

#include "utilities/parameters.h"
#include "utilities/solver_info.h"

#include <mpi.h>
#include <functional>
#include <string>

template <int dim>
class CahnHilliardSubsystem
{
public:
    // ========================================================================
    // Construction
    // ========================================================================
    CahnHilliardSubsystem(const Parameters& params,
                           MPI_Comm mpi_comm,
                           dealii::parallel::distributed::Triangulation<dim>& triangulation);

    // ========================================================================
    // Setup — call once after mesh is ready
    // ========================================================================
    void setup();

    // ========================================================================
    // Assemble — monolithic (phi, mu) system
    //
    //   old_solution_relevant: ghosted solution at previous time step
    //   dt: time step size
    //   ux/uy_relevant: velocity components from NS (optional, for convection)
    //   vel_dof_handler: DoFHandler for velocity
    // ========================================================================
    void assemble(
        const dealii::TrilinosWrappers::MPI::Vector& old_solution_relevant,
        double dt,
        const dealii::TrilinosWrappers::MPI::Vector& ux_relevant,
        const dealii::TrilinosWrappers::MPI::Vector& uy_relevant,
        const dealii::DoFHandler<dim>& vel_dof_handler);

    // Assemble without convection (standalone mode)
    void assemble(
        const dealii::TrilinosWrappers::MPI::Vector& old_solution_relevant,
        double dt);

    // ========================================================================
    // Solve
    // ========================================================================
    SolverInfo solve();

    // ========================================================================
    // VTK Output
    // ========================================================================
    void write_vtu(const std::string& output_dir,
                   unsigned int step,
                   double time) const;

    // ========================================================================
    // Initialize from functions (phi_0, mu_0)
    // ========================================================================
    void initialize(const dealii::Function<dim>& ic_function);

    // ========================================================================
    // Save old solution (call before assembling new time step)
    // ========================================================================
    void save_old_solution();

    // ========================================================================
    // Accessors
    // ========================================================================
    const dealii::DoFHandler<dim>& get_dof_handler() const;
    const dealii::FESystem<dim>& get_fe() const;
    const dealii::TrilinosWrappers::MPI::Vector& get_solution() const;
    const dealii::TrilinosWrappers::MPI::Vector& get_relevant() const;
    const dealii::TrilinosWrappers::MPI::Vector& get_old_relevant() const;

    // ========================================================================
    // Ghost management
    // ========================================================================
    void update_ghosts();
    void invalidate_ghosts();

    void initialize_zero();

    // ========================================================================
    // MMS source injection
    //
    // Callback: (point, t_new, dt, phi_old_disc) -> (f_phi, f_mu)
    // Uses discrete phi_old to avoid 1/tau and 1/eps^2 amplification.
    // ========================================================================
    using MmsSourceFn = std::function<std::pair<double, double>(
        const dealii::Point<dim>& p,
        double t_new,
        double dt,
        double phi_old_disc)>;

    void set_mms_source(const MmsSourceFn& fn);

    // ========================================================================
    // Diagnostics
    // ========================================================================
    struct Diagnostics
    {
        double phi_min = 0.0, phi_max = 0.0;
        double phi_L2 = 0.0;
        double phi_mass = 0.0;       // integral phi dOmega
        double mu_min = 0.0, mu_max = 0.0;
        double mu_L2 = 0.0;
        double free_energy = 0.0;    // integral [Psi(phi) + (eps^2/2)|grad phi|^2] dOmega
        int iterations = 0;
        double residual = 0.0;
        double solve_time = 0.0;
        double assemble_time = 0.0;
    };

    Diagnostics compute_diagnostics() const;

private:
    // ========================================================================
    // Internal setup
    // ========================================================================
    void distribute_dofs();
    void build_constraints();
    void build_sparsity_pattern();
    void allocate_vectors();

    // ========================================================================
    // References (not owned)
    // ========================================================================
    const Parameters& params_;
    MPI_Comm mpi_comm_;
    dealii::parallel::distributed::Triangulation<dim>& triangulation_;

    dealii::ConditionalOStream pcout_;

    // ========================================================================
    // Finite element and DoF
    // ========================================================================
    dealii::FESystem<dim> fe_;        // FESystem(FE_Q(degree), 2): [phi, mu]
    dealii::DoFHandler<dim> dof_handler_;

    dealii::IndexSet locally_owned_;
    dealii::IndexSet locally_relevant_;

    dealii::AffineConstraints<double> constraints_;

    // ========================================================================
    // Linear system (monolithic)
    // ========================================================================
    dealii::TrilinosWrappers::SparseMatrix system_matrix_;
    dealii::TrilinosWrappers::MPI::Vector system_rhs_;
    dealii::TrilinosWrappers::MPI::Vector solution_;

    dealii::TrilinosWrappers::MPI::Vector solution_relevant_;
    dealii::TrilinosWrappers::MPI::Vector old_solution_relevant_;

    bool ghosts_valid_ = false;

    // ========================================================================
    // MMS
    // ========================================================================
    MmsSourceFn mms_source_fn_;

    // ========================================================================
    // Solver info
    // ========================================================================
    SolverInfo last_solve_info_;
    double last_assemble_time_ = 0.0;
};

extern template class CahnHilliardSubsystem<2>;
extern template class CahnHilliardSubsystem<3>;

#endif // FHD_CAHN_HILLIARD_H
