// ============================================================================
// passive_scalar/passive_scalar.h - Passive Scalar Subsystem (Facade)
//
// PAPER EQUATION 104 (Nochetto, Salgado & Tomas, arXiv:1511.04381):
//
//   c_t + u · ∇c − α Δc = 0
//
// One-way coupled to FHD: velocity u drives scalar c, but c does not
// affect the flow. Used for Section 7.3 (ferromagnetic stirring).
//
// FE space:
//   CG Q_ℓ with homogeneous Neumann BCs (no-flux walls)
//
// System (backward Euler):
//   [(1/τ) M + α K + convection(u)] c^k = (1/τ) M c^{k-1}
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015)
// ============================================================================
#ifndef FHD_PASSIVE_SCALAR_H
#define FHD_PASSIVE_SCALAR_H

#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/function.h>

#include "utilities/parameters.h"
#include "utilities/solver_info.h"

#include <mpi.h>
#include <string>

template <int dim>
class PassiveScalarSubsystem
{
public:
    // ========================================================================
    // Construction
    // ========================================================================
    PassiveScalarSubsystem(const Parameters& params,
                           MPI_Comm mpi_comm,
                           dealii::parallel::distributed::Triangulation<dim>& triangulation);

    // ========================================================================
    // Setup — call once after mesh is ready
    // ========================================================================
    void setup();

    // ========================================================================
    // Assemble — convection-diffusion with velocity from NS
    //
    //   c_old_relevant: concentration at previous time step
    //   dt:             time step size
    //   ux/uy_relevant: velocity components (CG Q2, from NS)
    //   vel_dof_handler: DoF handler for velocity
    // ========================================================================
    void assemble(
        const dealii::TrilinosWrappers::MPI::Vector& c_old_relevant,
        double dt,
        const dealii::TrilinosWrappers::MPI::Vector& ux_relevant,
        const dealii::TrilinosWrappers::MPI::Vector& uy_relevant,
        const dealii::DoFHandler<dim>& vel_dof_handler);

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
    // Initialize from a function (e.g., step function IC)
    // ========================================================================
    void initialize(const dealii::Function<dim>& ic_function);

    // ========================================================================
    // Accessors
    // ========================================================================
    const dealii::DoFHandler<dim>& get_dof_handler() const;
    const dealii::TrilinosWrappers::MPI::Vector& get_solution() const;
    const dealii::TrilinosWrappers::MPI::Vector& get_relevant() const;

    // ========================================================================
    // Ghost management
    // ========================================================================
    void update_ghosts();
    void invalidate_ghosts();

    // ========================================================================
    // Initialization
    // ========================================================================
    void initialize_zero();

    // ========================================================================
    // Diagnostics
    // ========================================================================
    struct Diagnostics
    {
        double c_min = 0.0, c_max = 0.0;
        double c_L2 = 0.0;
        double c_mass = 0.0;   // ∫c dΩ (conservation check)
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
    dealii::FE_Q<dim> fe_;        // CG Q_ℓ
    dealii::DoFHandler<dim> dof_handler_;

    dealii::IndexSet locally_owned_;
    dealii::IndexSet locally_relevant_;

    dealii::AffineConstraints<double> constraints_;

    // ========================================================================
    // Linear system
    // ========================================================================
    dealii::TrilinosWrappers::SparseMatrix system_matrix_;
    dealii::TrilinosWrappers::MPI::Vector system_rhs_;
    dealii::TrilinosWrappers::MPI::Vector c_solution_;

    dealii::TrilinosWrappers::MPI::Vector c_relevant_;

    bool ghosts_valid_ = false;

    // ========================================================================
    // Diagnostics
    // ========================================================================
    SolverInfo last_solve_info_;
    double last_assemble_time_ = 0.0;
};

extern template class PassiveScalarSubsystem<2>;
extern template class PassiveScalarSubsystem<3>;

#endif // FHD_PASSIVE_SCALAR_H
