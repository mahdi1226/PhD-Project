// ============================================================================
// angular_momentum/angular_momentum.h - Angular Momentum Subsystem (Facade)
//
// PAPER EQUATION 42f (Nochetto, Salgado & Tomas, arXiv:1511.04381):
//
// In 2D, w is scalar (z-component of angular velocity):
//
//   j(w^k/τ, z) + j·b_h(u^{k-1}; w^k, z) + c₁(∇w^k, ∇z) + 4ν_r(w^k, z)
//     = j(w^{k-1}/τ, z) + 2ν_r(curl u^k, z) + μ₀(m^k × h^k, z) + (f_mms, z)
//
// FE space:
//   W_h ⊂ H¹₀(Ω) — CG Q_ℓ (continuous polynomial, homogeneous Dirichlet)
//
// System: [j/τ M + c₁ K + 4ν_r M] w^k = RHS
//   M: mass matrix
//   K: stiffness matrix (Poisson-like)
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015)
// ============================================================================
#ifndef FHD_ANGULAR_MOMENTUM_H
#define FHD_ANGULAR_MOMENTUM_H

#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>

#include "utilities/parameters.h"
#include "utilities/solver_info.h"

#include <mpi.h>
#include <functional>
#include <string>

template <int dim>
class AngularMomentumSubsystem
{
public:
    // ========================================================================
    // Construction
    // ========================================================================
    AngularMomentumSubsystem(const Parameters& params,
                              MPI_Comm mpi_comm,
                              dealii::parallel::distributed::Triangulation<dim>& triangulation);

    // ========================================================================
    // Setup — call once after mesh is ready
    // ========================================================================
    void setup();

    // ========================================================================
    // Assemble — reaction-diffusion with optional convection + torque
    //
    // Optional coupling inputs (pass empty vectors to disable):
    //   u:   velocity for curl coupling 2ν_r(∇×u, z) + optional convection
    //   M,φ: magnetization + potential for magnetic torque μ₀(m×h, z)
    // ========================================================================
    void assemble(
        const dealii::TrilinosWrappers::MPI::Vector& w_old_relevant,
        double dt,
        double current_time,
        const dealii::TrilinosWrappers::MPI::Vector& ux_relevant,
        const dealii::TrilinosWrappers::MPI::Vector& uy_relevant,
        const dealii::DoFHandler<dim>& vel_dof_handler,
        bool include_convection = false,
        const dealii::TrilinosWrappers::MPI::Vector& Mx_relevant
            = dealii::TrilinosWrappers::MPI::Vector(),
        const dealii::TrilinosWrappers::MPI::Vector& My_relevant
            = dealii::TrilinosWrappers::MPI::Vector(),
        const dealii::DoFHandler<dim>* M_dof_handler = nullptr,
        const dealii::TrilinosWrappers::MPI::Vector& phi_relevant
            = dealii::TrilinosWrappers::MPI::Vector(),
        const dealii::DoFHandler<dim>* phi_dof_handler = nullptr);

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
    // MMS source injection
    //
    // Signature: f(point, t_new, t_old, j_micro, c1, nu_r, w_old_disc,
    //              U_old_disc, div_U_old_disc, include_convection) → double
    //
    // CRITICAL: w_old_disc is the discrete old angular velocity at the
    // quadrature point. The source must use j(w*_new - w_old_disc)/τ to
    // avoid 1/τ amplification of pointwise error.
    // U_old_disc and div_U_old_disc: discrete velocity for convection source.
    // ========================================================================
    using MmsSourceFunction = std::function<
        double(const dealii::Point<dim>&,
               double, double,
               double, double, double,
               double,
               const dealii::Tensor<1, dim>&,
               double,
               bool)>;
    void set_mms_source(MmsSourceFunction source);

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
        double w_min = 0.0, w_max = 0.0;
        double w_L2 = 0.0;
        double w_max_abs = 0.0;
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
    dealii::TrilinosWrappers::MPI::Vector w_solution_;

    dealii::TrilinosWrappers::MPI::Vector w_relevant_;

    bool ghosts_valid_ = false;

    // ========================================================================
    // MMS and diagnostics
    // ========================================================================
    MmsSourceFunction mms_source_;
    SolverInfo last_solve_info_;
    double last_assemble_time_ = 0.0;
};

extern template class AngularMomentumSubsystem<2>;
extern template class AngularMomentumSubsystem<3>;

#endif // FHD_ANGULAR_MOMENTUM_H
