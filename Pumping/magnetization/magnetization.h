// ============================================================================
// magnetization/magnetization.h - Magnetization Subsystem (Public Facade)
//
// PAPER EQUATION 52d (Nochetto, Salgado & Tomas, arXiv:1511.04381):
//
//   (m^k/τ, z) + σ·a_h^m(m^k, z) + B_h^m(u^{k-1}; m^k, z)
//     + (m^k × w^k, z) + (1/𝒯)(m^k, z)
//     = (1/𝒯)(κ₀·h^k, z) + (m^{k-1}/τ, z) + f_mms
//
// The (m × w) spin-magnetization coupling is treated EXPLICITLY:
//   (m_old × w, z) moved to RHS → -(m_old × w, z) on RHS
//   In 2D: m × w = (w·My, -w·Mx), so RHS gets (-w·My_old, +w·Mx_old)
//
// where h^k = ∇φ^k is the total magnetic field (h_a encoded via Poisson).
//
// FE space: M_h = DG [Q_ℓ]^d (discontinuous Galerkin, componentwise)
//   Mx and My solved as separate scalar DG systems sharing one DoFHandler.
//
// DG forms:
//   a_h^m (Eq. 63-65): Interior penalty for magnetic diffusion (σ > 0)
//   B_h^m (Eq. 62):    Skew-symmetric transport + upwind penalty
//
// Energy identity: B_h^m(U, M, M) = 0 globally (Prop. 3.5)
//
// Parallel:
//   Trilinos vectors/matrices, p4est triangulation, MPI-aware assembly
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015)
// ============================================================================
#ifndef FHD_MAGNETIZATION_H
#define FHD_MAGNETIZATION_H

#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_dgq.h>
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
class MagnetizationSubsystem
{
public:
    // ========================================================================
    // Construction — does not own triangulation or parameters
    // ========================================================================
    MagnetizationSubsystem(const Parameters& params,
                           MPI_Comm mpi_comm,
                           dealii::parallel::distributed::Triangulation<dim>& triangulation);

    // ========================================================================
    // Setup — call once after mesh is ready
    //
    //   1. Distribute DoFs (FE_DGQ<dim>(degree_magnetization))
    //   2. Build sparsity pattern (DG: block-diagonal + face couplings)
    //   3. Allocate vectors (Mx, My, Mx_old, My_old, RHS)
    // ========================================================================
    void setup();

    // ========================================================================
    // Assemble — matrix + RHS (call each Picard iteration / timestep)
    //
    // Eq. 52d: LHS has mass + diffusion + transport + relaxation
    //          RHS has old-time mass + relaxation source + spin coupling + MMS
    //
    // Inputs from other subsystems:
    //   phi_relevant, phi_dof_handler: potential from Poisson (CG)
    //   ux_relevant, uy_relevant, u_dof_handler: velocity from NS (CG)
    //   dt: time step size
    //   current_time: for h_a ramp and MMS time dependence
    //   w_relevant, w_dof_handler: angular velocity from AngMom (CG)
    //     (optional: if empty/nullptr, spin-magnetization coupling is skipped)
    //
    // If velocity is empty (size 0), assembles without transport (standalone).
    // ========================================================================
    void assemble(
        const dealii::TrilinosWrappers::MPI::Vector& Mx_old_relevant,
        const dealii::TrilinosWrappers::MPI::Vector& My_old_relevant,
        const dealii::TrilinosWrappers::MPI::Vector& phi_relevant,
        const dealii::DoFHandler<dim>& phi_dof_handler,
        const dealii::TrilinosWrappers::MPI::Vector& ux_relevant,
        const dealii::TrilinosWrappers::MPI::Vector& uy_relevant,
        const dealii::DoFHandler<dim>& u_dof_handler,
        double dt,
        double current_time,
        const dealii::TrilinosWrappers::MPI::Vector& w_relevant
            = dealii::TrilinosWrappers::MPI::Vector(),
        const dealii::DoFHandler<dim>* w_dof_handler = nullptr,
        // Phase B: Cahn-Hilliard for phase-dependent χ(φ)
        const dealii::TrilinosWrappers::MPI::Vector& ch_solution_relevant
            = dealii::TrilinosWrappers::MPI::Vector(),
        const dealii::DoFHandler<dim>* ch_dof_handler = nullptr);

    // ========================================================================
    // Solve — call after assemble
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
    // Source callback for MMS testing. Signature:
    //   f(point, t_new, t_old, tau_M, kappa_0, H_disc, U, div_U, M_old_disc,
    //     w_disc)
    //   → returns Tensor<1,dim> = (f_Mx, f_My)
    //
    // CRITICAL: Uses discrete H, M_old, and w (from assembly), not analytical.
    // w_disc: angular velocity at quadrature point (for spin-mag coupling).
    // ========================================================================
    using MmsSourceFunction = std::function<
        dealii::Tensor<1, dim>(const dealii::Point<dim>&,
                               double, double, double, double,
                               const dealii::Tensor<1, dim>&,
                               const dealii::Tensor<1, dim>&,
                               double,
                               const dealii::Tensor<1, dim>&,
                               double)>;
    void set_mms_source(MmsSourceFunction source);

    // ========================================================================
    // Accessors — for other subsystems to read M
    //
    // Consumed by:
    //   Poisson: (h_a − M, ∇X) in RHS
    //   NS:      Kelvin force μ₀·B_h^m(v, h, m)
    //   Angular: magnetic torque μ₀(m × h, ξ)
    // ========================================================================
    const dealii::DoFHandler<dim>& get_dof_handler() const;
    const dealii::TrilinosWrappers::MPI::Vector& get_Mx_solution() const;
    const dealii::TrilinosWrappers::MPI::Vector& get_My_solution() const;
    const dealii::TrilinosWrappers::MPI::Vector& get_Mx_relevant() const;
    const dealii::TrilinosWrappers::MPI::Vector& get_My_relevant() const;

    // Mutable accessors (for AMR solution transfer)
    dealii::TrilinosWrappers::MPI::Vector& get_Mx_solution_mutable();
    dealii::TrilinosWrappers::MPI::Vector& get_My_solution_mutable();

    // ========================================================================
    // Ghost management
    // ========================================================================
    void update_ghosts();
    void invalidate_ghosts();
    void save_old_solution();

    // ========================================================================
    // Diagnostics
    // ========================================================================
    struct Diagnostics
    {
        double Mx_min = 0.0, Mx_max = 0.0;
        double My_min = 0.0, My_max = 0.0;
        double M_L2 = 0.0;
        double M_max = 0.0;
        int iterations = 0;
        double residual = 0.0;
        double solve_time = 0.0;
        double assemble_time = 0.0;
    };

    Diagnostics compute_diagnostics() const;

private:
    // ========================================================================
    // Internal methods
    // ========================================================================

    // magnetization_setup.cc
    void distribute_dofs();
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
    // FE and DoFHandler: DG Q_ℓ (one DoFHandler, two solution vectors)
    // ========================================================================
    dealii::FE_DGQ<dim> fe_;
    dealii::DoFHandler<dim> dof_handler_;

    dealii::IndexSet locally_owned_dofs_;
    dealii::IndexSet locally_relevant_dofs_;

    // DG: no hanging node constraints needed
    dealii::AffineConstraints<double> constraints_;

    // ========================================================================
    // Linear system (shared between Mx and My)
    // ========================================================================
    dealii::TrilinosWrappers::SparseMatrix system_matrix_;
    dealii::TrilinosWrappers::MPI::Vector Mx_rhs_;
    dealii::TrilinosWrappers::MPI::Vector My_rhs_;

    // ========================================================================
    // Solution vectors (separate Mx, My)
    // ========================================================================
    dealii::TrilinosWrappers::MPI::Vector Mx_solution_;
    dealii::TrilinosWrappers::MPI::Vector My_solution_;
    dealii::TrilinosWrappers::MPI::Vector Mx_relevant_;
    dealii::TrilinosWrappers::MPI::Vector My_relevant_;

    bool ghosts_valid_ = false;

    // ========================================================================
    // MMS source callback
    // ========================================================================
    MmsSourceFunction mms_source_;

    // Cached diagnostics
    SolverInfo last_solve_info_;
    double last_assemble_time_ = 0.0;
};

extern template class MagnetizationSubsystem<2>;
extern template class MagnetizationSubsystem<3>;

#endif // FHD_MAGNETIZATION_H
