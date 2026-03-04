// ============================================================================
// navier_stokes/navier_stokes.h - Micropolar NS Subsystem (Public Facade)
//
// PAPER EQUATION 42e (Nochetto, Salgado & Tomas, arXiv:1511.04381):
//
//   (u^k/τ, v) + (ν+ν_r)(D(u^k), D(v)) + b_h(u^{k-1}; u^k, v)
//     - (p^k, ∇·v) + (∇·u^k, q)
//     = (u^{k-1}/τ, v) + μ₀·B_h^m(v, h^k, m^k)
//       + 2ν_r(w^{k-1}, ∇×v) + (f_mms, v)
//
// FE spaces:
//   V_h = CG [Q_ℓ]^d (continuous, Taylor-Hood velocity)
//   Π_h = DG P_{ℓ-1} (discontinuous polynomial pressure)
//
// Saddle-point system (monolithic):
//   [A   B^T] [u^k]   [F_u]
//   [B   0  ] [p^k] = [0  ]
//
// A: mass(1/τ) + viscous(ν+ν_r) + convection(u^{k-1})
// B^T: pressure gradient
// B: continuity (divergence)
//
// Reference: Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015)
// ============================================================================
#ifndef FHD_NAVIER_STOKES_H
#define FHD_NAVIER_STOKES_H

#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgp.h>
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
class NavierStokesSubsystem
{
public:
    // ========================================================================
    // Construction
    // ========================================================================
    NavierStokesSubsystem(const Parameters& params,
                          MPI_Comm mpi_comm,
                          dealii::parallel::distributed::Triangulation<dim>& triangulation);

    // ========================================================================
    // Setup — call once after mesh is ready
    // ========================================================================
    void setup();

    // ========================================================================
    // Assemble — monolithic saddle-point system
    //
    // Optional coupling inputs (pass empty vectors to disable):
    //   w:   angular velocity for micropolar: 2ν_r(w, ∇×v)
    //   M,φ: magnetization + potential for Kelvin force: μ₀·B_h^m(v, h, m)
    // ========================================================================
    void assemble(
        const dealii::TrilinosWrappers::MPI::Vector& ux_old_relevant,
        const dealii::TrilinosWrappers::MPI::Vector& uy_old_relevant,
        double dt,
        double current_time,
        bool include_convection,
        const dealii::TrilinosWrappers::MPI::Vector& w_relevant,
        const dealii::DoFHandler<dim>& w_dof_handler,
        const dealii::TrilinosWrappers::MPI::Vector& Mx_relevant
            = dealii::TrilinosWrappers::MPI::Vector(),
        const dealii::TrilinosWrappers::MPI::Vector& My_relevant
            = dealii::TrilinosWrappers::MPI::Vector(),
        const dealii::DoFHandler<dim>* M_dof_handler = nullptr,
        const dealii::TrilinosWrappers::MPI::Vector& phi_relevant
            = dealii::TrilinosWrappers::MPI::Vector(),
        const dealii::DoFHandler<dim>* phi_dof_handler = nullptr,
        // Phase B: Cahn-Hilliard capillary force σ μ ∇φ
        const dealii::TrilinosWrappers::MPI::Vector& ch_solution_relevant
            = dealii::TrilinosWrappers::MPI::Vector(),
        const dealii::DoFHandler<dim>* ch_dof_handler = nullptr);

    // ========================================================================
    // Solve — direct solver for saddle-point system
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
    // Signature: f(point, t_new, t_old, nu_eff, U_old_disc,
    //              div_U_old_disc, include_convection) → (fx, fy)
    //
    // CRITICAL: U_old_disc is the discrete old velocity at the quadrature
    // point. The source must use (u*_new - U_old_disc)/τ to avoid 1/τ
    // amplification of pointwise error between u*_old and U_old_disc.
    //
    // When include_convection=true, the source must include the skew-symmetric
    // convection contribution: (U_old_disc · ∇)u* + ½ div_U_old_disc · u*
    // ========================================================================
    using MmsSourceFunction = std::function<
        dealii::Tensor<1, dim>(const dealii::Point<dim>&,
                               double, double, double,
                               const dealii::Tensor<1, dim>&,
                               double, bool)>;
    void set_mms_source(MmsSourceFunction source);

    // ========================================================================
    // Accessors
    // ========================================================================
    const dealii::DoFHandler<dim>& get_ux_dof_handler() const;
    const dealii::DoFHandler<dim>& get_uy_dof_handler() const;
    const dealii::DoFHandler<dim>& get_p_dof_handler() const;

    const dealii::TrilinosWrappers::MPI::Vector& get_ux_solution() const;
    const dealii::TrilinosWrappers::MPI::Vector& get_uy_solution() const;
    const dealii::TrilinosWrappers::MPI::Vector& get_p_solution() const;

    const dealii::TrilinosWrappers::MPI::Vector& get_ux_relevant() const;
    const dealii::TrilinosWrappers::MPI::Vector& get_uy_relevant() const;
    const dealii::TrilinosWrappers::MPI::Vector& get_p_relevant() const;

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
        double ux_min = 0.0, ux_max = 0.0;
        double uy_min = 0.0, uy_max = 0.0;
        double p_min = 0.0, p_max = 0.0;
        double U_max = 0.0;
        double E_kin = 0.0;      // (1/2)||u||²
        double divU_L2 = 0.0;   // ||∇·u||
        int iterations = 0;
        double residual = 0.0;
        double solve_time = 0.0;
        double assemble_time = 0.0;

        // Kelvin force diagnostics (mesh-dependence tracking)
        double kelvin_cell_L2 = 0.0;   // ||f_cell||_L2 (cell force density)
        double kelvin_face_L2 = 0.0;   // ||f_face||_L2 (face force density)
        double kelvin_Fx = 0.0;        // Total resultant Kelvin Fx (cell+face)
        double kelvin_Fy = 0.0;        // Total resultant Kelvin Fy (cell+face)
    };

    Diagnostics compute_diagnostics() const;

private:
    // ========================================================================
    // Internal setup methods (navier_stokes_setup.cc)
    // ========================================================================
    void distribute_dofs();
    void build_constraints();
    void build_coupled_system();
    void build_block_sparsity_patterns();
    void allocate_vectors();

    // ========================================================================
    // Internal solve methods (navier_stokes_solve.cc)
    // ========================================================================
    void extract_solutions();

    // ========================================================================
    // References (not owned)
    // ========================================================================
    const Parameters& params_;
    MPI_Comm mpi_comm_;
    dealii::parallel::distributed::Triangulation<dim>& triangulation_;

    dealii::ConditionalOStream pcout_;

    // ========================================================================
    // Finite elements
    // ========================================================================
    dealii::FE_Q<dim>   fe_velocity_;   // CG Q_ℓ
    dealii::FE_DGP<dim> fe_pressure_;   // DG P_{ℓ-1}

    // ========================================================================
    // Component DoFHandlers (separate scalar systems)
    // ========================================================================
    dealii::DoFHandler<dim> ux_dof_handler_;
    dealii::DoFHandler<dim> uy_dof_handler_;
    dealii::DoFHandler<dim> p_dof_handler_;

    // Component index sets
    dealii::IndexSet ux_locally_owned_, ux_locally_relevant_;
    dealii::IndexSet uy_locally_owned_, uy_locally_relevant_;
    dealii::IndexSet p_locally_owned_,  p_locally_relevant_;

    // Component constraints
    dealii::AffineConstraints<double> ux_constraints_;
    dealii::AffineConstraints<double> uy_constraints_;
    dealii::AffineConstraints<double> p_constraints_;

    // ========================================================================
    // Coupled saddle-point system
    //
    // Global DoF numbering: [ux | uy | p]
    //   ux DoFs: [0, n_ux)
    //   uy DoFs: [n_ux, n_ux + n_uy)
    //   p DoFs:  [n_ux + n_uy, n_ux + n_uy + n_p)
    // ========================================================================
    dealii::IndexSet ns_locally_owned_;
    dealii::IndexSet ns_locally_relevant_;
    dealii::AffineConstraints<double> ns_constraints_;

    // Index maps: component local DoF → coupled system DoF
    std::vector<dealii::types::global_dof_index> ux_to_ns_map_;
    std::vector<dealii::types::global_dof_index> uy_to_ns_map_;
    std::vector<dealii::types::global_dof_index> p_to_ns_map_;

    dealii::types::global_dof_index n_ux_ = 0, n_uy_ = 0, n_p_ = 0;

    // Monolithic system
    dealii::TrilinosWrappers::SparseMatrix ns_matrix_;
    dealii::TrilinosWrappers::MPI::Vector ns_rhs_;
    dealii::TrilinosWrappers::MPI::Vector ns_solution_;

    // ========================================================================
    // Block matrices for Schur complement preconditioner
    // ========================================================================
    bool use_block_schur_ = false;

    dealii::TrilinosWrappers::SparseMatrix A_ux_ux_;   // Velocity diagonal block (ux)
    dealii::TrilinosWrappers::SparseMatrix A_uy_uy_;   // Velocity diagonal block (uy)
    dealii::TrilinosWrappers::SparseMatrix Bt_ux_;     // B^T: pressure gradient (ux)
    dealii::TrilinosWrappers::SparseMatrix Bt_uy_;     // B^T: pressure gradient (uy)
    dealii::TrilinosWrappers::SparseMatrix B_ux_;      // B: divergence (ux)
    dealii::TrilinosWrappers::SparseMatrix B_uy_;      // B: divergence (uy)
    dealii::TrilinosWrappers::SparseMatrix M_p_;       // Pressure mass matrix

    // ========================================================================
    // Component solution vectors
    // ========================================================================
    dealii::TrilinosWrappers::MPI::Vector ux_solution_;
    dealii::TrilinosWrappers::MPI::Vector uy_solution_;
    dealii::TrilinosWrappers::MPI::Vector p_solution_;

    dealii::TrilinosWrappers::MPI::Vector ux_relevant_;
    dealii::TrilinosWrappers::MPI::Vector uy_relevant_;
    dealii::TrilinosWrappers::MPI::Vector p_relevant_;

    bool ghosts_valid_ = false;

    // ========================================================================
    // MMS and diagnostics
    // ========================================================================
    MmsSourceFunction mms_source_;
    SolverInfo last_solve_info_;
    double last_assemble_time_ = 0.0;

    // Kelvin force diagnostics (populated during assemble)
    double last_kelvin_cell_L2_sq_ = 0.0;
    double last_kelvin_face_L2_sq_ = 0.0;
    double last_kelvin_Fx_ = 0.0;
    double last_kelvin_Fy_ = 0.0;
};

extern template class NavierStokesSubsystem<2>;
extern template class NavierStokesSubsystem<3>;

#endif // FHD_NAVIER_STOKES_H
