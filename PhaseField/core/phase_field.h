// ============================================================================
// core/phase_field.h - Phase Field Problem (CH + Poisson + M + NS)
//
// Full ferrofluid solver implementing Nochetto et al. CMAME 309 (2016)
//
// Subsystems:
//   - Cahn-Hilliard (θ, ψ): phase separation with convection
//   - Poisson (φ): magnetostatic potential, H = ∇φ
//   - Magnetization (Mx, My): DG transport of M (Eq. 42d)
//   - Navier-Stokes (ux, uy, p): fluid flow with Kelvin force B_h^m
//
// Time stepping (Paper Algorithm 1):
//   1. Solve CH → θ^k, ψ^k
//   2. Solve Poisson → φ^k (H^k = ∇φ^k)
//   3. Solve Magnetization → M^k (DG transport)
//   4. Solve NS → u^k, p^k (uses θ^{k-1}, ψ^k, H^k, M^k)
//
// CRITICAL: θ is LAGGED in NS (θ^{k-1}) for energy stability!
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef PHASE_FIELD_H
#define PHASE_FIELD_H

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/affine_constraints.h>

#include "utilities/parameters.h"

#include <vector>

/**
 * @brief Full ferrofluid phase field solver
 *
 * Implements all subsystems from the paper:
 *   - Cahn-Hilliard (θ, ψ): phase separation
 *   - Poisson (φ): magnetostatic potential
 *   - Magnetization (Mx, My): DG transport (NEW!)
 *   - Navier-Stokes (ux, uy, p): fluid flow with full Kelvin B_h^m
 */
template <int dim>
class PhaseFieldProblem
{
public:
    explicit PhaseFieldProblem(const Parameters& params);

    /// Main entry point
    void run();

private:
    // ========================================================================
    // Setup methods (in phase_field_setup.cc)
    // ========================================================================
    void setup_mesh();
    void setup_dof_handlers();
    void setup_constraints();
    void setup_ch_system();
    void setup_poisson_system();
    void setup_magnetization_system();
    void setup_ns_system();
    void initialize_solutions();
    void refine_mesh();                 //  AMR

    // ========================================================================
    // Solve methods (in phase_field.cc)
    // ========================================================================
    void do_time_step(double dt);
    void solve_ch();
    void solve_poisson();
    void solve_magnetization();
    void solve_ns();

    // ========================================================================
    // Constraint updates (for MMS or time-varying BCs)
    // ========================================================================
    void update_mms_boundary_constraints(double time);
    void update_ns_constraints();

    // ========================================================================
    // Output and diagnostics
    // ========================================================================
    void output_results(unsigned int step) const;
    double compute_mass() const;
    double compute_ch_energy() const;
    double compute_kinetic_energy() const;
    double compute_magnetic_energy() const;
    double get_min_h() const;

    // ========================================================================
    // MMS verification
    // ========================================================================
    void compute_mms_errors() const;

    // ========================================================================
    // Data members
    // ========================================================================

    // Parameters
    const Parameters& params_;

    // Mesh
    dealii::Triangulation<dim> triangulation_;

    // Finite elements
    dealii::FE_Q<dim> fe_Q2_;    // Q2 for velocity, θ, ψ, φ
    dealii::FE_Q<dim> fe_Q1_;    // Q1 for pressure
    dealii::FE_DGQ<dim> fe_DG_;  // DG0 for magnetization M

    // ========================================================================
    // Cahn-Hilliard system (θ, ψ)
    // ========================================================================
    dealii::DoFHandler<dim> theta_dof_handler_;
    dealii::DoFHandler<dim> psi_dof_handler_;

    dealii::AffineConstraints<double> theta_constraints_;
    dealii::AffineConstraints<double> psi_constraints_;
    dealii::AffineConstraints<double> ch_combined_constraints_;

    std::vector<dealii::types::global_dof_index> theta_to_ch_map_;
    std::vector<dealii::types::global_dof_index> psi_to_ch_map_;

    dealii::SparsityPattern ch_sparsity_;
    dealii::SparseMatrix<double> ch_matrix_;
    dealii::Vector<double> ch_rhs_;
    dealii::Vector<double> ch_solution_;

    dealii::Vector<double> theta_solution_;
    dealii::Vector<double> theta_old_solution_;  // θ^{k-1} for lagging
    dealii::Vector<double> psi_solution_;

    // ========================================================================
    // Poisson system (φ)
    // ========================================================================
    dealii::DoFHandler<dim> phi_dof_handler_;
    dealii::AffineConstraints<double> phi_constraints_;
    dealii::SparsityPattern phi_sparsity_;
    dealii::SparseMatrix<double> phi_matrix_;
    dealii::Vector<double> phi_rhs_;
    dealii::Vector<double> phi_solution_;

    // ========================================================================
    // Magnetization system (Mx, My) - DG transport (NEW!)
    //
    // Paper Eq. 42d: ∂M/∂t + (u·∇)M = (1/τ_M)(χ(θ)H - M)
    // Discretized with DG upwind for stability.
    // ========================================================================
    dealii::DoFHandler<dim> mx_dof_handler_;
    dealii::DoFHandler<dim> my_dof_handler_;

    dealii::SparsityPattern mx_sparsity_;
    dealii::SparsityPattern my_sparsity_;
    dealii::SparseMatrix<double> mx_matrix_;
    dealii::SparseMatrix<double> my_matrix_;
    dealii::Vector<double> mx_rhs_;
    dealii::Vector<double> my_rhs_;

    dealii::Vector<double> mx_solution_;
    dealii::Vector<double> my_solution_;
    dealii::Vector<double> mx_old_solution_;
    dealii::Vector<double> my_old_solution_;

    // ========================================================================
    // Navier-Stokes system (ux, uy, p)
    // ========================================================================
    dealii::DoFHandler<dim> ux_dof_handler_;
    dealii::DoFHandler<dim> uy_dof_handler_;
    dealii::DoFHandler<dim> p_dof_handler_;

    dealii::AffineConstraints<double> ux_constraints_;
    dealii::AffineConstraints<double> uy_constraints_;
    dealii::AffineConstraints<double> p_constraints_;
    dealii::AffineConstraints<double> ns_combined_constraints_;

    std::vector<dealii::types::global_dof_index> ux_to_ns_map_;
    std::vector<dealii::types::global_dof_index> uy_to_ns_map_;
    std::vector<dealii::types::global_dof_index> p_to_ns_map_;

    // ========================================================================
    // Pressure mass matrix (for Schur complement preconditioner)
    // ========================================================================
    dealii::SparsityPattern pressure_mass_sparsity_;
    dealii::SparseMatrix<double> pressure_mass_matrix_;

    dealii::SparsityPattern ns_sparsity_;
    dealii::SparseMatrix<double> ns_matrix_;
    dealii::Vector<double> ns_rhs_;
    dealii::Vector<double> ns_solution_;

    dealii::Vector<double> ux_solution_;
    dealii::Vector<double> ux_old_solution_;
    dealii::Vector<double> uy_solution_;
    dealii::Vector<double> uy_old_solution_;
    dealii::Vector<double> p_solution_;

    // ========================================================================
    // Time state
    // ========================================================================
    double time_;
    unsigned int timestep_number_;
};

#endif // PHASE_FIELD_H