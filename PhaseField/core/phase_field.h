// ============================================================================
// core/phase_field.h - Phase Field Problem (CH + Poisson + NS)
//
// Full ferrofluid solver: Cahn-Hilliard + Poisson + Navier-Stokes
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef PHASE_FIELD_H
#define PHASE_FIELD_H

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/affine_constraints.h>

#include "utilities/parameters.h"

#include <vector>

/**
 * @brief Full ferrofluid phase field solver
 *
 * Implements all subsystems:
 *   - Cahn-Hilliard (θ, ψ): phase separation
 *   - Poisson (φ): magnetostatic potential
 *   - Navier-Stokes (ux, uy, p): fluid flow with Kelvin force
 *
 * Staggered time stepping:
 *   1. Solve CH → new θ, ψ
 *   2. Solve Poisson → new φ (uses θ for permeability)
 *   3. Solve NS → new u, p (uses θ, ψ, φ for forces)
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
    void setup_ns_system();
    void initialize_solutions();

    // ========================================================================
    // Time stepping (in phase_field.cc)
    // ========================================================================
    void do_time_step(double dt);
    void solve_poisson();
    void solve_ns();
    void update_poisson_constraints(double time);
    void update_ns_constraints();

    // ========================================================================
    // Output
    // ========================================================================
    void output_results(unsigned int step) const;

    // ========================================================================
    // Utilities
    // ========================================================================
    double compute_mass() const;
    double get_min_h() const;

    // ========================================================================
    // MMS verification (in phase_field.cc)
    // ========================================================================
    void compute_mms_errors() const;
    void update_mms_boundary_constraints(double time);

    // ========================================================================
    // Data members
    // ========================================================================

    // Parameters
    const Parameters& params_;

    // Mesh
    dealii::Triangulation<dim> triangulation_;

    // Finite elements
    dealii::FE_Q<dim> fe_Q2_;  // Q2 for velocity, θ, ψ, φ
    dealii::FE_Q<dim> fe_Q1_;  // Q1 for pressure

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

    dealii::Vector<double> theta_solution_;
    dealii::Vector<double> theta_old_solution_;
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

    dealii::SparsityPattern ns_sparsity_;
    dealii::SparseMatrix<double> ns_matrix_;
    dealii::Vector<double> ns_rhs_;
    dealii::Vector<double> ns_solution_;

    dealii::Vector<double> ux_solution_;
    dealii::Vector<double> ux_old_solution_;
    dealii::Vector<double> uy_solution_;
    dealii::Vector<double> uy_old_solution_;
    dealii::Vector<double> p_solution_;

    // Time state
    double time_;
    unsigned int timestep_number_;
};

#endif // PHASE_FIELD_H