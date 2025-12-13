// ============================================================================
// core/phase_field.h - Phase Field Problem (CH-only for now)
//
// Standalone Cahn-Hilliard solver. No NS/Poisson/Magnetization dependencies.
// These will be added incrementally after CH is verified.
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
 * @brief Cahn-Hilliard phase field solver
 *
 * Currently implements only the Cahn-Hilliard subsystem.
 * With u = 0 (no flow), this reduces to pure Allen-Cahn/Cahn-Hilliard diffusion.
 *
 * Data layout:
 *   - Separate DoFHandlers for θ and ψ (AMR-compatible)
 *   - Coupled system with index maps
 *   - Combined constraints for hanging nodes
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
    void initialize_solutions();

    // ========================================================================
    // Time stepping (in phase_field.cc)
    // ========================================================================
    void do_time_step(double dt);

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
    dealii::FE_Q<dim> fe_phase_;  // Q2 for θ, ψ

    // DoF handlers (separate for AMR compatibility)
    dealii::DoFHandler<dim> theta_dof_handler_;
    dealii::DoFHandler<dim> psi_dof_handler_;

    // Individual field constraints (hanging nodes + BCs)
    dealii::AffineConstraints<double> theta_constraints_;
    dealii::AffineConstraints<double> psi_constraints_;

    // Combined constraints for coupled CH system (critical for AMR!)
    dealii::AffineConstraints<double> ch_combined_constraints_;

    // Index maps: field DoF index → coupled system index
    // θ occupies [0, n_theta), ψ occupies [n_theta, n_theta + n_psi)
    std::vector<dealii::types::global_dof_index> theta_to_ch_map_;
    std::vector<dealii::types::global_dof_index> psi_to_ch_map_;

    // Coupled CH system
    dealii::SparsityPattern ch_sparsity_;
    dealii::SparseMatrix<double> ch_matrix_;
    dealii::Vector<double> ch_rhs_;

    // Solution vectors
    dealii::Vector<double> theta_solution_;
    dealii::Vector<double> theta_old_solution_;
    dealii::Vector<double> psi_solution_;

    // Dummy velocity (zero for standalone CH test)
    // Will be replaced with actual velocity when NS is added
    dealii::Vector<double> ux_dummy_;
    dealii::Vector<double> uy_dummy_;

    // Time state
    double time_;
    unsigned int timestep_number_;
};

#endif // PHASE_FIELD_H