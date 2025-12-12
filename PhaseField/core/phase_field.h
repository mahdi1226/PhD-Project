// ============================================================================
// core/phase_field.h - PhaseFieldProblem Class Definition
//
// FIXED VERSION: Added index maps and combined constraints for coupled systems
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

// Forward declarations
template <int dim> class CHAssembler;
template <int dim> class CHSolver;
template <int dim> class NSAssembler;
template <int dim> class NSSolver;
template <int dim> class MagnetizationAssembler;
template <int dim> class MagnetizationSolver;
template <int dim> class PoissonAssembler;
template <int dim> class PoissonSolver;
template <int dim> class VTKWriter;

/**
 * @brief Main problem class for two-phase ferrofluid simulation
 */
template <int dim>
class PhaseFieldProblem
{
public:
    PhaseFieldProblem(const Parameters& params);
    void run();

private:
    // =========================================================================
    // Friend declarations for assemblers/solvers (direct member access)
    // =========================================================================
    friend class CHAssembler<dim>;
    friend class CHSolver<dim>;
    friend class NSAssembler<dim>;
    friend class NSSolver<dim>;
    friend class MagnetizationAssembler<dim>;
    friend class MagnetizationSolver<dim>;
    friend class PoissonAssembler<dim>;
    friend class PoissonSolver<dim>;
    friend class VTKWriter<dim>;

    // =========================================================================
    // Setup methods
    // =========================================================================
    void setup_mesh();
    void setup_dof_handlers();
    void setup_constraints();
    void setup_sparsity_patterns();
    void initialize_solutions();

    // =========================================================================
    // Time stepping
    // =========================================================================
    void do_time_step();
    void update_old_solutions();
    void output_results(unsigned int step);
    void refine_mesh();
    std::string generate_run_folder();

    // =========================================================================
    // Parameters
    // =========================================================================
    const Parameters& params_;

    // =========================================================================
    // Mesh and finite elements
    // =========================================================================
    dealii::Triangulation<dim> triangulation_;
    dealii::FE_Q<dim> fe_Q2_;  // Q2 for velocity, θ, ψ, m, φ
    dealii::FE_Q<dim> fe_Q1_;  // Q1 for pressure

    // =========================================================================
    // DoF handlers (one per scalar field)
    // =========================================================================
    dealii::DoFHandler<dim> theta_dof_handler_;  // Phase field θ
    dealii::DoFHandler<dim> psi_dof_handler_;    // Chemical potential ψ
    dealii::DoFHandler<dim> mx_dof_handler_;     // Magnetization x
    dealii::DoFHandler<dim> my_dof_handler_;     // Magnetization y
    dealii::DoFHandler<dim> phi_dof_handler_;    // Magnetic potential φ
    dealii::DoFHandler<dim> ux_dof_handler_;     // Velocity x
    dealii::DoFHandler<dim> uy_dof_handler_;     // Velocity y
    dealii::DoFHandler<dim> p_dof_handler_;      // Pressure

    // =========================================================================
    // Constraints (per field)
    // =========================================================================
    dealii::AffineConstraints<double> theta_constraints_;
    dealii::AffineConstraints<double> psi_constraints_;
    dealii::AffineConstraints<double> mx_constraints_;
    dealii::AffineConstraints<double> my_constraints_;
    dealii::AffineConstraints<double> phi_constraints_;
    dealii::AffineConstraints<double> ux_constraints_;
    dealii::AffineConstraints<double> uy_constraints_;
    dealii::AffineConstraints<double> p_constraints_;

    // =========================================================================
    // INDEX MAPS: scalar DoF -> coupled system index
    // =========================================================================
    std::vector<dealii::types::global_dof_index> theta_to_ch_map_;
    std::vector<dealii::types::global_dof_index> psi_to_ch_map_;
    std::vector<dealii::types::global_dof_index> ux_to_ns_map_;
    std::vector<dealii::types::global_dof_index> uy_to_ns_map_;
    std::vector<dealii::types::global_dof_index> p_to_ns_map_;

    // =========================================================================
    // COMBINED CONSTRAINTS for coupled systems
    // =========================================================================
    dealii::AffineConstraints<double> ch_combined_constraints_;
    dealii::AffineConstraints<double> ns_combined_constraints_;

    // =========================================================================
    // Solution vectors
    // =========================================================================
    dealii::Vector<double> theta_solution_, theta_old_solution_;
    dealii::Vector<double> psi_solution_;
    dealii::Vector<double> mx_solution_, mx_old_solution_;
    dealii::Vector<double> my_solution_, my_old_solution_;
    dealii::Vector<double> phi_solution_;
    dealii::Vector<double> ux_solution_, ux_old_solution_;
    dealii::Vector<double> uy_solution_, uy_old_solution_;
    dealii::Vector<double> p_solution_;

    // =========================================================================
    // Linear systems - CH: [θ | ψ] coupled
    // =========================================================================
    dealii::SparsityPattern ch_sparsity_;
    dealii::SparseMatrix<double> ch_matrix_;
    dealii::Vector<double> ch_rhs_;

    // =========================================================================
    // Linear systems - NS: [u_x | u_y | p] coupled
    // =========================================================================
    dealii::SparsityPattern ns_sparsity_;
    dealii::SparseMatrix<double> ns_matrix_;
    dealii::Vector<double> ns_rhs_;

    // =========================================================================
    // Linear systems - Magnetization (lumped mass)
    // =========================================================================
    dealii::SparsityPattern mag_sparsity_;
    dealii::SparseMatrix<double> mag_matrix_;
    dealii::Vector<double> mag_rhs_x_, mag_rhs_y_;

    // =========================================================================
    // Linear systems - Poisson
    // =========================================================================
    dealii::SparsityPattern poisson_sparsity_;
    dealii::SparseMatrix<double> poisson_matrix_;
    dealii::Vector<double> poisson_rhs_;

    // =========================================================================
    // Assemblers and Solvers (owned via unique_ptr)
    // =========================================================================
    std::unique_ptr<CHAssembler<dim>> ch_assembler_;
    std::unique_ptr<CHSolver<dim>> ch_solver_;
    std::unique_ptr<NSAssembler<dim>> ns_assembler_;
    std::unique_ptr<NSSolver<dim>> ns_solver_;
    std::unique_ptr<MagnetizationAssembler<dim>> mag_assembler_;
    std::unique_ptr<MagnetizationSolver<dim>> mag_solver_;
    std::unique_ptr<PoissonAssembler<dim>> poisson_assembler_;
    std::unique_ptr<PoissonSolver<dim>> poisson_solver_;
    std::unique_ptr<VTKWriter<dim>> vtk_writer_;

    // =========================================================================
    // Output folder
    // =========================================================================
    std::string output_folder_;
};

#endif // PHASE_FIELD_H