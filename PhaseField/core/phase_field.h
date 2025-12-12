// ============================================================================
// core/phase_field.h - Main Orchestrator for Two-Phase Ferrofluid System
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef PHASE_FIELD_H
#define PHASE_FIELD_H

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/affine_constraints.h>

#include "utilities/parameters.h"

// Forward declarations
template <int dim> class CHAssembler;
template <int dim> class NSAssembler;
template <int dim> class MagnetizationAssembler;
template <int dim> class PoissonAssembler;
template <int dim> class CHSolver;
template <int dim> class NSSolver;
template <int dim> class MagnetizationSolver;
template <int dim> class PoissonSolver;
template <int dim> class VTKWriter;

/**
 * @brief Main problem class for coupled NS/CH/Magnetization/Poisson system
 */
template <int dim>
class PhaseFieldProblem
{
public:
    explicit PhaseFieldProblem(const Parameters& params);
    ~PhaseFieldProblem();

    void run();

    // Minimal public accessors (for output/diagnostics only)
    const Parameters& get_params() const { return params_; }
    double get_time() const { return time_; }
    unsigned int get_timestep_number() const { return timestep_number_; }

private:
    // Friend classes have direct access - no accessor bloat
    friend class CHAssembler<dim>;
    friend class NSAssembler<dim>;
    friend class MagnetizationAssembler<dim>;
    friend class PoissonAssembler<dim>;
    friend class CHSolver<dim>;
    friend class NSSolver<dim>;
    friend class MagnetizationSolver<dim>;
    friend class PoissonSolver<dim>;
    friend class VTKWriter<dim>;

    // Setup methods (phase_field_setup.cc)
    void setup_mesh();
    void setup_dof_handlers();
    void setup_constraints();
    void setup_sparsity_patterns();
    void initialize_solutions();

    // Time stepping (phase_field.cc)
    void do_time_step();

    // AMR (phase_field_amr.cc)
    void refine_mesh();

    // Output
    void output_results() const;

    // ========================================================================
    // Data members
    // ========================================================================
    Parameters params_;

    // Mesh
    dealii::Triangulation<dim> triangulation_;

    // Finite elements
    dealii::FE_Q<dim> fe_Q2_;  // θ, ψ, m, φ, u
    dealii::FE_Q<dim> fe_Q1_;  // p

    // θ (phase field)
    dealii::DoFHandler<dim> theta_dof_handler_;
    dealii::Vector<double>  theta_solution_, theta_old_solution_;
    dealii::AffineConstraints<double> theta_constraints_;

    // ψ (chemical potential)
    dealii::DoFHandler<dim> psi_dof_handler_;
    dealii::Vector<double>  psi_solution_;
    dealii::AffineConstraints<double> psi_constraints_;

    // m (magnetization)
    dealii::DoFHandler<dim> mx_dof_handler_, my_dof_handler_;
    dealii::Vector<double>  mx_solution_, my_solution_;
    dealii::Vector<double>  mx_old_solution_, my_old_solution_;
    dealii::AffineConstraints<double> mx_constraints_, my_constraints_;

    // φ (magnetic potential)
    dealii::DoFHandler<dim> phi_dof_handler_;
    dealii::Vector<double>  phi_solution_;
    dealii::AffineConstraints<double> phi_constraints_;

    // u (velocity)
    dealii::DoFHandler<dim> ux_dof_handler_, uy_dof_handler_;
    dealii::Vector<double>  ux_solution_, uy_solution_;
    dealii::Vector<double>  ux_old_solution_, uy_old_solution_;
    dealii::AffineConstraints<double> ux_constraints_, uy_constraints_;

    // p (pressure)
    dealii::DoFHandler<dim> p_dof_handler_;
    dealii::Vector<double>  p_solution_;
    dealii::AffineConstraints<double> p_constraints_;

    // Linear systems
    dealii::SparsityPattern ch_sparsity_, ns_sparsity_, mag_sparsity_, poisson_sparsity_;
    dealii::SparseMatrix<double> ch_matrix_, ns_matrix_, mag_matrix_, poisson_matrix_;
    dealii::Vector<double> ch_rhs_, ns_rhs_, mag_rhs_, poisson_rhs_;

    // Time state
    double time_;
    unsigned int timestep_number_;

    // Assemblers and solvers
    std::unique_ptr<CHAssembler<dim>> ch_assembler_;
    std::unique_ptr<NSAssembler<dim>> ns_assembler_;
    std::unique_ptr<MagnetizationAssembler<dim>> mag_assembler_;
    std::unique_ptr<PoissonAssembler<dim>> poisson_assembler_;

    std::unique_ptr<CHSolver<dim>> ch_solver_;
    std::unique_ptr<NSSolver<dim>> ns_solver_;
    std::unique_ptr<MagnetizationSolver<dim>> mag_solver_;
    std::unique_ptr<PoissonSolver<dim>> poisson_solver_;

    std::unique_ptr<VTKWriter<dim>> vtk_writer_;
};

#endif // PHASE_FIELD_H