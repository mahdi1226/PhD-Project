// ============================================================================
// core/nsch_problem.h - Coupled NS/CH/Poisson Problem Class for Ferrofluids
//
// REFACTORED VERSION: Separate DoFHandlers for each scalar field
// This enables proper SolutionTransfer during AMR (fixes deal.II 9.7 issue)
//
// Based on: Nochetto, Salgado & Tomas (2016)
// "A diffuse interface model for two-phase ferrofluid flows"
// ============================================================================
#ifndef NSCH_PROBLEM_H
#define NSCH_PROBLEM_H

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/affine_constraints.h>

#include "utilities/nsch_parameters.h"
#include "utilities/nsch_linear_algebra.h"

/**
 * @brief Main problem class for coupled NS/CH/Poisson ferrofluid system
 *
 * ARCHITECTURE: Separate DoFHandlers for each scalar field
 * =========================================================
 * This design enables proper AMR with SolutionTransfer by avoiding
 * the BlockVector hanging node issue in deal.II 9.7.
 *
 * All DoFHandlers share the SAME triangulation but have independent:
 *   - Finite element (Q2 or Q1)
 *   - DoF numbering
 *   - Constraints
 *   - Solution vectors
 *
 * Coupled systems (NS, CH) are assembled using values from separate
 * DoFHandlers but solved as monolithic systems for stability.
 *
 * FIELDS:
 *   - c  : Concentration (phase field), Q2
 *   - mu : Chemical potential, Q2
 *   - ux : Velocity x-component, Q2
 *   - uy : Velocity y-component, Q2
 *   - p  : Pressure, Q1
 *   - phi: Magnetic potential, Q2
 */
template <int dim>
class NSCHProblem
{
public:
    explicit NSCHProblem(const NSCHParameters& params);

    /// Main entry point
    void run();

    // ========================================================================
    // Accessors - Mesh (shared by all fields)
    // ========================================================================
    dealii::Triangulation<dim>&       get_triangulation()       { return triangulation_; }
    const dealii::Triangulation<dim>& get_triangulation() const { return triangulation_; }

    // ========================================================================
    // Accessors - Concentration (c)
    // ========================================================================
    dealii::DoFHandler<dim>&               get_c_dof_handler()       { return c_dof_handler_; }
    const dealii::DoFHandler<dim>&         get_c_dof_handler() const { return c_dof_handler_; }
    dealii::Vector<double>&                get_c_solution()          { return c_solution_; }
    const dealii::Vector<double>&          get_c_solution()    const { return c_solution_; }
    dealii::Vector<double>&                get_c_old_solution()      { return c_old_solution_; }
    const dealii::Vector<double>&          get_c_old_solution() const { return c_old_solution_; }
    dealii::AffineConstraints<double>&     get_c_constraints()       { return c_constraints_; }

    // ========================================================================
    // Accessors - Chemical potential (mu)
    // ========================================================================
    dealii::DoFHandler<dim>&               get_mu_dof_handler()       { return mu_dof_handler_; }
    const dealii::DoFHandler<dim>&         get_mu_dof_handler() const { return mu_dof_handler_; }
    dealii::Vector<double>&                get_mu_solution()          { return mu_solution_; }
    const dealii::Vector<double>&          get_mu_solution()    const { return mu_solution_; }
    dealii::AffineConstraints<double>&     get_mu_constraints()       { return mu_constraints_; }

    // ========================================================================
    // Accessors - Velocity x (ux)
    // ========================================================================
    dealii::DoFHandler<dim>&               get_ux_dof_handler()       { return ux_dof_handler_; }
    const dealii::DoFHandler<dim>&         get_ux_dof_handler() const { return ux_dof_handler_; }
    dealii::Vector<double>&                get_ux_solution()          { return ux_solution_; }
    const dealii::Vector<double>&          get_ux_solution()    const { return ux_solution_; }
    dealii::Vector<double>&                get_ux_old_solution()      { return ux_old_solution_; }
    dealii::AffineConstraints<double>&     get_ux_constraints()       { return ux_constraints_; }

    // ========================================================================
    // Accessors - Velocity y (uy)
    // ========================================================================
    dealii::DoFHandler<dim>&               get_uy_dof_handler()       { return uy_dof_handler_; }
    const dealii::DoFHandler<dim>&         get_uy_dof_handler() const { return uy_dof_handler_; }
    dealii::Vector<double>&                get_uy_solution()          { return uy_solution_; }
    const dealii::Vector<double>&          get_uy_solution()    const { return uy_solution_; }
    dealii::Vector<double>&                get_uy_old_solution()      { return uy_old_solution_; }
    dealii::AffineConstraints<double>&     get_uy_constraints()       { return uy_constraints_; }

    // ========================================================================
    // Accessors - Pressure (p)
    // ========================================================================
    dealii::DoFHandler<dim>&               get_p_dof_handler()       { return p_dof_handler_; }
    const dealii::DoFHandler<dim>&         get_p_dof_handler() const { return p_dof_handler_; }
    dealii::Vector<double>&                get_p_solution()          { return p_solution_; }
    const dealii::Vector<double>&          get_p_solution()    const { return p_solution_; }
    dealii::AffineConstraints<double>&     get_p_constraints()       { return p_constraints_; }

    // ========================================================================
    // Accessors - Magnetic potential (phi)
    // ========================================================================
    dealii::DoFHandler<dim>&               get_phi_dof_handler()       { return phi_dof_handler_; }
    const dealii::DoFHandler<dim>&         get_phi_dof_handler() const { return phi_dof_handler_; }
    dealii::Vector<double>&                get_phi_solution()          { return phi_solution_; }
    const dealii::Vector<double>&          get_phi_solution()    const { return phi_solution_; }
    dealii::AffineConstraints<double>&     get_phi_constraints()       { return phi_constraints_; }

    // ========================================================================
    // Accessors - Parameters and state
    // ========================================================================
    const NSCHParameters& get_params() const { return params_; }
    double get_time()      const { return time_; }
    double get_time_step() const { return params_.dt; }
    double get_h()         const;

    unsigned int get_c_n_dofs()   const { return c_dof_handler_.n_dofs(); }
    unsigned int get_mu_n_dofs()  const { return mu_dof_handler_.n_dofs(); }
    unsigned int get_ux_n_dofs()  const { return ux_dof_handler_.n_dofs(); }
    unsigned int get_uy_n_dofs()  const { return uy_dof_handler_.n_dofs(); }
    unsigned int get_p_n_dofs()   const { return p_dof_handler_.n_dofs(); }
    unsigned int get_phi_n_dofs() const { return phi_dof_handler_.n_dofs(); }

    // Total DoFs for reporting
    unsigned int get_total_dofs() const {
        return c_dof_handler_.n_dofs() + mu_dof_handler_.n_dofs() +
               ux_dof_handler_.n_dofs() + uy_dof_handler_.n_dofs() +
               p_dof_handler_.n_dofs() +
               (params_.enable_magnetic ? phi_dof_handler_.n_dofs() : 0);
    }

private:
    // ========================================================================
    // Setup methods (nsch_problem_setup.cc)
    // ========================================================================
    void setup_mesh();

    // Individual scalar field setup
    void setup_c_system();
    void setup_mu_system();
    void setup_ux_system();
    void setup_uy_system();
    void setup_p_system();
    void setup_phi_system();

    // Coupled system sparsity patterns (for monolithic solves)
    void setup_ch_coupled_system();  // Sparsity for [c, mu] coupling
    void setup_ns_coupled_system();  // Sparsity for [ux, uy, p] coupling

    void setup_all_systems();

    // Initialization
    void initialize_c();
    void initialize_mu();
    void initialize_velocity();
    void initialize_phi();
    void initialize_all();

    // ========================================================================
    // Solver methods (nsch_problem_solver.cc)
    // ========================================================================
    void solve_cahn_hilliard(double dt);
    void solve_poisson();
    void solve_navier_stokes(double dt);
    void update_phi_constraints_dipole(double current_time);
    void update_mms_boundary_conditions(double new_time);

    // Assembly methods (nsch_problem_solver.cc)
    void assemble_ch_system(double dt, double current_time);
    void assemble_poisson_system();
    void assemble_ns_system(double dt, double current_time);

    // Adapter functions for BlockVector compatibility (legacy)
    CHVector pack_ch_solution(const dealii::Vector<double>& c,
                              const dealii::Vector<double>& mu) const;
    void unpack_ch_solution(const CHVector& ch_vec,
                            dealii::Vector<double>& c,
                            dealii::Vector<double>& mu) const;
    NSVector pack_ns_solution(const dealii::Vector<double>& ux,
                              const dealii::Vector<double>& uy,
                              const dealii::Vector<double>& p) const;
    void unpack_ns_solution(const NSVector& ns_vec,
                            dealii::Vector<double>& ux,
                            dealii::Vector<double>& uy,
                            dealii::Vector<double>& p) const;

    // ========================================================================
    // Time stepping (nsch_problem.cc)
    // ========================================================================
    void do_time_step(double dt);

    // ========================================================================
    // AMR methods (nsch_problem_amr.cc)
    // ========================================================================
    void refine_mesh();
    void compute_refinement_indicators(dealii::Vector<float>& indicators) const;

    // ========================================================================
    // Output (nsch_problem_output.cc)
    // ========================================================================
    void output_results(unsigned int step) const;

    // ========================================================================
    // Data members - Parameters
    // ========================================================================
    NSCHParameters params_;

    // ========================================================================
    // Data members - Mesh (shared by all fields)
    // ========================================================================
    dealii::Triangulation<dim> triangulation_;

    // ========================================================================
    // Data members - Finite elements (shared references)
    // Q2 for velocity, concentration, chemical potential, magnetic potential
    // Q1 for pressure
    // ========================================================================
    dealii::FE_Q<dim> fe_Q2_;
    dealii::FE_Q<dim> fe_Q1_;

    // ========================================================================
    // Data members - Concentration (c)
    // ========================================================================
    dealii::DoFHandler<dim> c_dof_handler_;
    dealii::Vector<double>  c_solution_;
    dealii::Vector<double>  c_old_solution_;
    dealii::AffineConstraints<double> c_constraints_;

    // ========================================================================
    // Data members - Chemical potential (mu)
    // ========================================================================
    dealii::DoFHandler<dim> mu_dof_handler_;
    dealii::Vector<double>  mu_solution_;
    dealii::AffineConstraints<double> mu_constraints_;

    // ========================================================================
    // Data members - Velocity x (ux)
    // ========================================================================
    dealii::DoFHandler<dim> ux_dof_handler_;
    dealii::Vector<double>  ux_solution_;
    dealii::Vector<double>  ux_old_solution_;
    dealii::AffineConstraints<double> ux_constraints_;

    // ========================================================================
    // Data members - Velocity y (uy)
    // ========================================================================
    dealii::DoFHandler<dim> uy_dof_handler_;
    dealii::Vector<double>  uy_solution_;
    dealii::Vector<double>  uy_old_solution_;
    dealii::AffineConstraints<double> uy_constraints_;

    // ========================================================================
    // Data members - Pressure (p)
    // ========================================================================
    dealii::DoFHandler<dim> p_dof_handler_;
    dealii::Vector<double>  p_solution_;
    dealii::AffineConstraints<double> p_constraints_;

    // ========================================================================
    // Data members - Magnetic potential (phi)
    // ========================================================================
    dealii::DoFHandler<dim> phi_dof_handler_;
    dealii::Vector<double>  phi_solution_;
    dealii::AffineConstraints<double> phi_constraints_;

    // ========================================================================
    // Data members - Coupled CH system (for monolithic solve)
    // Couples c and mu equations
    // ========================================================================
    dealii::SparsityPattern      ch_sparsity_;
    dealii::SparseMatrix<double> ch_matrix_;
    dealii::Vector<double>       ch_rhs_;

    // Index mapping: local scalar DoF -> coupled system index
    std::vector<dealii::types::global_dof_index> c_to_ch_map_;
    std::vector<dealii::types::global_dof_index> mu_to_ch_map_;

    // Combined constraints for coupled CH system (for AMR with hanging nodes)
    dealii::AffineConstraints<double> ch_combined_constraints_;

    // ========================================================================
    // Data members - Coupled NS system (for monolithic solve)
    // Couples ux, uy, and p equations
    // ========================================================================
    dealii::SparsityPattern      ns_sparsity_;
    dealii::SparseMatrix<double> ns_matrix_;
    dealii::Vector<double>       ns_rhs_;

    // Index mapping for extracting ux, uy, p from coupled solution
    std::vector<dealii::types::global_dof_index> ux_to_ns_map_;
    std::vector<dealii::types::global_dof_index> uy_to_ns_map_;
    std::vector<dealii::types::global_dof_index> p_to_ns_map_;

    // Combined constraints for coupled NS system (for AMR with hanging nodes)
    dealii::AffineConstraints<double> ns_combined_constraints_;

    // ========================================================================
    // Data members - Poisson system (already scalar)
    // ========================================================================
    dealii::SparsityPattern      phi_sparsity_;
    dealii::SparseMatrix<double> phi_matrix_;
    dealii::Vector<double>       phi_rhs_;
    
    // ========================================================================
    // Data members - Time stepping state
    // ========================================================================
    double       time_;
    unsigned int timestep_number_;
};

#endif // NSCH_PROBLEM_H