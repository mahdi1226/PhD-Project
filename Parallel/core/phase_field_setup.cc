// ============================================================================
// core/phase_field_setup.cc - Setup Methods (PARALLEL)
//
// OPTIMIZED VERSION:
//   - Cached magnetization assembler/solver (avoid recreation each timestep)
//   - Follows Poisson pattern for caching
//
// Setup follows MMSContext pattern with inline CH setup.
// Other subsystems use free functions from setup/*.h
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "core/phase_field.h"
#include "setup/poisson_setup.h"
#include "setup/magnetization_setup.h"
#include "setup/ns_setup.h"
#include "solvers/ns_solver.h"
#include "assembly/poisson_assembler.h"
#include "assembly/magnetization_assembler.h"
#include "solvers/magnetization_solver.h"
#include "solvers/ns_solver.h"

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/affine_constraints.h>

#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Constructor
// ============================================================================
template <int dim>
PhaseFieldProblem<dim>::PhaseFieldProblem(const Parameters& params)
    : mpi_communicator_(MPI_COMM_WORLD)
    , pcout_(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_communicator_) == 0)
    , params_(params)
    , triangulation_(mpi_communicator_,
                     typename dealii::Triangulation<dim>::MeshSmoothing(
                         dealii::Triangulation<dim>::smoothing_on_refinement |
                         dealii::Triangulation<dim>::smoothing_on_coarsening))
    , fe_Q2_(params.fe.degree_velocity)
    , fe_Q1_(params.fe.degree_pressure)
    , fe_DG_(params.fe.degree_magnetization)
    , theta_dof_handler_(triangulation_)
    , psi_dof_handler_(triangulation_)
    , phi_dof_handler_(triangulation_)
    , M_dof_handler_(triangulation_)
    , ux_dof_handler_(triangulation_)
    , uy_dof_handler_(triangulation_)
    , p_dof_handler_(triangulation_)
    , time_(params.mms_t_init)
    , timestep_number_(0)
    , last_picard_iterations_(0)
    , last_picard_residual_(0.0)
{
}

// ============================================================================
// setup_mesh()
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::setup_mesh()
{
    dealii::Point<dim> p1(params_.domain.x_min, params_.domain.y_min);
    dealii::Point<dim> p2(params_.domain.x_max, params_.domain.y_max);

    std::vector<unsigned int> subdivisions(dim);
    subdivisions[0] = params_.domain.initial_cells_x;
    subdivisions[1] = params_.domain.initial_cells_y;

    dealii::GridGenerator::subdivided_hyper_rectangle(triangulation_, subdivisions, p1, p2);

    // Assign boundary IDs: 0=bottom, 1=right, 2=top, 3=left
    for (auto& cell : triangulation_.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        for (unsigned int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell; ++f)
        {
            if (!cell->face(f)->at_boundary())
                continue;

            const auto center = cell->face(f)->center();
            const double tol = 1e-10;

            if (std::abs(center[1] - params_.domain.y_min) < tol)
                cell->face(f)->set_boundary_id(0);
            else if (std::abs(center[0] - params_.domain.x_max) < tol)
                cell->face(f)->set_boundary_id(1);
            else if (std::abs(center[1] - params_.domain.y_max) < tol)
                cell->face(f)->set_boundary_id(2);
            else if (std::abs(center[0] - params_.domain.x_min) < tol)
                cell->face(f)->set_boundary_id(3);
        }
    }

    triangulation_.refine_global(params_.mesh.initial_refinement);

    pcout_ << "[Setup] Mesh: " << triangulation_.n_global_active_cells() << " cells, "
           << dealii::Utilities::MPI::n_mpi_processes(mpi_communicator_) << " MPI ranks\n";
}

// ============================================================================
// setup_dof_handlers()
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::setup_dof_handlers()
{
    // CH fields (θ, ψ)
    theta_dof_handler_.distribute_dofs(fe_Q2_);
    psi_dof_handler_.distribute_dofs(fe_Q2_);

    theta_locally_owned_ = theta_dof_handler_.locally_owned_dofs();
    theta_locally_relevant_ = dealii::DoFTools::extract_locally_relevant_dofs(theta_dof_handler_);
    psi_locally_owned_ = psi_dof_handler_.locally_owned_dofs();
    psi_locally_relevant_ = dealii::DoFTools::extract_locally_relevant_dofs(psi_dof_handler_);

    // Poisson (φ)
    if (params_.enable_magnetic)
    {
        phi_dof_handler_.distribute_dofs(fe_Q2_);
        phi_locally_owned_ = phi_dof_handler_.locally_owned_dofs();
        phi_locally_relevant_ = dealii::DoFTools::extract_locally_relevant_dofs(phi_dof_handler_);
    }

    // Magnetization (M) - single DoFHandler for DG
    if (params_.enable_magnetic && params_.use_dg_transport)
    {
        M_dof_handler_.distribute_dofs(fe_DG_);
        M_locally_owned_ = M_dof_handler_.locally_owned_dofs();
        M_locally_relevant_ = dealii::DoFTools::extract_locally_relevant_dofs(M_dof_handler_);
    }

    // NS fields (ux, uy, p)
    if (params_.enable_ns)
    {
        ux_dof_handler_.distribute_dofs(fe_Q2_);
        uy_dof_handler_.distribute_dofs(fe_Q2_);
        p_dof_handler_.distribute_dofs(fe_Q1_);

        ux_locally_owned_ = ux_dof_handler_.locally_owned_dofs();
        ux_locally_relevant_ = dealii::DoFTools::extract_locally_relevant_dofs(ux_dof_handler_);
        uy_locally_owned_ = uy_dof_handler_.locally_owned_dofs();
        uy_locally_relevant_ = dealii::DoFTools::extract_locally_relevant_dofs(uy_dof_handler_);
        p_locally_owned_ = p_dof_handler_.locally_owned_dofs();
        p_locally_relevant_ = dealii::DoFTools::extract_locally_relevant_dofs(p_dof_handler_);
    }

    pcout_ << "[Setup] DoFs: θ=" << theta_dof_handler_.n_dofs();
    if (params_.enable_magnetic)
        pcout_ << ", φ=" << phi_dof_handler_.n_dofs();
    if (params_.enable_magnetic && params_.use_dg_transport)
        pcout_ << ", M=" << M_dof_handler_.n_dofs();
    if (params_.enable_ns)
        pcout_ << ", ux=" << ux_dof_handler_.n_dofs()
               << ", p=" << p_dof_handler_.n_dofs();
    pcout_ << "\n";
}

// ============================================================================
// setup_ch_system() - Inline setup following MMSContext pattern
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::setup_ch_system()
{
    const unsigned int n_theta = theta_dof_handler_.n_dofs();
    const unsigned int n_psi = psi_dof_handler_.n_dofs();
    const unsigned int n_total = n_theta + n_psi;

    // Build combined IndexSets: θ in [0, n_theta), ψ in [n_theta, n_total)
    ch_locally_owned_.clear();
    ch_locally_owned_.set_size(n_total);
    ch_locally_relevant_.clear();
    ch_locally_relevant_.set_size(n_total);

    for (auto idx = theta_locally_owned_.begin(); idx != theta_locally_owned_.end(); ++idx)
        ch_locally_owned_.add_index(*idx);
    for (auto idx = psi_locally_owned_.begin(); idx != psi_locally_owned_.end(); ++idx)
        ch_locally_owned_.add_index(n_theta + *idx);

    for (auto idx = theta_locally_relevant_.begin(); idx != theta_locally_relevant_.end(); ++idx)
        ch_locally_relevant_.add_index(*idx);
    for (auto idx = psi_locally_relevant_.begin(); idx != psi_locally_relevant_.end(); ++idx)
        ch_locally_relevant_.add_index(n_theta + *idx);

    ch_locally_owned_.compress();
    ch_locally_relevant_.compress();

    // Index maps
    theta_to_ch_map_.resize(n_theta);
    psi_to_ch_map_.resize(n_psi);
    for (unsigned int i = 0; i < n_theta; ++i)
        theta_to_ch_map_[i] = i;
    for (unsigned int i = 0; i < n_psi; ++i)
        psi_to_ch_map_[i] = n_theta + i;

    // Individual constraints (Neumann = hanging nodes only for physical runs)
    theta_constraints_.clear();
    theta_constraints_.reinit(theta_locally_owned_, theta_locally_relevant_);
    dealii::DoFTools::make_hanging_node_constraints(theta_dof_handler_, theta_constraints_);
    theta_constraints_.close();

    psi_constraints_.clear();
    psi_constraints_.reinit(psi_locally_owned_, psi_locally_relevant_);
    dealii::DoFTools::make_hanging_node_constraints(psi_dof_handler_, psi_constraints_);
    psi_constraints_.close();

    // Combined constraints
    ch_constraints_.clear();
    ch_constraints_.reinit(ch_locally_owned_, ch_locally_relevant_);

    // Map θ constraints to coupled system
    for (auto idx = theta_locally_relevant_.begin(); idx != theta_locally_relevant_.end(); ++idx)
    {
        if (theta_constraints_.is_constrained(*idx))
        {
            const auto coupled_i = theta_to_ch_map_[*idx];
            const auto* entries = theta_constraints_.get_constraint_entries(*idx);
            ch_constraints_.add_line(coupled_i);
            if (entries)
                for (const auto& e : *entries)
                    ch_constraints_.add_entry(coupled_i, theta_to_ch_map_[e.first], e.second);
            ch_constraints_.set_inhomogeneity(coupled_i, theta_constraints_.get_inhomogeneity(*idx));
        }
    }

    // Map ψ constraints to coupled system
    for (auto idx = psi_locally_relevant_.begin(); idx != psi_locally_relevant_.end(); ++idx)
    {
        if (psi_constraints_.is_constrained(*idx))
        {
            const auto coupled_i = psi_to_ch_map_[*idx];
            const auto* entries = psi_constraints_.get_constraint_entries(*idx);
            ch_constraints_.add_line(coupled_i);
            if (entries)
                for (const auto& e : *entries)
                    ch_constraints_.add_entry(coupled_i, psi_to_ch_map_[e.first], e.second);
            ch_constraints_.set_inhomogeneity(coupled_i, psi_constraints_.get_inhomogeneity(*idx));
        }
    }
    ch_constraints_.close();

    // Sparsity pattern
    dealii::TrilinosWrappers::SparsityPattern ch_sparsity(
        ch_locally_owned_, ch_locally_owned_, ch_locally_relevant_,
        mpi_communicator_);

    // Build sparsity for coupled blocks
    std::vector<dealii::types::global_dof_index> theta_dofs(fe_Q2_.dofs_per_cell);
    std::vector<dealii::types::global_dof_index> psi_dofs(fe_Q2_.dofs_per_cell);

    for (auto cell = theta_dof_handler_.begin_active(); cell != theta_dof_handler_.end(); ++cell)
    {
        if (!cell->is_locally_owned())
            continue;

        cell->get_dof_indices(theta_dofs);

        // Find matching psi cell
        typename dealii::DoFHandler<dim>::active_cell_iterator psi_cell(&triangulation_,
            cell->level(), cell->index(), &psi_dof_handler_);
        psi_cell->get_dof_indices(psi_dofs);

        // Map to combined indices
        std::vector<dealii::types::global_dof_index> combined_dofs;
        combined_dofs.reserve(2 * fe_Q2_.dofs_per_cell);
        for (auto d : theta_dofs)
            combined_dofs.push_back(theta_to_ch_map_[d]);
        for (auto d : psi_dofs)
            combined_dofs.push_back(psi_to_ch_map_[d]);

        ch_constraints_.add_entries_local_to_global(combined_dofs, ch_sparsity);
    }
    ch_sparsity.compress();

    // Initialize matrix
    ch_matrix_.reinit(ch_sparsity);

    // Initialize vectors
    ch_rhs_.reinit(ch_locally_owned_, mpi_communicator_);
    ch_solution_.reinit(ch_locally_owned_, mpi_communicator_);

    theta_solution_.reinit(theta_locally_owned_, mpi_communicator_);
    theta_old_solution_.reinit(theta_locally_owned_, mpi_communicator_);
    psi_solution_.reinit(psi_locally_owned_, mpi_communicator_);

    theta_relevant_.reinit(theta_locally_owned_, theta_locally_relevant_, mpi_communicator_);
    theta_old_relevant_.reinit(theta_locally_owned_, theta_locally_relevant_, mpi_communicator_);
    psi_relevant_.reinit(psi_locally_owned_, psi_locally_relevant_, mpi_communicator_);

    pcout_ << "[CH Setup] n_dofs=" << n_total << ", owned=" << ch_locally_owned_.n_elements() << "\n";
}

// ============================================================================
// setup_poisson_system()
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::setup_poisson_system()
{
    setup_poisson_constraints_and_sparsity<dim>(
        phi_dof_handler_,
        phi_locally_owned_,
        phi_locally_relevant_,
        phi_constraints_,
        phi_matrix_,
        mpi_communicator_,
        pcout_);

    phi_rhs_.reinit(phi_locally_owned_, mpi_communicator_);
    phi_solution_.reinit(phi_locally_owned_, mpi_communicator_);
    phi_relevant_.reinit(phi_locally_owned_, phi_locally_relevant_, mpi_communicator_);

    // OPTIMIZATION: Assemble Laplacian matrix ONCE (it's constant)
    assemble_poisson_matrix<dim>(phi_dof_handler_, phi_constraints_, phi_matrix_);

    // OPTIMIZATION: Initialize solver with cached AMG preconditioner
    poisson_solver_ = std::make_unique<PoissonSolver>(
        params_.solvers.poisson, phi_locally_owned_, mpi_communicator_);
    poisson_solver_->initialize(phi_matrix_);
}

// ============================================================================
// setup_magnetization_system()
//
// OPTIMIZED: Creates cached assembler and solver that persist across timesteps
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::setup_magnetization_system()
{
    M_constraints_.clear();
    M_constraints_.reinit(M_locally_owned_, M_locally_relevant_);
    M_constraints_.close();

    setup_magnetization_sparsity<dim>(
        M_dof_handler_,
        M_locally_owned_,
        M_locally_relevant_,
        M_matrix_,
        mpi_communicator_,
        pcout_);

    Mx_rhs_.reinit(M_locally_owned_, mpi_communicator_);
    My_rhs_.reinit(M_locally_owned_, mpi_communicator_);
    Mx_solution_.reinit(M_locally_owned_, mpi_communicator_);
    My_solution_.reinit(M_locally_owned_, mpi_communicator_);
    Mx_old_solution_.reinit(M_locally_owned_, mpi_communicator_);
    My_old_solution_.reinit(M_locally_owned_, mpi_communicator_);

    Mx_relevant_.reinit(M_locally_owned_, M_locally_relevant_, mpi_communicator_);
    My_relevant_.reinit(M_locally_owned_, M_locally_relevant_, mpi_communicator_);

    // ========================================================================
    // OPTIMIZATION: Create cached assembler (avoids recreation each timestep)
    // ========================================================================
    magnetization_assembler_ = std::make_unique<MagnetizationAssembler<dim>>(
        params_,
        M_dof_handler_,
        ux_dof_handler_,
        phi_dof_handler_,
        theta_dof_handler_,
        mpi_communicator_);

    // ========================================================================
    // OPTIMIZATION: Create cached solver (avoids recreation each timestep)
    // The solver will cache MUMPS factorization between Mx/My solves
    // ========================================================================
    magnetization_solver_ = std::make_unique<MagnetizationSolver<dim>>(
        params_.solvers.magnetization,
        M_locally_owned_,
        mpi_communicator_);

    pcout_ << "[Magnetization Setup] Cached assembler and solver initialized\n";
}

// ============================================================================
// setup_ns_system()
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::setup_ns_system()
{
    // Velocity constraints (Dirichlet u=0 on all boundaries)
    ux_constraints_.clear();
    ux_constraints_.reinit(ux_locally_owned_, ux_locally_relevant_);
    dealii::DoFTools::make_hanging_node_constraints(ux_dof_handler_, ux_constraints_);
    for (unsigned int bid = 0; bid <= 3; ++bid)
        dealii::VectorTools::interpolate_boundary_values(
            ux_dof_handler_, bid, dealii::Functions::ZeroFunction<dim>(), ux_constraints_);
    ux_constraints_.close();

    uy_constraints_.clear();
    uy_constraints_.reinit(uy_locally_owned_, uy_locally_relevant_);
    dealii::DoFTools::make_hanging_node_constraints(uy_dof_handler_, uy_constraints_);
    for (unsigned int bid = 0; bid <= 3; ++bid)
        dealii::VectorTools::interpolate_boundary_values(
            uy_dof_handler_, bid, dealii::Functions::ZeroFunction<dim>(), uy_constraints_);
    uy_constraints_.close();

    // Pressure constraints (pin DoF 0)
    p_constraints_.clear();
    p_constraints_.reinit(p_locally_owned_, p_locally_relevant_);
    dealii::DoFTools::make_hanging_node_constraints(p_dof_handler_, p_constraints_);
    if (p_locally_owned_.is_element(0))
    {
        p_constraints_.add_line(0);
        p_constraints_.set_inhomogeneity(0, 0.0);
    }
    p_constraints_.close();

    // Coupled NS system
    dealii::TrilinosWrappers::SparsityPattern ns_sparsity;
    setup_ns_coupled_system_parallel<dim>(
        ux_dof_handler_, uy_dof_handler_, p_dof_handler_,
        ux_constraints_, uy_constraints_, p_constraints_,
        ux_to_ns_map_, uy_to_ns_map_, p_to_ns_map_,
        ns_locally_owned_, ns_locally_relevant_,
        ns_constraints_, ns_sparsity,
        mpi_communicator_, pcout_);

    ns_matrix_.reinit(ns_sparsity);
    ns_rhs_.reinit(ns_locally_owned_, mpi_communicator_);
    ns_solution_.reinit(ns_locally_owned_, mpi_communicator_);

    ux_solution_.reinit(ux_locally_owned_, mpi_communicator_);
    ux_old_solution_.reinit(ux_locally_owned_, mpi_communicator_);
    uy_solution_.reinit(uy_locally_owned_, mpi_communicator_);
    uy_old_solution_.reinit(uy_locally_owned_, mpi_communicator_);
    p_solution_.reinit(p_locally_owned_, mpi_communicator_);

    ux_relevant_.reinit(ux_locally_owned_, ux_locally_relevant_, mpi_communicator_);
    uy_relevant_.reinit(uy_locally_owned_, uy_locally_relevant_, mpi_communicator_);
    p_relevant_.reinit(p_locally_owned_, p_locally_relevant_, mpi_communicator_);

    // Pressure mass matrix for Schur preconditioner
    assemble_pressure_mass_matrix_parallel<dim>(
        p_dof_handler_, p_constraints_,
        p_locally_owned_, p_locally_relevant_,
        pressure_mass_matrix_, mpi_communicator_);
}

// ============================================================================
// initialize_solutions()
// ============================================================================
template <int dim>
void PhaseFieldProblem<dim>::initialize_solutions()
{
    const double interface_y = params_.ic.pool_depth;
    const double eps = params_.physics.epsilon;

    class FlatLayerIC : public dealii::Function<dim>
    {
    public:
        double interface_y_, eps_;
        FlatLayerIC(double y, double e) : dealii::Function<dim>(1), interface_y_(y), eps_(e) {}
        double value(const dealii::Point<dim>& p, unsigned int = 0) const override
        {
            return -std::tanh((p[1] - interface_y_) / (std::sqrt(2.0) * eps_));
        }
    };

    FlatLayerIC ic_func(interface_y, eps);
    dealii::VectorTools::interpolate(theta_dof_handler_, ic_func, theta_solution_);
    theta_constraints_.distribute(theta_solution_);

    theta_old_solution_ = theta_solution_;
    psi_solution_ = 0;

    theta_relevant_ = theta_solution_;
    theta_old_relevant_ = theta_old_solution_;

    // Initialize zero velocity for CH assembly when NS disabled
    if (!params_.enable_ns)
    {
        ux_relevant_.reinit(theta_locally_owned_, theta_locally_relevant_, mpi_communicator_);
        uy_relevant_.reinit(theta_locally_owned_, theta_locally_relevant_, mpi_communicator_);
        ux_relevant_ = 0;
        uy_relevant_ = 0;
    }

    if (params_.enable_magnetic && params_.use_dg_transport)
    {
        Mx_solution_ = 0;
        My_solution_ = 0;
        Mx_old_solution_ = 0;
        My_old_solution_ = 0;
    }

    if (params_.enable_ns)
    {
        ux_solution_ = 0;
        uy_solution_ = 0;
        ux_old_solution_ = 0;
        uy_old_solution_ = 0;
        p_solution_ = 0;

        ux_relevant_ = 0;
        uy_relevant_ = 0;
    }

    pcout_ << "[Setup] IC: flat layer at y=" << interface_y << "\n";
}

// ============================================================================
// Explicit instantiation
// ============================================================================
template class PhaseFieldProblem<2>;