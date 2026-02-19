// ============================================================================
// cahn_hilliard/cahn_hilliard.cc - Orchestration, Accessors, Diagnostics
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
//            Equations 42a-42b
// ============================================================================

#include "cahn_hilliard/cahn_hilliard.h"
#include "physics/material_properties.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>

#include <chrono>
#include <iostream>

// ============================================================================
// Constructor
// ============================================================================
template <int dim>
CahnHilliardSubsystem<dim>::CahnHilliardSubsystem(
    const Parameters& params,
    MPI_Comm mpi_comm,
    dealii::parallel::distributed::Triangulation<dim>& triangulation)
    : params_(params)
    , mpi_comm_(mpi_comm)
    , triangulation_(triangulation)
    , fe_(params.fe.degree_phase)
    , theta_dof_handler_(triangulation)
    , psi_dof_handler_(triangulation)
    , pcout_(std::cout,
             dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
{
}

// ============================================================================
// Setup — orchestrate all initialization steps
// ============================================================================
template <int dim>
void CahnHilliardSubsystem<dim>::setup()
{
    pcout_ << "[CH] Setting up Cahn-Hilliard subsystem (CG Q"
           << params_.fe.degree_phase << ")...\n";

    distribute_dofs();
    build_constraints();
    build_index_maps();
    build_coupled_sparsity();
    allocate_vectors();

    pcout_ << "[CH] Setup complete: "
           << theta_dof_handler_.n_dofs() << " θ DoFs + "
           << psi_dof_handler_.n_dofs() << " ψ DoFs = "
           << (theta_dof_handler_.n_dofs() + psi_dof_handler_.n_dofs())
           << " coupled DoFs\n";
}

// ============================================================================
// Assemble — public entry point
// ============================================================================
template <int dim>
void CahnHilliardSubsystem<dim>::assemble(
    const dealii::TrilinosWrappers::MPI::Vector& theta_old_relevant,
    const std::vector<const dealii::TrilinosWrappers::MPI::Vector*>& velocity_components,
    const dealii::DoFHandler<dim>& u_dof_handler,
    double dt,
    double current_time)
{
    Assert(velocity_components.size() == dim,
           dealii::ExcMessage("velocity_components must have exactly dim entries"));

    assemble_system(theta_old_relevant,
                    velocity_components, u_dof_handler,
                    dt, current_time);
}

// ============================================================================
// Solve — public entry point
// ============================================================================
template <int dim>
SolverInfo CahnHilliardSubsystem<dim>::solve()
{
    last_solve_info_ = solve_coupled_system();
    invalidate_ghosts();
    return last_solve_info_;
}

// ============================================================================
// MMS source injection
// ============================================================================
template <int dim>
void CahnHilliardSubsystem<dim>::set_mms_source(
    MmsSourceFunction theta_source,
    MmsSourceFunction psi_source)
{
    mms_source_theta_ = std::move(theta_source);
    mms_source_psi_ = std::move(psi_source);
}

// ============================================================================
// Apply Dirichlet boundary conditions (MMS testing)
// ============================================================================
template <int dim>
void CahnHilliardSubsystem<dim>::apply_dirichlet_boundary(
    const dealii::Function<dim>& theta_bc,
    const dealii::Function<dim>& psi_bc)
{
    // Rebuild θ constraints: hanging nodes + Dirichlet BCs
    theta_constraints_.clear();
    theta_constraints_.reinit(theta_locally_owned_, theta_locally_relevant_);
    dealii::DoFTools::make_hanging_node_constraints(theta_dof_handler_,
                                                     theta_constraints_);

    std::map<dealii::types::boundary_id, const dealii::Function<dim>*> theta_bc_map;
    for (unsigned int bid = 0; bid < 2 * dim; ++bid)
        theta_bc_map[bid] = &theta_bc;

    dealii::VectorTools::interpolate_boundary_values(theta_dof_handler_,
                                                      theta_bc_map,
                                                      theta_constraints_);
    theta_constraints_.close();

    // Rebuild ψ constraints: hanging nodes + Dirichlet BCs
    psi_constraints_.clear();
    psi_constraints_.reinit(psi_locally_owned_, psi_locally_relevant_);
    dealii::DoFTools::make_hanging_node_constraints(psi_dof_handler_,
                                                     psi_constraints_);

    std::map<dealii::types::boundary_id, const dealii::Function<dim>*> psi_bc_map;
    for (unsigned int bid = 0; bid < 2 * dim; ++bid)
        psi_bc_map[bid] = &psi_bc;

    dealii::VectorTools::interpolate_boundary_values(psi_dof_handler_,
                                                      psi_bc_map,
                                                      psi_constraints_);
    psi_constraints_.close();

    // Rebuild coupled constraints from updated individual constraints
    rebuild_coupled_constraints();
}

// ============================================================================
// Project initial condition
// ============================================================================
template <int dim>
void CahnHilliardSubsystem<dim>::project_initial_condition(
    const dealii::Function<dim>& f_theta,
    const dealii::Function<dim>& f_psi)
{
    dealii::VectorTools::project(theta_dof_handler_,
                                 theta_constraints_,
                                 dealii::QGauss<dim>(fe_.degree + 2),
                                 f_theta,
                                 theta_solution_);

    dealii::VectorTools::project(psi_dof_handler_,
                                 psi_constraints_,
                                 dealii::QGauss<dim>(fe_.degree + 2),
                                 f_psi,
                                 psi_solution_);

    invalidate_ghosts();
    pcout_ << "[CH] Initial condition projected\n";
}

// ============================================================================
// Initialize constant θ
// ============================================================================
template <int dim>
void CahnHilliardSubsystem<dim>::initialize_constant(double theta_value)
{
    theta_solution_ = theta_value;
    theta_constraints_.distribute(theta_solution_);

    // ψ = f(θ)/ε in equilibrium, but for θ = 0 or ±1 this gives simple values
    const double psi_value = double_well_derivative(theta_value) / params_.physics.epsilon;
    psi_solution_ = psi_value;
    psi_constraints_.distribute(psi_solution_);

    invalidate_ghosts();
    pcout_ << "[CH] Initialized θ = " << theta_value
           << ", ψ = " << psi_value << "\n";
}

// ============================================================================
// Accessors
// ============================================================================
template <int dim>
const dealii::DoFHandler<dim>&
CahnHilliardSubsystem<dim>::get_theta_dof_handler() const
{
    return theta_dof_handler_;
}

template <int dim>
const dealii::DoFHandler<dim>&
CahnHilliardSubsystem<dim>::get_psi_dof_handler() const
{
    return psi_dof_handler_;
}

template <int dim>
dealii::TrilinosWrappers::MPI::Vector&
CahnHilliardSubsystem<dim>::get_theta_solution()
{
    return theta_solution_;
}

template <int dim>
const dealii::TrilinosWrappers::MPI::Vector&
CahnHilliardSubsystem<dim>::get_theta_solution() const
{
    return theta_solution_;
}

template <int dim>
dealii::TrilinosWrappers::MPI::Vector&
CahnHilliardSubsystem<dim>::get_psi_solution()
{
    return psi_solution_;
}

template <int dim>
const dealii::TrilinosWrappers::MPI::Vector&
CahnHilliardSubsystem<dim>::get_psi_solution() const
{
    return psi_solution_;
}

template <int dim>
const dealii::TrilinosWrappers::MPI::Vector&
CahnHilliardSubsystem<dim>::get_theta_relevant() const
{
    Assert(ghosts_valid_, dealii::ExcMessage("Call update_ghosts() first"));
    return theta_relevant_;
}

template <int dim>
const dealii::TrilinosWrappers::MPI::Vector&
CahnHilliardSubsystem<dim>::get_psi_relevant() const
{
    Assert(ghosts_valid_, dealii::ExcMessage("Call update_ghosts() first"));
    return psi_relevant_;
}

template <int dim>
void CahnHilliardSubsystem<dim>::update_ghosts()
{
    if (!ghosts_valid_)
    {
        theta_relevant_ = theta_solution_;
        psi_relevant_ = psi_solution_;
        ghosts_valid_ = true;
    }
}

template <int dim>
void CahnHilliardSubsystem<dim>::invalidate_ghosts()
{
    ghosts_valid_ = false;
}

// ============================================================================
// Diagnostics
// ============================================================================
template <int dim>
typename CahnHilliardSubsystem<dim>::Diagnostics
CahnHilliardSubsystem<dim>::compute_diagnostics() const
{
    Diagnostics diag;

    // Ensure ghosts are available
    dealii::TrilinosWrappers::MPI::Vector theta_ghost(
        theta_locally_owned_, theta_locally_relevant_, mpi_comm_);
    theta_ghost = theta_solution_;

    dealii::TrilinosWrappers::MPI::Vector psi_ghost(
        psi_locally_owned_, psi_locally_relevant_, mpi_comm_);
    psi_ghost = psi_solution_;

    const unsigned int quad_degree = fe_.degree + 2;
    dealii::QGauss<dim> quadrature(quad_degree);
    dealii::FEValues<dim> theta_fe_values(fe_, quadrature,
                                           dealii::update_values |
                                           dealii::update_gradients |
                                           dealii::update_JxW_values);
    dealii::FEValues<dim> psi_fe_values(fe_, quadrature,
                                         dealii::update_values |
                                         dealii::update_JxW_values);

    const unsigned int n_q = quadrature.size();
    std::vector<double> theta_vals(n_q);
    std::vector<dealii::Tensor<1, dim>> theta_grads(n_q);
    std::vector<double> psi_vals(n_q);

    const double eps = params_.physics.epsilon;
    const double lambda = params_.physics.lambda;

    double local_mass = 0.0;
    double local_E_grad = 0.0;
    double local_E_bulk = 0.0;
    double local_psi_L2_sq = 0.0;
    double local_interface = 0.0;
    double local_theta_min = std::numeric_limits<double>::max();
    double local_theta_max = -std::numeric_limits<double>::max();
    double local_psi_min = std::numeric_limits<double>::max();
    double local_psi_max = -std::numeric_limits<double>::max();
    double local_volume = 0.0;
    double local_theta_sum = 0.0;

    for (const auto& cell : theta_dof_handler_.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        theta_fe_values.reinit(cell);

        // Construct matching psi cell
        const typename dealii::DoFHandler<dim>::active_cell_iterator psi_cell(
            &triangulation_, cell->level(), cell->index(), &psi_dof_handler_);
        psi_fe_values.reinit(psi_cell);

        theta_fe_values.get_function_values(theta_ghost, theta_vals);
        theta_fe_values.get_function_gradients(theta_ghost, theta_grads);
        psi_fe_values.get_function_values(psi_ghost, psi_vals);

        for (unsigned int q = 0; q < n_q; ++q)
        {
            const double JxW = theta_fe_values.JxW(q);
            const double th = theta_vals[q];
            const double ps = psi_vals[q];
            const double grad_sq = theta_grads[q] * theta_grads[q];

            local_mass += th * JxW;
            local_volume += JxW;
            local_theta_sum += th * JxW;

            // CH energy: E = λ [ε/2 |∇θ|² + (1/ε) F(θ)]
            local_E_grad += lambda * 0.5 * eps * grad_sq * JxW;
            local_E_bulk += lambda * (1.0 / eps) * double_well_potential(th) * JxW;

            local_psi_L2_sq += ps * ps * JxW;

            // Interface measure: ∫|∇θ| dΩ
            local_interface += std::sqrt(grad_sq) * JxW;

            local_theta_min = std::min(local_theta_min, th);
            local_theta_max = std::max(local_theta_max, th);
            local_psi_min = std::min(local_psi_min, ps);
            local_psi_max = std::max(local_psi_max, ps);
        }
    }

    // MPI reductions
    diag.mass_integral = dealii::Utilities::MPI::sum(local_mass, mpi_comm_);
    diag.E_gradient = dealii::Utilities::MPI::sum(local_E_grad, mpi_comm_);
    diag.E_bulk = dealii::Utilities::MPI::sum(local_E_bulk, mpi_comm_);
    diag.E_total = diag.E_gradient + diag.E_bulk;
    diag.psi_L2 = std::sqrt(dealii::Utilities::MPI::sum(local_psi_L2_sq, mpi_comm_));
    diag.interface_length = dealii::Utilities::MPI::sum(local_interface, mpi_comm_);
    diag.theta_min = dealii::Utilities::MPI::min(local_theta_min, mpi_comm_);
    diag.theta_max = dealii::Utilities::MPI::max(local_theta_max, mpi_comm_);
    diag.psi_min = dealii::Utilities::MPI::min(local_psi_min, mpi_comm_);
    diag.psi_max = dealii::Utilities::MPI::max(local_psi_max, mpi_comm_);

    const double total_volume = dealii::Utilities::MPI::sum(local_volume, mpi_comm_);
    const double theta_sum = dealii::Utilities::MPI::sum(local_theta_sum, mpi_comm_);
    diag.theta_mean = (total_volume > 0.0) ? (theta_sum / total_volume) : 0.0;

    // Solver stats from last solve
    diag.iterations = last_solve_info_.iterations;
    diag.residual = last_solve_info_.residual;
    diag.solve_time = last_solve_info_.solve_time;
    diag.assemble_time = last_assemble_time_;

    return diag;
}

// ============================================================================
// Explicit instantiations
// ============================================================================
template class CahnHilliardSubsystem<2>;
template class CahnHilliardSubsystem<3>;