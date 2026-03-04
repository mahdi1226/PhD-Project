// ============================================================================
// mms/coupled/ns_mag_mms_test.cc - NS + Magnetization Coupled MMS Test
//
// Tests the Kelvin force coupling from magnetization to Navier-Stokes:
//   NS: ρ(∂U/∂t + (U·∇)U) - ν∇²U + ∇p = μ₀[(M·∇)H + ½(∇·M)H] + f_NS
//
// This is the CRITICAL coupling for Rosensweig instability!
// The Kelvin force μ₀(M·∇)H drives ferrofluid spike formation.
//
// Strategy:
//   1. Initialize M, φ from exact solutions (updated each timestep)
//   2. Kelvin force F_K is computed from these exact-interpolated fields
//   3. Standard NS MMS source f_NS manufactures U*, p*
//   4. Total RHS: f_NS + F_K(M_h, H_h)
//   5. Errors converge at optimal rates
//
// PRODUCTION CODE PATHS USED:
//   - setup_ns_velocity_constraints_parallel()
//   - setup_ns_pressure_constraints_parallel()
//   - setup_ns_coupled_system_parallel()
//   - assemble_ns_system_with_kelvin_force_parallel()
//   - solve_ns_system_direct_parallel()
//   - extract_ns_solutions_parallel()
//
// This verifies:
//   - Kelvin force assembly: μ₀[(M·∇)H + ½(∇·M)H]
//   - Correct coupling between magnetization and momentum equation
//   - Foundation for variable viscosity ν(θ)
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) Eq. 38
// ============================================================================

#include "mms/coupled/coupled_mms_test.h"
#include "mms/ns/ns_mms.h"
#include "mms/poisson/poisson_mms.h"
#include "mms/magnetization/magnetization_mms.h"

#include "setup/ns_setup.h"
#include "assembly/ns_assembler.h"
#include "solvers/ns_solver.h"

#include <deal.II/base/mpi.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/numerics/vector_tools.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <iomanip>

constexpr int dim = 2;

// ============================================================================
// NS + Magnetization single refinement test
// ============================================================================
static CoupledMMSResult run_ns_mag_single(
    unsigned int refinement,
    const Parameters& params,
    unsigned int n_time_steps,
    MPI_Comm mpi_communicator)
{
    CoupledMMSResult result;
    result.refinement = refinement;

    const unsigned int this_rank = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
    dealii::ConditionalOStream pcout(std::cout, this_rank == 0);

    // Parameters
    const double L_y = params.domain.y_max - params.domain.y_min;
    const double t_start = 0.1;
    const double t_end = 0.2;
    const double dt = (t_end - t_start) / n_time_steps;
    const double nu = params.physics.nu_ferro;
    const double mu_0 = params.physics.mu_0;

    Parameters mms_params = params;
    mms_params.enable_mms = true;

    auto total_start = std::chrono::high_resolution_clock::now();

    // ========================================================================
    // Create mesh
    // ========================================================================
    dealii::parallel::distributed::Triangulation<dim> triangulation(
        mpi_communicator,
        typename dealii::Triangulation<dim>::MeshSmoothing(
            dealii::Triangulation<dim>::smoothing_on_refinement |
            dealii::Triangulation<dim>::smoothing_on_coarsening));

    dealii::Point<dim> p1(params.domain.x_min, params.domain.y_min);
    dealii::Point<dim> p2(params.domain.x_max, params.domain.y_max);
    std::vector<unsigned int> subdivisions = {
        params.domain.initial_cells_x,
        params.domain.initial_cells_y
    };
    dealii::GridGenerator::subdivided_hyper_rectangle(triangulation, subdivisions, p1, p2);

    // Set boundary IDs
    for (const auto& cell : triangulation.active_cell_iterators())
    {
        if (!cell->is_locally_owned()) continue;
        for (const auto& face : cell->face_iterators())
        {
            if (!face->at_boundary()) continue;
            const auto center = face->center();
            if (std::abs(center[1] - params.domain.y_min) < 1e-10) face->set_boundary_id(0);
            else if (std::abs(center[0] - params.domain.x_max) < 1e-10) face->set_boundary_id(1);
            else if (std::abs(center[1] - params.domain.y_max) < 1e-10) face->set_boundary_id(2);
            else if (std::abs(center[0] - params.domain.x_min) < 1e-10) face->set_boundary_id(3);
        }
    }
    triangulation.refine_global(refinement);

    // Compute h
    double local_h = std::numeric_limits<double>::max();
    for (const auto& cell : triangulation.active_cell_iterators())
        if (cell->is_locally_owned())
            local_h = std::min(local_h, cell->diameter());
    MPI_Allreduce(&local_h, &result.h, 1, MPI_DOUBLE, MPI_MIN, mpi_communicator);

    // ========================================================================
    // NS DoF handlers
    // ========================================================================
    dealii::FE_Q<dim> fe_vel(params.fe.degree_velocity);
    dealii::FE_Q<dim> fe_p(params.fe.degree_pressure);

    dealii::DoFHandler<dim> ux_dof(triangulation);
    dealii::DoFHandler<dim> uy_dof(triangulation);
    dealii::DoFHandler<dim> p_dof(triangulation);

    ux_dof.distribute_dofs(fe_vel);
    uy_dof.distribute_dofs(fe_vel);
    p_dof.distribute_dofs(fe_p);

    dealii::IndexSet ux_owned = ux_dof.locally_owned_dofs();
    dealii::IndexSet uy_owned = uy_dof.locally_owned_dofs();
    dealii::IndexSet p_owned = p_dof.locally_owned_dofs();
    dealii::IndexSet ux_relevant = dealii::DoFTools::extract_locally_relevant_dofs(ux_dof);
    dealii::IndexSet uy_relevant = dealii::DoFTools::extract_locally_relevant_dofs(uy_dof);
    dealii::IndexSet p_relevant = dealii::DoFTools::extract_locally_relevant_dofs(p_dof);

    // ========================================================================
    // Poisson/Magnetization DoFs (for Kelvin force)
    // ========================================================================
    dealii::FE_Q<dim> fe_phi(params.fe.degree_potential);
    dealii::FE_DGQ<dim> fe_M(params.fe.degree_magnetization);

    dealii::DoFHandler<dim> phi_dof(triangulation);
    dealii::DoFHandler<dim> M_dof(triangulation);

    phi_dof.distribute_dofs(fe_phi);
    M_dof.distribute_dofs(fe_M);

    dealii::IndexSet phi_owned = phi_dof.locally_owned_dofs();
    dealii::IndexSet phi_relevant = dealii::DoFTools::extract_locally_relevant_dofs(phi_dof);
    dealii::IndexSet M_owned = M_dof.locally_owned_dofs();
    dealii::IndexSet M_relevant = dealii::DoFTools::extract_locally_relevant_dofs(M_dof);

    result.n_dofs = ux_dof.n_dofs() + uy_dof.n_dofs() + p_dof.n_dofs()
                  + phi_dof.n_dofs() + 2 * M_dof.n_dofs();

    // ========================================================================
    // NS setup (PRODUCTION CODE)
    // ========================================================================
    dealii::AffineConstraints<double> ux_constraints, uy_constraints, p_constraints;
    setup_ns_velocity_constraints_parallel<dim>(ux_dof, uy_dof, ux_constraints, uy_constraints);
    setup_ns_pressure_constraints_parallel<dim>(p_dof, p_constraints);

    std::vector<dealii::types::global_dof_index> ux_to_ns, uy_to_ns, p_to_ns;
    dealii::IndexSet ns_owned, ns_relevant;
    dealii::AffineConstraints<double> ns_constraints;
    dealii::TrilinosWrappers::SparsityPattern ns_sparsity;

    setup_ns_coupled_system_parallel<dim>(
        ux_dof, uy_dof, p_dof,
        ux_constraints, uy_constraints, p_constraints,
        ux_to_ns, uy_to_ns, p_to_ns,
        ns_owned, ns_relevant, ns_constraints, ns_sparsity,
        mpi_communicator, pcout);

    dealii::TrilinosWrappers::SparseMatrix ns_matrix;
    ns_matrix.reinit(ns_sparsity);
    dealii::TrilinosWrappers::MPI::Vector ns_rhs(ns_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector ns_solution(ns_owned, mpi_communicator);

    // NS solution vectors
    dealii::TrilinosWrappers::MPI::Vector ux_sol(ux_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector uy_sol(uy_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector p_sol(p_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector ux_old(ux_owned, ux_relevant, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector uy_old(uy_owned, uy_relevant, mpi_communicator);

    // M, φ vectors (for Kelvin force)
    dealii::TrilinosWrappers::MPI::Vector phi_vec(phi_owned, phi_relevant, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector Mx_vec(M_owned, M_relevant, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector My_vec(M_owned, M_relevant, mpi_communicator);

    // Temporary vectors for interpolation
    dealii::TrilinosWrappers::MPI::Vector phi_tmp(phi_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector Mx_tmp(M_owned, mpi_communicator);
    dealii::TrilinosWrappers::MPI::Vector My_tmp(M_owned, mpi_communicator);

    // ========================================================================
    // Initialize at t_start
    // ========================================================================
    double current_time = t_start;

    // Initialize U from exact
    NSExactVelocityX<dim> ux_exact(current_time, L_y);
    NSExactVelocityY<dim> uy_exact(current_time, L_y);
    dealii::VectorTools::interpolate(ux_dof, ux_exact, ux_sol);
    dealii::VectorTools::interpolate(uy_dof, uy_exact, uy_sol);
    ux_old = ux_sol;
    uy_old = uy_sol;

    // Initialize M, φ from exact
    MagExactMx<dim> Mx_exact_fn(current_time, L_y);
    MagExactMy<dim> My_exact_fn(current_time, L_y);
    PoissonExactSolution<dim> phi_exact_fn(current_time, L_y);

    dealii::VectorTools::interpolate(M_dof, Mx_exact_fn, Mx_tmp);
    dealii::VectorTools::interpolate(M_dof, My_exact_fn, My_tmp);
    dealii::VectorTools::interpolate(phi_dof, phi_exact_fn, phi_tmp);
    Mx_vec = Mx_tmp;
    My_vec = My_tmp;
    phi_vec = phi_tmp;

    // ========================================================================
    // Time stepping
    // ========================================================================
    for (unsigned int step = 0; step < n_time_steps; ++step)
    {
        current_time += dt;
        const double t_old = current_time - dt;

        // Update M, φ to exact at current time
        Mx_exact_fn.set_time(current_time);
        My_exact_fn.set_time(current_time);
        phi_exact_fn.set_time(current_time);

        dealii::VectorTools::interpolate(M_dof, Mx_exact_fn, Mx_tmp);
        dealii::VectorTools::interpolate(M_dof, My_exact_fn, My_tmp);
        dealii::VectorTools::interpolate(phi_dof, phi_exact_fn, phi_tmp);
        Mx_vec = Mx_tmp;
        My_vec = My_tmp;
        phi_vec = phi_tmp;

        // Assemble NS with Kelvin force (PRODUCTION CODE)
        ns_matrix = 0;
        ns_rhs = 0;

        assemble_ns_system_with_kelvin_force_parallel<dim>(
            ux_dof, uy_dof, p_dof,
            ux_old, uy_old,
            nu, dt, true, true,  // include_time_derivative, include_convection
            ux_to_ns, uy_to_ns, p_to_ns,
            ns_owned, ns_constraints,
            ns_matrix, ns_rhs, mpi_communicator,
            // Kelvin force inputs
            phi_dof, M_dof,
            phi_vec, Mx_vec, My_vec,
            mu_0,
            // MMS options
            true, current_time, t_old, L_y);

        // Solve NS (PRODUCTION CODE - direct solver for speed in MMS)
        ns_solution = 0;
        solve_ns_system_direct_parallel(
            ns_matrix, ns_rhs, ns_solution, ns_constraints,
            p_to_ns, ns_owned, mpi_communicator, false);

        // Extract solutions (PRODUCTION CODE)
        extract_ns_solutions_parallel(
            ns_solution, ux_to_ns, uy_to_ns, p_to_ns,
            ux_owned, uy_owned, p_owned,
            ns_owned, ns_relevant,
            ux_sol, uy_sol, p_sol,
            mpi_communicator);

        ux_old = ux_sol;
        uy_old = uy_sol;
    }

    // ========================================================================
    // Compute errors
    // ========================================================================
    ux_exact.set_time(current_time);
    uy_exact.set_time(current_time);
    NSExactPressure<dim> p_exact_fn(current_time, L_y);

    dealii::QGauss<dim> quad(params.fe.degree_velocity + 2);
    dealii::Vector<double> cell_err(triangulation.n_active_cells());

    // U_x L2 error
    dealii::TrilinosWrappers::MPI::Vector ux_gh(ux_owned, ux_relevant, mpi_communicator);
    ux_gh = ux_sol;
    dealii::VectorTools::integrate_difference(
        ux_dof, ux_gh, ux_exact, cell_err, quad, dealii::VectorTools::L2_norm);
    double local_sq = cell_err.norm_sqr(), global_sq;
    MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
    result.ux_L2 = std::sqrt(global_sq);

    // U_x H1 error
    dealii::VectorTools::integrate_difference(
        ux_dof, ux_gh, ux_exact, cell_err, quad, dealii::VectorTools::H1_seminorm);
    local_sq = cell_err.norm_sqr();
    MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
    result.ux_H1 = std::sqrt(global_sq);

    // Pressure L2 error (WITH MEAN SUBTRACTION)
// Incompressible NS: pressure determined up to a constant
dealii::TrilinosWrappers::MPI::Vector p_gh(p_owned, p_relevant, mpi_communicator);
p_gh = p_sol;

// Compute mean pressures
dealii::FEValues<dim> fe_values_p(p_dof.get_fe(), quad,
    dealii::update_values | dealii::update_quadrature_points | dealii::update_JxW_values);
double local_p_integral = 0.0, local_exact_p_integral = 0.0, local_volume = 0.0;
double local_p_err_sq = 0.0;
std::vector<double> p_vals(quad.size());

for (const auto& cell : p_dof.active_cell_iterators())
{
    if (!cell->is_locally_owned()) continue;
    fe_values_p.reinit(cell);
    fe_values_p.get_function_values(p_gh, p_vals);
    for (unsigned int q = 0; q < quad.size(); ++q)
    {
        const double JxW = fe_values_p.JxW(q);
        local_p_integral += p_vals[q] * JxW;
        local_exact_p_integral += p_exact_fn.value(fe_values_p.quadrature_point(q)) * JxW;
        local_volume += JxW;
    }
}
double global_p_int = 0, global_exact_p_int = 0, global_vol = 0;
MPI_Allreduce(&local_p_integral, &global_p_int, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
MPI_Allreduce(&local_exact_p_integral, &global_exact_p_int, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
MPI_Allreduce(&local_volume, &global_vol, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);

const double p_mean = global_p_int / global_vol;
const double exact_p_mean = global_exact_p_int / global_vol;

// Now compute error with mean-corrected pressures
for (const auto& cell : p_dof.active_cell_iterators())
{
    if (!cell->is_locally_owned()) continue;
    fe_values_p.reinit(cell);
    fe_values_p.get_function_values(p_gh, p_vals);
    for (unsigned int q = 0; q < quad.size(); ++q)
    {
        const double JxW = fe_values_p.JxW(q);
        const double p_err = (p_vals[q] - p_mean) - (p_exact_fn.value(fe_values_p.quadrature_point(q)) - exact_p_mean);
        local_p_err_sq += p_err * p_err * JxW;
    }
}
MPI_Allreduce(&local_p_err_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
result.p_L2 = std::sqrt(global_sq);

    auto total_end = std::chrono::high_resolution_clock::now();
    result.total_time = std::chrono::duration<double>(total_end - total_start).count();

    return result;
}

// ============================================================================
// Public interface
// ============================================================================
CoupledMMSConvergenceResult run_ns_magnetization_mms(
    const std::vector<unsigned int>& refinements,
    Parameters params,
    unsigned int n_time_steps,
    MPI_Comm mpi_communicator)
{
    CoupledMMSConvergenceResult result;
    result.level = CoupledMMSLevel::NS_MAGNETIZATION;
    result.expected_L2_rate = params.fe.degree_velocity + 1;  // Q2 -> 3
    result.expected_H1_rate = params.fe.degree_velocity;      // Q2 -> 2

    const unsigned int this_rank = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);

    params.enable_mms = true;

    if (this_rank == 0)
    {
        std::cout << "\n[NS_MAG] Running NS + Magnetization (Kelvin force) MMS test...\n";
        std::cout << "  Tests: Kelvin force mu_0[(M.grad)H + 0.5(div M)H]\n";
        std::cout << "  MPI ranks: " << dealii::Utilities::MPI::n_mpi_processes(mpi_communicator) << "\n";
        std::cout << "  Time steps: " << n_time_steps << "\n";
        std::cout << "  mu_0: " << params.physics.mu_0 << "\n";
        std::cout << "  Expected rates: U L2 = 3, U H1 = 2, p L2 = 2\n\n";
    }

    for (unsigned int ref : refinements)
    {
        if (this_rank == 0)
            std::cout << "  Ref " << ref << "... " << std::flush;

        Parameters iter_params = params;
        iter_params.enable_mms = true;

        CoupledMMSResult r = run_ns_mag_single(ref, iter_params, n_time_steps, mpi_communicator);
        result.results.push_back(r);

        if (this_rank == 0)
        {
            std::cout << "ux_L2=" << std::scientific << std::setprecision(2) << r.ux_L2
                      << ", p_L2=" << r.p_L2
                      << ", time=" << std::fixed << std::setprecision(1) << r.total_time << "s\n";
        }
    }

    result.compute_rates();
    return result;
}