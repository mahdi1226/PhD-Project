// ============================================================================
// solvers/magnetic_solver.cc - Monolithic Magnetics Solver (PARALLEL)
//
// Direct solver cascade: MUMPS → SuperLU_DIST → KLU
// Iterative solver: GMRES + AMG (or ILU)
// Block PC solver: FGMRES + block-triangular [A_M, C_Mφ; 0, A_φ]
// Fallback: iterative/block ↔ direct if primary fails
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "solvers/magnetic_solver.h"
#include "solvers/direct_solver_utils.h"
#include "solvers/magnetic_block_preconditioner.h"

#include <deal.II/base/utilities.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_precondition.h>

#include <chrono>
#include <iostream>
#include <iomanip>
#include <memory>

// ============================================================================
// Constructor
// ============================================================================
template <int dim>
MagneticSolver<dim>::MagneticSolver(
    const dealii::IndexSet& locally_owned,
    MPI_Comm mpi_communicator)
    : locally_owned_(locally_owned)
    , mpi_communicator_(mpi_communicator)
    , last_n_iterations_(0)
    , n_M_dofs_(0)
    , n_phi_dofs_(0)
    , block_structure_set_(false)
{
}

// ============================================================================
// set_block_structure
// ============================================================================
template <int dim>
void MagneticSolver<dim>::set_block_structure(
    dealii::types::global_dof_index n_M_dofs,
    dealii::types::global_dof_index n_phi_dofs)
{
    n_M_dofs_ = n_M_dofs;
    n_phi_dofs_ = n_phi_dofs;
    block_structure_set_ = true;
}

// ============================================================================
// solve (with params) — primary interface
// ============================================================================
template <int dim>
void MagneticSolver<dim>::solve(
    const dealii::TrilinosWrappers::SparseMatrix& system_matrix,
    dealii::TrilinosWrappers::MPI::Vector& solution,
    const dealii::TrilinosWrappers::MPI::Vector& rhs,
    const LinearSolverParams& params,
    bool verbose)
{
    last_info_.reset();
    last_info_.solver_name = "Mag";
    last_info_.matrix_size = system_matrix.m();
    last_info_.nnz = system_matrix.n_nonzero_elements();

    // Zero RHS guard
    const double rhs_norm = rhs.l2_norm();
    if (rhs_norm < 1e-14)
    {
        solution = 0;
        last_n_iterations_ = 0;
        last_info_.converged = true;
        last_info_.iterations = 0;
        return;
    }

    auto start = std::chrono::high_resolution_clock::now();

    if (!params.use_iterative)
    {
        // Direct solver path
        solve_direct(system_matrix, solution, rhs, verbose);

        // Fallback to iterative if direct failed
        if (!last_info_.converged && params.fallback_to_direct)
        {
            if (block_structure_set_ &&
                params.preconditioner == LinearSolverParams::Preconditioner::BlockSchur)
                solve_block_preconditioned(system_matrix, solution, rhs, params, verbose);
            else
                solve_iterative(system_matrix, solution, rhs, params, verbose);
        }
    }
    else
    {
        // Iterative solver path — choose block PC or monolithic
        if (block_structure_set_ &&
            params.preconditioner == LinearSolverParams::Preconditioner::BlockSchur)
        {
            solve_block_preconditioned(system_matrix, solution, rhs, params, verbose);
        }
        else
        {
            solve_iterative(system_matrix, solution, rhs, params, verbose);
        }

        // Fallback to direct if iterative failed
        if (!last_info_.converged && params.fallback_to_direct)
        {
            const unsigned int rank =
                dealii::Utilities::MPI::this_mpi_process(mpi_communicator_);
            if (verbose && rank == 0)
                std::cerr << "[Mag] Iterative failed, falling back to direct\n";

            solve_direct(system_matrix, solution, rhs, verbose);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    last_info_.solve_time = std::chrono::duration<double>(end - start).count();
    last_n_iterations_ = last_info_.iterations;
}

// ============================================================================
// solve (legacy — backward compatible, always direct)
// ============================================================================
template <int dim>
void MagneticSolver<dim>::solve(
    const dealii::TrilinosWrappers::SparseMatrix& system_matrix,
    dealii::TrilinosWrappers::MPI::Vector& solution,
    const dealii::TrilinosWrappers::MPI::Vector& rhs)
{
    // Legacy: direct solver only
    LinearSolverParams params;
    params.use_iterative = false;
    params.fallback_to_direct = true;
    solve(system_matrix, solution, rhs, params, false);
}

// ============================================================================
// Direct solver cascade
// ============================================================================
template <int dim>
void MagneticSolver<dim>::solve_direct(
    const dealii::TrilinosWrappers::SparseMatrix& system_matrix,
    dealii::TrilinosWrappers::MPI::Vector& solution,
    const dealii::TrilinosWrappers::MPI::Vector& rhs,
    bool verbose)
{
    const unsigned int rank =
        dealii::Utilities::MPI::this_mpi_process(mpi_communicator_);

    bool converged = DirectSolverUtils::try_direct_solver_chain(
        system_matrix, rhs, solution, "Mag", rank, verbose);

    if (converged)
    {
        last_info_.iterations = 1;
        last_info_.residual = 0.0;
        last_info_.converged = true;
        last_info_.used_direct = true;
    }
    else
    {
        last_info_.converged = false;
        if (verbose && rank == 0)
            std::cerr << "[Mag] All direct solvers failed\n";
    }
}

// ============================================================================
// Iterative solver: GMRES + AMG or ILU (monolithic, no block structure)
// ============================================================================
template <int dim>
void MagneticSolver<dim>::solve_iterative(
    const dealii::TrilinosWrappers::SparseMatrix& system_matrix,
    dealii::TrilinosWrappers::MPI::Vector& solution,
    const dealii::TrilinosWrappers::MPI::Vector& rhs,
    const LinearSolverParams& params,
    bool verbose)
{
    const unsigned int rank =
        dealii::Utilities::MPI::this_mpi_process(mpi_communicator_);
    const double rhs_norm = rhs.l2_norm();

    dealii::SolverControl solver_control(
        params.max_iterations,
        params.rel_tolerance * rhs_norm,
        false, false);

    dealii::SolverGMRES<dealii::TrilinosWrappers::MPI::Vector>::AdditionalData gmres_data;
    gmres_data.max_n_tmp_vectors = params.gmres_restart;
    gmres_data.right_preconditioning = true;

    dealii::SolverGMRES<dealii::TrilinosWrappers::MPI::Vector> solver(solver_control, gmres_data);

    // Build preconditioner
    std::unique_ptr<dealii::TrilinosWrappers::PreconditionBase> preconditioner;

    if (params.preconditioner == LinearSolverParams::Preconditioner::ILU)
    {
        auto ilu = std::make_unique<dealii::TrilinosWrappers::PreconditionILU>();
        dealii::TrilinosWrappers::PreconditionILU::AdditionalData ilu_data;
        ilu_data.ilu_fill = 2;
        ilu->initialize(system_matrix, ilu_data);
        preconditioner = std::move(ilu);

        if (verbose && rank == 0)
            std::cout << "[Mag] Using GMRES + ILU(2)\n";
    }
    else
    {
        // AMG — non-elliptic, higher-order for the mixed M+φ system
        auto amg = std::make_unique<dealii::TrilinosWrappers::PreconditionAMG>();
        dealii::TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
        amg_data.smoother_sweeps = 2;
        amg_data.aggregation_threshold = 0.02;
        amg_data.elliptic = false;           // Mixed DG+CG, non-symmetric
        amg_data.higher_order_elements = true;
        amg->initialize(system_matrix, amg_data);
        preconditioner = std::move(amg);

        if (verbose && rank == 0)
            std::cout << "[Mag] Using GMRES + AMG\n";
    }

    try
    {
        solver.solve(system_matrix, solution, rhs, *preconditioner);
        last_info_.iterations = solver_control.last_step();
        last_info_.residual = solver_control.last_value() / rhs_norm;
        last_info_.converged = true;
    }
    catch (dealii::SolverControl::NoConvergence&)
    {
        last_info_.iterations = solver_control.last_step();
        last_info_.residual = solver_control.last_value() / rhs_norm;
        last_info_.converged = false;

        if (verbose && rank == 0)
        {
            std::cerr << "[Mag] GMRES failed after " << last_info_.iterations
                      << " iterations, residual = " << std::scientific
                      << last_info_.residual << "\n";
        }
    }
}

// ============================================================================
// Block preconditioned solver: FGMRES + block-triangular PC
//
// Uses MagneticBlockPreconditioner which exploits the block structure:
//   | A_M      C_Mφ  | | M |   | f_M |
//   | C_φM     A_φ   | | φ | = | f_φ |
//
// FGMRES (flexible GMRES) is required because the block PC uses
// inner iterative solvers (variable preconditioning).
// ============================================================================
template <int dim>
void MagneticSolver<dim>::solve_block_preconditioned(
    const dealii::TrilinosWrappers::SparseMatrix& system_matrix,
    dealii::TrilinosWrappers::MPI::Vector& solution,
    const dealii::TrilinosWrappers::MPI::Vector& rhs,
    const LinearSolverParams& params,
    bool verbose)
{
    const unsigned int rank =
        dealii::Utilities::MPI::this_mpi_process(mpi_communicator_);
    const double rhs_norm = rhs.l2_norm();

    if (verbose && rank == 0)
        std::cout << "[Mag] Using FGMRES + Block PC (M=" << n_M_dofs_
                  << ", phi=" << n_phi_dofs_ << ")\n";

    // Build block preconditioner
    const bool use_ilu =
        (params.preconditioner == LinearSolverParams::Preconditioner::ILU);

    MagneticBlockPreconditioner block_pc(
        system_matrix,
        locally_owned_,
        n_M_dofs_,
        n_phi_dofs_,
        mpi_communicator_,
        use_ilu);

    block_pc.inner_tolerance = params.schur_inner_tolerance;
    block_pc.max_inner_iterations = params.schur_max_inner_iters;
    block_pc.verbose_ = verbose;

    // Outer FGMRES solver (flexible for variable preconditioning)
    dealii::SolverControl solver_control(
        params.max_iterations,
        params.rel_tolerance * rhs_norm,
        false, false);

    dealii::SolverFGMRES<dealii::TrilinosWrappers::MPI::Vector>::AdditionalData fgmres_data;
    fgmres_data.max_basis_size = params.gmres_restart;

    dealii::SolverFGMRES<dealii::TrilinosWrappers::MPI::Vector> solver(
        solver_control, fgmres_data);

    try
    {
        solver.solve(system_matrix, solution, rhs, block_pc);
        last_info_.iterations = solver_control.last_step();
        last_info_.residual = solver_control.last_value() / rhs_norm;
        last_info_.converged = true;
    }
    catch (dealii::SolverControl::NoConvergence&)
    {
        last_info_.iterations = solver_control.last_step();
        last_info_.residual = solver_control.last_value() / rhs_norm;
        last_info_.converged = false;

        if (verbose && rank == 0)
        {
            std::cerr << "[Mag] FGMRES+Block failed after " << last_info_.iterations
                      << " iterations, residual = " << std::scientific
                      << last_info_.residual << "\n";
        }
    }

    // Capture inner iteration diagnostics
    last_info_.inner_iterations_A = block_pc.n_iterations_M;
    last_info_.inner_iterations_S = block_pc.n_iterations_phi;
}

// ============================================================================
// Explicit instantiation
// ============================================================================
template class MagneticSolver<2>;
