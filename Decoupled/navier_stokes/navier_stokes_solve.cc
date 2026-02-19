// ============================================================================
// navier_stokes/navier_stokes_solve.cc — Solver Implementation
//
// Implements NSSubsystem<dim>::solve():
//   1. Direct solver (MUMPS → UMFPACK → SuperLU cascade) with pressure pinning
//   2. Extract component solutions (ux, uy, p) from monolithic vector
//   3. Update ghosted vectors
//
// Block Schur iterative solver will be added separately.
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

// --- Pressure null-space fix selector ---
// Must match navier_stokes_setup.cc
#ifndef NS_PRESSURE_FIX
#define NS_PRESSURE_FIX 2
#endif

#include "navier_stokes/navier_stokes.h"

#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/read_write_vector.h>

#include <Epetra_CrsMatrix.h>

#if defined(NS_PRESSURE_FIX) && NS_PRESSURE_FIX == 3
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#endif

#include <chrono>
#include <iostream>
#include <iomanip>

// ============================================================================
// Static helper: Try a specific direct solver with verbose error reporting
// ============================================================================
static bool try_direct_solver(
    const std::string& solver_type,
    const dealii::TrilinosWrappers::SparseMatrix& matrix,
    const dealii::TrilinosWrappers::MPI::Vector& rhs,
    dealii::TrilinosWrappers::MPI::Vector& solution,
    int rank,
    bool verbose)
{
    if (verbose && rank == 0)
        std::cout << "[NS Direct] Trying " << solver_type << "...\n";

    try
    {
        dealii::SolverControl solver_control(1, 0);
        dealii::TrilinosWrappers::SolverDirect::AdditionalData data;
        data.solver_type = solver_type;

        dealii::TrilinosWrappers::SolverDirect direct_solver(solver_control, data);

        if (verbose && rank == 0)
            std::cout << "[NS Direct]   Initializing (symbolic + numeric factorization)...\n";

        direct_solver.initialize(matrix);

        if (verbose && rank == 0)
            std::cout << "[NS Direct]   Solving...\n";

        direct_solver.solve(solution, rhs);

        if (verbose && rank == 0)
            std::cout << "[NS Direct] SUCCESS with " << solver_type << "\n";

        return true;
    }
    catch (const dealii::ExceptionBase& e)
    {
        if (verbose && rank == 0)
            std::cout << "[NS Direct]   " << solver_type << " failed: " << e.what() << "\n";
        return false;
    }
    catch (const std::exception& e)
    {
        if (verbose && rank == 0)
            std::cout << "[NS Direct]   " << solver_type << " failed: " << e.what() << "\n";
        return false;
    }
    catch (...)
    {
        if (verbose && rank == 0)
            std::cout << "[NS Direct]   " << solver_type << " failed: unknown exception\n";
        return false;
    }
}

// ============================================================================
// Static helper: Pin pressure DoF in matrix to remove null space
// ============================================================================
static bool pin_pressure_dof(
    dealii::TrilinosWrappers::SparseMatrix& matrix,
    dealii::TrilinosWrappers::MPI::Vector& rhs,
    dealii::types::global_dof_index pin_dof,
    const dealii::IndexSet& owned_dofs,
    int rank,
    bool verbose)
{
    if (verbose && rank == 0)
        std::cout << "[NS Direct] Pinning pressure DoF " << pin_dof << " to remove null space\n";

    bool diagonal_found = false;

    if (owned_dofs.is_element(pin_dof))
    {
        Epetra_CrsMatrix& epetra_mat = const_cast<Epetra_CrsMatrix&>(matrix.trilinos_matrix());
        int local_row = epetra_mat.LRID(static_cast<long long>(pin_dof));

        if (local_row >= 0)
        {
            int num_entries;
            double* values;
            int* col_indices;
            epetra_mat.ExtractMyRowView(local_row, num_entries, values, col_indices);

            if (verbose && rank == 0)
                std::cout << "[NS Direct]   Row " << pin_dof << " has " << num_entries << " entries\n";

            for (int k = 0; k < num_entries; ++k)
            {
                long long global_col = epetra_mat.GCID64(col_indices[k]);
                if (global_col == static_cast<long long>(pin_dof))
                {
                    values[k] = 1.0;
                    diagonal_found = true;
                    if (verbose && rank == 0)
                        std::cout << "[NS Direct]   Found diagonal at position " << k << ", setting to 1.0\n";
                }
                else
                {
                    values[k] = 0.0;
                }
            }

            if (!diagonal_found && rank == 0)
                std::cerr << "[NS Direct] ERROR: No diagonal entry for pressure DoF " << pin_dof << "!\n";
        }

        rhs[pin_dof] = 0.0;
    }

    rhs.compress(dealii::VectorOperation::insert);
    return diagonal_found;
}


// ============================================================================
// solve_direct() — Direct solver with pressure pinning
//
// Tries: MUMPS → UMFPACK → SuperLU_dist → SuperLU → KLU → LAPACK
// Pins pressure DoF 0 in coupled system to remove null space.
// ============================================================================
template <int dim>
SolverInfo NSSubsystem<dim>::solve_direct(bool verbose)
{
    constexpr unsigned int LAPACK_SIZE_LIMIT = 50000;

    SolverInfo info;
    info.solver_name = "NS-Direct";
    info.matrix_size = ns_matrix_.m();
    info.nnz = ns_matrix_.n_nonzero_elements();
    info.used_direct = true;

    int rank;
    MPI_Comm_rank(mpi_comm_, &rank);

    const double rhs_norm_before = ns_rhs_.l2_norm();
    if (rhs_norm_before < 1e-14)
    {
        ns_solution_ = 0;
        ns_constraints_.distribute(ns_solution_);
        info.converged = true;
        info.residual = 0.0;
        info.iterations = 0;
        return info;
    }

    auto start = std::chrono::high_resolution_clock::now();

    if (verbose && rank == 0)
    {
        std::cout << "[NS Direct] Matrix size: " << info.matrix_size
                  << " x " << ns_matrix_.n()
                  << ", nnz: " << info.nnz << "\n";
    }

    // Pin pressure DoF 0 to remove null space
    // (only for FIX modes 0 and 2)
    bool diagonal_pinned = false;
#if !defined(NS_PRESSURE_FIX) || NS_PRESSURE_FIX == 0 || NS_PRESSURE_FIX == 2
    if (!p_to_ns_map_.empty())
    {
        const auto pin_dof = p_to_ns_map_[0];
        diagonal_pinned = pin_pressure_dof(ns_matrix_, ns_rhs_, pin_dof, ns_locally_owned_, rank, verbose);
    }
    else
    {
        if (rank == 0)
            std::cerr << "[NS Direct] WARNING: p_to_ns_map is empty, cannot pin pressure!\n";
    }

    if (!diagonal_pinned && rank == 0)
        std::cerr << "[NS Direct] WARNING: Pressure diagonal not found - matrix may still be singular!\n";
#else
    if (verbose && rank == 0)
        std::cout << "[NS Direct] Skipping pressure pin (NS_PRESSURE_FIX="
                  << NS_PRESSURE_FIX << ")\n";
    (void)diagonal_pinned;
#endif

    const double rhs_norm = ns_rhs_.l2_norm();

    if (verbose && rank == 0)
        std::cout << "[NS Direct] Trying solvers in order of preference...\n";

    bool solved = false;

    if (!solved)
        solved = try_direct_solver("Amesos_Mumps", ns_matrix_, ns_rhs_, ns_solution_, rank, verbose);
    if (!solved)
        solved = try_direct_solver("Amesos_Umfpack", ns_matrix_, ns_rhs_, ns_solution_, rank, verbose);
    if (!solved)
        solved = try_direct_solver("Amesos_Superludist", ns_matrix_, ns_rhs_, ns_solution_, rank, verbose);
    if (!solved)
        solved = try_direct_solver("Amesos_Superlu", ns_matrix_, ns_rhs_, ns_solution_, rank, verbose);
    if (!solved)
        solved = try_direct_solver("Amesos_Klu", ns_matrix_, ns_rhs_, ns_solution_, rank, verbose);
    if (!solved && info.matrix_size <= LAPACK_SIZE_LIMIT)
        solved = try_direct_solver("Amesos_Lapack", ns_matrix_, ns_rhs_, ns_solution_, rank, verbose);

    info.converged = solved;
    info.iterations = solved ? 1 : 0;

    if (!solved && rank == 0)
        std::cerr << "[NS Direct] ALL DIRECT SOLVERS FAILED!\n";

    auto end = std::chrono::high_resolution_clock::now();
    info.solve_time = std::chrono::duration<double>(end - start).count();

    ns_constraints_.distribute(ns_solution_);

    // Compute true residual
    dealii::TrilinosWrappers::MPI::Vector residual(ns_rhs_);
    ns_matrix_.vmult(residual, ns_solution_);
    residual -= ns_rhs_;
    info.residual = (rhs_norm > 1e-14) ? residual.l2_norm() / rhs_norm : 0.0;

    if (verbose && rank == 0 && solved)
    {
        std::cout << "[NS Direct] rel_res: " << std::scientific << std::setprecision(2)
                  << info.residual << ", time: " << std::fixed << std::setprecision(2)
                  << info.solve_time << "s\n";
    }

    return info;
}


// ============================================================================
// extract_solutions() — Extract ux, uy, p from monolithic solution
// ============================================================================
template <int dim>
void NSSubsystem<dim>::extract_solutions()
{
    dealii::IndexSet needed_ns_indices(ns_solution_.size());
    for (auto it = ux_locally_owned_.begin(); it != ux_locally_owned_.end(); ++it)
        needed_ns_indices.add_index(ux_to_ns_map_[*it]);
    for (auto it = uy_locally_owned_.begin(); it != uy_locally_owned_.end(); ++it)
        needed_ns_indices.add_index(uy_to_ns_map_[*it]);
    for (auto it = p_locally_owned_.begin(); it != p_locally_owned_.end(); ++it)
        needed_ns_indices.add_index(p_to_ns_map_[*it]);
    needed_ns_indices.compress();

    dealii::LinearAlgebra::ReadWriteVector<double> ns_values(needed_ns_indices);
    ns_values.import_elements(ns_solution_, dealii::VectorOperation::insert);

    for (auto it = ux_locally_owned_.begin(); it != ux_locally_owned_.end(); ++it)
        ux_solution_[*it] = ns_values[ux_to_ns_map_[*it]];
    for (auto it = uy_locally_owned_.begin(); it != uy_locally_owned_.end(); ++it)
        uy_solution_[*it] = ns_values[uy_to_ns_map_[*it]];
    for (auto it = p_locally_owned_.begin(); it != p_locally_owned_.end(); ++it)
        p_solution_[*it] = ns_values[p_to_ns_map_[*it]];

    ux_solution_.compress(dealii::VectorOperation::insert);
    uy_solution_.compress(dealii::VectorOperation::insert);
    p_solution_.compress(dealii::VectorOperation::insert);
}


// ============================================================================
// PUBLIC: solve() — Main entry point
//
// 1. Direct solve with pressure pinning
// 2. Extract component solutions
// 3. Update ghosted vectors
// ============================================================================
template <int dim>
SolverInfo NSSubsystem<dim>::solve()
{
    const bool verbose = params_.solvers.ns.verbose;

    // Direct solver only for now
    SolverInfo info = solve_direct(verbose);

    // Extract component solutions from monolithic vector
    extract_solutions();

#if defined(NS_PRESSURE_FIX) && NS_PRESSURE_FIX == 3
    // ====================================================================
    // Post-solve pressure mean subtraction
    //
    // With no pinning, the pressure is determined only up to a constant.
    // Subtract the mean to enforce ∫p dx = 0.
    // ====================================================================
    {
        dealii::QGauss<dim> quad_p(fe_pressure_.degree + 2);
        const unsigned int nq = quad_p.size();

        dealii::FEValues<dim> fv_p(fe_pressure_, quad_p,
            dealii::update_values | dealii::update_JxW_values);

        // Need ghosted vector for get_function_values
        dealii::TrilinosWrappers::MPI::Vector p_ghost(
            p_locally_owned_, p_locally_relevant_, mpi_comm_);
        p_ghost = p_solution_;

        std::vector<double> p_vals(nq);
        double loc_integral = 0.0, loc_volume = 0.0;

        for (const auto& cell : p_dof_handler_.active_cell_iterators())
        {
            if (!cell->is_locally_owned()) continue;
            fv_p.reinit(cell);
            fv_p.get_function_values(p_ghost, p_vals);
            for (unsigned int q = 0; q < nq; ++q)
            {
                const double w = fv_p.JxW(q);
                loc_integral += p_vals[q] * w;
                loc_volume   += w;
            }
        }

        double g_integral = 0, g_volume = 0;
        MPI_Allreduce(&loc_integral, &g_integral, 1, MPI_DOUBLE, MPI_SUM, mpi_comm_);
        MPI_Allreduce(&loc_volume,   &g_volume,   1, MPI_DOUBLE, MPI_SUM, mpi_comm_);

        const double p_mean = g_integral / g_volume;

        // Subtract mean from owned pressure vector.
        // Use buffer to avoid Trilinos add/insert state tracking issues:
        // operator[] may return a proxy that triggers state changes on read.
        {
            const unsigned int n_owned = p_locally_owned_.n_elements();
            std::vector<dealii::types::global_dof_index> indices(n_owned);
            std::vector<double> values(n_owned);

            unsigned int idx = 0;
            for (auto it = p_locally_owned_.begin();
                 it != p_locally_owned_.end(); ++it, ++idx)
            {
                indices[idx] = *it;
                values[idx]  = p_ghost[*it] - p_mean;  // read from ghost copy
            }

            // Clean write to owned vector
            p_solution_ = 0;  // reset internal state
            for (unsigned int i = 0; i < n_owned; ++i)
                p_solution_[indices[i]] = values[i];
            p_solution_.compress(dealii::VectorOperation::insert);
        }

        pcout_ << "  [NS] pressure mean subtracted: " << std::scientific
               << std::setprecision(3) << p_mean << std::defaultfloat << "\n";
    }
#endif

    // Update ghosted vectors
    ux_relevant_ = ux_solution_;
    uy_relevant_ = uy_solution_;
    p_relevant_  = p_solution_;

    last_solve_info_ = info;

    if (verbose)
    {
        pcout_ << "  [NS] iterations=" << info.iterations
               << ", residual=" << std::scientific << info.residual
               << std::defaultfloat << "\n";
    }

    return info;
}


// ============================================================================
// Explicit instantiations
// ============================================================================
template SolverInfo NSSubsystem<2>::solve();
template SolverInfo NSSubsystem<3>::solve();
template SolverInfo NSSubsystem<2>::solve_direct(bool);
template SolverInfo NSSubsystem<3>::solve_direct(bool);
template void NSSubsystem<2>::extract_solutions();
template void NSSubsystem<3>::extract_solutions();