// ============================================================================
// solvers/ns_solver.cc - Parallel Navier-Stokes Solver Implementation
//
// FIX (2025-01-11): Direct solver for saddle-point systems
// FIX (2026-01-12): Added pressure pinning to remove null space
// FIX (2026-01-13): Don't override Block Schur defaults (use improved values)
// ============================================================================

#include "solvers/ns_solver.h"
#include "solvers/ns_block_preconditioner.h"

#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/read_write_vector.h>

#include <Epetra_CrsMatrix.h>

#include <chrono>
#include <iostream>
#include <iomanip>

// ============================================================================
// Block Schur solver (recommended for saddle-point)
// ============================================================================
SolverInfo solve_ns_system_schur_parallel(
    const dealii::TrilinosWrappers::SparseMatrix& matrix,
    const dealii::TrilinosWrappers::MPI::Vector& rhs,
    dealii::TrilinosWrappers::MPI::Vector& solution,
    const dealii::AffineConstraints<double>& constraints,
    const dealii::TrilinosWrappers::SparseMatrix& pressure_mass,
    const std::vector<dealii::types::global_dof_index>& ux_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& uy_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& p_to_ns_map,
    const dealii::IndexSet& ns_owned,
    const dealii::IndexSet& vel_owned,
    const dealii::IndexSet& p_owned,
    MPI_Comm mpi_comm,
    double viscosity,
    bool verbose)
{
    SolverInfo info;
    info.solver_name = "NS-FGMRES-Schur";
    info.matrix_size = matrix.m();
    info.nnz = matrix.n_nonzero_elements();

    int rank;
    MPI_Comm_rank(mpi_comm, &rank);

    const double rhs_norm = rhs.l2_norm();
    if (rhs_norm < 1e-14)
    {
        solution = 0;
        constraints.distribute(solution);
        info.converged = true;
        info.residual = 0.0;
        info.iterations = 0;
        return info;
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Create block Schur preconditioner
    // Uses improved defaults from constructor: inner_tolerance=1e-1, max_inner_iterations=20
    BlockSchurPreconditionerParallel preconditioner(
        matrix, pressure_mass,
        ux_to_ns_map, uy_to_ns_map, p_to_ns_map,
        ns_owned, vel_owned, p_owned,
        mpi_comm, viscosity);

    // DO NOT override defaults - they are already optimized in the constructor!
    // OLD (slow): preconditioner.inner_tolerance = 1e-3;
    // OLD (slow): preconditioner.max_inner_iterations = 500;

    // FGMRES (flexible GMRES) because preconditioner changes each iteration
    const double rel_tol = 1e-8;
    const double tol = rel_tol * rhs_norm;

    dealii::SolverControl solver_control(1500, tol, false, false);

    dealii::SolverFGMRES<dealii::TrilinosWrappers::MPI::Vector>::AdditionalData fgmres_data;
    fgmres_data.max_basis_size = 100;

    dealii::SolverFGMRES<dealii::TrilinosWrappers::MPI::Vector> solver(solver_control, fgmres_data);

    solution = 0;

    try
    {
        solver.solve(matrix, solution, rhs, preconditioner);
        info.converged = true;
        info.iterations = solver_control.last_step();
    }
    catch (const dealii::SolverControl::NoConvergence& e)
    {
        info.converged = false;
        info.iterations = e.last_step;
        if (rank == 0)
        {
            std::cout << "[NS Schur] FGMRES did not converge in " << e.last_step
                      << " iterations, res = " << e.last_residual << "\n";
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    info.solve_time = std::chrono::duration<double>(end - start).count();

    // Distribute constraints
    constraints.distribute(solution);

    // Compute true residual
    dealii::TrilinosWrappers::MPI::Vector residual(rhs);
    matrix.vmult(residual, solution);
    residual -= rhs;
    info.residual = residual.l2_norm() / rhs_norm;

    if (verbose && rank == 0)
    {
        std::cout << "[NS Schur] FGMRES: " << info.iterations << " iters"
                  << ", inner A: " << preconditioner.n_iterations_A
                  << ", inner S: " << preconditioner.n_iterations_S
                  << ", rel_res: " << std::scientific << std::setprecision(2) << info.residual
                  << ", time: " << std::fixed << std::setprecision(2) << info.solve_time << "s\n";
    }

    return info;
}

// ============================================================================
// Helper: Try a specific direct solver with verbose error reporting
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
// Helper: Pin pressure DoF in matrix to remove null space
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
            {
                std::cerr << "[NS Direct] ERROR: No diagonal entry for pressure DoF " << pin_dof << "!\n";
            }
        }

        rhs[pin_dof] = 0.0;
    }

    rhs.compress(dealii::VectorOperation::insert);
    return diagonal_found;
}

// ============================================================================
// Direct solver WITH pressure pinning (RECOMMENDED)
// ============================================================================
SolverInfo solve_ns_system_direct_parallel(
    const dealii::TrilinosWrappers::SparseMatrix& matrix,
    const dealii::TrilinosWrappers::MPI::Vector& rhs,
    dealii::TrilinosWrappers::MPI::Vector& solution,
    const dealii::AffineConstraints<double>& constraints,
    const std::vector<dealii::types::global_dof_index>& p_to_ns_map,
    const dealii::IndexSet& ns_owned,
    MPI_Comm mpi_comm,
    bool verbose)
{
    constexpr unsigned int LAPACK_SIZE_LIMIT = 50000;

    SolverInfo info;
    info.solver_name = "NS-Direct";
    info.matrix_size = matrix.m();
    info.nnz = matrix.n_nonzero_elements();
    info.used_direct = true;

    int rank;
    MPI_Comm_rank(mpi_comm, &rank);

    auto& mutable_matrix = const_cast<dealii::TrilinosWrappers::SparseMatrix&>(matrix);
    auto& mutable_rhs = const_cast<dealii::TrilinosWrappers::MPI::Vector&>(rhs);

    const double rhs_norm_before_pinning = rhs.l2_norm();
    if (rhs_norm_before_pinning < 1e-14)
    {
        solution = 0;
        constraints.distribute(solution);
        info.converged = true;
        info.residual = 0.0;
        info.iterations = 0;
        return info;
    }

    auto start = std::chrono::high_resolution_clock::now();

    if (verbose && rank == 0)
    {
        std::cout << "[NS Direct] Matrix size: " << info.matrix_size
                  << " x " << matrix.n()
                  << ", nnz: " << info.nnz << "\n";
    }

    bool diagonal_pinned = false;
    if (!p_to_ns_map.empty())
    {
        const auto pin_dof = p_to_ns_map[0];
        diagonal_pinned = pin_pressure_dof(mutable_matrix, mutable_rhs, pin_dof, ns_owned, rank, verbose);
    }
    else
    {
        if (rank == 0)
            std::cerr << "[NS Direct] WARNING: p_to_ns_map is empty, cannot pin pressure!\n";
    }

    if (!diagonal_pinned && rank == 0)
    {
        std::cerr << "[NS Direct] WARNING: Pressure diagonal not found - matrix may still be singular!\n";
    }

    const double rhs_norm = mutable_rhs.l2_norm();

    if (verbose && rank == 0)
        std::cout << "[NS Direct] Trying solvers in order of preference...\n";

    bool solved = false;

    if (!solved)
        solved = try_direct_solver("Amesos_Mumps", mutable_matrix, mutable_rhs, solution, rank, verbose);
    if (!solved)
        solved = try_direct_solver("Amesos_Umfpack", mutable_matrix, mutable_rhs, solution, rank, verbose);
    if (!solved)
        solved = try_direct_solver("Amesos_Superludist", mutable_matrix, mutable_rhs, solution, rank, verbose);
    if (!solved)
        solved = try_direct_solver("Amesos_Superlu", mutable_matrix, mutable_rhs, solution, rank, verbose);
    if (!solved)
        solved = try_direct_solver("Amesos_Klu", mutable_matrix, mutable_rhs, solution, rank, verbose);
    if (!solved && info.matrix_size <= LAPACK_SIZE_LIMIT)
        solved = try_direct_solver("Amesos_Lapack", mutable_matrix, mutable_rhs, solution, rank, verbose);

    if (solved)
    {
        info.converged = true;
        info.iterations = 1;
    }
    else
    {
        info.converged = false;
        info.iterations = 0;
        if (rank == 0)
            std::cerr << "[NS Direct] ALL DIRECT SOLVERS FAILED!\n";
    }

    auto end = std::chrono::high_resolution_clock::now();
    info.solve_time = std::chrono::duration<double>(end - start).count();

    constraints.distribute(solution);

    dealii::TrilinosWrappers::MPI::Vector residual(mutable_rhs);
    mutable_matrix.vmult(residual, solution);
    residual -= mutable_rhs;
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
// Direct solver WITHOUT pressure pinning (LEGACY)
// ============================================================================
SolverInfo solve_ns_system_direct_parallel(
    const dealii::TrilinosWrappers::SparseMatrix& matrix,
    const dealii::TrilinosWrappers::MPI::Vector& rhs,
    dealii::TrilinosWrappers::MPI::Vector& solution,
    const dealii::AffineConstraints<double>& constraints,
    MPI_Comm mpi_comm,
    bool verbose)
{
    constexpr unsigned int LAPACK_SIZE_LIMIT = 50000;

    SolverInfo info;
    info.solver_name = "NS-Direct";
    info.matrix_size = matrix.m();
    info.nnz = matrix.n_nonzero_elements();
    info.used_direct = true;

    int rank;
    MPI_Comm_rank(mpi_comm, &rank);

    if (rank == 0)
    {
        std::cout << "[NS Direct] WARNING: Using legacy solver without pressure pinning.\n";
    }

    const double rhs_norm = rhs.l2_norm();
    if (rhs_norm < 1e-14)
    {
        solution = 0;
        constraints.distribute(solution);
        info.converged = true;
        info.residual = 0.0;
        info.iterations = 0;
        return info;
    }

    auto start = std::chrono::high_resolution_clock::now();

    bool solved = false;

    if (!solved)
        solved = try_direct_solver("Amesos_Umfpack", matrix, rhs, solution, rank, verbose);
    if (!solved)
        solved = try_direct_solver("Amesos_Mumps", matrix, rhs, solution, rank, verbose);
    if (!solved)
        solved = try_direct_solver("Amesos_Superludist", matrix, rhs, solution, rank, verbose);
    if (!solved)
        solved = try_direct_solver("Amesos_Superlu", matrix, rhs, solution, rank, verbose);
    if (!solved)
        solved = try_direct_solver("Amesos_Klu", matrix, rhs, solution, rank, verbose);
    if (!solved && info.matrix_size <= LAPACK_SIZE_LIMIT)
        solved = try_direct_solver("Amesos_Lapack", matrix, rhs, solution, rank, verbose);

    if (solved)
    {
        info.converged = true;
        info.iterations = 1;
    }
    else
    {
        info.converged = false;
        info.iterations = 0;
        if (rank == 0)
            std::cerr << "[NS Direct] ALL DIRECT SOLVERS FAILED!\n";
    }

    auto end = std::chrono::high_resolution_clock::now();
    info.solve_time = std::chrono::duration<double>(end - start).count();

    constraints.distribute(solution);

    dealii::TrilinosWrappers::MPI::Vector residual(rhs);
    matrix.vmult(residual, solution);
    residual -= rhs;
    info.residual = residual.l2_norm() / rhs_norm;

    return info;
}

// ============================================================================
// Extract solutions
// ============================================================================
void extract_ns_solutions_parallel(
    const dealii::TrilinosWrappers::MPI::Vector& ns_solution,
    const std::vector<dealii::types::global_dof_index>& ux_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& uy_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& p_to_ns_map,
    const dealii::IndexSet& ux_owned,
    const dealii::IndexSet& uy_owned,
    const dealii::IndexSet& p_owned,
    const dealii::IndexSet& ns_owned,
    const dealii::IndexSet& ns_relevant,
    dealii::TrilinosWrappers::MPI::Vector& ux_solution,
    dealii::TrilinosWrappers::MPI::Vector& uy_solution,
    dealii::TrilinosWrappers::MPI::Vector& p_solution,
    MPI_Comm mpi_comm)
{
    if (ux_solution.size() != ux_to_ns_map.size())
        ux_solution.reinit(ux_owned, mpi_comm);
    if (uy_solution.size() != uy_to_ns_map.size())
        uy_solution.reinit(uy_owned, mpi_comm);
    if (p_solution.size() != p_to_ns_map.size())
        p_solution.reinit(p_owned, mpi_comm);

    dealii::IndexSet needed_ns_indices(ns_solution.size());
    for (auto it = ux_owned.begin(); it != ux_owned.end(); ++it)
        needed_ns_indices.add_index(ux_to_ns_map[*it]);
    for (auto it = uy_owned.begin(); it != uy_owned.end(); ++it)
        needed_ns_indices.add_index(uy_to_ns_map[*it]);
    for (auto it = p_owned.begin(); it != p_owned.end(); ++it)
        needed_ns_indices.add_index(p_to_ns_map[*it]);
    needed_ns_indices.compress();

    dealii::LinearAlgebra::ReadWriteVector<double> ns_values(needed_ns_indices);
    ns_values.import_elements(ns_solution, dealii::VectorOperation::insert);

    for (auto it = ux_owned.begin(); it != ux_owned.end(); ++it)
        ux_solution[*it] = ns_values[ux_to_ns_map[*it]];
    for (auto it = uy_owned.begin(); it != uy_owned.end(); ++it)
        uy_solution[*it] = ns_values[uy_to_ns_map[*it]];
    for (auto it = p_owned.begin(); it != p_owned.end(); ++it)
        p_solution[*it] = ns_values[p_to_ns_map[*it]];

    ux_solution.compress(dealii::VectorOperation::insert);
    uy_solution.compress(dealii::VectorOperation::insert);
    p_solution.compress(dealii::VectorOperation::insert);
}

// ============================================================================
// Assemble pressure mass matrix
// ============================================================================
template <int dim>
void assemble_pressure_mass_matrix_parallel(
    const dealii::DoFHandler<dim>& p_dof_handler,
    const dealii::AffineConstraints<double>& p_constraints,
    const dealii::IndexSet& p_owned,
    const dealii::IndexSet& p_relevant,
    dealii::TrilinosWrappers::SparseMatrix& pressure_mass,
    MPI_Comm mpi_comm)
{
    const dealii::FiniteElement<dim>& fe = p_dof_handler.get_fe();
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

    dealii::DynamicSparsityPattern dsp(p_relevant);
    dealii::DoFTools::make_sparsity_pattern(p_dof_handler, dsp, p_constraints, false);

    dealii::TrilinosWrappers::SparsityPattern sp;
    sp.reinit(p_owned, p_owned, dsp, mpi_comm);

    pressure_mass.reinit(sp);

    dealii::QGauss<dim> quadrature(fe.degree + 1);
    dealii::FEValues<dim> fe_values(fe, quadrature,
        dealii::update_values | dealii::update_JxW_values);

    const unsigned int n_q_points = quadrature.size();
    dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto& cell : p_dof_handler.active_cell_iterators())
    {
        if (cell->is_locally_owned())
        {
            fe_values.reinit(cell);
            cell_matrix = 0;

            for (unsigned int q = 0; q < n_q_points; ++q)
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        cell_matrix(i, j) += fe_values.shape_value(i, q) *
                                             fe_values.shape_value(j, q) *
                                             fe_values.JxW(q);

            cell->get_dof_indices(local_dof_indices);
            p_constraints.distribute_local_to_global(cell_matrix, local_dof_indices, pressure_mass);
        }
    }

    pressure_mass.compress(dealii::VectorOperation::add);
}

// ============================================================================
// Unified NS solver with auto-selection
// ============================================================================
SolverInfo solve_ns_system(
    const dealii::TrilinosWrappers::SparseMatrix& matrix,
    const dealii::TrilinosWrappers::MPI::Vector& rhs,
    dealii::TrilinosWrappers::MPI::Vector& solution,
    const dealii::AffineConstraints<double>& constraints,
    const dealii::TrilinosWrappers::SparseMatrix& pressure_mass,
    const std::vector<dealii::types::global_dof_index>& ux_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& uy_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& p_to_ns_map,
    const dealii::IndexSet& ns_owned,
    const dealii::IndexSet& vel_owned,
    const dealii::IndexSet& p_owned,
    MPI_Comm mpi_comm,
    double viscosity,
    bool verbose,
    bool force_direct,
    bool force_iterative,
    unsigned int direct_threshold)
{
    const unsigned int n_dofs = matrix.m();

    int rank;
    MPI_Comm_rank(mpi_comm, &rank);

    bool use_direct = false;

    if (force_direct && force_iterative)
    {
        if (rank == 0)
            std::cerr << "[NS] Warning: both force_direct and force_iterative set, using auto-selection\n";
        use_direct = (n_dofs < direct_threshold);
    }
    else if (force_direct)
    {
        use_direct = true;
    }
    else if (force_iterative)
    {
        use_direct = false;
    }
    else
    {
        use_direct = (n_dofs < direct_threshold);
    }

    if (verbose && rank == 0)
    {
        std::cout << "[NS] Auto-select: " << n_dofs << " DoFs "
                  << (use_direct ? "<" : ">=") << " " << direct_threshold
                  << " threshold -> " << (use_direct ? "DIRECT" : "ITERATIVE") << "\n";
    }

    SolverInfo info;

    if (use_direct)
    {
        info = solve_ns_system_direct_parallel(
            matrix, rhs, solution, constraints,
            p_to_ns_map, ns_owned, mpi_comm, verbose);

        if (!info.converged && !force_direct)
        {
            if (rank == 0)
                std::cout << "[NS] Direct solver failed, falling back to iterative\n";

            info = solve_ns_system_schur_parallel(
                matrix, rhs, solution, constraints,
                pressure_mass,
                ux_to_ns_map, uy_to_ns_map, p_to_ns_map,
                ns_owned, vel_owned, p_owned,
                mpi_comm, viscosity, verbose);
        }
    }
    else
    {
        info = solve_ns_system_schur_parallel(
            matrix, rhs, solution, constraints,
            pressure_mass,
            ux_to_ns_map, uy_to_ns_map, p_to_ns_map,
            ns_owned, vel_owned, p_owned,
            mpi_comm, viscosity, verbose);

        if (!info.converged && !force_iterative && n_dofs < 2 * direct_threshold)
        {
            if (rank == 0)
                std::cout << "[NS] Iterative solver failed, falling back to direct\n";

            info = solve_ns_system_direct_parallel(
                matrix, rhs, solution, constraints,
                p_to_ns_map, ns_owned, mpi_comm, verbose);
        }
    }

    return info;
}

template void assemble_pressure_mass_matrix_parallel<2>(
    const dealii::DoFHandler<2>&,
    const dealii::AffineConstraints<double>&,
    const dealii::IndexSet&,
    const dealii::IndexSet&,
    dealii::TrilinosWrappers::SparseMatrix&,
    MPI_Comm);

template void assemble_pressure_mass_matrix_parallel<3>(
    const dealii::DoFHandler<3>&,
    const dealii::AffineConstraints<double>&,
    const dealii::IndexSet&,
    const dealii::IndexSet&,
    dealii::TrilinosWrappers::SparseMatrix&,
    MPI_Comm);