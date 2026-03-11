// ============================================================================
// diagnostics/parallel_data.h - Per-Step Parallel Performance Data
//
// Captures all metrics needed for parallel computing analysis:
//   - Assembly vs solve timing breakdown per subsystem
//   - Sparsity pattern metrics (nnz per matrix)
//   - Load balance (local cells, ghost cells, DoFs per rank)
//   - MPI-reduced imbalance ratios
//   - Memory usage per rank
//
// Each rank records its local values. The ParallelDiagnosticsLogger
// performs MPI reductions and writes the combined data.
// ============================================================================
#ifndef PARALLEL_DATA_H
#define PARALLEL_DATA_H

#include <mpi.h>
#include <algorithm>
#include <cmath>

/**
 * @brief Per-rank timing breakdown for a single time step
 *
 * Each subsystem has assembly + solve times measured separately.
 * These are LOCAL to each MPI rank.
 */
struct ParallelStepData
{
    // ========================================================================
    // Time step info
    // ========================================================================
    unsigned int step = 0;
    double time = 0.0;
    int mpi_size = 1;

    // ========================================================================
    // Per-subsystem timing breakdown (seconds, THIS rank)
    //   assembly = matrix + RHS construction
    //   solve    = linear solver (preconditioner setup + iterations)
    // ========================================================================

    // Cahn-Hilliard
    double ch_assemble_time = 0.0;
    double ch_solve_time = 0.0;

    // Poisson (per Picard iteration, accumulated)
    double poisson_assemble_time = 0.0;
    double poisson_solve_time = 0.0;

    // Magnetization (DG transport or L2 projection)
    double mag_time = 0.0;           // Total mag time (projection has no assembly/solve split)

    // Navier-Stokes
    double ns_assemble_time = 0.0;
    double ns_solve_time = 0.0;

    // Overhead
    double diagnostics_time = 0.0;   // compute_system_diagnostics()
    double amr_time = 0.0;           // refine_mesh()
    double output_time = 0.0;        // VTK output
    double communication_time = 0.0; // Ghost vector updates (approximate)

    // Total step time on this rank
    double step_total = 0.0;

    // ========================================================================
    // BGS iteration counts (picard_iterations kept for CSV compatibility, always 0)
    // ========================================================================
    unsigned int picard_iterations = 0;  // Legacy: monolithic system has no Picard
    unsigned int bgs_iterations = 0;

    // ========================================================================
    // Solver iteration counts (on this rank — same across ranks for global solvers)
    // ========================================================================
    unsigned int ch_solver_iters = 0;
    unsigned int poisson_solver_iters = 0;
    unsigned int mag_solver_iters = 0;
    unsigned int ns_solver_iters = 0;

    // ========================================================================
    // Sparsity metrics (LOCAL to this rank's portion)
    //   nnz = number of nonzero entries in the locally owned rows
    //   For Trilinos distributed matrices: local_nnz = matrix.local_nnz()
    // ========================================================================
    unsigned long long ch_nnz = 0;
    unsigned long long poisson_nnz = 0;
    unsigned long long mag_nnz = 0;
    unsigned long long ns_nnz = 0;

    // Global nnz (sum across ranks — but note Trilinos counts local entries)
    unsigned long long ch_nnz_global = 0;
    unsigned long long poisson_nnz_global = 0;
    unsigned long long mag_nnz_global = 0;
    unsigned long long ns_nnz_global = 0;

    // Bandwidth (max |i-j| for nonzero entries, per matrix, LOCAL to this rank)
    unsigned int ch_bandwidth = 0;
    unsigned int poisson_bandwidth = 0;
    unsigned int mag_bandwidth = 0;
    unsigned int ns_bandwidth = 0;

    // Global bandwidth (max across all ranks)
    unsigned int ch_bandwidth_global = 0;
    unsigned int poisson_bandwidth_global = 0;
    unsigned int mag_bandwidth_global = 0;
    unsigned int ns_bandwidth_global = 0;

    // ========================================================================
    // Load balance metrics (per-rank)
    // ========================================================================
    unsigned int local_cells = 0;       // Locally owned active cells
    unsigned int ghost_cells = 0;       // Ghost cells (level 0 artificial)
    unsigned int local_dofs_theta = 0;  // CH DoFs on this rank
    unsigned int local_dofs_phi = 0;    // Poisson DoFs on this rank
    unsigned int local_dofs_M = 0;      // Magnetization DoFs on this rank
    unsigned int local_dofs_ns = 0;     // NS (ux+uy+p) DoFs on this rank
    unsigned int total_local_dofs = 0;  // Sum of all subsystem local DoFs

    // Global counts (for reference)
    unsigned int global_cells = 0;
    unsigned int global_dofs = 0;

    // ========================================================================
    // MPI-reduced metrics (computed by logger via MPI_Allreduce)
    // ========================================================================

    // Step time across ranks
    double step_time_min = 0.0;
    double step_time_max = 0.0;
    double step_time_avg = 0.0;
    double imbalance_ratio = 1.0;   // max / avg (1.0 = perfect balance)

    // Per-subsystem imbalance
    double ch_imbalance = 1.0;
    double poisson_imbalance = 1.0;
    double mag_imbalance = 1.0;
    double ns_imbalance = 1.0;

    // Cell/DoF balance
    unsigned int cells_min = 0;
    unsigned int cells_max = 0;
    unsigned int dofs_min = 0;
    unsigned int dofs_max = 0;

    // ========================================================================
    // Memory (MB, this rank)
    // ========================================================================
    double memory_mb = 0.0;
    double memory_min = 0.0;
    double memory_max = 0.0;

    // ========================================================================
    // AMR level info
    // ========================================================================
    unsigned int amr_min_level = 0;
    unsigned int amr_max_level = 0;

    // ========================================================================
    // Helper: compute derived imbalance metrics via MPI
    // Call this AFTER all local values are set.
    // ========================================================================
    void compute_mpi_reductions(MPI_Comm comm)
    {
        int rank, size;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
        mpi_size = size;

        // --- Step time imbalance ---
        MPI_Allreduce(&step_total, &step_time_min, 1, MPI_DOUBLE, MPI_MIN, comm);
        MPI_Allreduce(&step_total, &step_time_max, 1, MPI_DOUBLE, MPI_MAX, comm);
        double step_sum = 0.0;
        MPI_Allreduce(&step_total, &step_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
        step_time_avg = step_sum / size;
        imbalance_ratio = (step_time_avg > 1e-12) ? step_time_max / step_time_avg : 1.0;

        // --- Per-subsystem imbalance (max/avg) ---
        auto compute_imbalance = [&](double local_time) -> double {
            double t_max = 0.0, t_sum = 0.0;
            MPI_Allreduce(&local_time, &t_max, 1, MPI_DOUBLE, MPI_MAX, comm);
            MPI_Allreduce(&local_time, &t_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
            double t_avg = t_sum / size;
            return (t_avg > 1e-12) ? t_max / t_avg : 1.0;
        };

        double ch_total = ch_assemble_time + ch_solve_time;
        double poisson_total = poisson_assemble_time + poisson_solve_time;
        double ns_total = ns_assemble_time + ns_solve_time;

        ch_imbalance = compute_imbalance(ch_total);
        poisson_imbalance = compute_imbalance(poisson_total);
        mag_imbalance = compute_imbalance(mag_time);
        ns_imbalance = compute_imbalance(ns_total);

        // --- Cell/DoF balance ---
        MPI_Allreduce(&local_cells, &cells_min, 1, MPI_UNSIGNED, MPI_MIN, comm);
        MPI_Allreduce(&local_cells, &cells_max, 1, MPI_UNSIGNED, MPI_MAX, comm);
        MPI_Allreduce(&total_local_dofs, &dofs_min, 1, MPI_UNSIGNED, MPI_MIN, comm);
        MPI_Allreduce(&total_local_dofs, &dofs_max, 1, MPI_UNSIGNED, MPI_MAX, comm);

        // --- Memory balance ---
        MPI_Allreduce(&memory_mb, &memory_min, 1, MPI_DOUBLE, MPI_MIN, comm);
        MPI_Allreduce(&memory_mb, &memory_max, 1, MPI_DOUBLE, MPI_MAX, comm);

        // --- Global sparsity (sum of local nnz) ---
        MPI_Allreduce(&ch_nnz, &ch_nnz_global, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);
        MPI_Allreduce(&poisson_nnz, &poisson_nnz_global, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);
        MPI_Allreduce(&mag_nnz, &mag_nnz_global, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);
        MPI_Allreduce(&ns_nnz, &ns_nnz_global, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);

        // --- Global bandwidth (max across ranks) ---
        MPI_Allreduce(&ch_bandwidth, &ch_bandwidth_global, 1, MPI_UNSIGNED, MPI_MAX, comm);
        MPI_Allreduce(&poisson_bandwidth, &poisson_bandwidth_global, 1, MPI_UNSIGNED, MPI_MAX, comm);
        MPI_Allreduce(&mag_bandwidth, &mag_bandwidth_global, 1, MPI_UNSIGNED, MPI_MAX, comm);
        MPI_Allreduce(&ns_bandwidth, &ns_bandwidth_global, 1, MPI_UNSIGNED, MPI_MAX, comm);
    }
};

#endif // PARALLEL_DATA_H
