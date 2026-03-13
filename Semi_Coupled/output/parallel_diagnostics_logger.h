// ============================================================================
// output/parallel_diagnostics_logger.h - Parallel Performance Logger
//
// Records per-step parallel computing metrics to parallel_diagnostics.csv:
//   - Assembly vs solve timing breakdown per subsystem (per-rank)
//   - Sparsity pattern metrics (nnz per matrix, local + global)
//   - Load balance (cells, DoFs, ghost cells per rank)
//   - MPI-reduced imbalance ratios (max/avg)
//   - Memory usage per rank
//
// DESIGN:
//   - Each rank records local values into ParallelStepData
//   - compute_mpi_reductions() performs MPI_Allreduce for cross-rank stats
//   - Rank 0 writes the CSV with local + reduced columns
//
// For multi-rank analysis, enable --parallel-diag-all-ranks to write
// per-rank CSV files (parallel_diagnostics_rank_N.csv).
//
// MPI rank-0 only for primary output. Optional per-rank files.
// ============================================================================
#ifndef PARALLEL_DIAGNOSTICS_LOGGER_H
#define PARALLEL_DIAGNOSTICS_LOGGER_H

#include "diagnostics/parallel_data.h"
#include "utilities/mpi_tools.h"
#include "utilities/tools.h"

#include <string>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

#ifdef __APPLE__
#include <mach/mach.h>
#elif defined(__linux__)
#include <sys/resource.h>
#endif

/**
 * @brief Logger for parallel computing performance metrics
 *
 * Usage:
 *   ParallelDiagnosticsLogger plog(output_dir, params, comm);
 *   for (step...) {
 *       ParallelStepData pdata;
 *       // ... fill timing, sparsity, load balance ...
 *       pdata.compute_mpi_reductions(comm);
 *       plog.log_step(pdata);
 *   }
 *   // Destructor writes summary
 */
class ParallelDiagnosticsLogger
{
public:
    /**
     * @brief Constructor - opens CSV and writes header
     * @param output_dir Directory for output file
     * @param params Simulation parameters (for header stamp)
     * @param comm MPI communicator
     * @param write_per_rank If true, each rank writes its own CSV
     */
    ParallelDiagnosticsLogger(const std::string& output_dir,
                               const Parameters& params,
                               MPI_Comm comm = MPI_COMM_WORLD,
                               bool write_per_rank = false)
        : output_dir_(output_dir)
        , is_root_(MPIUtils::is_root(comm))
        , write_per_rank_(write_per_rank)
    {
        MPI_Comm_rank(comm, &rank_);
        MPI_Comm_size(comm, &n_ranks_);

        // Primary file (rank 0 only)
        if (is_root_)
        {
            file_.open(output_dir + "/parallel_diagnostics.csv");
            if (!file_.is_open())
                throw std::runtime_error("Could not open parallel_diagnostics.csv");

            write_header(file_);
            file_ << get_csv_header_stamp(params) << "\n";
            file_.flush();
        }

        // Per-rank files (optional)
        if (write_per_rank_)
        {
            std::string rank_file = output_dir + "/parallel_diagnostics_rank_"
                                    + std::to_string(rank_) + ".csv";
            rank_file_.open(rank_file);
            if (rank_file_.is_open())
            {
                write_rank_header(rank_file_);
                rank_file_.flush();
            }
        }
    }

    /**
     * @brief Destructor - writes summary and closes files
     */
    ~ParallelDiagnosticsLogger()
    {
        if (is_root_ && file_.is_open())
        {
            write_summary();
            file_.close();
        }
        if (rank_file_.is_open())
        {
            rank_file_.close();
        }
    }

    // No copy
    ParallelDiagnosticsLogger(const ParallelDiagnosticsLogger&) = delete;
    ParallelDiagnosticsLogger& operator=(const ParallelDiagnosticsLogger&) = delete;

    /**
     * @brief Log parallel diagnostics for one step
     *
     * IMPORTANT: Call pdata.compute_mpi_reductions(comm) BEFORE this!
     */
    void log_step(const ParallelStepData& d)
    {
        step_count_++;
        last_step_ = d.step;

        // Accumulate for summary
        cumul_ch_assemble_ += d.ch_assemble_time;
        cumul_ch_solve_ += d.ch_solve_time;
        cumul_poisson_assemble_ += d.poisson_assemble_time;
        cumul_poisson_solve_ += d.poisson_solve_time;
        cumul_mag_ += d.mag_time;
        cumul_ns_assemble_ += d.ns_assemble_time;
        cumul_ns_solve_ += d.ns_solve_time;
        cumul_diagnostics_ += d.diagnostics_time;
        cumul_total_ += d.step_total;
        max_imbalance_ = std::max(max_imbalance_, d.imbalance_ratio);
        sum_imbalance_ += d.imbalance_ratio;

        // Write rank 0 row
        if (is_root_)
        {
            write_row(file_, d);
            if (step_count_ % 10 == 0)
                file_.flush();
        }

        // Write per-rank row
        if (write_per_rank_ && rank_file_.is_open())
        {
            write_rank_row(rank_file_, d);
            if (step_count_ % 10 == 0)
                rank_file_.flush();
        }
    }

    void flush()
    {
        if (is_root_ && file_.is_open())
            file_.flush();
        if (rank_file_.is_open())
            rank_file_.flush();
    }

private:
    std::string output_dir_;
    bool is_root_;
    bool write_per_rank_;
    int rank_ = 0;
    int n_ranks_ = 1;

    std::ofstream file_;       // Primary (rank 0)
    std::ofstream rank_file_;  // Per-rank (optional)

    unsigned int step_count_ = 0;
    unsigned int last_step_ = 0;

    // Cumulative totals for summary
    double cumul_ch_assemble_ = 0.0;
    double cumul_ch_solve_ = 0.0;
    double cumul_poisson_assemble_ = 0.0;
    double cumul_poisson_solve_ = 0.0;
    double cumul_mag_ = 0.0;
    double cumul_ns_assemble_ = 0.0;
    double cumul_ns_solve_ = 0.0;
    double cumul_diagnostics_ = 0.0;
    double cumul_total_ = 0.0;
    double max_imbalance_ = 1.0;
    double sum_imbalance_ = 0.0;

    // ========================================================================
    // Primary CSV (rank 0): timing breakdown + MPI-reduced imbalance
    // ========================================================================
    void write_header(std::ofstream& f)
    {
        f << "step,time,mpi_size,"
          // Timing breakdown (rank 0 local)
          << "ch_assemble,ch_solve,"
          << "poisson_assemble,poisson_solve,"
          << "mag_time,"
          << "ns_assemble,ns_solve,"
          << "diagnostics_time,amr_time,output_time,"
          << "step_total,"
          // Picard/BGS
          << "picard_iters,bgs_iters,"
          // Solver iterations
          << "ch_solver_iters,poisson_solver_iters,mag_solver_iters,ns_solver_iters,"
          // Sparsity (global nnz)
          << "ch_nnz,poisson_nnz,mag_nnz,ns_nnz,"
          // Load balance (rank 0 local)
          << "local_cells,ghost_cells,"
          << "local_dofs_theta,local_dofs_phi,local_dofs_M,local_dofs_ns,"
          << "global_cells,global_dofs,"
          // MPI-reduced: step time
          << "step_time_min,step_time_max,step_time_avg,imbalance_ratio,"
          // MPI-reduced: per-subsystem imbalance
          << "ch_imbalance,poisson_imbalance,mag_imbalance,ns_imbalance,"
          // MPI-reduced: cells/DoFs balance
          << "cells_min,cells_max,dofs_min,dofs_max,"
          // Memory
          << "memory_mb,memory_min,memory_max,"
          // AMR
          << "amr_min_level,amr_max_level,"
          // Bandwidth (global max |i-j|)
          << "ch_bandwidth,poisson_bandwidth,mag_bandwidth,ns_bandwidth"
          << "\n";
    }

    void write_row(std::ofstream& f, const ParallelStepData& d)
    {
        f << d.step << ","
          << std::scientific << std::setprecision(6) << d.time << ","
          << d.mpi_size << ","
          // Timing
          << std::fixed << std::setprecision(6)
          << d.ch_assemble_time << "," << d.ch_solve_time << ","
          << d.poisson_assemble_time << "," << d.poisson_solve_time << ","
          << d.mag_time << ","
          << d.ns_assemble_time << "," << d.ns_solve_time << ","
          << d.diagnostics_time << "," << d.amr_time << "," << d.output_time << ","
          << d.step_total << ","
          // Picard/BGS
          << d.picard_iterations << "," << d.bgs_iterations << ","
          // Solver iterations
          << d.ch_solver_iters << "," << d.poisson_solver_iters << ","
          << d.mag_solver_iters << "," << d.ns_solver_iters << ","
          // Sparsity (global)
          << d.ch_nnz_global << "," << d.poisson_nnz_global << ","
          << d.mag_nnz_global << "," << d.ns_nnz_global << ","
          // Load balance (local)
          << d.local_cells << "," << d.ghost_cells << ","
          << d.local_dofs_theta << "," << d.local_dofs_phi << ","
          << d.local_dofs_M << "," << d.local_dofs_ns << ","
          << d.global_cells << "," << d.global_dofs << ","
          // MPI-reduced: step time
          << std::setprecision(6)
          << d.step_time_min << "," << d.step_time_max << ","
          << d.step_time_avg << ","
          << std::setprecision(4) << d.imbalance_ratio << ","
          // MPI-reduced: subsystem imbalance
          << d.ch_imbalance << "," << d.poisson_imbalance << ","
          << d.mag_imbalance << "," << d.ns_imbalance << ","
          // Cell/DoF balance
          << d.cells_min << "," << d.cells_max << ","
          << d.dofs_min << "," << d.dofs_max << ","
          // Memory
          << std::setprecision(1)
          << d.memory_mb << "," << d.memory_min << "," << d.memory_max << ","
          // AMR
          << d.amr_min_level << "," << d.amr_max_level << ","
          // Bandwidth
          << d.ch_bandwidth_global << "," << d.poisson_bandwidth_global << ","
          << d.mag_bandwidth_global << "," << d.ns_bandwidth_global
          << "\n";
    }

    // ========================================================================
    // Per-rank CSV: timing details for each rank individually
    // ========================================================================
    void write_rank_header(std::ofstream& f)
    {
        f << "step,time,rank,"
          << "ch_assemble,ch_solve,"
          << "poisson_assemble,poisson_solve,"
          << "mag_time,"
          << "ns_assemble,ns_solve,"
          << "diagnostics_time,step_total,"
          << "local_cells,ghost_cells,total_local_dofs,"
          << "ch_nnz_local,poisson_nnz_local,mag_nnz_local,ns_nnz_local,"
          << "memory_mb"
          << "\n";
    }

    void write_rank_row(std::ofstream& f, const ParallelStepData& d)
    {
        f << d.step << ","
          << std::scientific << std::setprecision(6) << d.time << ","
          << rank_ << ","
          << std::fixed << std::setprecision(6)
          << d.ch_assemble_time << "," << d.ch_solve_time << ","
          << d.poisson_assemble_time << "," << d.poisson_solve_time << ","
          << d.mag_time << ","
          << d.ns_assemble_time << "," << d.ns_solve_time << ","
          << d.diagnostics_time << "," << d.step_total << ","
          << d.local_cells << "," << d.ghost_cells << "," << d.total_local_dofs << ","
          << d.ch_nnz << "," << d.poisson_nnz << ","
          << d.mag_nnz << "," << d.ns_nnz << ","
          << std::setprecision(1) << d.memory_mb
          << "\n";
    }

    // ========================================================================
    // Summary (rank 0, at end of simulation)
    // ========================================================================
    void write_summary()
    {
        if (step_count_ == 0) return;

        double avg_imbalance = sum_imbalance_ / step_count_;

        file_ << "\n# ========== PARALLEL PERFORMANCE SUMMARY ==========\n";
        file_ << "# Steps: " << step_count_ << "\n";
        file_ << "# MPI ranks: " << n_ranks_ << "\n";
        file_ << "#\n";
        file_ << "# Cumulative timing (rank 0, seconds):\n";
        file_ << "#   CH  assembly: " << std::fixed << std::setprecision(2) << cumul_ch_assemble_
              << " (" << std::setprecision(1) << 100.0 * cumul_ch_assemble_ / std::max(cumul_total_, 1e-12) << "%)\n";
        file_ << "#   CH  solve:    " << std::setprecision(2) << cumul_ch_solve_
              << " (" << std::setprecision(1) << 100.0 * cumul_ch_solve_ / std::max(cumul_total_, 1e-12) << "%)\n";
        file_ << "#   Poi assembly: " << std::setprecision(2) << cumul_poisson_assemble_
              << " (" << std::setprecision(1) << 100.0 * cumul_poisson_assemble_ / std::max(cumul_total_, 1e-12) << "%)\n";
        file_ << "#   Poi solve:    " << std::setprecision(2) << cumul_poisson_solve_
              << " (" << std::setprecision(1) << 100.0 * cumul_poisson_solve_ / std::max(cumul_total_, 1e-12) << "%)\n";
        file_ << "#   Mag total:    " << std::setprecision(2) << cumul_mag_
              << " (" << std::setprecision(1) << 100.0 * cumul_mag_ / std::max(cumul_total_, 1e-12) << "%)\n";
        file_ << "#   NS  assembly: " << std::setprecision(2) << cumul_ns_assemble_
              << " (" << std::setprecision(1) << 100.0 * cumul_ns_assemble_ / std::max(cumul_total_, 1e-12) << "%)\n";
        file_ << "#   NS  solve:    " << std::setprecision(2) << cumul_ns_solve_
              << " (" << std::setprecision(1) << 100.0 * cumul_ns_solve_ / std::max(cumul_total_, 1e-12) << "%)\n";
        file_ << "#   Diagnostics:  " << std::setprecision(2) << cumul_diagnostics_
              << " (" << std::setprecision(1) << 100.0 * cumul_diagnostics_ / std::max(cumul_total_, 1e-12) << "%)\n";
        file_ << "#\n";
        file_ << "# Load imbalance:\n";
        file_ << "#   Average imbalance ratio: " << std::setprecision(3) << avg_imbalance << "\n";
        file_ << "#   Peak imbalance ratio:    " << std::setprecision(3) << max_imbalance_ << "\n";
        file_ << "#\n";
        file_ << "# Total wall time (rank 0): " << std::setprecision(2) << cumul_total_ << " s\n";
        file_ << "# Avg step time (rank 0):   " << std::setprecision(4) << cumul_total_ / step_count_ << " s\n";
        file_ << "# ====================================================\n";
    }

public:
    /**
     * @brief Get current memory usage in MB (cross-platform)
     */
    static double get_memory_usage_mb()
    {
#ifdef __APPLE__
        struct mach_task_basic_info info;
        mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
        if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                      (task_info_t)&info, &count) == KERN_SUCCESS)
        {
            return static_cast<double>(info.resident_size) / (1024.0 * 1024.0);
        }
#elif defined(__linux__)
        struct rusage usage;
        if (getrusage(RUSAGE_SELF, &usage) == 0)
        {
            return static_cast<double>(usage.ru_maxrss) / 1024.0;  // KB -> MB
        }
#endif
        return 0.0;
    }
};

#endif // PARALLEL_DIAGNOSTICS_LOGGER_H
