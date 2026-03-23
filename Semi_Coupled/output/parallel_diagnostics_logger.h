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
                               bool write_per_rank = false);

    /**
     * @brief Destructor - writes summary and closes files
     */
    ~ParallelDiagnosticsLogger();

    // No copy
    ParallelDiagnosticsLogger(const ParallelDiagnosticsLogger&) = delete;
    ParallelDiagnosticsLogger& operator=(const ParallelDiagnosticsLogger&) = delete;

    /**
     * @brief Log parallel diagnostics for one step
     *
     * IMPORTANT: Call pdata.compute_mpi_reductions(comm) BEFORE this!
     */
    void log_step(const ParallelStepData& d);

    void flush();

    /**
     * @brief Get current memory usage in MB (cross-platform)
     */
    static double get_memory_usage_mb();

private:
    std::string output_dir_;
    MPI_Comm comm_;
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

    // CSV writing helpers
    void write_header(std::ofstream& f);
    void write_row(std::ofstream& f, const ParallelStepData& d);
    void write_rank_header(std::ofstream& f);
    void write_rank_row(std::ofstream& f, const ParallelStepData& d);
    void write_summary();
};

#endif // PARALLEL_DIAGNOSTICS_LOGGER_H
