// ============================================================================
// utilities/tools.h - Utility Functions
//
// Provides:
//   - Timestamped folder generation for unique output directories
//   - CSV header stamps for reproducibility
//   - Run info writer for full configuration logging
// ============================================================================
#ifndef TOOLS_H
#define TOOLS_H

#include "utilities/parameters.h"
#include "utilities/mpi_tools.h"

#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <string>


// ============================================================================
// TEST CONFIGURATION - Change this before each test run!
// ============================================================================
#define TEST_ID "A"
#define TEST_H_FORMULA ": A: H = grad_phi (both assemblers)"
// Options:
//   A: H = grad_phi (both assemblers)
//   B: H = h_a - grad_phi (NS), grad_phi (Mag)
//   C: H = h_a + grad_phi (NS), grad_phi (Mag)


// ============================================================================
// Timestamped Folder Generation
// ============================================================================

/**
 * @brief Generate timestamp (YYYYMMDD_HHMMSS) — local only, no MPI.
 *
 * Safe to call from any single rank without matching calls on other ranks.
 * Use this in rank-0-only contexts (loggers, write_run_info, etc.).
 */
std::string get_timestamp_compact_local();

/**
 * @brief Generate timestamp (YYYY-MM-DD HH:MM:SS) — local only, no MPI.
 *
 * Safe to call from any single rank.
 */
std::string get_timestamp_local();

/**
 * @brief Generate timestamp string in format YYYYMMDD_HHMMSS
 *
 * COLLECTIVE: ALL ranks must call this (uses MPI_Bcast).
 * Rank 0 generates the timestamp and broadcasts it.
 * For rank-0-only contexts, use get_timestamp_compact_local() instead.
 */
std::string get_timestamp_compact(MPI_Comm comm = MPI_COMM_WORLD);

/**
 * @brief Generate timestamp string in format YYYY-MM-DD HH:MM:SS
 *
 * COLLECTIVE: ALL ranks must call this (uses MPI_Bcast).
 * For rank-0-only contexts, use get_timestamp_local() instead.
 */
std::string get_timestamp(MPI_Comm comm = MPI_COMM_WORLD);

/**
 * @brief Generate unique timestamped output folder path
 *
 * Format: base/YYYYMMDD_HHMMSS_preset_rN_solver_amr/
 *
 * Examples:
 *   20260107_143215_rosen_r4_direct_amr
 *   20260107_143215_hedge_r5_iter_Namr
 *
 * @param base Base output directory (e.g., "../Results")
 * @param params Simulation parameters
 * @return Full path to timestamped output folder
 */
std::string timestamped_folder(const std::string& base,
                               const Parameters& params,
                               MPI_Comm comm = MPI_COMM_WORLD);

/**
 * @brief Legacy overload for backward compatibility
 */
std::string timestamped_folder(const std::string& base,
                               const std::string& run_name = "",
                               MPI_Comm comm = MPI_COMM_WORLD);

/**
 * @brief Create directory if it doesn't exist
 * @param path Directory path
 * @return true if directory exists or was created successfully
 */
bool ensure_directory(const std::string& path);


// ============================================================================
// CSV Header Stamp
// ============================================================================

/**
 * @brief Get compact CSV header comment with critical run parameters
 *
 * Returns a comment line for CSV files containing all critical run parameters.
 * Format: # key=value,key=value,...
 *
 * @param params Current simulation parameters
 * @return Comment string starting with #
 */
std::string get_csv_header_stamp(const Parameters& params, MPI_Comm comm = MPI_COMM_WORLD);

/**
 * @brief Get short flag summary for console output header
 *
 * Returns: rosen_r4_direct_amr
 */
std::string get_run_name(const Parameters& params);


// ============================================================================
// Run Info Writer
// ============================================================================

/**
 * @brief Write complete run configuration to run_info.txt
 *
 * Call this at the start of run() after output directory is created.
 * Only rank 0 should call this function.
 *
 * @param output_dir Directory to write run_info.txt
 * @param params Current parameters
 * @param argc Command line argument count
 * @param argv Command line arguments
 * @param np Number of MPI processes
 */
void write_run_info(const std::string& output_dir,
                    const Parameters& params,
                    int argc, char* argv[],
                    int np = 1);

/**
 * @brief Overload for when argc/argv not available
 */
void write_run_info(const std::string& output_dir,
                    const Parameters& params,
                    int np = 1);

#endif // TOOLS_H
