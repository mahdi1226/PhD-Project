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

#include <string>
#include <ctime>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <sys/stat.h>

// ============================================================================
// Timestamped Folder Generation
// ============================================================================

/**
 * @brief Generate timestamp string in format YYYYMMDD_HHMMSS
 */
inline std::string get_timestamp_compact()
{
    std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", std::localtime(&t));
    return std::string(buf);
}

/**
 * @brief Generate timestamp string in format YYYY-MM-DD HH:MM:SS
 */
inline std::string get_timestamp()
{
    std::time_t now = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    return std::string(buf);
}

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
inline std::string timestamped_folder(const std::string& base,
                                       const Parameters& params)
{
    std::ostringstream ss;

    // Timestamp first (for chronological sorting)
    ss << get_timestamp_compact();

    // Preset name
    ss << "_" << params.preset_name;

    // Refinement level
    ss << "_r" << params.mesh.initial_refinement;

    // Solver type
    if (!params.solvers.ch.use_iterative || !params.solvers.ns.use_iterative)
        ss << "_direct";
    else
        ss << "_iter";

    // AMR flag
    if (params.mesh.use_amr)
        ss << "_amr";
    else
        ss << "_Namr";

    return base + "/" + ss.str();
}

/**
 * @brief Legacy overload for backward compatibility
 */
inline std::string timestamped_folder(const std::string& base,
                                       const std::string& run_name = "")
{
    std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", std::localtime(&t));

    if (run_name.empty())
        return base + "/" + buf;
    else
        return base + "/" + std::string(buf) + "_" + run_name;
}

/**
 * @brief Create directory if it doesn't exist
 * @param path Directory path
 * @return true if directory exists or was created successfully
 */
inline bool ensure_directory(const std::string& path)
{
    struct stat st;
    if (stat(path.c_str(), &st) == 0)
    {
        return S_ISDIR(st.st_mode);
    }

    // Create directory with parents
    std::string cmd = "mkdir -p " + path;
    return system(cmd.c_str()) == 0;
}

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
inline std::string get_csv_header_stamp(const Parameters& params)
{
    std::ostringstream ss;
    ss << "# ";
    ss << "date=" << get_timestamp_compact();
    ss << ",preset=" << params.preset_name;
    ss << ",dt=" << std::scientific << std::setprecision(1) << params.time.dt;
    ss << ",t_final=" << std::fixed << std::setprecision(2) << params.time.t_final;
    ss << ",ref=" << params.mesh.initial_refinement;
    ss << ",eps=" << std::scientific << std::setprecision(1) << params.physics.epsilon;
    ss << ",lambda=" << params.physics.lambda;
    ss << ",chi=" << std::fixed << std::setprecision(2) << params.physics.chi_0;
    ss << ",mms=" << (params.enable_mms ? 1 : 0);
    ss << ",mag=" << (params.enable_magnetic ? 1 : 0);
    ss << ",ns=" << (params.enable_ns ? 1 : 0);
    ss << ",grav=" << (params.enable_gravity ? 1 : 0);
    ss << ",reduced_h=" << (params.use_reduced_magnetic_field ? 1 : 0);
    ss << ",amr=" << (params.mesh.use_amr ? 1 : 0);
    ss << ",dg=" << (params.use_dg_transport ? 1 : 0);
    ss << ",direct=" << (!params.solvers.ch.use_iterative ? 1 : 0);
    ss << ",picard=" << params.picard_iterations;

    return ss.str();
}

/**
 * @brief Get short flag summary for console output header
 *
 * Returns: rosen_r4_direct_amr
 */
inline std::string get_run_name(const Parameters& params)
{
    std::ostringstream ss;
    ss << params.preset_name;
    ss << "_r" << params.mesh.initial_refinement;

    if (!params.solvers.ch.use_iterative || !params.solvers.ns.use_iterative)
        ss << "_direct";
    else
        ss << "_iter";

    if (params.mesh.use_amr)
        ss << "_amr";
    else
        ss << "_Namr";

    return ss.str();
}

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
inline void write_run_info(const std::string& output_dir,
                           const Parameters& params,
                           int argc, char* argv[],
                           int np = 1)
{
    std::ofstream file(output_dir + "/run_info.txt");
    if (!file.is_open())
    {
        std::cerr << "[Warning] Could not create run_info.txt\n";
        return;
    }

    file << "============================================================\n";
    file << "  Ferrofluid Phase Field Solver - Run Configuration\n";
    file << "  Generated: " << get_timestamp() << "\n";
    file << "============================================================\n\n";

    // Command line (for exact reproduction)
    file << "COMMAND LINE:\n  ";
    for (int i = 0; i < argc; ++i)
        file << argv[i] << " ";
    file << "\n\n";

    // MPI configuration
    file << "MPI CONFIGURATION:\n";
    file << "  Number of processes (np): " << np << "\n\n";

    // Preset and run name
    file << "PRESET: " << params.preset_name << "\n";
    file << "RUN NAME: " << get_run_name(params) << "\n\n";

    // Critical flags (the most important for debugging!)
    file << "FLAGS (Critical for debugging):\n";
    file << "  enable_mms              = " << (params.enable_mms ? "TRUE  <-- MMS MODE!" : "false") << "\n";
    file << "  enable_magnetic         = " << (params.enable_magnetic ? "true" : "FALSE") << "\n";
    file << "  enable_ns               = " << (params.enable_ns ? "true" : "FALSE") << "\n";
    file << "  enable_gravity          = " << (params.enable_gravity ? "true" : "FALSE") << "\n";
    file << "  use_dg_transport        = " << (params.use_dg_transport ? "true" : "false") << "\n";
    file << "  use_reduced_mag_field   = " << (params.use_reduced_magnetic_field ? "TRUE (h=ha only)" : "false (h=ha+hd)") << "\n";
    file << "\n";

    // Time stepping
    file << "TIME STEPPING:\n";
    file << "  dt                = " << std::scientific << std::setprecision(6) << params.time.dt << "\n";
    file << "  t_final           = " << std::fixed << std::setprecision(4) << params.time.t_final << "\n";
    file << "  max_steps         = " << params.time.max_steps << "\n";
    file << "  use_adaptive_dt   = " << (params.time.use_adaptive_dt ? "true" : "false") << "\n";
    file << "\n";

    // Mesh / AMR
    file << "MESH:\n";
    file << "  initial_refinement = " << params.mesh.initial_refinement << "\n";
    file << "  use_amr            = " << (params.mesh.use_amr ? "true" : "false") << "\n";
    if (params.mesh.use_amr)
    {
        file << "  amr_min_level      = " << params.mesh.amr_min_level << "\n";
        file << "  amr_max_level      = " << params.mesh.amr_max_level << "\n";
        file << "  amr_interval       = " << params.mesh.amr_interval << "\n";
    }
    file << "\n";

    // Physics - Cahn-Hilliard
    file << "PHYSICS (Cahn-Hilliard):\n";
    file << "  epsilon  = " << std::scientific << params.physics.epsilon << "\n";
    file << "  mobility = " << params.physics.mobility << "\n";
    file << "  lambda   = " << params.physics.lambda << "\n";
    file << "\n";

    // Physics - Navier-Stokes
    file << "PHYSICS (Navier-Stokes):\n";
    file << "  nu_water = " << std::fixed << std::setprecision(4) << params.physics.nu_water << "\n";
    file << "  nu_ferro = " << params.physics.nu_ferro << "\n";
    file << "  rho      = " << params.physics.rho << "\n";
    file << "  r        = " << params.physics.r << "\n";
    file << "  gravity  = " << std::scientific << params.physics.gravity << "\n";
    file << "\n";

    // Physics - Magnetic
    file << "PHYSICS (Magnetic):\n";
    file << "  chi_0    = " << std::fixed << std::setprecision(4) << params.physics.chi_0 << "\n";
    file << "  mu_0     = " << params.physics.mu_0 << "\n";
    file << "  tau_M    = " << std::scientific << params.physics.tau_M << "\n";
    file << "\n";

    // Dipoles
    file << "DIPOLES:\n";
    file << "  count         = " << params.dipoles.positions.size() << "\n";
    file << "  intensity_max = " << std::fixed << std::setprecision(2) << params.dipoles.intensity_max << "\n";
    file << "  ramp_time     = " << params.dipoles.ramp_time << "\n";
    file << "  direction     = [" << params.dipoles.direction[0] << ", "
         << params.dipoles.direction[1] << "]\n";
    file << "\n";

    // Solver settings
    file << "SOLVERS:\n";
    file << "  CH:\n";
    file << "    use_iterative = " << (params.solvers.ch.use_iterative ? "true" : "false (DIRECT)") << "\n";
    file << "    rel_tolerance = " << std::scientific << params.solvers.ch.rel_tolerance << "\n";
    file << "    max_iter      = " << params.solvers.ch.max_iterations << "\n";
    file << "  NS:\n";
    file << "    use_iterative = " << (params.solvers.ns.use_iterative ? "true (Schur)" : "false (DIRECT)") << "\n";
    file << "    rel_tolerance = " << std::scientific << params.solvers.ns.rel_tolerance << "\n";
    file << "    max_iter      = " << params.solvers.ns.max_iterations << "\n";
    file << "  Poisson:\n";
    file << "    rel_tolerance = " << std::scientific << params.solvers.poisson.rel_tolerance << "\n";
    file << "    max_iter      = " << params.solvers.poisson.max_iterations << "\n";
    file << "\n";

    // Coupling
    file << "COUPLING:\n";
    file << "  picard_iterations = " << params.picard_iterations << "\n";
    file << "\n";

    // Domain
    file << "DOMAIN:\n";
    file << "  x: [" << std::fixed << std::setprecision(2) << params.domain.x_min << ", " << params.domain.x_max << "]\n";
    file << "  y: [" << params.domain.y_min << ", " << params.domain.y_max << "]\n";
    file << "\n";

    // Initial condition
    file << "INITIAL CONDITION:\n";
    file << "  type       = " << params.ic.type << " (0=pool, 1=droplet)\n";
    file << "  pool_depth = " << params.ic.pool_depth << "\n";
    file << "\n";

    file << "============================================================\n";
    file << "CSV Header Stamp:\n";
    file << get_csv_header_stamp(params) << "\n";
    file << "============================================================\n";

    file.close();
    std::cout << "[Info] Run configuration saved to: " << output_dir << "/run_info.txt\n";
}

/**
 * @brief Overload for when argc/argv not available
 */
inline void write_run_info(const std::string& output_dir,
                           const Parameters& params,
                           int np = 1)
{
    const char* dummy_argv[] = {"(command line not captured)"};
    write_run_info(output_dir, params, 1, const_cast<char**>(dummy_argv), np);
}

#endif // TOOLS_H