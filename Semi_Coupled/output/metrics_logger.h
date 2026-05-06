// ============================================================================
// output/metrics_logger.h - Metrics Logger for Simulation Diagnostics
//
// Logs all diagnostic data to CSV files for post-processing and analysis.
// MPI-aware: only rank 0 writes to files.
//
// Output files:
//   - diagnostics.csv: Per-step full data
//   - energy.csv: Energy components for stability analysis
//   - warnings.csv: Timestamped warnings and alerts
//   - convergence.csv: MMS convergence data (if MMS mode)
//   - validation_metrics.csv: Rosensweig validation data
// ============================================================================
#ifndef METRICS_LOGGER_H
#define METRICS_LOGGER_H

#include "diagnostics/step_data.h"
#include "utilities/mpi_tools.h"
#include "utilities/tools.h"

#include <string>
#include <fstream>

/**
 * @brief Metrics logger for simulation diagnostics
 *
 * Usage:
 *   MetricsLogger logger(output_dir, params_, mpi_communicator_);
 *   logger.log_step(step_data);
 *   logger.log_warning(step, time, "theta exceeded bounds");
 */
class MetricsLogger
{
public:
    /**
     * @brief Constructor - opens all CSV files and writes headers
     * @param output_dir Directory for output files
     * @param params Simulation parameters (for header stamp)
     * @param comm MPI communicator
     */
    MetricsLogger(const std::string& output_dir,
                  const Parameters& params,
                  MPI_Comm comm = MPI_COMM_WORLD);

    /**
     * @brief Destructor - closes all files
     */
    ~MetricsLogger();

    // No copy
    MetricsLogger(const MetricsLogger&) = delete;
    MetricsLogger& operator=(const MetricsLogger&) = delete;

    /**
     * @brief Log all data for a time step
     * @param data Step diagnostic data
     */
    void log_step(const StepData& data);

    /**
     * @brief Log a warning message
     */
    void log_warning(unsigned int step, double time, const std::string& message);

    /**
     * @brief Log MMS convergence data
     */
    void log_convergence(const ConvergenceData& data);

    /**
     * @brief Flush all files
     */
    void flush();

    /**
     * @brief Get path to diagnostics file
     */
    std::string get_diagnostics_path() const;

private:
    std::string output_dir_;
    MPI_Comm comm_;
    bool is_root_;
    bool mms_mode_;        // gate convergence.csv on this

    std::ofstream diagnostics_file_;
    std::ofstream energy_file_;
    std::ofstream validation_file_;
    std::ofstream warnings_file_;
    std::ofstream convergence_file_;  // only opened when mms_mode_ is true

    unsigned int step_count_;
    double E_internal_prev_;

    void open_files();
    void write_headers();
    void write_config_stamp(const Parameters& params);
    void close_files();

    /**
     * @brief Format double in scientific notation for messages
     */
    std::string format_sci(double value) const;
};

#endif // METRICS_LOGGER_H
