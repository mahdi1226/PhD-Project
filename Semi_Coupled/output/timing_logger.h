// ============================================================================
// output/timing_logger.h - Timing Logger for Performance Tracking
//
// Logs per-step and cumulative timing data to timing.csv:
//   - Per-subsystem times (CH, Poisson, Magnetization, NS)
//   - Step totals and cumulative totals
//   - Memory usage
//
// MPI rank-0 only for file output.
// ============================================================================
#ifndef TIMING_LOGGER_H
#define TIMING_LOGGER_H

#include "diagnostics/step_data.h"
#include "utilities/mpi_tools.h"
#include "utilities/tools.h"

#include <string>
#include <fstream>
#include <iomanip>
#include <sstream>

#ifdef __linux__
#include <sys/resource.h>
#endif

/**
 * @brief Timing logger for performance tracking
 *
 * Usage:
 *   TimingLogger timing(output_dir, params, comm);
 *   for (step...) {
 *       StepTiming t;
 *       t.ch_time = ...;
 *       timing.log_step(step, time, t);
 *   }
 *   timing.write_summary();
 */
class TimingLogger
{
public:
    /**
     * @brief Constructor - opens timing.csv and writes header
     * @param output_dir Directory for output file
     * @param params Simulation parameters (for header stamp)
     * @param comm MPI communicator
     */
    TimingLogger(const std::string& output_dir,
                 const Parameters& params,
                 MPI_Comm comm = MPI_COMM_WORLD)
        : output_dir_(output_dir)
        , comm_(comm)
        , is_root_(MPIUtils::is_root(comm))
    {
        if (!is_root_) return;

        file_.open(output_dir + "/timing.csv");
        if (!file_.is_open())
        {
            throw std::runtime_error("Could not open timing.csv");
        }

        // Write header
        write_header();

        // Write config stamp
        file_ << get_csv_header_stamp(params) << "\n";
        file_.flush();
    }

    /**
     * @brief Destructor - writes summary and closes file
     */
    ~TimingLogger()
    {
        if (is_root_ && file_.is_open())
        {
            write_summary();
            file_.close();
        }
    }

    // No copy
    TimingLogger(const TimingLogger&) = delete;
    TimingLogger& operator=(const TimingLogger&) = delete;

    /**
     * @brief Log timing data for a step
     * @param step Step number
     * @param time Simulation time
     * @param timing Timing data for this step
     */
    void log_step(unsigned int step, double time, const StepTiming& timing)
    {
        if (!is_root_) return;

        // Update cumulative totals
        cumul_ch_ += timing.ch_time;
        cumul_poisson_ += timing.poisson_time;
        cumul_mag_ += timing.mag_time;
        cumul_ns_ += timing.ns_time;
        cumul_output_ += timing.output_time;
        cumul_total_ += timing.step_total;
        step_count_++;

        // Get memory usage
        double memory_mb = get_memory_usage_mb();

        // Write row
        file_ << step << ","
              << std::scientific << std::setprecision(6) << time << ","
              << std::fixed << std::setprecision(4)
              << timing.ch_time << ","
              << timing.poisson_time << ","
              << timing.mag_time << ","
              << timing.ns_time << ","
              << timing.output_time << ","
              << timing.step_total << ","
              << cumul_ch_ << ","
              << cumul_poisson_ << ","
              << cumul_mag_ << ","
              << cumul_ns_ << ","
              << cumul_total_ << ","
              << std::setprecision(1) << memory_mb
              << "\n";

        // Flush periodically
        if (step_count_ % 10 == 0)
        {
            file_.flush();
        }

        // Track for summary
        last_step_ = step;
        last_time_ = time;
    }

    /**
     * @brief Force flush to disk
     */
    void flush()
    {
        if (is_root_ && file_.is_open())
        {
            file_.flush();
        }
    }

    /**
     * @brief Get cumulative totals
     */
    double total_ch_time() const { return cumul_ch_; }
    double total_poisson_time() const { return cumul_poisson_; }
    double total_mag_time() const { return cumul_mag_; }
    double total_ns_time() const { return cumul_ns_; }
    double total_time() const { return cumul_total_; }

    /**
     * @brief Get average times per step
     */
    double avg_ch_time() const { return step_count_ > 0 ? cumul_ch_ / step_count_ : 0.0; }
    double avg_poisson_time() const { return step_count_ > 0 ? cumul_poisson_ / step_count_ : 0.0; }
    double avg_mag_time() const { return step_count_ > 0 ? cumul_mag_ / step_count_ : 0.0; }
    double avg_ns_time() const { return step_count_ > 0 ? cumul_ns_ / step_count_ : 0.0; }
    double avg_step_time() const { return step_count_ > 0 ? cumul_total_ / step_count_ : 0.0; }

private:
    std::string output_dir_;
    MPI_Comm comm_;
    bool is_root_;
    std::ofstream file_;

    // Cumulative totals
    double cumul_ch_ = 0.0;
    double cumul_poisson_ = 0.0;
    double cumul_mag_ = 0.0;
    double cumul_ns_ = 0.0;
    double cumul_output_ = 0.0;
    double cumul_total_ = 0.0;
    unsigned int step_count_ = 0;

    // For summary
    unsigned int last_step_ = 0;
    double last_time_ = 0.0;

    /**
     * @brief Write CSV header
     */
    void write_header()
    {
        file_ << "step,time,"
              << "ch_time,poisson_time,mag_time,ns_time,output_time,step_total,"
              << "cumul_ch,cumul_poisson,cumul_mag,cumul_ns,cumul_total,"
              << "memory_mb\n";
    }

    /**
     * @brief Write summary section at end of file
     */
    void write_summary()
    {
        if (step_count_ == 0) return;

        file_ << "\n# ========== TIMING SUMMARY ==========\n";
        file_ << "# Steps completed: " << step_count_ << "\n";
        file_ << "# Final simulation time: " << std::scientific << std::setprecision(4) << last_time_ << "\n";
        file_ << "#\n";
        file_ << "# Subsystem timing (seconds):\n";
        file_ << "#   CH:       total=" << std::fixed << std::setprecision(2) << cumul_ch_
              << ", avg/step=" << std::setprecision(4) << avg_ch_time()
              << ", fraction=" << std::setprecision(1) << (100.0 * cumul_ch_ / cumul_total_) << "%\n";
        file_ << "#   Poisson:  total=" << std::fixed << std::setprecision(2) << cumul_poisson_
              << ", avg/step=" << std::setprecision(4) << avg_poisson_time()
              << ", fraction=" << std::setprecision(1) << (100.0 * cumul_poisson_ / cumul_total_) << "%\n";
        file_ << "#   Mag:      total=" << std::fixed << std::setprecision(2) << cumul_mag_
              << ", avg/step=" << std::setprecision(4) << avg_mag_time()
              << ", fraction=" << std::setprecision(1) << (100.0 * cumul_mag_ / cumul_total_) << "%\n";
        file_ << "#   NS:       total=" << std::fixed << std::setprecision(2) << cumul_ns_
              << ", avg/step=" << std::setprecision(4) << avg_ns_time()
              << ", fraction=" << std::setprecision(1) << (100.0 * cumul_ns_ / cumul_total_) << "%\n";
        file_ << "#   Output:   total=" << std::fixed << std::setprecision(2) << cumul_output_ << "\n";
        file_ << "#\n";
        file_ << "# Total wall time: " << std::fixed << std::setprecision(2) << cumul_total_ << " s\n";
        file_ << "# Average per step: " << std::setprecision(4) << avg_step_time() << " s\n";
        file_ << "# =====================================\n";
    }

    /**
     * @brief Get current memory usage in MB (Linux only)
     */
    double get_memory_usage_mb() const
    {
#ifdef __linux__
        struct rusage usage;
        if (getrusage(RUSAGE_SELF, &usage) == 0)
        {
            // ru_maxrss is in KB on Linux
            return static_cast<double>(usage.ru_maxrss) / 1024.0;
        }
#endif
        return 0.0;
    }
};

#endif // TIMING_LOGGER_H