// ============================================================================
// utilities/run_tracker.h - Run Timing and Termination Tracking
//
// Tracks:
//   - Simulation start/end wall time
//   - Termination reason (complete, error, interrupted)
//   - Writes final summary to run_info.txt
//
// Usage:
//   RunTracker tracker;
//   tracker.start(argc, argv, params, output_dir);
//   // ... simulation loop ...
//   tracker.end("complete");  // or "error: solver failed", etc.
// ============================================================================
#ifndef RUN_TRACKER_H
#define RUN_TRACKER_H

#include "utilities/mpi_tools.h"

#include <string>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <csignal>

/**
 * @brief Tracks run timing and termination status
 */
class RunTracker
{
public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;

    RunTracker() = default;

    /**
     * @brief Start tracking (call at beginning of run)
     * @param output_dir Directory where run_info.txt will be updated
     * @param comm MPI communicator
     */
    void start(const std::string& output_dir, MPI_Comm comm = MPI_COMM_WORLD)
    {
        output_dir_ = output_dir;
        comm_ = comm;
        start_time_ = Clock::now();
        running_ = true;
        termination_reason_ = "running";

        // Record start timestamp
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        char buf[32];
        std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&time_t_now));
        start_timestamp_ = buf;
    }

    /**
     * @brief Mark simulation end with reason
     * @param reason Termination reason: "complete", "error: ...", "interrupted", etc.
     */
    void end(const std::string& reason)
    {
        if (!running_)
            return;

        end_time_ = Clock::now();
        running_ = false;
        termination_reason_ = reason;

        // Record end timestamp
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        char buf[32];
        std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&time_t_now));
        end_timestamp_ = buf;

        // Write final info (rank 0 only)
        if (MPIUtils::is_root(comm_))
        {
            write_termination_info();
        }
    }

    /**
     * @brief Get elapsed wall time in seconds
     */
    double elapsed_seconds() const
    {
        TimePoint end = running_ ? Clock::now() : end_time_;
        return std::chrono::duration<double>(end - start_time_).count();
    }

    /**
     * @brief Get elapsed time formatted as HH:MM:SS
     */
    std::string elapsed_formatted() const
    {
        double total_seconds = elapsed_seconds();
        int hours = static_cast<int>(total_seconds) / 3600;
        int minutes = (static_cast<int>(total_seconds) % 3600) / 60;
        int seconds = static_cast<int>(total_seconds) % 60;

        std::ostringstream ss;
        ss << std::setfill('0')
           << std::setw(2) << hours << ":"
           << std::setw(2) << minutes << ":"
           << std::setw(2) << seconds;
        return ss.str();
    }

    /**
     * @brief Check if simulation is still running
     */
    bool is_running() const { return running_; }

    /**
     * @brief Get termination reason
     */
    const std::string& termination_reason() const { return termination_reason_; }

    /**
     * @brief Get start timestamp string
     */
    const std::string& start_timestamp() const { return start_timestamp_; }

private:
    TimePoint start_time_;
    TimePoint end_time_;
    std::string start_timestamp_;
    std::string end_timestamp_;
    std::string output_dir_;
    std::string termination_reason_ = "not started";
    MPI_Comm comm_ = MPI_COMM_WORLD;
    bool running_ = false;

    /**
     * @brief Append termination info to run_info.txt
     */
    void write_termination_info()
    {
        std::ofstream file(output_dir_ + "/run_info.txt", std::ios::app);
        if (!file.is_open())
            return;

        file << "\n";
        file << "============================================================\n";
        file << "  RUN TERMINATION\n";
        file << "============================================================\n";
        file << "  Start time:   " << start_timestamp_ << "\n";
        file << "  End time:     " << end_timestamp_ << "\n";
        file << "  Wall time:    " << elapsed_formatted()
             << " (" << std::fixed << std::setprecision(1)
             << elapsed_seconds() << " s)\n";
        file << "  Status:       " << termination_reason_ << "\n";
        file << "============================================================\n";

        file.close();
    }
};

// ============================================================================
// Step Timer - RAII timer for measuring individual step/subsystem times
// ============================================================================

/**
 * @brief RAII timer that records elapsed time to a reference variable
 *
 * Usage:
 *   double ch_time;
 *   {
 *       StepTimer timer(ch_time);
 *       // ... do CH solve ...
 *   }
 *   // ch_time now contains elapsed seconds
 */
class StepTimer
{
public:
    using Clock = std::chrono::high_resolution_clock;

    explicit StepTimer(double& result)
        : result_(result)
        , start_(Clock::now())
    {}

    ~StepTimer()
    {
        auto end = Clock::now();
        result_ = std::chrono::duration<double>(end - start_).count();
    }

    // Non-copyable
    StepTimer(const StepTimer&) = delete;
    StepTimer& operator=(const StepTimer&) = delete;

private:
    double& result_;
    std::chrono::time_point<Clock> start_;
};

// ============================================================================
// Cumulative Timer - Tracks both per-call and cumulative time
// ============================================================================

/**
 * @brief Tracks cumulative time across multiple calls
 *
 * Usage:
 *   CumulativeTimer ch_timer;
 *   for (step...) {
 *       ch_timer.start();
 *       // ... do CH solve ...
 *       ch_timer.stop();
 *       double this_step = ch_timer.last();
 *   }
 *   double total = ch_timer.total();
 */
class CumulativeTimer
{
public:
    using Clock = std::chrono::high_resolution_clock;

    CumulativeTimer() = default;

    void start()
    {
        start_ = Clock::now();
    }

    void stop()
    {
        auto end = Clock::now();
        last_ = std::chrono::duration<double>(end - start_).count();
        total_ += last_;
        ++count_;
    }

    double last() const { return last_; }
    double total() const { return total_; }
    unsigned int count() const { return count_; }
    double average() const { return count_ > 0 ? total_ / count_ : 0.0; }

    void reset()
    {
        last_ = 0.0;
        total_ = 0.0;
        count_ = 0;
    }

private:
    std::chrono::time_point<Clock> start_;
    double last_ = 0.0;
    double total_ = 0.0;
    unsigned int count_ = 0;
};

#endif // RUN_TRACKER_H