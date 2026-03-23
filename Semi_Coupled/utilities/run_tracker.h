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
    void start(const std::string& output_dir, MPI_Comm comm = MPI_COMM_WORLD);

    /**
     * @brief Mark simulation end with reason
     * @param reason Termination reason: "complete", "error: ...", "interrupted", etc.
     */
    void end(const std::string& reason);

    /**
     * @brief Get elapsed wall time in seconds
     */
    double elapsed_seconds() const;

    /**
     * @brief Get elapsed time formatted as HH:MM:SS
     */
    std::string elapsed_formatted() const;

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
    void write_termination_info();
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

    void start();
    void stop();

    double last() const { return last_; }
    double total() const { return total_; }
    unsigned int count() const { return count_; }
    double average() const { return count_ > 0 ? total_ / count_ : 0.0; }

    void reset();

private:
    std::chrono::time_point<Clock> start_;
    double last_ = 0.0;
    double total_ = 0.0;
    unsigned int count_ = 0;
};

#endif // RUN_TRACKER_H
