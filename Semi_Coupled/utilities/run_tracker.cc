// ============================================================================
// utilities/run_tracker.cc - Run Timing and Termination Tracking
// ============================================================================
#include "utilities/run_tracker.h"

#include <fstream>
#include <iomanip>
#include <sstream>

// ============================================================================
// RunTracker
// ============================================================================

void RunTracker::start(const std::string& output_dir, MPI_Comm comm)
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

void RunTracker::end(const std::string& reason)
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

double RunTracker::elapsed_seconds() const
{
    TimePoint end = running_ ? Clock::now() : end_time_;
    return std::chrono::duration<double>(end - start_time_).count();
}

std::string RunTracker::elapsed_formatted() const
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

void RunTracker::write_termination_info()
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

// ============================================================================
// CumulativeTimer
// ============================================================================

void CumulativeTimer::start()
{
    start_ = Clock::now();
}

void CumulativeTimer::stop()
{
    auto end = Clock::now();
    last_ = std::chrono::duration<double>(end - start_).count();
    total_ += last_;
    ++count_;
}

void CumulativeTimer::reset()
{
    last_ = 0.0;
    total_ = 0.0;
    count_ = 0;
}
