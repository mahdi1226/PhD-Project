// ============================================================================
// output/console_logger.h - Console Logging Utilities
//
// Provides:
//   - Colored console output (info, success, warning, error)
//   - Formatted step summary (minimal, every N steps)
//   - Progress indicator
//
// ============================================================================
#ifndef CONSOLE_LOGGER_H
#define CONSOLE_LOGGER_H

#include "diagnostics/step_data.h"

#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <chrono>

/**
 * @brief Console logger with colored output and step summaries
 */
class ConsoleLogger
{
public:
    // ========================================================================
    // Basic logging (static methods)
    // ========================================================================

    /// Log info message
    static void info(const std::string& message)
    {
        std::cout << "[Info] " << message << std::endl;
    }

    /// Log success message (green)
    static void success(const std::string& message)
    {
        std::cout << "\033[32m[Success] " << message << "\033[0m" << std::endl;
    }

    /// Log warning message (yellow)
    static void warning(const std::string& message)
    {
        std::cout << "\033[33m[Warning] " << message << "\033[0m" << std::endl;
    }

    /// Log error message (red)
    static void error(const std::string& message)
    {
        std::cerr << "\033[31m[Error] " << message << "\033[0m" << std::endl;
    }

    // ========================================================================
    // Step summary logging
    // ========================================================================

    /**
     * @brief Print table header for step summaries
     */
    static void print_step_header()
    {
        std::cout << std::string(100, '-') << "\n";
        std::cout << std::setw(6) << "Step"
                  << std::setw(10) << "Time"
                  << std::setw(22) << "θ ∈ [min, max]"
                  << std::setw(12) << "mass"
                  << std::setw(12) << "E_total"
                  << std::setw(12) << "divU"
                  << std::setw(12) << "CFL"
                  << std::setw(12) << "|U|_max"
                  << "\n";
        std::cout << std::string(100, '-') << "\n";
    }

    /**
     * @brief Print one-line step summary
     * @param data Step diagnostic data
     * @param force Print even if not at output interval
     */
    static void print_step_summary(const StepData& data, bool force = false)
    {
        std::cout << std::setw(6) << data.step
                  << std::setw(10) << std::fixed << std::setprecision(4) << data.time
                  << "  [" << std::scientific << std::setprecision(3)
                  << std::setw(9) << data.theta_min << ", "
                  << std::setw(9) << data.theta_max << "]"
                  << std::setw(12) << std::fixed << std::setprecision(4) << data.mass
                  << std::scientific << std::setprecision(3)
                  << std::setw(12) << data.E_total
                  << std::setw(12) << data.divU_L2
                  << std::setw(12) << data.CFL
                  << std::setw(12) << data.U_max
                  << "\n";
    }

    /**
     * @brief Print warnings if any thresholds exceeded
     * @param data Step diagnostic data
     */
    static void print_warnings(const StepData& data)
    {
        if (data.theta_min < -1.001 || data.theta_max > 1.001)
        {
            std::cout << "\033[33m[Warning] step=" << data.step
                      << ": θ bounds violated: ["
                      << data.theta_min << ", " << data.theta_max << "]\033[0m\n";
        }

        if (data.divU_L2 > 0.1)
        {
            std::cout << "\033[33m[Warning] step=" << data.step
                      << ": divU large: " << data.divU_L2 << "\033[0m\n";
        }

        if (data.dE_total_dt > 1e-3)
        {
            std::cout << "\033[33m[Warning] step=" << data.step
                      << ": energy increasing: dE/dt="
                      << data.dE_total_dt << "\033[0m\n";
        }
    }

    /**
     * @brief Print detailed subsystem info (verbose mode)
     * @param data Step diagnostic data
     */
    static void print_verbose(const StepData& data)
    {
        std::cout << "[CH] step=" << data.step
                  << " θ∈[" << std::scientific << std::setprecision(4)
                  << data.theta_min << "," << data.theta_max << "]"
                  << " mass=" << std::fixed << std::setprecision(4) << data.mass
                  << " E=" << std::scientific << data.E_CH
                  << " iters=" << data.ch_iterations
                  << " time=" << std::fixed << std::setprecision(3) << data.ch_time << "s\n";

        std::cout << "[Poisson] step=" << data.step
                  << " φ∈[" << std::scientific << std::setprecision(2)
                  << data.phi_min << "," << data.phi_max << "]"
                  << " |H|_max=" << data.H_max
                  << " |M|_max=" << data.M_max
                  << " E_mag=" << data.E_mag
                  << " iters=" << data.poisson_iterations
                  << " time=" << std::fixed << std::setprecision(3) << data.poisson_time << "s\n";

        std::cout << "[NS] step=" << data.step
                  << " |U|_max=" << std::scientific << std::setprecision(3) << data.U_max
                  << " E_kin=" << data.E_kin
                  << " |divU|=" << data.divU_L2
                  << " CFL=" << data.CFL
                  << " p∈[" << std::setprecision(2) << data.p_min << "," << data.p_max << "]"
                  << " time=" << std::fixed << std::setprecision(2) << data.ns_time << "s\n";

        std::cout << "[Forces] |F_cap|=" << std::scientific << std::setprecision(2)
                  << data.F_cap_max
                  << " |F_mag|=" << data.F_mag_max
                  << " |F_grav|=" << data.F_grav_max << "\n";
    }

    /**
     * @brief Print progress bar
     * @param current Current step
     * @param total Total steps
     * @param width Bar width in characters
     */
    static void print_progress(unsigned int current, unsigned int total, int width = 50)
    {
        float progress = static_cast<float>(current) / total;
        int pos = static_cast<int>(width * progress);

        std::cout << "\r[";
        for (int i = 0; i < width; ++i)
        {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << std::setw(3) << int(progress * 100.0) << "% "
                  << "(" << current << "/" << total << ")" << std::flush;
    }

    /**
     * @brief Print simulation summary at end
     * @param final_data Final step data
     * @param total_time Total wall clock time in seconds
     */
    static void print_summary(const StepData& final_data, double total_time)
    {
        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << "  SIMULATION COMPLETE\n";
        std::cout << std::string(60, '=') << "\n";

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  Final time:     " << final_data.time << "\n";
        std::cout << "  Total steps:    " << final_data.step << "\n";
        std::cout << "  Wall time:      " << total_time << " s\n";
        std::cout << "  Time per step:  " << total_time / final_data.step << " s\n";

        std::cout << "\n  Final state:\n";
        std::cout << "    θ ∈ [" << std::scientific << std::setprecision(4)
                  << final_data.theta_min << ", " << final_data.theta_max << "]\n";
        std::cout << "    mass = " << std::fixed << std::setprecision(6)
                  << final_data.mass << "\n";
        std::cout << "    E_total = " << std::scientific << final_data.E_total << "\n";
        std::cout << "    |divU| = " << final_data.divU_L2 << "\n";
        std::cout << "    |U|_max = " << final_data.U_max << "\n";

        std::cout << std::string(60, '=') << "\n";
    }
};

/**
 * @brief RAII timer for measuring execution time
 */
class ScopedTimer
{
public:
    explicit ScopedTimer(double& result)
        : result_(result)
        , start_(std::chrono::high_resolution_clock::now())
    {}

    ~ScopedTimer()
    {
        auto end = std::chrono::high_resolution_clock::now();
        result_ = std::chrono::duration<double>(end - start_).count();
    }

private:
    double& result_;
    std::chrono::high_resolution_clock::time_point start_;
};

#endif // CONSOLE_LOGGER_H