// ============================================================================
// output/console_logger.h - Console Output for Simulation Progress
//
// Provides clean, system-level console output:
//   - Run header with configuration summary
//   - Per-step table with key diagnostics
//   - Inline warnings and notes
//   - Final summary
//
// All output is MPI rank-0 only.
// ============================================================================
#ifndef CONSOLE_LOGGER_H
#define CONSOLE_LOGGER_H

#include "diagnostics/step_data.h"
#include "utilities/mpi_tools.h"
#include "utilities/tools.h"

#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>

/**
 * @brief Console logger with clean system-level output
 *
 * Usage:
 *   ConsoleLogger console(params, comm);
 *   console.print_header();
 *   for (step...) {
 *       console.print_step(data);      // Every output_frequency steps
 *       console.print_warnings(data);  // Prints if any warnings
 *       console.print_notes(data, prev_data);  // Interface changes, etc.
 *   }
 *   console.print_footer("complete", final_data);
 */
class ConsoleLogger
{
public:
    /**
     * @brief Constructor
     * @param params Simulation parameters (for header info)
     * @param comm MPI communicator
     */
    ConsoleLogger(const Parameters& params, MPI_Comm comm = MPI_COMM_WORLD)
        : params_(params)
          , comm_(comm)
          , is_root_(MPIUtils::is_root(comm))
          , np_(MPIUtils::size(comm))
          , initial_interface_y_(params.ic.pool_depth)
          , last_spike_count_(0)
    {
    }

    /**
     * @brief Print run header with configuration
     */
    void print_header()
    {
        if (!is_root_) return;

        std::string run_name = get_run_name(params_);
        std::string timestamp = get_timestamp();

        std::cout << "\n";
        std::cout << std::string(91, '=') << "\n";
        std::cout << "  Ferrofluid Solver | " << run_name
            << " | np=" << np_
            << " | " << timestamp << "\n";
        std::cout << std::string(91, '=') << "\n";

        // Column headers
        std::cout << std::setw(6) << "Step"
            << std::setw(10) << "Time"
            << std::setw(12) << "Mass"
            << std::setw(11) << "E_int"
            << std::setw(11) << "dE/dt"
            << std::setw(11) << "|divU|"
            << std::setw(9) << "CFL"
            << std::setw(11) << "Res"
            << "   Wall(Tot)\n";

        std::cout << std::string(91, '-') << "\n";
    }

    /**
     * @brief Print one-line step summary
     * @param data Step diagnostic data
     */
    void print_step(const StepData& data)
    {
        if (!is_root_) return;

        std::cout << std::setw(6) << data.step
            << std::setw(10) << std::fixed << std::setprecision(4) << data.time
            << std::scientific << std::setprecision(2)
            << std::setw(12) << data.mass
            << std::setw(11) << data.E_internal
            << std::setw(11) << data.dE_internal_dt
            << std::setw(11) << data.divU_L2
            << std::setw(9) << std::setprecision(1) << data.CFL
            << std::setw(11) << std::setprecision(2) << data.system_residual
            << std::fixed << std::setprecision(1)
            << "   " << std::setw(5) << data.wall_time_step
            << " (" << std::setw(6) << data.wall_time_total << ")\n";
    }

    /**
     * @brief Print warnings if any thresholds exceeded
     * @param data Step diagnostic data
     */
    void print_warnings(const StepData& data)
    {
        if (!is_root_) return;

        if (data.theta_bounds_violated)
        {
            std::cout << "[!] Step " << data.step
                << ": θ bounds violated ["
                << std::fixed << std::setprecision(3)
                << data.theta_min << ", " << data.theta_max << "]\n";
        }

        if (data.energy_increasing)
        {
            std::cout << "[!] Step " << data.step
                << ": dE/dt = " << std::scientific << std::setprecision(2)
                << data.dE_internal_dt << " > 0 (energy increasing)\n";
        }

        if (data.divU_large)
        {
            std::cout << "[!] Step " << data.step
                << ": |divU| = " << std::scientific << std::setprecision(2)
                << data.divU_L2 << " > 0.1\n";
        }

        if (data.solver_fallback_used)
        {
            std::cout << "[!] Step " << data.step
                << ": Solver fell back to direct method\n";
        }
    }

    /**
     * @brief Print notes (interface changes, spike detection, AMR)
     * @param data Current step data
     * @param amr_triggered Whether AMR was triggered this step
     * @param prev_dofs Previous DOF count (for AMR reporting)
     */
    void print_notes(const StepData& data,
                     bool amr_triggered = false,
                     unsigned int prev_dofs = 0)
    {
        if (!is_root_) return;

        // Interface change note (only if significant)
        double delta_y = data.interface_y_max - initial_interface_y_;
        if (std::abs(delta_y) > 0.005 && data.step % 10 == 0) // ADD: && data.step % 10 == 0
        {
            std::cout << "[+] Step " << data.step
                << ": Interface y_max = " << std::fixed << std::setprecision(3)
                << data.interface_y_max
                << " (Δ = " << std::showpos << std::setprecision(3) << delta_y
                << std::noshowpos << " from initial)\n";
        }

        // Spike count change
        if (data.spike_count != last_spike_count_ && data.spike_count > 0)
        {
            std::cout << "[+] Step " << data.step
                << ": " << data.spike_count << " spike(s) detected\n";
            last_spike_count_ = data.spike_count;
        }

        // AMR note
        if (amr_triggered && prev_dofs > 0)
        {
            std::cout << "[+] Step " << data.step
                << ": AMR triggered, DOFs: " << prev_dofs
                << " → " << data.n_dofs_total << "\n";
        }
    }

    /**
     * @brief Print separator line
     */
    void print_separator()
    {
        if (!is_root_) return;
        std::cout << std::string(91, '-') << "\n";
    }

    /**
     * @brief Print final summary
     * @param reason Termination reason ("complete", "error: ...", etc.)
     * @param data Final step data
     */
    void print_footer(const std::string& reason, const StepData& data)
    {
        if (!is_root_) return;

        std::cout << std::string(91, '-') << "\n";

        // Status line
        std::cout << "  " << reason_to_status(reason)
            << " | t = " << std::fixed << std::setprecision(4) << data.time
            << " | " << data.step << " steps"
            << " | " << std::setprecision(1) << data.wall_time_total << "s wall"
            << " | " << std::setprecision(2) << (data.wall_time_total / std::max(1u, data.step)) << "s/step\n";

        std::cout << std::string(91, '=') << "\n\n";
    }

    /**
     * @brief Print a simple info message
     */
    void info(const std::string& message)
    {
        if (!is_root_) return;
        std::cout << "[Info] " << message << "\n";
    }

    /**
     * @brief Print a warning message
     */
    void warning(const std::string& message)
    {
        if (!is_root_) return;
        std::cout << "\033[33m[Warning] " << message << "\033[0m\n";
    }

    /**
     * @brief Print an error message
     */
    void error(const std::string& message)
    {
        if (!is_root_) return;
        std::cerr << "\033[31m[Error] " << message << "\033[0m\n";
    }

    /**
     * @brief Set initial interface position (for delta computation)
     */
    void set_initial_interface(double y_initial)
    {
        initial_interface_y_ = y_initial;
    }

private:
    const Parameters& params_;
    MPI_Comm comm_;
    bool is_root_;
    int np_;
    double initial_interface_y_;
    mutable unsigned int last_spike_count_;

    /**
     * @brief Convert termination reason to display status
     */
    std::string reason_to_status(const std::string& reason) const
    {
        if (reason == "complete" || reason.find("complete") != std::string::npos)
            return "COMPLETE";
        else if (reason.find("error") != std::string::npos)
            return "ERROR: " + reason.substr(reason.find(":") + 2);
        else if (reason == "interrupted")
            return "INTERRUPTED";
        else if (reason == "max_steps")
            return "MAX STEPS REACHED";
        else
            return reason;
    }
};

#endif // CONSOLE_LOGGER_H
