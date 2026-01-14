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
//
// ============================================================================
#ifndef METRICS_LOGGER_H
#define METRICS_LOGGER_H

#include "diagnostics/step_data.h"
#include "utilities/mpi_tools.h"
#include "utilities/tools.h"

#include <string>
#include <fstream>
#include <iomanip>
#include <sstream>

/**
 * @brief Metrics logger for simulation diagnostics
 *
 * Usage:
 *   MetricsLogger logger(output_dir, params, comm);
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
                  MPI_Comm comm = MPI_COMM_WORLD)
        : output_dir_(output_dir)
        , comm_(comm)
        , is_root_(MPIUtils::is_root(comm))
        , step_count_(0)
        , E_internal_prev_(0.0)
    {
        if (!is_root_) return;

        open_files();
        write_headers();
        write_config_stamp(params);
    }

    /**
     * @brief Destructor - closes all files
     */
    ~MetricsLogger()
    {
        if (is_root_)
        {
            close_files();
        }
    }

    // No copy
    MetricsLogger(const MetricsLogger&) = delete;
    MetricsLogger& operator=(const MetricsLogger&) = delete;

    /**
     * @brief Log all data for a time step
     * @param data Step diagnostic data
     */
    void log_step(const StepData& data)
    {
        if (!is_root_) return;

        // Main diagnostics file
        diagnostics_file_
            << data.step << ","
            << std::scientific << std::setprecision(6)
            << data.time << ","
            << data.dt << ","
            // CH
            << data.theta_min << ","
            << data.theta_max << ","
            << data.mass << ","
            << data.E_CH << ","
            << data.dE_CH_dt << ","
            << data.ch_iterations << ","
            << data.ch_residual << ","
            << data.ch_time << ","
            // Poisson
            << data.phi_min << ","
            << data.phi_max << ","
            << data.H_max << ","
            << data.M_max << ","
            << data.E_mag << ","
            << data.mu_min << ","
            << data.mu_max << ","
            << data.poisson_iterations << ","
            << data.poisson_residual << ","
            << data.poisson_time << ","
            // NS
            << data.ux_min << ","
            << data.ux_max << ","
            << data.uy_min << ","
            << data.uy_max << ","
            << data.U_max << ","
            << data.E_kin << ","
            << data.divU_L2 << ","
            << data.divU_Linf << ","
            << data.CFL << ","
            << data.p_min << ","
            << data.p_max << ","
            << data.F_cap_max << ","
            << data.F_mag_max << ","
            << data.F_grav_max << ","
            << data.ns_time << ","
            // Energy
            << data.E_internal << ","
            << data.E_total << ","
            << data.dE_internal_dt << ","
            << data.dE_total_dt << ","
            // Interface
            << data.interface_y_min << ","
            << data.interface_y_max << ","
            << data.interface_y_mean << ","
            // Mesh
            << data.n_active_cells << ","
            << data.n_dofs_total
            << "\n";

        // Energy file (simpler, for plotting)
        energy_file_
            << data.step << ","
            << std::scientific << std::setprecision(6)
            << data.time << ","
            << data.E_CH << ","
            << data.E_kin << ","
            << data.E_mag << ","
            << data.E_internal << ","
            << data.E_total << ","
            << data.dE_internal_dt << ","
            << data.dE_total_dt
            << "\n";

        // Auto-log warnings
        if (data.theta_bounds_violated)
        {
            log_warning(data.step, data.time,
                "theta bounds violated: [" + format_sci(data.theta_min) +
                ", " + format_sci(data.theta_max) + "]");
        }
        if (data.divU_large)
        {
            log_warning(data.step, data.time,
                "divU large: L2=" + format_sci(data.divU_L2) +
                ", Linf=" + format_sci(data.divU_Linf));
        }
        if (data.energy_increasing)
        {
            log_warning(data.step, data.time,
                "internal energy increasing: dE/dt=" + format_sci(data.dE_internal_dt));
        }
        if (data.solver_fallback_used)
        {
            log_warning(data.step, data.time, "solver fell back to direct method");
        }

        ++step_count_;
        E_internal_prev_ = data.E_internal;

        // Flush periodically
        if (step_count_ % 10 == 0)
        {
            flush();
        }
    }

    /**
     * @brief Log a warning message
     */
    void log_warning(unsigned int step, double time, const std::string& message)
    {
        if (!is_root_) return;

        warnings_file_
            << step << ","
            << std::scientific << std::setprecision(6)
            << time << ","
            << "\"" << message << "\""
            << "\n";
        warnings_file_.flush();
    }

    /**
     * @brief Log MMS convergence data
     */
    void log_convergence(const ConvergenceData& data)
    {
        if (!is_root_) return;

        convergence_file_
            << data.refinement << ","
            << std::scientific << std::setprecision(6)
            << data.h << ","
            << data.theta_L2 << ","
            << data.theta_H1 << ","
            << data.psi_L2 << ","
            << data.phi_L2 << ","
            << data.phi_H1 << ","
            << data.ux_L2 << ","
            << data.ux_H1 << ","
            << data.uy_L2 << ","
            << data.uy_H1 << ","
            << data.p_L2 << ","
            << data.divU_L2
            << "\n";
        convergence_file_.flush();
    }

    /**
     * @brief Flush all files
     */
    void flush()
    {
        if (!is_root_) return;

        diagnostics_file_.flush();
        energy_file_.flush();
        warnings_file_.flush();
        convergence_file_.flush();
    }

    /**
     * @brief Get path to diagnostics file
     */
    std::string get_diagnostics_path() const
    {
        return output_dir_ + "/diagnostics.csv";
    }

private:
    std::string output_dir_;
    MPI_Comm comm_;
    bool is_root_;

    std::ofstream diagnostics_file_;
    std::ofstream energy_file_;
    std::ofstream warnings_file_;
    std::ofstream convergence_file_;

    unsigned int step_count_;
    double E_internal_prev_;

    void open_files()
    {
        diagnostics_file_.open(output_dir_ + "/diagnostics.csv");
        energy_file_.open(output_dir_ + "/energy.csv");
        warnings_file_.open(output_dir_ + "/warnings.csv");
        convergence_file_.open(output_dir_ + "/convergence.csv");

        if (!diagnostics_file_.is_open())
        {
            throw std::runtime_error("Could not open diagnostics.csv");
        }
    }

    void write_headers()
    {
        // Main diagnostics header
        diagnostics_file_
            << "step,time,dt,"
            << "theta_min,theta_max,mass,E_CH,dE_CH_dt,"
            << "ch_iterations,ch_residual,ch_time,"
            << "phi_min,phi_max,H_max,M_max,E_mag,mu_min,mu_max,"
            << "poisson_iterations,poisson_residual,poisson_time,"
            << "ux_min,ux_max,uy_min,uy_max,U_max,E_kin,"
            << "divU_L2,divU_Linf,CFL,p_min,p_max,"
            << "F_cap_max,F_mag_max,F_grav_max,ns_time,"
            << "E_internal,E_total,dE_internal_dt,dE_total_dt,"
            << "interface_y_min,interface_y_max,interface_y_mean,"
            << "n_cells,n_dofs"
            << "\n";

        // Energy header
        energy_file_
            << "step,time,E_CH,E_kin,E_mag,E_internal,E_total,dE_internal_dt,dE_total_dt"
            << "\n";

        // Warnings header
        warnings_file_
            << "step,time,message"
            << "\n";

        // Convergence header
        convergence_file_
            << "refinement,h,"
            << "theta_L2,theta_H1,psi_L2,"
            << "phi_L2,phi_H1,"
            << "ux_L2,ux_H1,uy_L2,uy_H1,p_L2,divU_L2"
            << "\n";
    }

    /**
     * @brief Write config stamp ONCE to all files
     */
    void write_config_stamp(const Parameters& params)
    {
        std::string stamp = get_csv_header_stamp(params);

        // Write stamp as comment line in each file (ONLY ONCE - fixed double-stamp bug)
        if (diagnostics_file_.is_open())
            diagnostics_file_ << stamp << "\n";

        if (energy_file_.is_open())
            energy_file_ << stamp << "\n";

        if (convergence_file_.is_open())
            convergence_file_ << stamp << "\n";

        if (warnings_file_.is_open())
            warnings_file_ << stamp << "\n";
    }

    void close_files()
    {
        if (diagnostics_file_.is_open()) diagnostics_file_.close();
        if (energy_file_.is_open()) energy_file_.close();
        if (warnings_file_.is_open()) warnings_file_.close();
        if (convergence_file_.is_open()) convergence_file_.close();
    }

    /**
     * @brief Format double in scientific notation for messages
     */
    std::string format_sci(double value) const
    {
        std::ostringstream ss;
        ss << std::scientific << std::setprecision(3) << value;
        return ss.str();
    }
};

#endif // METRICS_LOGGER_H