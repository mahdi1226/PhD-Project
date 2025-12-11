// ============================================================================
// output/nsch_output.h - Clean console output + CSV logging (SELF-CONTAINED)
//
// NO external dependencies - just standard library.
// Put this in your output/ folder.
//
// Usage in solver:
//   #include "output/nsch_output.h"
//
//   NSCHOutput output;
//   output.initialize("../ReSuLtS");  // "../" = project root, not cmake-build
//   output.set_intervals(10, 1);      // console every 10, CSV every 1
//
//   // In time loop:
//   StepMetrics m;
//   m.step = step; m.time = time;
//   m.mass = ...; m.total_energy = ...; // fill from your verification
//   output.record(m);
//
//   // At end:
//   output.finalize();
// ============================================================================
#ifndef NSCH_OUTPUT_H
#define NSCH_OUTPUT_H

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <cmath>
#include <sys/stat.h>
#include <ctime>

// ============================================================================
// StepMetrics - All data for one timestep
// ============================================================================
struct StepMetrics
{
    unsigned int step = 0;
    double time = 0.0;

    // Conservation
    double mass = 0.0;
    double ch_energy = 0.0;
    double kinetic_energy = 0.0;
    double total_energy = 0.0;

    // Navier-Stokes
    double divergence_L2 = 0.0;
    double u_max = 0.0;
    double cfl_number = 0.0;

    // Phase field bounds
    double c_min = 0.0, c_max = 0.0;
    double mu_min = 0.0, mu_max = 0.0;
    double p_min = 0.0, p_max = 0.0;

    // Interface tracking
    double amplitude = 0.0;
    double y_min = 0.0, y_max = 0.0;
    int n_spikes = 0;

    // Solution norms (for detailed CSV)
    double c_norm = 0.0, mu_norm = 0.0;
    double ux_norm = 0.0, uy_norm = 0.0, p_norm = 0.0;

    // Force magnitudes
    double F_cap_max = 0.0, F_mag_max = 0.0, F_grav_max = 0.0;
};

// ============================================================================
// Console table formatting - Simple ASCII header for step output
// ============================================================================
namespace console
{
    inline void print_header()
    {
        std::cout << "\n";
        std::cout << " step       time        Mass     E_total   E_kinetic       div_u     c_min     c_max   |u|_max     CFL\n";
        std::cout << std::string(111, '-') << "\n";
    }

    inline void print_row(const StepMetrics& m)
    {
        std::cout << std::setw(5) << m.step << "  "
                  << std::setw(10) << std::setprecision(4) << std::scientific << m.time << "  "
                  << std::setw(10) << std::setprecision(4) << std::scientific << m.mass << "  "
                  << std::setw(10) << std::setprecision(4) << std::scientific << m.total_energy << "  "
                  << std::setw(10) << std::setprecision(4) << std::scientific << m.kinetic_energy << "  "
                  << std::setw(10) << std::setprecision(2) << std::scientific << m.divergence_L2 << "  "
                  << std::setw(8) << std::setprecision(4) << std::fixed << m.c_min << "  "
                  << std::setw(8) << std::setprecision(4) << std::fixed << m.c_max << "  "
                  << std::setw(8) << std::setprecision(2) << std::scientific << m.u_max << "  "
                  << std::setw(6) << std::setprecision(3) << std::fixed << m.cfl_number << "\n";
    }

    inline void print_interface(const StepMetrics& m)
    {
        if (m.amplitude > 0.001)
        {
            std::cout << "       Interface: amp=" << std::fixed << std::setprecision(4) << m.amplitude
                      << "  y=[" << std::setprecision(3) << m.y_min << "," << m.y_max << "]"
                      << "  spikes=" << m.n_spikes << "\n";
        }
    }

    inline void print_footer()
    {
        std::cout << std::string(111, '-') << "\n";
    }

    // Summary table keeps nice Unicode formatting
    inline void print_summary(const StepMetrics& initial, const StepMetrics& final_m)
    {
        double mass_change = final_m.mass - initial.mass;
        double mass_pct = (initial.mass != 0) ? 100.0 * std::abs(mass_change / initial.mass) : 0.0;
        double energy_change = final_m.total_energy - initial.total_energy;

        std::cout << "\n";
        std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
        std::cout << "║                    SIMULATION SUMMARY                         ║\n";
        std::cout << "╠═══════════════════════════════════════════════════════════════╣\n";
        std::cout << std::scientific << std::setprecision(6);
        std::cout << "║  Conservation:                                                ║\n";
        std::cout << "║    Initial mass    : " << std::setw(14) << initial.mass << "                    ║\n";
        std::cout << "║    Final mass      : " << std::setw(14) << final_m.mass << "                    ║\n";
        std::cout << "║    Mass change     : " << std::setw(14) << mass_change
                  << " (" << std::setprecision(2) << mass_pct << "%)        ║\n";
        std::cout << "╟───────────────────────────────────────────────────────────────╢\n";
        std::cout << std::setprecision(6);
        std::cout << "║  Energy:                                                      ║\n";
        std::cout << "║    Initial E_total : " << std::setw(14) << initial.total_energy << "                    ║\n";
        std::cout << "║    Final E_total   : " << std::setw(14) << final_m.total_energy << "                    ║\n";
        std::cout << "║    Energy change   : " << std::setw(14) << energy_change;
        std::cout << (energy_change <= 0 ? "  [OK] dissipating   ║\n" : "  [!] INCREASING     ║\n");
        std::cout << "╟───────────────────────────────────────────────────────────────╢\n";
        std::cout << "║  Final state:                                                 ║\n";
        std::cout << "║    c in [" << std::setw(9) << std::setprecision(4) << std::fixed << final_m.c_min
                  << ", " << std::setw(8) << final_m.c_max << "]";
        std::cout << (final_m.c_min >= -1.01 && final_m.c_max <= 1.01
                      ? "  [OK] bounded           ║\n" : "  [!] OVERSHOOT          ║\n");
        std::cout << std::scientific << std::setprecision(3);
        std::cout << "║    |u|_max         : " << std::setw(10) << final_m.u_max << "                         ║\n";
        std::cout << "║    ||div(u)||_L2   : " << std::setw(10) << final_m.divergence_L2 << "                         ║\n";
        std::cout << "║    Interface amp   : " << std::setw(10) << std::fixed << std::setprecision(4) << final_m.amplitude << "                         ║\n";
        std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";
    }
}

// ============================================================================
// NSCHOutput - Main output handler (SELF-CONTAINED)
// ============================================================================
class NSCHOutput
{
public:
    NSCHOutput() = default;

    ~NSCHOutput()
    {
        if (csv_file_.is_open())
            csv_file_.close();
    }

    /// Initialize output system
    /// Creates: base_dir/run_YYYYMMDD_HHMMSS/
    /// DEFAULT: "../ReSuLtS" puts results in PROJECT ROOT (not cmake-build)
    void initialize(const std::string& base_dir = "../ReSuLtS")
    {
        base_dir_ = base_dir;

        // Create base directory
        mkdir(base_dir_.c_str(), 0755);

        // Create timestamped run directory
        std::time_t now = std::time(nullptr);
        char timestamp[64];
        std::strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", std::localtime(&now));

        run_dir_ = base_dir_ + "/run_" + timestamp;
        mkdir(run_dir_.c_str(), 0755);

        // Open CSV file
        std::string csv_path = run_dir_ + "/metrics.csv";
        csv_file_.open(csv_path);

        if (csv_file_.is_open())
        {
            csv_file_ << "step,time,mass,ch_energy,kinetic_energy,total_energy,"
                      << "divergence_L2,c_min,c_max,u_max,cfl_number,"
                      << "amplitude,y_min,y_max,n_spikes,"
                      << "mu_min,mu_max,p_min,p_max,"
                      << "c_norm,mu_norm,ux_norm,uy_norm,p_norm,"
                      << "F_cap_max,F_mag_max,F_grav_max\n";
        }

        std::cout << "[OUTPUT] Results directory: " << run_dir_ << "\n";
        std::cout << "[OUTPUT] CSV logging to: " << csv_path << "\n";
    }

    /// Set output intervals
    void set_intervals(int console_every, int csv_every)
    {
        console_interval_ = console_every;
        csv_interval_ = csv_every;
    }

    /// Record a timestep
    void record(const StepMetrics& m)
    {
        if (m.step == 0)
            initial_ = m;

        // CSV logging
        if (csv_file_.is_open() && (m.step % csv_interval_ == 0))
        {
            csv_file_ << std::setprecision(12) << std::scientific;
            csv_file_ << m.step << "," << m.time << "," << m.mass << ","
                      << m.ch_energy << "," << m.kinetic_energy << "," << m.total_energy << ","
                      << m.divergence_L2 << "," << m.c_min << "," << m.c_max << ","
                      << m.u_max << "," << m.cfl_number << ","
                      << m.amplitude << "," << m.y_min << "," << m.y_max << "," << m.n_spikes << ","
                      << m.mu_min << "," << m.mu_max << "," << m.p_min << "," << m.p_max << ","
                      << m.c_norm << "," << m.mu_norm << "," << m.ux_norm << ","
                      << m.uy_norm << "," << m.p_norm << ","
                      << m.F_cap_max << "," << m.F_mag_max << "," << m.F_grav_max << "\n";
            csv_file_.flush();
        }

        // Console output
        if (m.step % console_interval_ == 0)
        {
            if (!header_printed_)
            {
                console::print_header();
                header_printed_ = true;
            }
            console::print_row(m);
            console::print_interface(m);
        }

        last_ = m;
    }

    /// Finalize and print summary
    void finalize()
    {
        console::print_footer();
        console::print_summary(initial_, last_);
        std::cout << "\n[OUTPUT] Results saved to: " << run_dir_ << "\n";

        if (csv_file_.is_open())
            csv_file_.close();
    }

    /// Get run directory path (for VTK output, etc.)
    std::string get_run_dir() const { return run_dir_; }

private:
    std::string base_dir_;
    std::string run_dir_;
    std::ofstream csv_file_;

    int console_interval_ = 10;
    int csv_interval_ = 1;
    bool header_printed_ = false;

    StepMetrics initial_;
    StepMetrics last_;
};

#endif // NSCH_OUTPUT_H