// ============================================================================
// mms/mms_verification.h - MMS Verification Interface
// ============================================================================
#ifndef MMS_VERIFICATION_H
#define MMS_VERIFICATION_H

#include "utilities/parameters.h"
#include <vector>
#include <string>

// ============================================================================
// MMS Verification Levels
// ============================================================================
enum class MMSLevel
{
    CH_STANDALONE,
    POISSON_STANDALONE,
    NS_STANDALONE,
    MAGNETIZATION_STANDALONE,
    CH_WITH_CONVECTION,
    NS_WITH_VARIABLE_NU,
    NS_WITH_KELVIN_FORCE,
    POISSON_MAGNETIZATION,
    CH_NS_CAPILLARY,
    MAGNETIC_NS,
    FULL_SYSTEM
};

// Inline definition - only in header
inline std::string to_string(MMSLevel level)
{
    switch (level)
    {
        case MMSLevel::CH_STANDALONE:            return "CH_STANDALONE";
        case MMSLevel::POISSON_STANDALONE:       return "POISSON_STANDALONE";
        case MMSLevel::NS_STANDALONE:            return "NS_STANDALONE";
        case MMSLevel::MAGNETIZATION_STANDALONE: return "MAGNETIZATION_STANDALONE";
        case MMSLevel::CH_WITH_CONVECTION:       return "CH_WITH_CONVECTION";
        case MMSLevel::NS_WITH_VARIABLE_NU:      return "NS_WITH_VARIABLE_NU";
        case MMSLevel::NS_WITH_KELVIN_FORCE:     return "NS_WITH_KELVIN_FORCE";
        case MMSLevel::POISSON_MAGNETIZATION:    return "POISSON_MAGNETIZATION";
        case MMSLevel::CH_NS_CAPILLARY:          return "CH_NS_CAPILLARY";
        case MMSLevel::MAGNETIC_NS:              return "MAGNETIC_NS";
        case MMSLevel::FULL_SYSTEM:              return "FULL_SYSTEM";
        default:                                 return "UNKNOWN";
    }
}

// ============================================================================
// Convergence Results
// ============================================================================
struct MMSConvergenceResult
{
    MMSLevel level;
    std::vector<unsigned int> refinements;
    std::vector<double> h_values;

    // Metadata
    unsigned int fe_degree = 1;
    unsigned int n_time_steps = 0;
    double dt = 0.0;
    double expected_L2_rate = 2.0;  // p+1
    double expected_H1_rate = 1.0;  // p

    // DoF counts and timing
    std::vector<unsigned int> n_dofs;
    std::vector<double> wall_times;

    // CH errors
    std::vector<double> theta_L2, theta_H1, psi_L2;

    // Poisson errors
    std::vector<double> phi_L2, phi_H1;

    // NS errors
    std::vector<double> ux_L2, ux_H1, uy_L2, uy_H1, p_L2, div_u_L2;

    // Magnetization errors
    std::vector<double> M_L2;

    // Computed rates
    std::vector<double> theta_L2_rate, theta_H1_rate, psi_L2_rate;
    std::vector<double> phi_L2_rate, phi_H1_rate;
    std::vector<double> ux_L2_rate, ux_H1_rate, uy_L2_rate, uy_H1_rate, p_L2_rate;
    std::vector<double> M_L2_rate;

    /// Compute convergence rates from error data
    void compute_rates();

    /// Print formatted table to console
    void print() const;

    /// Check if rates match expected values (within tolerance)
    /// Returns true if rate >= expected - tol (allows superconvergence)
    bool passes(double tol = 0.3) const;

    /// Write results to CSV file
    void write_csv(const std::string& filename) const;
};

// ============================================================================
// Main Entry Point
// ============================================================================

/// Run MMS convergence study for specified level
MMSConvergenceResult run_mms_convergence_study(
    MMSLevel level,
    const std::vector<unsigned int>& refinements,
    const Parameters& params,
    unsigned int n_time_steps = 10);

#endif // MMS_VERIFICATION_H