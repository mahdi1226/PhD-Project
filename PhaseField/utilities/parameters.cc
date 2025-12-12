// ============================================================================
// utilities/parameters.cc - Parameter Parsing Implementation
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Section 6.2, p.520-522
// ============================================================================

#include "parameters.h"
#include "output/logger.h"
#include <stdexcept>
#include <cstring>

Parameters Parameters::parse_command_line(int argc, char* argv[])
{
    Logger::info("Parsing command line parameters...");
    
    Parameters params;  // Defaults from Section 6.2
    
    // Simple command line parsing
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        
        // Domain parameters
        if (arg == "--x_min" && i + 1 < argc)
            params.domain.x_min = std::stod(argv[++i]);
        else if (arg == "--x_max" && i + 1 < argc)
            params.domain.x_max = std::stod(argv[++i]);
        else if (arg == "--y_min" && i + 1 < argc)
            params.domain.y_min = std::stod(argv[++i]);
        else if (arg == "--y_max" && i + 1 < argc)
            params.domain.y_max = std::stod(argv[++i]);
        else if (arg == "--refinement" && i + 1 < argc)
            params.domain.initial_refinement = std::stoul(argv[++i]);

        // Time parameters (accept both hyphen and underscore)
        else if (arg == "--dt" && i + 1 < argc)
            params.time.dt = std::stod(argv[++i]);
        else if ((arg == "--t_final" || arg == "--t-final") && i + 1 < argc)
            params.time.t_final = std::stod(argv[++i]);

        // CH parameters
        else if (arg == "--epsilon" && i + 1 < argc)
            params.ch.epsilon = std::stod(argv[++i]);
        else if (arg == "--gamma" && i + 1 < argc)
            params.ch.gamma = std::stod(argv[++i]);
        else if (arg == "--lambda" && i + 1 < argc)
            params.ch.lambda = std::stod(argv[++i]);
        else if (arg == "--eta" && i + 1 < argc)
            params.ch.eta = std::stod(argv[++i]);

        // Magnetization parameters
        else if (arg == "--chi_0" && i + 1 < argc)
            params.magnetization.chi_0 = std::stod(argv[++i]);
        else if (arg == "--T_relax" && i + 1 < argc)
            params.magnetization.T_relax = std::stod(argv[++i]);

        // NS parameters
        else if (arg == "--nu_water" && i + 1 < argc)
            params.ns.nu_water = std::stod(argv[++i]);
        else if (arg == "--nu_ferro" && i + 1 < argc)
            params.ns.nu_ferro = std::stod(argv[++i]);
        else if (arg == "--mu_0" && i + 1 < argc)
            params.ns.mu_0 = std::stod(argv[++i]);
        else if (arg == "--grad_div" && i + 1 < argc)
            params.ns.grad_div = std::stod(argv[++i]);

        // Dipole parameters
        else if (arg == "--dipole_intensity" && i + 1 < argc)
            params.dipoles.intensity_max = std::stod(argv[++i]);
        else if (arg == "--dipole_ramp" && i + 1 < argc)
            params.dipoles.ramp_time = std::stod(argv[++i]);

        // Gravity parameters
        else if (arg == "--g" && i + 1 < argc)
            params.gravity.magnitude = std::stod(argv[++i]);
        else if (arg == "--no_gravity")
            params.gravity.enabled = false;

        // AMR parameters
        else if (arg == "--amr")
            params.amr.enabled = true;
        else if (arg == "--no_amr")
            params.amr.enabled = false;
        else if (arg == "--amr_min" && i + 1 < argc)
            params.amr.min_level = std::stoul(argv[++i]);
        else if (arg == "--amr_max" && i + 1 < argc)
            params.amr.max_level = std::stoul(argv[++i]);
        else if (arg == "--amr_interval" && i + 1 < argc)
            params.amr.interval = std::stoul(argv[++i]);

        // IC parameters
        else if (arg == "--pool_depth" && i + 1 < argc)
            params.ic.pool_depth = std::stod(argv[++i]);

        // Output parameters
        else if (arg == "--output_dir" && i + 1 < argc)
            params.output.folder = argv[++i];
        else if (arg == "--output_frequency" && i + 1 < argc)
            params.output.frequency = std::stoul(argv[++i]);

        // Help
        else if (arg == "--help" || arg == "-h")
        {
            Logger::info("Usage: ferrofluid [options]");
            Logger::info("Options:");
            Logger::info("  --epsilon <val>      Interface thickness ε (default: 0.01)");
            Logger::info("  --gamma <val>        Mobility γ (default: 0.0002)");
            Logger::info("  --lambda <val>       Capillary coefficient λ (default: 0.05)");
            Logger::info("  --chi_0 <val>        Susceptibility χ₀ (default: 0.5)");
            Logger::info("  --dt <val>           Time step (default: 5e-4)");
            Logger::info("  --t-final <val>      Final time (default: 2.0)");
            Logger::info("  --g <val>            Gravity magnitude (default: 30000)");
            Logger::info("  --refinement <val>   Initial mesh refinement (default: 5)");
            Logger::info("  --amr / --no_amr     Enable/disable AMR");
            Logger::info("  --help               Show this help");
            std::exit(0);
        }
    }

    // Validate parameters
    if (params.magnetization.chi_0 > 4.0)
    {
        Logger::warning("χ₀ = " + std::to_string(params.magnetization.chi_0) +
                        " > 4 violates energy stability (Proposition 3.1, p.502)");
    }

    if (params.ch.eta > params.ch.epsilon)
    {
        Logger::warning("η = " + std::to_string(params.ch.eta) +
                        " > ε = " + std::to_string(params.ch.epsilon) +
                        " violates stability (Proposition 4.1, p.505)");
    }

    // Print summary
    Logger::info("Parameters loaded:");
    Logger::info("  Domain: [" + std::to_string(params.domain.x_min) + ", " +
                 std::to_string(params.domain.x_max) + "] × [" +
                 std::to_string(params.domain.y_min) + ", " +
                 std::to_string(params.domain.y_max) + "]");
    Logger::info("  Time: dt = " + std::to_string(params.time.dt) +
                 ", t_final = " + std::to_string(params.time.t_final));
    Logger::info("  CH: ε = " + std::to_string(params.ch.epsilon) +
                 ", γ = " + std::to_string(params.ch.gamma) +
                 ", λ = " + std::to_string(params.ch.lambda));
    Logger::info("  Mag: χ₀ = " + std::to_string(params.magnetization.chi_0));
    Logger::info("  NS: ν_w = " + std::to_string(params.ns.nu_water) +
                 ", ν_f = " + std::to_string(params.ns.nu_ferro));
    Logger::info("  Gravity: " + std::string(params.gravity.enabled ? "enabled" : "disabled") +
                 ", g = " + std::to_string(params.gravity.magnitude));
    Logger::info("  AMR: " + std::string(params.amr.enabled ? "enabled" : "disabled"));

    return params;
}