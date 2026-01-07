// ============================================================================
// test_mms.cc - MMS Verification Test Program
//
// Usage:
//   ./test_mms                           # Run CH_STANDALONE with defaults
//   ./test_mms --level CH_STANDALONE     # Explicit level
//   ./test_mms --refinements 3,4,5,6,7   # Custom refinements
//   ./test_mms --steps 1000              # More time steps
//   ./test_mms --degree 2                # Use Q2 elements
//   ./test_mms --csv results.csv         # Write results to CSV
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "mms/mms_verification.h"
#include "utilities/parameters.h"

#include <iostream>
#include <sstream>
#include <cstring>

void print_usage(const char* prog)
{
    std::cout << "MMS Verification Test\n\n";
    std::cout << "Usage: " << prog << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --level LEVEL         MMS level to test (default: CH_STANDALONE)\n";
    std::cout << "                        Standalone: CH_STANDALONE, POISSON_STANDALONE,\n";
    std::cout << "                                    NS_STANDALONE, MAGNETIZATION_STANDALONE\n";
    std::cout << "                        Coupled:    POISSON_MAGNETIZATION\n";
    std::cout << "                        Coupled:    POISSON_MAGNETIZATION, CH_NS_CAPILLARY\n";
    std::cout << "  --refinements R1,R2,  Refinement levels (default: 3,4,5,6)\n";
    std::cout << "  --steps N             Time steps (default: 10)\n";
    std::cout << "  --degree D            FE polynomial degree (default: 1 for Q1)\n";
    std::cout << "  --epsilon E           Interface thickness (default: 0.1)\n";
    std::cout << "  --gamma G             Mobility coefficient (default: 0.01)\n";
    std::cout << "  --csv FILE            Write results to CSV file\n";
    std::cout << "  --help                Show this help\n";
}

MMSLevel parse_level(const std::string& s)
{
    if (s == "CH_STANDALONE") return MMSLevel::CH_STANDALONE;
    if (s == "POISSON_STANDALONE") return MMSLevel::POISSON_STANDALONE;
    if (s == "NS_STANDALONE") return MMSLevel::NS_STANDALONE;
    if (s == "MAGNETIZATION_STANDALONE") return MMSLevel::MAGNETIZATION_STANDALONE;
    if (s == "POISSON_MAGNETIZATION") return MMSLevel::POISSON_MAGNETIZATION;
    if (s == "CH_NS_CAPILLARY") return MMSLevel::CH_NS_CAPILLARY;

    std::cerr << "Unknown MMS level: " << s << "\n";
    std::exit(1);
}

int main(int argc, char* argv[])
{
    // Defaults
    MMSLevel level = MMSLevel::CH_STANDALONE;
    std::vector<unsigned int> refinements = {3, 4, 5, 6};
    unsigned int n_steps = 10;
    unsigned int fe_degree = 1;
    double epsilon = 0.1;
    double gamma = 0.01;
    std::string csv_file;

    // Parse command line
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0)
        {
            print_usage(argv[0]);
            return 0;
        }
        else if (std::strcmp(argv[i], "--level") == 0 && i + 1 < argc)
        {
            level = parse_level(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--refinements") == 0 && i + 1 < argc)
        {
            refinements.clear();
            std::stringstream ss(argv[++i]);
            std::string token;
            while (std::getline(ss, token, ','))
                refinements.push_back(std::stoul(token));
        }
        else if (std::strcmp(argv[i], "--steps") == 0 && i + 1 < argc)
        {
            n_steps = std::stoul(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--degree") == 0 && i + 1 < argc)
        {
            fe_degree = std::stoul(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--epsilon") == 0 && i + 1 < argc)
        {
            epsilon = std::stod(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--gamma") == 0 && i + 1 < argc)
        {
            gamma = std::stod(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--csv") == 0 && i + 1 < argc)
        {
            csv_file = argv[++i];
        }
        else
        {
            std::cerr << "Unknown option: " << argv[i] << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // Setup parameters
    Parameters params;
    params.physics.epsilon = epsilon;
    params.physics.mobility = gamma;
    params.fe.degree_phase = fe_degree;
    params.fe.degree_velocity = 2;   // Q2 for NS velocity
    params.fe.degree_pressure = 1;   // Q1 for NS pressure
    params.fe.degree_potential = 1;  // Q1 for Poisson

    // Print configuration
    std::cout << "================================================================\n";
    std::cout << "  MMS VERIFICATION TEST\n";
    std::cout << "================================================================\n";
    std::cout << "Level:       " << to_string(level) << "\n";
    std::cout << "Refinements: ";
    for (auto r : refinements) std::cout << r << " ";
    std::cout << "\n";
    std::cout << "Time steps:  " << n_steps << "\n";
    std::cout << "FE degree:   Q" << fe_degree << "\n";
    std::cout << "ε (epsilon): " << epsilon << "\n";
    std::cout << "γ (gamma):   " << gamma << "\n";
    if (!csv_file.empty())
        std::cout << "CSV output:  " << csv_file << "\n";
    std::cout << "================================================================\n";

    // Run test
    auto result = run_mms_convergence_study(level, refinements, params, n_steps);

    // Print results
    result.print();

    // Write CSV if requested
    if (!csv_file.empty())
        result.write_csv(csv_file);

    // Check rates - FIX: was check_rates(), should be passes()
    bool pass = result.passes(0.3);

    std::cout << "\n================================================================\n";
    if (pass)
        std::cout << "  TEST PASSED\n";
    else
        std::cout << "  TEST FAILED\n";
    std::cout << "================================================================\n\n";

    return pass ? 0 : 1;
}