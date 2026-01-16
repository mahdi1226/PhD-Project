// ============================================================================
// mms/mms_core/test_mms.cc - MMS Test Driver (PARALLEL)
//
// Uses Parameters defaults from parameters.h - NO OVERRIDES!
// MMS verifies production code with production parameters.
//
// Standalone tests (mms_verification.h):
//   CH_STANDALONE, POISSON_STANDALONE, NS_STANDALONE, MAGNETIZATION_STANDALONE
//
// Coupled tests (coupled_mms_test.h):
//   CH_NS, POISSON_MAGNETIZATION, FULL_SYSTEM
//
// CSV output: mms/Rates/YYMMDD-HHMMSS-level_name.csv
// ============================================================================

#include "mms_verification.h"
#include "mms/coupled/coupled_mms_test.h"
#include "utilities/parameters.h"

#include <deal.II/base/utilities.h>
#include <deal.II/base/mpi.h>

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>
#include <ctime>
#include <sys/stat.h>

// ============================================================================
// Test type enum (combines standalone and coupled)
// ============================================================================
enum class TestType
{
    // Standalone
    CH_STANDALONE,
    POISSON_STANDALONE,
    NS_STANDALONE,
    MAGNETIZATION_STANDALONE,
    // Coupled
    CH_NS,
    POISSON_MAGNETIZATION,
    //FULL_SYSTEM
};

static std::string test_type_to_string(TestType type)
{
    switch (type)
    {
    case TestType::CH_STANDALONE:           return "CH_STANDALONE";
    case TestType::POISSON_STANDALONE:      return "POISSON_STANDALONE";
    case TestType::NS_STANDALONE:           return "NS_STANDALONE";
    case TestType::MAGNETIZATION_STANDALONE: return "MAGNETIZATION_STANDALONE";
    case TestType::CH_NS:                   return "CH_NS";
    case TestType::POISSON_MAGNETIZATION:   return "POISSON_MAGNETIZATION";
    //case TestType::FULL_SYSTEM:             return "FULL_SYSTEM";
    default:                                return "UNKNOWN";
    }
}

// ============================================================================
// Helper: Generate timestamp in YYMMDD-HHMMSS format
// ============================================================================
static std::string get_mms_timestamp()
{
    std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%y%m%d-%H%M%S", std::localtime(&t));
    return std::string(buf);
}

// ============================================================================
// Helper: Convert level name to lowercase with underscores
// ============================================================================
static std::string level_to_filename(TestType type)
{
    std::string name = test_type_to_string(type);
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return name;
}

// ============================================================================
// Helper: Ensure directory exists
// ============================================================================
static bool ensure_dir(const std::string& path)
{
    struct stat st;
    if (stat(path.c_str(), &st) == 0)
        return S_ISDIR(st.st_mode);

    std::string cmd = "mkdir -p " + path;
    return system(cmd.c_str()) == 0;
}

// ============================================================================
// Usage
// ============================================================================
void print_usage(const char* program_name)
{
    std::cout << "Usage: " << program_name << " [options]\n"
        << "\nOptions:\n"
        << "  --level <LEVEL>      MMS test level (default: CH_STANDALONE)\n"
        << "\n"
        << "  Standalone tests:\n"
        << "    CH_STANDALONE           - Cahn-Hilliard only\n"
        << "    POISSON_STANDALONE      - Poisson only\n"
        << "    NS_STANDALONE           - Navier-Stokes only\n"
        << "    MAGNETIZATION_STANDALONE - Magnetization only\n"
        << "\n"
        << "  Coupled tests:\n"
        << "    CH_NS                   - CH + NS (θ advected by U)\n"
        << "    POISSON_MAGNETIZATION   - Poisson + Magnetization (φ ↔ M Picard)\n"
        << "    FULL_SYSTEM             - All four subsystems coupled\n"
        << "\n"
        << "  --refs <r1> <r2> ... Refinement levels (default: 3 4 5)\n"
        << "  --steps <n>          Number of time steps (default: 10)\n"
        << "  --help               Show this help\n"
        << "\n"
        << "Output:\n"
        << "  CSV files written to: mms/Rates/YYMMDD-HHMMSS-level_name.csv\n"
        << "\n"
        << "Examples:\n"
        << "  mpirun -np 4 " << program_name << " --level CH_STANDALONE --refs 3 4 5\n"
        << "  mpirun -np 4 " << program_name << " --level POISSON_MAGNETIZATION --refs 3 4 5 --steps 50\n"
        << "  mpirun -np 4 " << program_name << " --level CH_NS --refs 3 4 5 --steps 50\n";
}

int main(int argc, char* argv[])
{
    std::cout << "[STARTUP TEST] Creating Parameters...\n";
    Parameters test1;
    std::cout << "[STARTUP TEST] tau_M = " << test1.physics.tau_M << "\n";

    // Initialize MPI
    dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

    const unsigned int this_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    const unsigned int n_ranks = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

    // Parse command line
    TestType test_type = TestType::CH_STANDALONE;
    std::vector<unsigned int> refinements = {3, 4, 5};
    unsigned int n_time_steps = 10;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        if (arg == "--level" && i + 1 < argc)
        {
            std::string level_str = argv[++i];

            // Standalone tests
            if (level_str == "CH_STANDALONE")
                test_type = TestType::CH_STANDALONE;
            else if (level_str == "POISSON_STANDALONE")
                test_type = TestType::POISSON_STANDALONE;
            else if (level_str == "NS_STANDALONE")
                test_type = TestType::NS_STANDALONE;
            else if (level_str == "MAGNETIZATION_STANDALONE")
                test_type = TestType::MAGNETIZATION_STANDALONE;

            // Coupled tests
            else if (level_str == "CH_NS" || level_str == "CH_NS_CAPILLARY")
                test_type = TestType::CH_NS;
            else if (level_str == "POISSON_MAGNETIZATION" || level_str == "POISSON_MAG")
                test_type = TestType::POISSON_MAGNETIZATION;
            //else if (level_str == "FULL_SYSTEM" || level_str == "FULL")
              //  test_type = TestType::FULL_SYSTEM;

            else
            {
                if (this_rank == 0)
                    std::cerr << "Unknown level: " << level_str << "\n"
                        << "Use --help for available options.\n";
                return 1;
            }
        }
        else if (arg == "--refs" && i + 1 < argc)
        {
            refinements.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-')
            {
                refinements.push_back(std::stoi(argv[++i]));
            }
        }
        else if (arg == "--steps" && i + 1 < argc)
        {
            n_time_steps = std::stoi(argv[++i]);
        }
        else if (arg == "--help" || arg == "-h")
        {
            if (this_rank == 0)
                print_usage(argv[0]);
            return 0;
        }
    }

    // Production parameters from parameters.h - no overrides
    Parameters params;

    if (this_rank == 0)
    {
        std::cout << "\n=== Parallel MMS Test ===\n";
        std::cout << "MPI ranks: " << n_ranks << "\n";
        std::cout << "Level: " << test_type_to_string(test_type) << "\n";
        std::cout << "Refinements:";
        for (auto r : refinements) std::cout << " " << r;
        std::cout << "\n";
        std::cout << "Time steps: " << n_time_steps << "\n";
        std::cout << "Domain: [" << params.domain.x_min << "," << params.domain.x_max
            << "] x [" << params.domain.y_min << "," << params.domain.y_max << "]\n";
        std::cout << "ε = " << params.physics.epsilon
            << ", γ = " << params.physics.mobility << "\n";
        std::cout << "=========================\n\n";
    }

    // Run test
    try
    {
        bool passed = false;

        // Ensure output directory exists
        const std::string rates_dir = "../mms/Rates";
        if (this_rank == 0)
            ensure_dir(rates_dir);

        // Generate timestamped CSV filename
        std::string timestamp = get_mms_timestamp();
        std::string level_name = level_to_filename(test_type);
        std::string csv_name = rates_dir + "/" + timestamp + "-" + level_name + ".csv";

        // Run appropriate test
        switch (test_type)
        {
        // ====== STANDALONE TESTS ======
        case TestType::CH_STANDALONE:
        case TestType::POISSON_STANDALONE:
        case TestType::NS_STANDALONE:
        case TestType::MAGNETIZATION_STANDALONE:
        {
            MMSLevel level;
            if (test_type == TestType::CH_STANDALONE)
                level = MMSLevel::CH_STANDALONE;
            else if (test_type == TestType::POISSON_STANDALONE)
                level = MMSLevel::POISSON_STANDALONE;
            else if (test_type == TestType::NS_STANDALONE)
                level = MMSLevel::NS_STANDALONE;
            else
                level = MMSLevel::MAGNETIZATION_STANDALONE;

            MMSConvergenceResult result = run_mms_test(
                level, refinements, params, n_time_steps, MPI_COMM_WORLD);

            if (this_rank == 0)
            {
                result.print();
                result.write_csv(csv_name);
            }
            passed = result.passes();
            break;
        }

        // ====== COUPLED TESTS ======
        case TestType::CH_NS:
        {
            CoupledMMSConvergenceResult result = run_ch_ns_mms(
                refinements, params, n_time_steps, MPI_COMM_WORLD);

            if (this_rank == 0)
            {
                result.print();
                result.write_csv(csv_name);
            }
            passed = result.passes();
            break;
        }

        case TestType::POISSON_MAGNETIZATION:
        {
            CoupledMMSConvergenceResult result = run_poisson_magnetization_mms(
                refinements, params, n_time_steps, MPI_COMM_WORLD);

            if (this_rank == 0)
            {
                result.print();
                result.write_csv(csv_name);
            }
            passed = result.passes();
            break;
        }

        /* case TestType::FULL_SYSTEM:
        {
            CoupledMMSConvergenceResult result = run_full_system_mms(
                refinements, params, n_time_steps, MPI_COMM_WORLD);

            if (this_rank == 0)
            {
                result.print();
                result.write_csv(csv_name);
            }
            passed = result.passes();
            break;
        }*/
        }

        return passed ? 0 : 1;
    }
    catch (const std::exception& e)
    {
        if (this_rank == 0)
            std::cerr << "\n[ERROR] " << e.what() << "\n";
        return 1;
    }
}