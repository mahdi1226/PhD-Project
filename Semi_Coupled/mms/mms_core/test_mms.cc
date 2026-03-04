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
//   CH_NS, POISSON_MAGNETIZATION, NS_MAGNETIZATION, NS_POISSON_MAG,
//   MAG_CH, FULL_SYSTEM
//
// Temporal tests (temporal_convergence.h):
//   CH_TEMPORAL, NS_TEMPORAL, MAG_TEMPORAL, FULL_TEMPORAL
//
// CSV output: mms/Rates/YYMMDD-HHMMSS-level_name.csv
// ============================================================================

#include "mms_verification.h"
#include "mms/coupled/coupled_mms_test.h"
#include "mms/magnetization/magnetization_mms_test.h"
#include "temporal_convergence.h"
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
// Test type enum (combines standalone, coupled, and temporal)
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
    NS_MAGNETIZATION,
    NS_POISSON_MAG,
    MAG_CH,
    FULL_SYSTEM,
    // Transport isolation test
    MAG_TRANSPORT,
    // Temporal convergence
    CH_TEMPORAL,
    NS_TEMPORAL,
    MAG_TEMPORAL,
    FULL_TEMPORAL
};

static std::string test_type_to_string(TestType type)
{
    switch (type)
    {
    case TestType::CH_STANDALONE: return "CH_STANDALONE";
    case TestType::POISSON_STANDALONE: return "POISSON_STANDALONE";
    case TestType::NS_STANDALONE: return "NS_STANDALONE";
    case TestType::MAGNETIZATION_STANDALONE: return "MAGNETIZATION_STANDALONE";
    case TestType::CH_NS: return "CH_NS";
    case TestType::POISSON_MAGNETIZATION: return "POISSON_MAGNETIZATION";
    case TestType::NS_MAGNETIZATION: return "NS_MAGNETIZATION";
    case TestType::NS_POISSON_MAG: return "NS_POISSON_MAG";
    case TestType::MAG_CH: return "MAG_CH";
    case TestType::FULL_SYSTEM: return "FULL_SYSTEM";
    case TestType::MAG_TRANSPORT: return "MAG_TRANSPORT";
    case TestType::CH_TEMPORAL: return "CH_TEMPORAL";
    case TestType::NS_TEMPORAL: return "NS_TEMPORAL";
    case TestType::MAG_TEMPORAL: return "MAG_TEMPORAL";
    case TestType::FULL_TEMPORAL: return "FULL_TEMPORAL";
    default: return "UNKNOWN";
    }
}

static bool is_temporal_test(TestType type)
{
    return type == TestType::CH_TEMPORAL ||
           type == TestType::NS_TEMPORAL ||
           type == TestType::MAG_TEMPORAL ||
           type == TestType::FULL_TEMPORAL;
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
        << "    CH_NS                   - CH + NS (theta advected by U)\n"
        << "    POISSON_MAGNETIZATION   - Poisson + Magnetization (phi <-> M Picard)\n"
        << "    NS_MAGNETIZATION        - NS + Magnetization (Kelvin force)\n"
        << "    NS_POISSON_MAG          - NS + Poisson + Magnetization\n"
        << "    MAG_CH                  - Magnetization + CH (chi(theta) coupling)\n"
        << "    FULL_SYSTEM             - All four subsystems coupled\n"
        << "\n"
        << "  Temporal convergence tests (fix mesh, vary dt):\n"
        << "    CH_TEMPORAL             - CH temporal O(tau)\n"
        << "    NS_TEMPORAL             - NS temporal O(tau)\n"
        << "    MAG_TEMPORAL            - Magnetization temporal O(tau)\n"
        << "    FULL_TEMPORAL           - Full system temporal O(tau)\n"
        << "\n"
        << "  --refs <r1> <r2> ...  Refinement levels (default: 3 4 5)\n"
        << "  --steps <n>           Number of time steps (default: 10)\n"
        << "  --temporal-ref <r>    Refinement for temporal tests (default: 5)\n"
        << "  --temporal-steps <n1> <n2> ...  Time step counts for temporal tests\n"
        << "                        (default: 10 20 40 80 160)\n"
        << "  --help                Show this help\n"
        << "\n"
        << "Output:\n"
        << "  CSV files written to: mms/Rates/YYMMDD-HHMMSS-level_name.csv\n"
        << "\n"
        << "Examples:\n"
        << "  mpirun -np 4 " << program_name << " --level CH_STANDALONE --refs 3 4 5\n"
        << "  mpirun -np 4 " << program_name << " --level POISSON_MAGNETIZATION --refs 3 4 5 --steps 50\n"
        << "  mpirun -np 4 " << program_name << " --level MAG_CH --refs 3 4 5 --steps 50\n"
        << "  mpirun -np 4 " << program_name << " --level CH_TEMPORAL --temporal-ref 5\n"
        << "  mpirun -np 4 " << program_name << " --level FULL_TEMPORAL --temporal-ref 4 --temporal-steps 10 20 40 80\n";
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
    unsigned int temporal_ref = 5;
    std::vector<unsigned int> temporal_steps = {10, 20, 40, 80, 160};

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
            else if (level_str == "NS_MAGNETIZATION" || level_str == "NS_MAG")
                test_type = TestType::NS_MAGNETIZATION;
            else if (level_str == "NS_POISSON_MAG")
                test_type = TestType::NS_POISSON_MAG;
            else if (level_str == "MAG_CH")
                test_type = TestType::MAG_CH;
            else if (level_str == "FULL_SYSTEM" || level_str == "FULL")
                test_type = TestType::FULL_SYSTEM;
            else if (level_str == "MAG_TRANSPORT")
                test_type = TestType::MAG_TRANSPORT;

                // Temporal tests
            else if (level_str == "CH_TEMPORAL")
                test_type = TestType::CH_TEMPORAL;
            else if (level_str == "NS_TEMPORAL")
                test_type = TestType::NS_TEMPORAL;
            else if (level_str == "MAG_TEMPORAL")
                test_type = TestType::MAG_TEMPORAL;
            else if (level_str == "FULL_TEMPORAL")
                test_type = TestType::FULL_TEMPORAL;

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
        else if (arg == "--temporal-ref" && i + 1 < argc)
        {
            temporal_ref = std::stoi(argv[++i]);
        }
        else if (arg == "--temporal-steps" && i + 1 < argc)
        {
            temporal_steps.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-')
            {
                temporal_steps.push_back(std::stoi(argv[++i]));
            }
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

        if (is_temporal_test(test_type))
        {
            std::cout << "Mode: TEMPORAL CONVERGENCE\n";
            std::cout << "Fixed refinement: " << temporal_ref << "\n";
            std::cout << "Time step counts:";
            for (auto n : temporal_steps) std::cout << " " << n;
            std::cout << "\n";
        }
        else
        {
            std::cout << "Mode: SPATIAL CONVERGENCE\n";
            std::cout << "Refinements:";
            for (auto r : refinements) std::cout << " " << r;
            std::cout << "\n";
            std::cout << "Time steps: " << n_time_steps << "\n";
        }

        std::cout << "Domain: [" << params.domain.x_min << "," << params.domain.x_max
            << "] x [" << params.domain.y_min << "," << params.domain.y_max << "]\n";
        std::cout << "epsilon = " << params.physics.epsilon
            << ", gamma = " << params.physics.mobility << "\n";
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

        case TestType::NS_MAGNETIZATION:
            {
                CoupledMMSConvergenceResult result = run_ns_magnetization_mms(
                    refinements, params, n_time_steps, MPI_COMM_WORLD);

                if (this_rank == 0)
                {
                    result.print();
                    result.write_csv(csv_name);
                }
                passed = result.passes();
                break;
            }

        case TestType::NS_POISSON_MAG:
            {
                CoupledMMSConvergenceResult result = run_ns_poisson_mag_mms(
                    refinements, params, n_time_steps, MPI_COMM_WORLD);

                if (this_rank == 0)
                {
                    result.print();
                    result.write_csv(csv_name);
                }
                passed = result.passes();
                break;
            }

        case TestType::MAG_CH:
            {
                CoupledMMSConvergenceResult result = run_mag_ch_mms(
                    refinements, params, n_time_steps, MPI_COMM_WORLD);

                if (this_rank == 0)
                {
                    result.print();
                    result.write_csv(csv_name);
                }
                passed = result.passes();
                break;
            }

        case TestType::FULL_SYSTEM:
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
            }

        case TestType::MAG_TRANSPORT:
            {
                MagMMSConvergenceResult result = run_magnetization_transport_mms(
                    refinements, params, n_time_steps, MPI_COMM_WORLD);

                if (this_rank == 0)
                {
                    result.print();
                    result.write_csv(csv_name);
                }
                passed = result.passes();
                break;
            }

        // ====== TEMPORAL CONVERGENCE TESTS ======
        case TestType::CH_TEMPORAL:
        case TestType::NS_TEMPORAL:
        case TestType::MAG_TEMPORAL:
        case TestType::FULL_TEMPORAL:
            {
                TemporalTestLevel temporal_level;
                if (test_type == TestType::CH_TEMPORAL)
                    temporal_level = TemporalTestLevel::CH_TEMPORAL;
                else if (test_type == TestType::NS_TEMPORAL)
                    temporal_level = TemporalTestLevel::NS_TEMPORAL;
                else if (test_type == TestType::MAG_TEMPORAL)
                    temporal_level = TemporalTestLevel::MAG_TEMPORAL;
                else
                    temporal_level = TemporalTestLevel::FULL_TEMPORAL;

                TemporalConvergenceResult result = run_temporal_mms_test(
                    temporal_level, temporal_ref, params, temporal_steps, MPI_COMM_WORLD);

                if (this_rank == 0)
                {
                    result.print();
                    result.write_csv(csv_name);
                }
                passed = result.passes();
                break;
            }
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
