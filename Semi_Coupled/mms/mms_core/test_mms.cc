// ============================================================================
// mms/mms_core/test_mms.cc - MMS Test Driver (PARALLEL)
//
// Uses Parameters defaults from parameters.h - NO OVERRIDES!
// MMS verifies production code with production parameters.
//
// Standalone tests (mms_verification.h):
//   CH_STANDALONE, NS_STANDALONE, MAGNETIC_STANDALONE
//
// Coupled tests (coupled_mms_test.h):
//   CH_MAGNETIC, MAGNETIC_NS, NS_CH, FULL_SYSTEM
//
// Temporal tests (temporal_convergence.h):
//   CH_TEMPORAL, NS_TEMPORAL, MAG_TEMPORAL, FULL_TEMPORAL
//
// CSV output: mms/Rates/YYMMDD-HHMMSS-level_name.csv
// ============================================================================

#include "mms_verification.h"
#include "mms/coupled/coupled_mms_test.h"
#include "mms/magnetic/magnetic_mms_test.h"
#include "temporal_convergence.h"
#include "long_duration_mms.h"
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
    NS_STANDALONE,
    MAGNETIC_STANDALONE,
    // Coupled (paper's algorithm order: source → target)
    CH_MAGNETIC,
    MAGNETIC_NS,
    NS_CH,
    FULL_SYSTEM,
    // Temporal convergence
    CH_TEMPORAL,
    NS_TEMPORAL,
    MAG_TEMPORAL,
    FULL_TEMPORAL,
    // Long-duration stability
    CH_LONG,
    CH_NS_LONG,
    FULL_LONG
};

static std::string test_type_to_string(TestType type)
{
    switch (type)
    {
    case TestType::CH_STANDALONE: return "CH_STANDALONE";
    case TestType::NS_STANDALONE: return "NS_STANDALONE";
    case TestType::MAGNETIC_STANDALONE: return "MAGNETIC_STANDALONE";
    case TestType::CH_MAGNETIC: return "CH_MAGNETIC";
    case TestType::MAGNETIC_NS: return "MAGNETIC_NS";
    case TestType::NS_CH: return "NS_CH";
    case TestType::FULL_SYSTEM: return "FULL_SYSTEM";
    case TestType::CH_TEMPORAL: return "CH_TEMPORAL";
    case TestType::NS_TEMPORAL: return "NS_TEMPORAL";
    case TestType::MAG_TEMPORAL: return "MAG_TEMPORAL";
    case TestType::FULL_TEMPORAL: return "FULL_TEMPORAL";
    case TestType::CH_LONG: return "CH_LONG";
    case TestType::CH_NS_LONG: return "CH_NS_LONG";
    case TestType::FULL_LONG: return "FULL_LONG";
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

static bool is_long_duration_test(TestType type)
{
    return type == TestType::CH_LONG ||
           type == TestType::CH_NS_LONG ||
           type == TestType::FULL_LONG;
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
        << "    NS_STANDALONE           - Navier-Stokes only\n"
        << "    MAGNETIC_STANDALONE     - Monolithic M+phi block system\n"
        << "\n"
        << "  Coupled tests (paper's algorithm order):\n"
        << "    CH_MAGNETIC             - CH → Magnetic: chi(theta) coupling\n"
        << "    MAGNETIC_NS             - Magnetic → NS: Kelvin force coupling\n"
        << "    NS_CH                   - NS → CH: advection coupling\n"
        << "    FULL_SYSTEM             - All subsystems coupled\n"
        << "\n"
        << "  Temporal convergence tests (fix mesh, vary dt):\n"
        << "    CH_TEMPORAL             - CH temporal O(tau)\n"
        << "    NS_TEMPORAL             - NS temporal O(tau)\n"
        << "    MAG_TEMPORAL            - Magnetization temporal O(tau)\n"
        << "    FULL_TEMPORAL           - Full system temporal O(tau)\n"
        << "\n"
        << "  Long-duration stability tests (fix mesh+dt, many steps, track error growth):\n"
        << "    CH_LONG                 - CH standalone, 500 steps, error per step\n"
        << "    CH_NS_LONG              - CH + NS coupled (TODO)\n"
        << "    FULL_LONG               - Full system coupled (TODO)\n"
        << "\n"
        << "  --refs <r1> <r2> ...  Refinement levels (default: 3 4 5)\n"
        << "  --steps <n>           Number of time steps (default: 10)\n"
        << "  --temporal-ref <r>    Refinement for temporal tests (default: 5)\n"
        << "  --mms-analytical      Use analytical d/dt in MMS sources (default: discrete\n"
        << "                        differences, which cancel BE truncation by construction).\n"
        << "                        Pass --mms-analytical to expose the BE temporal rate ~1.0.\n"
        << "  --tau-M <value>       Override params.physics.tau_M (default 1e-6 = stiff\n"
        << "                        equilibrium, masks M's BE rate). Set to ~0.01 to give\n"
        << "                        M its own dynamics and expose its temporal rate.\n"
        << "  --temporal-steps <n1> <n2> ...  Time step counts for temporal tests\n"
        << "                        (default: 10 20 40 80 160)\n"
        << "  --help                Show this help\n"
        << "\n"
        << "Output:\n"
        << "  CSV files written to: mms/Rates/YYMMDD-HHMMSS-level_name.csv\n"
        << "\n"
        << "Examples:\n"
        << "  mpirun -np 4 " << program_name << " --level CH_STANDALONE --refs 3 4 5\n"
        << "  mpirun -np 4 " << program_name << " --level MAGNETIC_STANDALONE --refs 3 4 5 --steps 10\n"
        << "  mpirun -np 4 " << program_name << " --level CH_MAGNETIC --refs 3 4 5 --steps 50\n"
        << "  mpirun -np 4 " << program_name << " --level MAGNETIC_NS --refs 3 4 5 --steps 50\n"
        << "  mpirun -np 4 " << program_name << " --level NS_CH --refs 3 4 5 --steps 50\n"
        << "  mpirun -np 4 " << program_name << " --level CH_TEMPORAL --temporal-ref 5\n"
        << "  mpirun -np 4 " << program_name << " --level FULL_TEMPORAL --temporal-ref 4 --temporal-steps 10 20 40 80\n";
}

int main(int argc, char* argv[])
{
    // Initialize MPI
    dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

    const unsigned int this_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    const unsigned int n_ranks = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

    // Parse command line
    TestType test_type = TestType::CH_STANDALONE;
    std::vector<unsigned int> refinements = {3, 4, 5};
    unsigned int n_time_steps = 10;
    unsigned int temporal_ref = 5;
    bool mms_analytical_dt = false;  // --mms-analytical: use analytical d/dt
                                     // in MMS sources (vs discrete differences).
                                     // Required to measure formal BE temporal rate.
    double tau_M_override = -1.0;    // --tau-M VAL: override params.physics.tau_M.
                                     // Default tau_M = 1e-6 puts M in stiff
                                     // equilibrium (M_n = chi*H_n forced at every
                                     // step) which masks any BE truncation error
                                     // in M. Use --tau-M >= 1e-2 to give M its own
                                     // dynamics and expose the formal rate ~1.0
                                     // in the temporal-convergence test. Negative
                                     // value (default) means no override.
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
            else if (level_str == "NS_STANDALONE")
                test_type = TestType::NS_STANDALONE;
            else if (level_str == "MAGNETIC_STANDALONE" || level_str == "MAGNETIC")
                test_type = TestType::MAGNETIC_STANDALONE;

            // Coupled tests (paper's algorithm order)
            else if (level_str == "CH_MAGNETIC")
                test_type = TestType::CH_MAGNETIC;
            else if (level_str == "MAGNETIC_NS")
                test_type = TestType::MAGNETIC_NS;
            else if (level_str == "NS_CH")
                test_type = TestType::NS_CH;
            else if (level_str == "FULL_SYSTEM" || level_str == "FULL")
                test_type = TestType::FULL_SYSTEM;

            // Temporal tests
            else if (level_str == "CH_TEMPORAL")
                test_type = TestType::CH_TEMPORAL;
            else if (level_str == "NS_TEMPORAL")
                test_type = TestType::NS_TEMPORAL;
            else if (level_str == "MAG_TEMPORAL")
                test_type = TestType::MAG_TEMPORAL;
            else if (level_str == "FULL_TEMPORAL")
                test_type = TestType::FULL_TEMPORAL;

            // Long-duration stability tests
            else if (level_str == "CH_LONG")
                test_type = TestType::CH_LONG;
            else if (level_str == "CH_NS_LONG")
                test_type = TestType::CH_NS_LONG;
            else if (level_str == "FULL_LONG")
                test_type = TestType::FULL_LONG;

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
        else if (arg == "--mms-analytical")
        {
            mms_analytical_dt = true;
        }
        else if (arg == "--tau-M" && i + 1 < argc)
        {
            tau_M_override = std::stod(argv[++i]);
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
    params.mms_analytical_dt = mms_analytical_dt;
    if (tau_M_override > 0.0)
        params.physics.tau_M = tau_M_override;

    if (this_rank == 0)
    {
        std::cout << "\n=== Parallel MMS Test ===\n";
        std::cout << "MPI ranks: " << n_ranks << "\n";
        std::cout << "Level: " << test_type_to_string(test_type) << "\n";
        if (tau_M_override > 0.0)
            std::cout << "tau_M:    " << params.physics.tau_M
                      << " (overridden via --tau-M)\n";
        std::cout << "MMS d/dt: "
                  << (mms_analytical_dt ? "ANALYTICAL (exposes BE rate)"
                                        : "discrete (matches discrete scheme exactly)")
                  << "\n";

        if (is_temporal_test(test_type))
        {
            std::cout << "Mode: TEMPORAL CONVERGENCE\n";
            std::cout << "Fixed refinement: " << temporal_ref << "\n";
            std::cout << "Time step counts:";
            for (auto n : temporal_steps) std::cout << " " << n;
            std::cout << "\n";
        }
        else if (is_long_duration_test(test_type))
        {
            std::cout << "Mode: LONG-DURATION STABILITY\n";
            std::cout << "Fixed refinement: " << temporal_ref << "\n";
            std::cout << "Time steps: " << n_time_steps << "\n";
            std::cout << "Time interval: [0.1, 0.6]\n";
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
        case TestType::NS_STANDALONE:
            {
                MMSLevel level;
                if (test_type == TestType::CH_STANDALONE)
                    level = MMSLevel::CH_STANDALONE;
                else
                    level = MMSLevel::NS_STANDALONE;

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

        case TestType::MAGNETIC_STANDALONE:
            {
                MagneticMMSConvergenceResult result = run_magnetic_mms_standalone(
                    refinements, params, n_time_steps, MPI_COMM_WORLD);

                if (this_rank == 0)
                {
                    result.print();
                    result.write_csv(csv_name);
                }
                passed = result.passes();
                break;
            }

        // ====== COUPLED TESTS (paper's algorithm order) ======
        case TestType::CH_MAGNETIC:
            {
                CoupledMMSConvergenceResult result = run_ch_magnetic_mms(
                    refinements, params, n_time_steps, MPI_COMM_WORLD);

                if (this_rank == 0)
                {
                    result.print();
                    result.write_csv(csv_name);
                }
                passed = result.passes();
                break;
            }

        case TestType::MAGNETIC_NS:
            {
                CoupledMMSConvergenceResult result = run_magnetic_ns_mms(
                    refinements, params, n_time_steps, MPI_COMM_WORLD);

                if (this_rank == 0)
                {
                    result.print();
                    result.write_csv(csv_name);
                }
                passed = result.passes();
                break;
            }

        case TestType::NS_CH:
            {
                CoupledMMSConvergenceResult result = run_ns_ch_mms(
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

        // ====== LONG-DURATION STABILITY TESTS ======
        case TestType::CH_LONG:
        case TestType::CH_NS_LONG:
        case TestType::FULL_LONG:
            {
                LongDurationLevel long_level;
                if (test_type == TestType::CH_LONG)
                    long_level = LongDurationLevel::CH_LONG;
                else if (test_type == TestType::CH_NS_LONG)
                    long_level = LongDurationLevel::CH_NS_LONG;
                else
                    long_level = LongDurationLevel::FULL_LONG;

                unsigned int long_steps = (n_time_steps == 10) ? 500 : n_time_steps;

                LongDurationResult result = run_long_duration_mms_test(
                    long_level, temporal_ref, params, long_steps, 1, MPI_COMM_WORLD);

                if (this_rank == 0)
                {
                    result.print_summary();
                    result.write_csv(csv_name);
                }
                // Stub runners set not_implemented=true; force a fail so CI
                // can't mistake an empty snapshot list for a passing run.
                passed = !result.is_exponential_growth && !result.not_implemented;
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
