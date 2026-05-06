// ============================================================================
// mms/magnetic/magnetic_mms_test.h - Magnetic Standalone MMS Test (STUB)
//
// STATUS: NOT IMPLEMENTED. The runners below return zero-initialized
// results and `passes()` always returns false. Any test driver invoking
// MAGNETIC_STANDALONE or MAG_TEMPORAL will be reported as FAILED by
// design — they cannot succeed until this file is fleshed out.
//
// Use ch_magnetic_mms_test or full_system_mms_test for end-to-end
// magnetic-coupled MMS coverage instead.
// ============================================================================
#ifndef MAGNETIC_MMS_TEST_H
#define MAGNETIC_MMS_TEST_H

#include "utilities/parameters.h"
#include <mpi.h>
#include <vector>
#include <string>
#include <iostream>

// Result struct for magnetic standalone MMS
struct MagneticMMSResult
{
    double phi_L2 = 0.0, phi_H1 = 0.0;
    double Mx_L2 = 0.0, My_L2 = 0.0;
    double M_L2 = 0.0;          // Combined M error (used by temporal convergence)
    double h = 0.0;
    unsigned int n_dofs = 0;
    double wall_time = 0.0;
    double total_time = 0.0;    // Alias for wall_time (used by temporal convergence)
};

struct MagneticMMSConvergenceResult
{
    std::vector<unsigned int> refinements;
    std::vector<MagneticMMSResult> results;

    void print_summary(std::ostream& out) const
    {
        out << "\n"
            << "  ============================================================\n"
            << "  *** MAGNETIC STANDALONE MMS — NOT IMPLEMENTED ***\n"
            << "  ============================================================\n"
            << "  This test cannot pass; passes() returns false unconditionally.\n"
            << "  Use --level CH_MAGNETIC or FULL_SYSTEM for magnetic coverage.\n"
            << "  See mms/magnetic/magnetic_mms_test.h for status.\n"
            << "  ============================================================\n\n";
    }

    void print() const
    {
        print_summary(std::cout);
    }

    void write_csv(const std::string& /*filename*/) const
    {
        // STUB: nothing to write yet
    }

    bool passed() const { return false; }
    bool passes() const { return false; }
};

// Standalone magnetic MMS convergence test (STUB — see file header).
inline MagneticMMSConvergenceResult run_magnetic_mms_standalone(
    const std::vector<unsigned int>& /*refinements*/,
    const Parameters& /*params*/,
    unsigned int /*n_time_steps*/,
    MPI_Comm /*comm*/)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        std::cerr << "\n  *** MAGNETIC_STANDALONE: not implemented (stub). ***\n"
                  << "  This call will produce a result struct with passes()==false.\n"
                  << "  See mms/magnetic/magnetic_mms_test.h header for context.\n\n";
    }
    return MagneticMMSConvergenceResult{};
}

// Single-level magnetic MMS for temporal convergence (STUB — see file header).
inline MagneticMMSResult run_magnetic_mms_single(
    unsigned int /*refinement*/,
    const Parameters& /*params*/,
    unsigned int /*n_time_steps*/,
    MPI_Comm /*comm*/)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
        std::cerr << "  [MAG_TEMPORAL] WARNING: stub — returning zero-error result.\n";
    return MagneticMMSResult{};
}

#endif // MAGNETIC_MMS_TEST_H
