// ============================================================================
// utilities/timestamp.h - Timestamped Filename Generation
//
// Provides consistent timestamp formatting for output files across all
// subsystems. Format: mmddyy_HHMMSS (e.g., 021126_143052)
//
// Usage:
//   std::string ts = get_timestamp();          // "021126_143052"
//   std::string f  = timestamped_filename(     // "021126_143052_poisson_convergence.csv"
//                      "poisson_convergence", ".csv");
//
// Only rank 0 should generate timestamps. Broadcast to other ranks
// to ensure all MPI processes use the same filename.
// ============================================================================
#ifndef TIMESTAMP_H
#define TIMESTAMP_H

#include <string>
#include <ctime>
#include <cstring>
#include <mpi.h>

// Returns timestamp string: mmddyy_HHMMSS
inline std::string get_timestamp()
{
    std::time_t now = std::time(nullptr);
    std::tm* lt = std::localtime(&now);

    char buf[16];
    std::snprintf(buf, sizeof(buf), "%02d%02d%02d_%02d%02d%02d",
                  lt->tm_mon + 1,   // month 01-12
                  lt->tm_mday,       // day 01-31
                  lt->tm_year % 100, // year 00-99
                  lt->tm_hour,       // hour 00-23
                  lt->tm_min,        // minute 00-59
                  lt->tm_sec);       // second 00-59
    return std::string(buf);
}

// Returns: "mmddyy_HHMMSS_<base><ext>"
// e.g., timestamped_filename("poisson_convergence", ".csv")
//     → "021126_143052_poisson_convergence.csv"
inline std::string timestamped_filename(const std::string& base,
                                        const std::string& ext = ".csv")
{
    return get_timestamp() + "_" + base + ext;
}

// MPI-safe version: rank 0 generates, broadcasts to all ranks
inline std::string timestamped_filename_mpi(const std::string& base,
                                            const std::string& ext,
                                            MPI_Comm mpi_comm)
{
    int rank;
    MPI_Comm_rank(mpi_comm, &rank);

    char buf[256];
    std::memset(buf, 0, sizeof(buf));

    if (rank == 0)
    {
        std::string name = timestamped_filename(base, ext);
        std::strncpy(buf, name.c_str(), sizeof(buf) - 1);
    }

    MPI_Bcast(buf, 256, MPI_CHAR, 0, mpi_comm);
    return std::string(buf);
}

#endif // TIMESTAMP_H