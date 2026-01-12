// ============================================================================
// utilities/mpi_utils.h - MPI Utility Functions
//
// Thin wrappers for common MPI operations used in parallel diagnostics.
// All functions are inline for header-only usage.
// ============================================================================
#ifndef MPI_TOOLS_H
#define MPI_TOOLS_H

#include <mpi.h>
#include <algorithm>
#include <limits>

namespace MPIUtils
{

// ============================================================================
// Rank and size queries
// ============================================================================

inline int rank(MPI_Comm comm = MPI_COMM_WORLD)
{
    int r;
    MPI_Comm_rank(comm, &r);
    return r;
}

inline int size(MPI_Comm comm = MPI_COMM_WORLD)
{
    int s;
    MPI_Comm_size(comm, &s);
    return s;
}

inline bool is_root(MPI_Comm comm = MPI_COMM_WORLD)
{
    return rank(comm) == 0;
}

// ============================================================================
// Scalar reductions
// ============================================================================

inline double reduce_sum(double local_value, MPI_Comm comm = MPI_COMM_WORLD)
{
    double global_value;
    MPI_Allreduce(&local_value, &global_value, 1, MPI_DOUBLE, MPI_SUM, comm);
    return global_value;
}

inline double reduce_min(double local_value, MPI_Comm comm = MPI_COMM_WORLD)
{
    double global_value;
    MPI_Allreduce(&local_value, &global_value, 1, MPI_DOUBLE, MPI_MIN, comm);
    return global_value;
}

inline double reduce_max(double local_value, MPI_Comm comm = MPI_COMM_WORLD)
{
    double global_value;
    MPI_Allreduce(&local_value, &global_value, 1, MPI_DOUBLE, MPI_MAX, comm);
    return global_value;
}

// ============================================================================
// Integer reductions
// ============================================================================

inline unsigned int reduce_sum(unsigned int local_value, MPI_Comm comm = MPI_COMM_WORLD)
{
    unsigned int global_value;
    MPI_Allreduce(&local_value, &global_value, 1, MPI_UNSIGNED, MPI_SUM, comm);
    return global_value;
}

inline unsigned int reduce_max(unsigned int local_value, MPI_Comm comm = MPI_COMM_WORLD)
{
    unsigned int global_value;
    MPI_Allreduce(&local_value, &global_value, 1, MPI_UNSIGNED, MPI_MAX, comm);
    return global_value;
}

// ============================================================================
// Multiple values at once (more efficient than separate calls)
// ============================================================================

/**
 * @brief Reduce min and max of a value simultaneously
 * @param local_value Local value on this rank
 * @param global_min Output: global minimum
 * @param global_max Output: global maximum
 * @param comm MPI communicator
 */
inline void reduce_minmax(double local_value,
                          double& global_min,
                          double& global_max,
                          MPI_Comm comm = MPI_COMM_WORLD)
{
    double local_data[2] = {local_value, -local_value};
    double global_data[2];

    // MPI_MIN on value gives min; MPI_MIN on -value gives -max
    MPI_Allreduce(local_data, global_data, 2, MPI_DOUBLE, MPI_MIN, comm);

    global_min = global_data[0];
    global_max = -global_data[1];
}

/**
 * @brief Reduce multiple sums at once
 * @param local_values Array of local values
 * @param global_values Output array of global sums
 * @param count Number of values
 * @param comm MPI communicator
 */
inline void reduce_sum_array(const double* local_values,
                             double* global_values,
                             int count,
                             MPI_Comm comm = MPI_COMM_WORLD)
{
    MPI_Allreduce(local_values, global_values, count, MPI_DOUBLE, MPI_SUM, comm);
}

// ============================================================================
// Barrier
// ============================================================================

inline void barrier(MPI_Comm comm = MPI_COMM_WORLD)
{
    MPI_Barrier(comm);
}

} // namespace MPIUtils

#endif // MPI_TOOLS_H