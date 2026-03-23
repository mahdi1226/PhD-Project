// ============================================================================
// utilities/sparsity_export.h - Sparsity Pattern Export & Analysis
//
// Exports sparsity patterns from Trilinos matrices as:
//   - SVG visualization (deal.II SparsityPattern::print_svg)
//   - Gnuplot format (SparsityPattern::print_gnuplot)
//   - Bandwidth CSV (per-row nnz distribution + global bandwidth)
//
// Works with parallel Trilinos matrices by extracting the local sparsity
// pattern on rank 0 (or collecting global pattern for small matrices).
//
// USAGE:
//   export_sparsity_pattern(matrix, "ch", output_dir, comm);
//   // Creates: ch_sparsity.svg, ch_sparsity.gnuplot, ch_bandwidth.csv
//
// For parallel: each rank exports its LOCAL portion. Rank 0 writes summary.
// ============================================================================
#ifndef SPARSITY_EXPORT_H
#define SPARSITY_EXPORT_H

#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/base/conditional_ostream.h>

#include <string>
#include <vector>
#include <mpi.h>

/**
 * @brief Sparsity pattern analysis results
 */
struct SparsityAnalysis
{
    std::string name;                    // Matrix name (e.g., "CH", "Poisson")
    unsigned int n_rows = 0;             // Total rows
    unsigned int n_cols = 0;             // Total columns
    unsigned long long total_nnz = 0;    // Total nonzero entries

    // Bandwidth metrics
    unsigned int bandwidth = 0;          // max |i-j| for nonzero a_ij
    double avg_bandwidth = 0.0;          // average |i-j| for nonzero a_ij
    unsigned int half_bandwidth = 0;     // max distance from diagonal in any row

    // Per-row statistics
    unsigned int min_nnz_per_row = 0;
    unsigned int max_nnz_per_row = 0;
    double avg_nnz_per_row = 0.0;
    double std_nnz_per_row = 0.0;        // Standard deviation

    // Density
    double density = 0.0;               // nnz / (n_rows * n_cols)

    // Per-row nnz distribution (for histogram)
    std::vector<unsigned int> nnz_per_row;
};

/**
 * @brief Analyze sparsity pattern of a Trilinos matrix (local rows on this rank)
 *
 * @param matrix The Trilinos sparse matrix
 * @param name Label for this matrix (e.g., "CH", "NS")
 * @return SparsityAnalysis struct with all metrics
 */
SparsityAnalysis
analyze_sparsity(const dealii::TrilinosWrappers::SparseMatrix& matrix,
                 const std::string& name);

/**
 * @brief Export sparsity pattern to SVG + gnuplot + bandwidth CSV
 *
 * Only rank 0 writes files. For small matrices (< 5000 rows), exports full SVG.
 * For large matrices, only exports bandwidth CSV and summary.
 *
 * @param matrix The Trilinos sparse matrix to export
 * @param name Label (used as filename prefix, e.g., "ch", "poisson")
 * @param output_dir Directory for output files
 * @param comm MPI communicator
 * @param pcout Conditional output stream (rank 0)
 */
void
export_sparsity_pattern(const dealii::TrilinosWrappers::SparseMatrix& matrix,
                        const std::string& name,
                        const std::string& output_dir,
                        MPI_Comm comm,
                        dealii::ConditionalOStream& pcout);

/**
 * @brief Write a combined sparsity summary CSV with all matrices
 *
 * Creates a single row per invocation with columns for each matrix's
 * bandwidth, nnz, avg_nnz_per_row, density.
 *
 * @param analyses Vector of SparsityAnalysis results
 * @param output_dir Directory for output file
 * @param renumbered Whether Cuthill-McKee was applied
 * @param pcout Conditional output stream
 */
void
write_sparsity_summary(const std::vector<SparsityAnalysis>& analyses,
                       const std::string& output_dir,
                       bool renumbered,
                       dealii::ConditionalOStream& pcout);

#endif // SPARSITY_EXPORT_H
