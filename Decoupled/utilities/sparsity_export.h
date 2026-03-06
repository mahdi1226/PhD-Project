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
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/base/conditional_ostream.h>

#include <Epetra_CrsMatrix.h>
#include <Epetra_Map.h>

#include <string>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>
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
inline SparsityAnalysis
analyze_sparsity(const dealii::TrilinosWrappers::SparseMatrix& matrix,
                 const std::string& name)
{
    SparsityAnalysis result;
    result.name = name;

    // Get global dimensions
    result.n_rows = matrix.m();
    result.n_cols = matrix.n();
    result.total_nnz = matrix.n_nonzero_elements();

    // Iterate over locally owned rows using the Epetra RowMap.
    // NOTE: local_range() assumes contiguous row ownership, which fails
    // for coupled systems like CH (θ block + shifted ψ block = non-contiguous).
    // Using the RowMap's GID list handles arbitrary distributions correctly.
    const Epetra_CrsMatrix& epetra = matrix.trilinos_matrix();
    const Epetra_Map& row_map = epetra.RowMap();
    const int n_local_rows = row_map.NumMyElements();

    result.nnz_per_row.resize(static_cast<unsigned int>(n_local_rows), 0);

    unsigned int max_bw = 0;
    double sum_bw = 0.0;
    unsigned long long local_nnz_count = 0;

    for (int local_i = 0; local_i < n_local_rows; ++local_i)
    {
        const unsigned int i = static_cast<unsigned int>(row_map.GID(local_i));

        // Use Epetra's native row access (bypasses deal.II iterators which
        // assume contiguous local_range and fail for non-contiguous maps)
        int num_entries = 0;
        double* values = nullptr;
        int* indices = nullptr;
        const int err = epetra.ExtractMyRowView(local_i, num_entries, values, indices);
        (void)err; // should be 0

        result.nnz_per_row[local_i] = static_cast<unsigned int>(num_entries);
        local_nnz_count += num_entries;

        // Column indices from ExtractMyRowView are LOCAL column indices;
        // convert to global using the ColMap
        const Epetra_Map& col_map = epetra.ColMap();
        for (int k = 0; k < num_entries; ++k)
        {
            const unsigned int j = static_cast<unsigned int>(col_map.GID(indices[k]));
            const unsigned int dist = (i > j) ? (i - j) : (j - i);
            if (dist > max_bw)
                max_bw = dist;
            sum_bw += dist;
        }
    }

    result.bandwidth = max_bw;
    result.half_bandwidth = max_bw;
    result.avg_bandwidth = (local_nnz_count > 0) ? sum_bw / local_nnz_count : 0.0;

    // Per-row nnz statistics
    if (n_local_rows > 0)
    {
        result.min_nnz_per_row = *std::min_element(result.nnz_per_row.begin(),
                                                    result.nnz_per_row.end());
        result.max_nnz_per_row = *std::max_element(result.nnz_per_row.begin(),
                                                    result.nnz_per_row.end());
        double sum = std::accumulate(result.nnz_per_row.begin(),
                                     result.nnz_per_row.end(), 0.0);
        result.avg_nnz_per_row = sum / n_local_rows;

        // Standard deviation
        double sq_sum = 0.0;
        for (auto v : result.nnz_per_row)
            sq_sum += (v - result.avg_nnz_per_row) * (v - result.avg_nnz_per_row);
        result.std_nnz_per_row = std::sqrt(sq_sum / n_local_rows);
    }

    result.density = (result.n_rows > 0 && result.n_cols > 0)
        ? static_cast<double>(result.total_nnz)
            / (static_cast<double>(result.n_rows) * result.n_cols)
        : 0.0;

    return result;
}

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
inline void
export_sparsity_pattern(const dealii::TrilinosWrappers::SparseMatrix& matrix,
                        const std::string& name,
                        const std::string& output_dir,
                        MPI_Comm comm,
                        dealii::ConditionalOStream& pcout)
{
    int rank;
    MPI_Comm_rank(comm, &rank);

    // Analyze sparsity (each rank analyzes its local rows)
    SparsityAnalysis analysis = analyze_sparsity(matrix, name);

    // MPI reduction for bandwidth (take global max)
    unsigned int global_bandwidth = 0;
    MPI_Reduce(&analysis.bandwidth, &global_bandwidth, 1,
               MPI_UNSIGNED, MPI_MAX, 0, comm);

    if (rank != 0)
        return;

    analysis.bandwidth = global_bandwidth;

    // ========================================================================
    // 1. Bandwidth CSV: per-row nnz distribution + summary
    // ========================================================================
    {
        std::string csv_path = output_dir + "/" + name + "_bandwidth.csv";
        std::ofstream csv(csv_path);
        if (csv.is_open())
        {
            // Header
            csv << "# Sparsity analysis: " << name << "\n";
            csv << "# Rows: " << analysis.n_rows
                << ", Cols: " << analysis.n_cols
                << ", NNZ: " << analysis.total_nnz << "\n";
            csv << "# Bandwidth: " << analysis.bandwidth
                << ", Avg bandwidth: " << std::fixed << std::setprecision(2)
                << analysis.avg_bandwidth << "\n";
            csv << "# Min nnz/row: " << analysis.min_nnz_per_row
                << ", Max nnz/row: " << analysis.max_nnz_per_row
                << ", Avg nnz/row: " << std::setprecision(2)
                << analysis.avg_nnz_per_row
                << ", Std nnz/row: " << analysis.std_nnz_per_row << "\n";
            csv << "# Density: " << std::scientific << std::setprecision(4)
                << analysis.density << "\n";
            csv << "row,nnz\n";

            // Use RowMap GIDs for correct row indices (handles non-contiguous maps)
            const Epetra_CrsMatrix& epetra_csv = matrix.trilinos_matrix();
            const Epetra_Map& rmap_csv = epetra_csv.RowMap();
            for (unsigned int i = 0; i < analysis.nnz_per_row.size(); ++i)
            {
                csv << rmap_csv.GID(static_cast<int>(i)) << ","
                    << analysis.nnz_per_row[i] << "\n";
            }
            csv.close();
            pcout << "[Sparsity] Wrote " << csv_path << "\n";
        }
    }

    // ========================================================================
    // 2. SVG visualization (only for small matrices to avoid huge files)
    // ========================================================================
    const unsigned int SVG_THRESHOLD = 5000;  // Max rows for SVG export
    if (analysis.n_rows <= SVG_THRESHOLD)
    {
        // Build a deal.II DynamicSparsityPattern from the Trilinos matrix
        // Use Epetra native API to handle non-contiguous row ownership
        dealii::DynamicSparsityPattern dsp(analysis.n_rows, analysis.n_cols);

        const Epetra_CrsMatrix& epetra_svg = matrix.trilinos_matrix();
        const Epetra_Map& rmap_svg = epetra_svg.RowMap();
        const Epetra_Map& cmap_svg = epetra_svg.ColMap();
        const int n_local_svg = rmap_svg.NumMyElements();
        for (int local_i = 0; local_i < n_local_svg; ++local_i)
        {
            const unsigned int i = static_cast<unsigned int>(rmap_svg.GID(local_i));
            int num_entries = 0;
            double* values = nullptr;
            int* indices = nullptr;
            epetra_svg.ExtractMyRowView(local_i, num_entries, values, indices);
            for (int k = 0; k < num_entries; ++k)
            {
                const unsigned int j = static_cast<unsigned int>(cmap_svg.GID(indices[k]));
                dsp.add(i, j);
            }
        }

        dealii::SparsityPattern sp;
        sp.copy_from(dsp);

        // SVG
        {
            std::string svg_path = output_dir + "/" + name + "_sparsity.svg";
            std::ofstream svg(svg_path);
            if (svg.is_open())
            {
                sp.print_svg(svg);
                svg.close();
                pcout << "[Sparsity] Wrote " << svg_path
                      << " (" << analysis.n_rows << "x" << analysis.n_cols << ")\n";
            }
        }

        // Gnuplot
        {
            std::string gp_path = output_dir + "/" + name + "_sparsity.gnuplot";
            std::ofstream gp(gp_path);
            if (gp.is_open())
            {
                sp.print_gnuplot(gp);
                gp.close();
                pcout << "[Sparsity] Wrote " << gp_path << "\n";
            }
        }
    }
    else
    {
        pcout << "[Sparsity] " << name << " matrix too large for SVG ("
              << analysis.n_rows << " rows > " << SVG_THRESHOLD
              << "). Only bandwidth CSV written.\n";
    }
}

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
inline void
write_sparsity_summary(const std::vector<SparsityAnalysis>& analyses,
                       const std::string& output_dir,
                       bool renumbered,
                       dealii::ConditionalOStream& pcout)
{
    std::string path = output_dir + "/sparsity_summary.csv";
    std::ofstream f(path);
    if (!f.is_open()) return;

    f << "# Sparsity Pattern Summary\n";
    f << "# Cuthill-McKee renumbering: " << (renumbered ? "ON" : "OFF") << "\n";
    f << "matrix,rows,cols,nnz,bandwidth,avg_bandwidth,"
      << "min_nnz_row,max_nnz_row,avg_nnz_row,std_nnz_row,density\n";

    for (const auto& a : analyses)
    {
        f << a.name << ","
          << a.n_rows << "," << a.n_cols << ","
          << a.total_nnz << ","
          << a.bandwidth << ","
          << std::fixed << std::setprecision(2) << a.avg_bandwidth << ","
          << a.min_nnz_per_row << "," << a.max_nnz_per_row << ","
          << std::setprecision(2) << a.avg_nnz_per_row << ","
          << std::setprecision(2) << a.std_nnz_per_row << ","
          << std::scientific << std::setprecision(4) << a.density << "\n";
    }
    f.close();
    pcout << "[Sparsity] Wrote " << path << "\n";
}

#endif // SPARSITY_EXPORT_H
