// ============================================================================
// utilities/sparsity_export.cc - Sparsity Pattern Export & Analysis
// ============================================================================

#include "sparsity_export.h"

#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <numeric>

SparsityAnalysis
analyze_sparsity(const dealii::TrilinosWrappers::SparseMatrix& matrix,
                 const std::string& name)
{
    SparsityAnalysis result;
    result.name = name;

    // Get global dimensions
    result.n_rows = matrix.m();
    result.n_cols = matrix.n();
    result.total_nnz = matrix.n_nonzero_elements();

    // We need to iterate over locally owned rows
    // Trilinos provides local_range()
    const auto range = matrix.local_range();
    const unsigned int local_start = range.first;
    const unsigned int local_end = range.second;
    const unsigned int n_local_rows = local_end - local_start;

    result.nnz_per_row.resize(n_local_rows, 0);

    unsigned int max_bw = 0;
    double sum_bw = 0.0;
    unsigned long long local_nnz_count = 0;

    for (unsigned int i = local_start; i < local_end; ++i)
    {
        const unsigned int local_i = i - local_start;

        // Get the number of entries in this row
        // Use row_length() for Trilinos matrices
        const int row_length = matrix.row_length(i);
        result.nnz_per_row[local_i] = static_cast<unsigned int>(row_length);
        local_nnz_count += row_length;

        // Iterate over entries to find bandwidth
        // For Trilinos, we iterate using the matrix iterator
        for (auto entry = matrix.begin(i); entry != matrix.end(i); ++entry)
        {
            const unsigned int j = entry->column();
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

void
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

            const auto range = matrix.local_range();
            for (unsigned int i = 0; i < analysis.nnz_per_row.size(); ++i)
            {
                csv << (range.first + i) << "," << analysis.nnz_per_row[i] << "\n";
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
        dealii::DynamicSparsityPattern dsp(analysis.n_rows, analysis.n_cols);

        const auto range = matrix.local_range();
        for (unsigned int i = range.first; i < range.second; ++i)
        {
            for (auto entry = matrix.begin(i); entry != matrix.end(i); ++entry)
            {
                dsp.add(i, entry->column());
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

void
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
