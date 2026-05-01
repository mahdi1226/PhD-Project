// ============================================================================
// solvers/magnetic_block_preconditioner.cc - Implementation
//
// Extracts M-block and phi-block sub-matrices from the monolithic
// [Mx | My | phi] system using Trilinos Epetra_CrsMatrix::ExtractMyRowView,
// then builds ILU on M and AMG on phi. The Epetra-direct path avoids a
// deal.II 9.7.1 SparseMatrix iterator bug at the M/phi block boundary.
// ============================================================================
#include "solvers/magnetic_block_preconditioner.h"

#include <Epetra_Comm.h>

#include <iostream>
#include <vector>

// ============================================================================
// Constructor
// ============================================================================
MagneticBlockPreconditioner::MagneticBlockPreconditioner(
    const dealii::TrilinosWrappers::SparseMatrix& system_matrix,
    const dealii::IndexSet& mag_locally_owned,
    dealii::types::global_dof_index n_M_dofs,
    MPI_Comm mpi_comm)
    : mag_locally_owned_(mag_locally_owned)
    , n_M_dofs_(n_M_dofs)
    , n_total_(system_matrix.m())
    , mpi_comm_(mpi_comm)
    , rank_(0)
{
    MPI_Comm_rank(mpi_comm_, &rank_);
    n_phi_dofs_ = n_total_ - n_M_dofs_;

    // Build per-block IndexSets (in monolithic GID numbering for M, reindexed
    // [0, n_phi_dofs_) for phi).
    M_owned_.set_size(n_M_dofs_);
    phi_owned_.set_size(n_phi_dofs_);
    for (auto it = mag_locally_owned_.begin(); it != mag_locally_owned_.end(); ++it)
    {
        const dealii::types::global_dof_index gid = *it;
        if (gid < n_M_dofs_)
            M_owned_.add_index(gid);
        else
            phi_owned_.add_index(gid - n_M_dofs_);
    }
    M_owned_.compress();
    phi_owned_.compress();

    // Extract sub-block matrices via Trilinos backend (no deal.II iterators)
    const Epetra_CrsMatrix& epetra_mat = system_matrix.trilinos_matrix();
    extract_M_block(epetra_mat);
    extract_phi_block(epetra_mat);

    // Working vectors for vmult
    r_M_.reinit(M_owned_, mpi_comm_);
    z_M_.reinit(M_owned_, mpi_comm_);
    r_phi_.reinit(phi_owned_, mpi_comm_);
    z_phi_.reinit(phi_owned_, mpi_comm_);

    // ILU on M block: mass coefficient (1/dt + 1/tau_M) ~ 1e6 dominates,
    // so ILU(0) is a strong preconditioner.
    {
        dealii::TrilinosWrappers::PreconditionILU::AdditionalData ilu_data;
        ilu_data.ilu_fill = 0;
        ilu_data.ilu_atol = 1e-10;
        ilu_data.ilu_rtol = 1.0;
        M_prec_.initialize(M_block_matrix_, ilu_data);
    }

    // AMG on phi block: standard elliptic Laplacian.
    {
        dealii::TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
        amg_data.elliptic = true;
        amg_data.higher_order_elements = false;
        amg_data.smoother_sweeps = 2;
        amg_data.aggregation_threshold = 0.02;
        amg_data.output_details = false;
        phi_prec_.initialize(phi_block_matrix_, amg_data);
    }
}

// ============================================================================
// Extract M-block: rows AND cols both in [0, n_M_dofs_), preserving GIDs.
// ============================================================================
void MagneticBlockPreconditioner::extract_M_block(
    const Epetra_CrsMatrix& epetra_mat)
{
    const Epetra_Comm& epetra_comm = epetra_mat.Comm();
    const int num_my_rows = epetra_mat.NumMyRows();

    // Collect this rank's M rows (those with GID < n_M_dofs_).
    std::vector<long long> my_M_gids;
    my_M_gids.reserve(num_my_rows);
    for (int local_row = 0; local_row < num_my_rows; ++local_row)
    {
        const long long gid = epetra_mat.GRID64(local_row);
        if (gid >= 0 && static_cast<dealii::types::global_dof_index>(gid) < n_M_dofs_)
            my_M_gids.push_back(gid);
    }
    std::sort(my_M_gids.begin(), my_M_gids.end());
    my_M_gids.erase(std::unique(my_M_gids.begin(), my_M_gids.end()), my_M_gids.end());
    const int n_my_M = static_cast<int>(my_M_gids.size());

    Epetra_Map M_row_map(static_cast<long long>(n_M_dofs_), n_my_M,
                         my_M_gids.data(), 0, epetra_comm);

    // First pass: count per-row M-block entries (cols in [0, n_M_dofs_)).
    std::vector<int> entries_per_row(n_my_M, 0);
    for (int i = 0; i < n_my_M; ++i)
    {
        const long long gid = my_M_gids[i];
        const int local_row = epetra_mat.LRID(gid);
        if (local_row < 0) continue;
        int num_entries = 0;
        double* values = nullptr;
        int* col_indices = nullptr;
        epetra_mat.ExtractMyRowView(local_row, num_entries, values, col_indices);
        for (int k = 0; k < num_entries; ++k)
        {
            const long long col_gid = epetra_mat.GCID64(col_indices[k]);
            if (col_gid >= 0 &&
                static_cast<dealii::types::global_dof_index>(col_gid) < n_M_dofs_)
                ++entries_per_row[i];
        }
    }

    // Build the Epetra CrsMatrix.
    auto M_crs = std::make_unique<Epetra_CrsMatrix>(
        Copy, M_row_map, entries_per_row.data(), /*static_profile=*/true);

    // Second pass: insert values.
    for (int i = 0; i < n_my_M; ++i)
    {
        const long long gid = my_M_gids[i];
        const int local_row = epetra_mat.LRID(gid);
        if (local_row < 0) continue;
        int num_entries = 0;
        double* values = nullptr;
        int* col_indices = nullptr;
        epetra_mat.ExtractMyRowView(local_row, num_entries, values, col_indices);

        std::vector<long long> col_gids;
        std::vector<double> col_vals;
        col_gids.reserve(num_entries);
        col_vals.reserve(num_entries);
        for (int k = 0; k < num_entries; ++k)
        {
            const long long col_gid = epetra_mat.GCID64(col_indices[k]);
            if (col_gid >= 0 &&
                static_cast<dealii::types::global_dof_index>(col_gid) < n_M_dofs_)
            {
                col_gids.push_back(col_gid);
                col_vals.push_back(values[k]);
            }
        }
        if (!col_gids.empty())
        {
            int err = M_crs->InsertGlobalValues(gid, static_cast<int>(col_gids.size()),
                                                col_vals.data(), col_gids.data());
            (void)err;
        }
    }

    M_crs->FillComplete(M_row_map, M_row_map);
    M_block_matrix_.reinit(*M_crs);  // copies into deal.II wrapper
}

// ============================================================================
// Extract phi-block: rows AND cols in [n_M_dofs_, n_total_), reindexed to
// [0, n_phi_dofs_).
// ============================================================================
void MagneticBlockPreconditioner::extract_phi_block(
    const Epetra_CrsMatrix& epetra_mat)
{
    const Epetra_Comm& epetra_comm = epetra_mat.Comm();
    const int num_my_rows = epetra_mat.NumMyRows();

    // Collect this rank's phi rows (those with GID >= n_M_dofs_), reindexed.
    std::vector<long long> my_phi_gids;
    my_phi_gids.reserve(num_my_rows);
    for (int local_row = 0; local_row < num_my_rows; ++local_row)
    {
        const long long gid = epetra_mat.GRID64(local_row);
        if (gid >= 0 &&
            static_cast<dealii::types::global_dof_index>(gid) >= n_M_dofs_)
        {
            my_phi_gids.push_back(gid - static_cast<long long>(n_M_dofs_));
        }
    }
    std::sort(my_phi_gids.begin(), my_phi_gids.end());
    my_phi_gids.erase(std::unique(my_phi_gids.begin(), my_phi_gids.end()), my_phi_gids.end());
    const int n_my_phi = static_cast<int>(my_phi_gids.size());

    Epetra_Map phi_row_map(static_cast<long long>(n_phi_dofs_), n_my_phi,
                           my_phi_gids.data(), 0, epetra_comm);

    // First pass: count per-row phi-block entries.
    std::vector<int> entries_per_row(n_my_phi, 0);
    for (int i = 0; i < n_my_phi; ++i)
    {
        const long long phi_gid = my_phi_gids[i];
        const long long orig_gid = phi_gid + static_cast<long long>(n_M_dofs_);
        const int local_row = epetra_mat.LRID(orig_gid);
        if (local_row < 0) continue;
        int num_entries = 0;
        double* values = nullptr;
        int* col_indices = nullptr;
        epetra_mat.ExtractMyRowView(local_row, num_entries, values, col_indices);
        for (int k = 0; k < num_entries; ++k)
        {
            const long long col_gid = epetra_mat.GCID64(col_indices[k]);
            if (col_gid >= 0 &&
                static_cast<dealii::types::global_dof_index>(col_gid) >= n_M_dofs_)
                ++entries_per_row[i];
        }
    }

    auto phi_crs = std::make_unique<Epetra_CrsMatrix>(
        Copy, phi_row_map, entries_per_row.data(), /*static_profile=*/true);

    for (int i = 0; i < n_my_phi; ++i)
    {
        const long long phi_gid = my_phi_gids[i];
        const long long orig_gid = phi_gid + static_cast<long long>(n_M_dofs_);
        const int local_row = epetra_mat.LRID(orig_gid);
        if (local_row < 0) continue;
        int num_entries = 0;
        double* values = nullptr;
        int* col_indices = nullptr;
        epetra_mat.ExtractMyRowView(local_row, num_entries, values, col_indices);

        std::vector<long long> col_gids;
        std::vector<double> col_vals;
        col_gids.reserve(num_entries);
        col_vals.reserve(num_entries);
        for (int k = 0; k < num_entries; ++k)
        {
            const long long col_gid = epetra_mat.GCID64(col_indices[k]);
            if (col_gid >= 0 &&
                static_cast<dealii::types::global_dof_index>(col_gid) >= n_M_dofs_)
            {
                col_gids.push_back(col_gid - static_cast<long long>(n_M_dofs_));
                col_vals.push_back(values[k]);
            }
        }
        if (!col_gids.empty())
        {
            int err = phi_crs->InsertGlobalValues(phi_gid,
                                                   static_cast<int>(col_gids.size()),
                                                   col_vals.data(),
                                                   col_gids.data());
            (void)err;
        }
    }

    phi_crs->FillComplete(phi_row_map, phi_row_map);
    phi_block_matrix_.reinit(*phi_crs);
}

// ============================================================================
// vmult — apply the block-diagonal preconditioner.
//
// Splits src into M and phi parts (using GID ranges), applies each per-block
// preconditioner, recombines.
// ============================================================================
void MagneticBlockPreconditioner::vmult(
    dealii::TrilinosWrappers::MPI::Vector& dst,
    const dealii::TrilinosWrappers::MPI::Vector& src) const
{
    r_M_ = 0.0;
    r_phi_ = 0.0;

    for (auto it = mag_locally_owned_.begin(); it != mag_locally_owned_.end(); ++it)
    {
        const dealii::types::global_dof_index gid = *it;
        const double v = src[gid];
        if (gid < n_M_dofs_)
            r_M_[gid] = v;
        else
            r_phi_[gid - n_M_dofs_] = v;
    }
    r_M_.compress(dealii::VectorOperation::insert);
    r_phi_.compress(dealii::VectorOperation::insert);

    M_prec_.vmult(z_M_, r_M_);
    phi_prec_.vmult(z_phi_, r_phi_);

    for (auto it = mag_locally_owned_.begin(); it != mag_locally_owned_.end(); ++it)
    {
        const dealii::types::global_dof_index gid = *it;
        if (gid < n_M_dofs_)
            dst[gid] = z_M_[gid];
        else
            dst[gid] = z_phi_[gid - n_M_dofs_];
    }
    dst.compress(dealii::VectorOperation::insert);
}
