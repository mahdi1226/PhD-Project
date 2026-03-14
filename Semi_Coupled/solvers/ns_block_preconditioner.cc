// ============================================================================
// solvers/ns_block_preconditioner.cc - Parallel Block Schur Preconditioner
//
// UPDATE (2026-03-14): FIXES
//   - Epetra 32-bit index fix: velocity block must match system matrix indexing
//   - Bounds check fix: skip invalid_dof_index entries in component-to-NS maps
//   - ILU preconditioner support via use_ilu flag (for HPC without ML/MueLu)
//
// UPDATE (2026-01-18): CRITICAL FIX for non-contiguous velocity IndexSets
//   - Uses Trilinos Epetra_Map directly (supports non-contiguous GIDs)
//   - Bypasses deal.II's SparsityPattern::reinit() which requires LinearMap
//   - Mathematically correct for separate ux/uy DoFHandlers
//
// PHASE-1 FIXES (earlier):
//   (1) Schur scaling accounts for time discretization
//   (2) Pressure pin consistency
//   (3) Improved defaults: inner_tolerance=1e-1, max_inner_iterations=20
//
// PHASE-2 FIXES (2026-01-20):
//   (1) Fixed Use-After-Free: Use existing communicator from system_matrix
//   (2) Memory Optimization: Avoid full copy of velocity block
// ============================================================================

#include "solvers/ns_block_preconditioner.h"

#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/read_write_vector.h>

#include <Epetra_CrsMatrix.h>
#include <Epetra_Map.h>
#include <Epetra_FECrsGraph.h>
#include <Epetra_FECrsMatrix.h>
#include <Epetra_Export.h>

#ifdef DEAL_II_WITH_MPI
#include <Epetra_MpiComm.h>
#else
#include <Epetra_SerialComm.h>
#endif

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <memory>

// ============================================================================
// Constructor
// ============================================================================
BlockSchurPreconditionerParallel::BlockSchurPreconditionerParallel(
    const dealii::TrilinosWrappers::SparseMatrix& system_matrix,
    const dealii::TrilinosWrappers::SparseMatrix& pressure_mass,
    const std::vector<dealii::types::global_dof_index>& ux_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& uy_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& p_to_ns_map,
    const dealii::IndexSet& ns_owned,
    const dealii::IndexSet& /* vel_owned - computed internally */,
    const dealii::IndexSet& p_owned,
    MPI_Comm mpi_comm,
    double viscosity,
    double dt,
    bool use_ilu)
    : n_iterations_A(0)
    , n_iterations_S(0)
    , inner_tolerance(1e-1)
    , max_inner_iterations(20)
    , system_matrix_ptr_(&system_matrix)
    , pressure_mass_ptr_(&pressure_mass)
    , ux_map_(ux_to_ns_map)
    , uy_map_(uy_to_ns_map)
    , p_map_(p_to_ns_map)
    , ns_owned_(ns_owned)
    , p_owned_(p_owned)
    , mpi_comm_(mpi_comm)
    , rank_(0)
    , viscosity_(viscosity > 0 ? viscosity : 1.0)
    , dt_(dt)
    , schur_alpha_(viscosity_ + ((dt_ > 0.0) ? (1.0 / dt_) : 0.0))
    , pinned_p_local_(-1)
{
    MPI_Comm_rank(mpi_comm_, &rank_);

    n_ux_ = ux_map_.size();
    n_uy_ = uy_map_.size();
    n_p_  = p_map_.size();
    n_vel_ = n_ux_ + n_uy_;
    n_total_ = n_vel_ + n_p_;

    // Pressure pin consistency in p-space:
    if (p_owned_.is_element(0))
        pinned_p_local_ = 0;

    // Build reverse mappings
    // FIX: Skip invalid_dof_index entries (remote DoFs not owned or ghosted)
    global_to_vel_.assign(n_total_, -1);
    global_to_p_.assign(n_total_, -1);

    for (dealii::types::global_dof_index i = 0; i < n_ux_; ++i)
    {
        if (ux_map_[i] < n_total_)
            global_to_vel_[ux_map_[i]] = static_cast<int>(i);
    }
    for (dealii::types::global_dof_index i = 0; i < n_uy_; ++i)
    {
        if (uy_map_[i] < n_total_)
            global_to_vel_[uy_map_[i]] = static_cast<int>(n_ux_ + i);
    }
    for (dealii::types::global_dof_index i = 0; i < n_p_; ++i)
    {
        if (p_map_[i] < n_total_)
            global_to_p_[p_map_[i]] = static_cast<int>(i);
    }

    // ========================================================================
    // Build vel_owned_ from matrix rows (determine which velocity DoFs we own)
    // ========================================================================
    const Epetra_CrsMatrix& epetra_mat = system_matrix_ptr_->trilinos_matrix();
    const int num_my_rows = epetra_mat.NumMyRows();

    vel_owned_.set_size(n_vel_);
    std::vector<long long> my_vel_gids;  // For Epetra_Map construction
    my_vel_gids.reserve(num_my_rows);

    for (int local_row = 0; local_row < num_my_rows; ++local_row)
    {
        const long long ns_idx = epetra_mat.GRID64(local_row);
        if (ns_idx >= 0 && static_cast<dealii::types::global_dof_index>(ns_idx) < n_total_)
        {
            const int vel_idx = global_to_vel_[static_cast<dealii::types::global_dof_index>(ns_idx)];
            if (vel_idx >= 0)
            {
                vel_owned_.add_index(static_cast<dealii::types::global_dof_index>(vel_idx));
                my_vel_gids.push_back(static_cast<long long>(vel_idx));
            }
        }
    }
    vel_owned_.compress();

    // Remove duplicates and sort (Epetra requires sorted unique GIDs)
    std::sort(my_vel_gids.begin(), my_vel_gids.end());
    my_vel_gids.erase(std::unique(my_vel_gids.begin(), my_vel_gids.end()), my_vel_gids.end());

    // ========================================================================
    // Precompute pressure indices needed for apply_BT
    // ========================================================================
    pressure_indices_for_BT_.set_size(n_p_);
    for (int local_row = 0; local_row < num_my_rows; ++local_row)
    {
        const long long ns_row = epetra_mat.GRID64(local_row);
        if (ns_row < 0 || static_cast<dealii::types::global_dof_index>(ns_row) >= n_total_)
            continue;

        const int vel_idx = global_to_vel_[static_cast<dealii::types::global_dof_index>(ns_row)];
        if (vel_idx < 0)
            continue;
        if (!vel_owned_.is_element(static_cast<dealii::types::global_dof_index>(vel_idx)))
            continue;

        int num_entries = 0;
        double* values = nullptr;
        int* col_indices = nullptr;
        epetra_mat.ExtractMyRowView(local_row, num_entries, values, col_indices);

        for (int k = 0; k < num_entries; ++k)
        {
            const long long ns_col = epetra_mat.GCID64(col_indices[k]);
            if (ns_col < 0 || static_cast<dealii::types::global_dof_index>(ns_col) >= n_total_)
                continue;

            const int p_idx = global_to_p_[static_cast<dealii::types::global_dof_index>(ns_col)];
            if (p_idx >= 0)
                pressure_indices_for_BT_.add_index(static_cast<dealii::types::global_dof_index>(p_idx));
        }
    }
    if (pinned_p_local_ >= 0)
        pressure_indices_for_BT_.add_index(0);

    pressure_indices_for_BT_.compress();

    // ========================================================================
    // Build velocity block using Trilinos directly
    //
    // deal.II's SparsityPattern::reinit() requires LinearMap (contiguous GIDs).
    // With separate ux/uy DoFHandlers, vel_owned_ is typically non-contiguous.
    // Solution: Use Epetra_Map directly which supports non-contiguous GIDs.
    //
    // CRITICAL: Must use 32-bit int GIDs to match system matrix index type.
    //           Using long long GIDs creates a 64-bit indexed matrix that is
    //           incompatible with deal.II's SparseMatrix::reinit().
    // ========================================================================

    const Epetra_Comm& epetra_comm = epetra_mat.Comm();

    // Create Epetra_Map with 32-bit int GIDs (matching system matrix)
    const int n_my_vel = static_cast<int>(my_vel_gids.size());
    std::vector<int> my_vel_gids_int(n_my_vel);
    for (int i = 0; i < n_my_vel; ++i)
        my_vel_gids_int[i] = static_cast<int>(my_vel_gids[i]);

    Epetra_Map vel_row_map(static_cast<int>(n_vel_), n_my_vel,
                           my_vel_gids_int.data(), 0, epetra_comm);

    // ========================================================================
    // Build sparsity pattern for velocity block
    // ========================================================================

    // First pass: count entries per row
    std::vector<int> entries_per_row(n_my_vel, 0);
    std::map<long long, int> gid_to_local;
    for (int i = 0; i < n_my_vel; ++i)
        gid_to_local[my_vel_gids[i]] = i;

    for (int local_row = 0; local_row < num_my_rows; ++local_row)
    {
        const long long ns_row = epetra_mat.GRID64(local_row);
        if (ns_row < 0 || static_cast<dealii::types::global_dof_index>(ns_row) >= n_total_)
            continue;

        const int vel_row = global_to_vel_[static_cast<dealii::types::global_dof_index>(ns_row)];
        if (vel_row < 0)
            continue;

        auto it = gid_to_local.find(vel_row);
        if (it == gid_to_local.end())
            continue;
        const int local_vel_row = it->second;

        int num_entries = 0;
        double* values = nullptr;
        int* col_indices = nullptr;
        epetra_mat.ExtractMyRowView(local_row, num_entries, values, col_indices);

        for (int k = 0; k < num_entries; ++k)
        {
            const long long ns_col = epetra_mat.GCID64(col_indices[k]);
            if (ns_col < 0 || static_cast<dealii::types::global_dof_index>(ns_col) >= n_total_)
                continue;

            const int vel_col = global_to_vel_[static_cast<dealii::types::global_dof_index>(ns_col)];
            if (vel_col >= 0)
                entries_per_row[local_vel_row]++;
        }
    }

    // Create CrsMatrix with estimated row sizes (32-bit int GIDs)
    auto vel_crs = std::make_unique<Epetra_CrsMatrix>(Copy, vel_row_map, entries_per_row.data(), true);

    // Second pass: fill matrix values
    for (int local_row = 0; local_row < num_my_rows; ++local_row)
    {
        const long long ns_row = epetra_mat.GRID64(local_row);
        if (ns_row < 0 || static_cast<dealii::types::global_dof_index>(ns_row) >= n_total_)
            continue;

        const int vel_row = global_to_vel_[static_cast<dealii::types::global_dof_index>(ns_row)];
        if (vel_row < 0)
            continue;

        auto it = gid_to_local.find(vel_row);
        if (it == gid_to_local.end())
            continue;

        int num_entries = 0;
        double* values = nullptr;
        int* col_indices = nullptr;
        epetra_mat.ExtractMyRowView(local_row, num_entries, values, col_indices);

        // Collect velocity entries (32-bit int column GIDs)
        std::vector<int> col_gids;
        std::vector<double> col_vals;
        col_gids.reserve(num_entries);
        col_vals.reserve(num_entries);

        for (int k = 0; k < num_entries; ++k)
        {
            const long long ns_col = epetra_mat.GCID64(col_indices[k]);
            if (ns_col < 0 || static_cast<dealii::types::global_dof_index>(ns_col) >= n_total_)
                continue;

            const int vel_col = global_to_vel_[static_cast<dealii::types::global_dof_index>(ns_col)];
            if (vel_col >= 0)
            {
                col_gids.push_back(vel_col);
                col_vals.push_back(values[k]);
            }
        }

        if (!col_gids.empty())
        {
            vel_crs->InsertGlobalValues(vel_row,
                                        static_cast<int>(col_gids.size()),
                                        col_vals.data(),
                                        col_gids.data());
        }
    }

    // Finalize the matrix
    vel_crs->FillComplete(vel_row_map, vel_row_map);

    // Wrap in deal.II matrix
    velocity_block_.reinit(*vel_crs);

    // ========================================================================
    // Initialize preconditioners (AMG or ILU depending on use_ilu flag)
    // ========================================================================

    if (use_ilu)
    {
        // ILU preconditioner — works on HPC without ML/MueLu
        auto A_ilu = std::make_unique<dealii::TrilinosWrappers::PreconditionILU>();
        dealii::TrilinosWrappers::PreconditionILU::AdditionalData ilu_data_A;
        ilu_data_A.ilu_fill = 1;
        ilu_data_A.ilu_atol = 0.0;
        ilu_data_A.ilu_rtol = 1.0;
        A_ilu->initialize(velocity_block_, ilu_data_A);
        A_preconditioner_ = std::move(A_ilu);

        auto S_ilu = std::make_unique<dealii::TrilinosWrappers::PreconditionILU>();
        dealii::TrilinosWrappers::PreconditionILU::AdditionalData ilu_data_S;
        ilu_data_S.ilu_fill = 0;
        ilu_data_S.ilu_atol = 0.0;
        ilu_data_S.ilu_rtol = 1.0;
        S_ilu->initialize(*pressure_mass_ptr_, ilu_data_S);
        S_preconditioner_ = std::move(S_ilu);
    }
    else
    {
        // AMG preconditioner — default (requires ML or MueLu in Trilinos)
        auto A_amg = std::make_unique<dealii::TrilinosWrappers::PreconditionAMG>();
        dealii::TrilinosWrappers::PreconditionAMG::AdditionalData amg_data_A;
        amg_data_A.elliptic = false;
        amg_data_A.higher_order_elements = true;
        amg_data_A.smoother_sweeps = 2;
        amg_data_A.aggregation_threshold = 0.02;
        amg_data_A.output_details = false;
        A_amg->initialize(velocity_block_, amg_data_A);
        A_preconditioner_ = std::move(A_amg);

        auto S_amg = std::make_unique<dealii::TrilinosWrappers::PreconditionAMG>();
        dealii::TrilinosWrappers::PreconditionAMG::AdditionalData amg_data_S;
        amg_data_S.elliptic = true;
        amg_data_S.higher_order_elements = false;
        amg_data_S.smoother_sweeps = 1;
        amg_data_S.aggregation_threshold = 0.02;
        amg_data_S.output_details = false;
        S_amg->initialize(*pressure_mass_ptr_, amg_data_S);
        S_preconditioner_ = std::move(S_amg);
    }

    if (rank_ == 0)
    {
        std::cout << "[Block Schur] Initialized: "
                  << "A = " << n_vel_ << "x" << n_vel_
                  << " (owned: " << vel_owned_.n_elements() << ")"
                  << ", S = " << n_p_ << "x" << n_p_
                  << ", precond = " << (use_ilu ? "ILU" : "AMG")
                  << ", alpha = " << std::scientific << std::setprecision(2) << schur_alpha_
                  << ", inner_tol = " << inner_tolerance
                  << ", max_inner = " << max_inner_iterations
                  << ", nu = " << viscosity_
                  << ", dt = " << dt_
                  << std::defaultfloat << "\n";
    }
}

// ============================================================================
// vmult - Apply preconditioner
// ============================================================================
void BlockSchurPreconditionerParallel::vmult(
    dealii::TrilinosWrappers::MPI::Vector& dst,
    const dealii::TrilinosWrappers::MPI::Vector& src) const
{
    // Temporary vectors
    dealii::TrilinosWrappers::MPI::Vector r_vel(vel_owned_, mpi_comm_);
    dealii::TrilinosWrappers::MPI::Vector r_p(p_owned_, mpi_comm_);

    extract_velocity(src, r_vel);
    extract_pressure(src, r_p);

    // Enforce pinned pressure mode consistency (p-space DoF 0)
    if (pinned_p_local_ >= 0)
    {
        r_p[0] = 0.0;
        r_p.compress(dealii::VectorOperation::insert);
    }

    // Step 1: Solve for pressure: M_p * z_p = r_p, then scale z_p *= -alpha
    dealii::TrilinosWrappers::MPI::Vector z_p(p_owned_, mpi_comm_);
    z_p = 0;

    {
        const double p_rhs_norm = r_p.l2_norm();
        const double tol = inner_tolerance * std::max(p_rhs_norm, 1e-30);

        dealii::SolverControl solver_control(max_inner_iterations, tol, false, false);
        dealii::SolverCG<dealii::TrilinosWrappers::MPI::Vector> cg(solver_control);

        try
        {
            cg.solve(*pressure_mass_ptr_, z_p, r_p, *S_preconditioner_);
        }
        catch (dealii::SolverControl::NoConvergence&)
        {
            // Fine for preconditioning
        }

        n_iterations_S += solver_control.last_step();

        if (pinned_p_local_ >= 0)
        {
            z_p[0] = 0.0;
            z_p.compress(dealii::VectorOperation::insert);
        }

        z_p *= (-schur_alpha_);
    }

    // Step 2: Update velocity RHS: rhs_vel = r_vel + B^T * z_p
    dealii::TrilinosWrappers::MPI::Vector Bt_zp(vel_owned_, mpi_comm_);
    apply_BT(z_p, Bt_zp);

    dealii::TrilinosWrappers::MPI::Vector rhs_vel(vel_owned_, mpi_comm_);
    rhs_vel = r_vel;
    rhs_vel += Bt_zp;

    // Step 3: Solve for velocity: A * z_vel = rhs_vel
    dealii::TrilinosWrappers::MPI::Vector z_vel(vel_owned_, mpi_comm_);
    z_vel = 0;

    {
        const double vel_rhs_norm = rhs_vel.l2_norm();
        const double tol = inner_tolerance * std::max(vel_rhs_norm, 1e-30);

        dealii::SolverControl solver_control(max_inner_iterations, tol, false, false);

        dealii::SolverGMRES<dealii::TrilinosWrappers::MPI::Vector>::AdditionalData gmres_data;
        gmres_data.max_n_tmp_vectors = 30;
        dealii::SolverGMRES<dealii::TrilinosWrappers::MPI::Vector> gmres(solver_control, gmres_data);

        try
        {
            gmres.solve(velocity_block_, z_vel, rhs_vel, *A_preconditioner_);
        }
        catch (dealii::SolverControl::NoConvergence&)
        {
            // Fine for preconditioning
        }

        n_iterations_A += solver_control.last_step();
    }

    // Step 4: Assemble output
    dst = 0;
    insert_velocity(z_vel, dst);
    insert_pressure(z_p, dst);
    dst.compress(dealii::VectorOperation::insert);
}

// ============================================================================
// Helper methods
// ============================================================================
void BlockSchurPreconditionerParallel::extract_velocity(
    const dealii::TrilinosWrappers::MPI::Vector& src,
    dealii::TrilinosWrappers::MPI::Vector& vel) const
{
    vel = 0;
    const Epetra_CrsMatrix& epetra_mat = system_matrix_ptr_->trilinos_matrix();
    const int num_my_rows = epetra_mat.NumMyRows();

    for (int local_row = 0; local_row < num_my_rows; ++local_row)
    {
        const long long ns_idx_ll = epetra_mat.GRID64(local_row);
        if (ns_idx_ll < 0)
            continue;

        const auto ns_idx = static_cast<dealii::types::global_dof_index>(ns_idx_ll);
        if (ns_idx >= n_total_)
            continue;

        const int vel_idx = global_to_vel_[ns_idx];
        if (vel_idx >= 0)
            vel[static_cast<dealii::types::global_dof_index>(vel_idx)] = src[ns_idx];
    }

    vel.compress(dealii::VectorOperation::insert);
}

void BlockSchurPreconditionerParallel::extract_pressure(
    const dealii::TrilinosWrappers::MPI::Vector& src,
    dealii::TrilinosWrappers::MPI::Vector& p) const
{
    p = 0;
    const Epetra_CrsMatrix& epetra_mat = system_matrix_ptr_->trilinos_matrix();
    const int num_my_rows = epetra_mat.NumMyRows();

    for (int local_row = 0; local_row < num_my_rows; ++local_row)
    {
        const long long ns_idx_ll = epetra_mat.GRID64(local_row);
        if (ns_idx_ll < 0)
            continue;

        const auto ns_idx = static_cast<dealii::types::global_dof_index>(ns_idx_ll);
        if (ns_idx >= n_total_)
            continue;

        const int p_idx = global_to_p_[ns_idx];
        if (p_idx >= 0)
            p[static_cast<dealii::types::global_dof_index>(p_idx)] = src[ns_idx];
    }

    p.compress(dealii::VectorOperation::insert);
}

void BlockSchurPreconditionerParallel::insert_velocity(
    const dealii::TrilinosWrappers::MPI::Vector& vel,
    dealii::TrilinosWrappers::MPI::Vector& dst) const
{
    const Epetra_CrsMatrix& epetra_mat = system_matrix_ptr_->trilinos_matrix();
    const int num_my_rows = epetra_mat.NumMyRows();

    for (int local_row = 0; local_row < num_my_rows; ++local_row)
    {
        const long long ns_idx_ll = epetra_mat.GRID64(local_row);
        if (ns_idx_ll < 0)
            continue;

        const auto ns_idx = static_cast<dealii::types::global_dof_index>(ns_idx_ll);
        if (ns_idx >= n_total_)
            continue;

        const int vel_idx = global_to_vel_[ns_idx];
        if (vel_idx >= 0)
            dst[ns_idx] = vel[static_cast<dealii::types::global_dof_index>(vel_idx)];
    }
}

void BlockSchurPreconditionerParallel::insert_pressure(
    const dealii::TrilinosWrappers::MPI::Vector& p,
    dealii::TrilinosWrappers::MPI::Vector& dst) const
{
    const Epetra_CrsMatrix& epetra_mat = system_matrix_ptr_->trilinos_matrix();
    const int num_my_rows = epetra_mat.NumMyRows();

    for (int local_row = 0; local_row < num_my_rows; ++local_row)
    {
        const long long ns_idx_ll = epetra_mat.GRID64(local_row);
        if (ns_idx_ll < 0)
            continue;

        const auto ns_idx = static_cast<dealii::types::global_dof_index>(ns_idx_ll);
        if (ns_idx >= n_total_)
            continue;

        const int p_idx = global_to_p_[ns_idx];
        if (p_idx >= 0)
            dst[ns_idx] = p[static_cast<dealii::types::global_dof_index>(p_idx)];
    }
}

void BlockSchurPreconditionerParallel::apply_BT(
    const dealii::TrilinosWrappers::MPI::Vector& p,
    dealii::TrilinosWrappers::MPI::Vector& vel) const
{
    vel = 0;

    dealii::LinearAlgebra::ReadWriteVector<double> p_values(pressure_indices_for_BT_);
    p_values.import_elements(p, dealii::VectorOperation::insert);

    const Epetra_CrsMatrix& epetra_mat = system_matrix_ptr_->trilinos_matrix();
    const int num_my_rows = epetra_mat.NumMyRows();

    for (int local_row = 0; local_row < num_my_rows; ++local_row)
    {
        const long long ns_row_ll = epetra_mat.GRID64(local_row);
        if (ns_row_ll < 0)
            continue;

        const auto ns_row = static_cast<dealii::types::global_dof_index>(ns_row_ll);
        if (ns_row >= n_total_)
            continue;

        const int vel_idx = global_to_vel_[ns_row];
        if (vel_idx < 0)
            continue;
        if (!vel_owned_.is_element(static_cast<dealii::types::global_dof_index>(vel_idx)))
            continue;

        int num_entries = 0;
        double* values = nullptr;
        int* col_indices = nullptr;
        epetra_mat.ExtractMyRowView(local_row, num_entries, values, col_indices);

        double sum = 0.0;
        for (int k = 0; k < num_entries; ++k)
        {
            const long long ns_col_ll = epetra_mat.GCID64(col_indices[k]);
            if (ns_col_ll < 0)
                continue;

            const auto ns_col = static_cast<dealii::types::global_dof_index>(ns_col_ll);
            if (ns_col >= n_total_)
                continue;

            const int p_idx = global_to_p_[ns_col];
            if (p_idx >= 0)
                sum += values[k] * p_values[static_cast<dealii::types::global_dof_index>(p_idx)];
        }

        vel[static_cast<dealii::types::global_dof_index>(vel_idx)] = sum;
    }

    vel.compress(dealii::VectorOperation::insert);
}
