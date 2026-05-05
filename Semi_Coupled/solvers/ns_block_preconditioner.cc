// ============================================================================
// solvers/ns_block_preconditioner.cc - Parallel Block Schur Preconditioner
//
// UPDATE (2026-01-18): CRITICAL FIX for non-contiguous velocity IndexSets
//   - Uses Trilinos Epetra_Map directly (supports non-contiguous GIDs)
//   - Bypasses deal.II's SparsityPattern::reinit() which requires LinearMap
//   - Mathematically correct for separate ux/uy DoFHandlers
//
// PHASE-1 FIXES (earlier):
//   (1) Schur scaling accounts for time discretization:
//         schur_alpha = nu                (steady / dt<=0)
//         schur_alpha = nu + 1/dt         (unsteady / dt>0)
//       and z_p = -(schur_alpha) * M_p^{-1} r_p
//
//   (2) Pressure pin consistency:
//       If global pressure DoF 0 is pinned, enforce r_p[0]=0 and z_p[0]=0
//
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
#include <Epetra_Vector.h>
#include <Epetra_FECrsGraph.h>
#include <Epetra_FECrsMatrix.h>
#include <Epetra_Export.h>
#include <EpetraExt_MatrixMatrix.h>  // [NEW] for B Q⁻¹ B^T product

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
    double dt)
    : n_iterations_A(0)
    , n_iterations_S(0)
    // =========================================================================
    // IMPROVED DEFAULTS: Looser tolerance OK for preconditioning.
    // 1e-2 / 30 was empirically the cheapest setting that gives LSC
    // convergence in ≤500 outer FGMRES iters on dome-r3.
    // =========================================================================
    , inner_tolerance(1e-2)
    , max_inner_iterations(30)
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
    global_to_vel_.assign(n_total_, -1);
    global_to_p_.assign(n_total_, -1);

    for (dealii::types::global_dof_index i = 0; i < n_ux_; ++i)
        global_to_vel_[ux_map_[i]] = static_cast<int>(i);
    for (dealii::types::global_dof_index i = 0; i < n_uy_; ++i)
        global_to_vel_[uy_map_[i]] = static_cast<int>(n_ux_ + i);
    for (dealii::types::global_dof_index i = 0; i < n_p_; ++i)
        global_to_p_[p_map_[i]] = static_cast<int>(i);

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
    // [FIXED] vmult MPI_ERR_TRUNCATE — root cause + fix (2026-05-05)
    //
    // Status: FIXED. The bug was an *asymmetric collective* in vmult, not a
    // partition mismatch and not a Trilinos AMG issue.
    //
    // ROOT CAUSE (was hidden under several wrong hypotheses):
    //   The pressure pinning block in vmult called compress(insert) only
    //   on ranks where pinned_p_local_ >= 0 (i.e., the owner of pressure
    //   DoF 0 — typically rank 0 only). Other ranks skipped the entire
    //   block, including the compress. compress() is a COLLECTIVE
    //   operation — every rank in the communicator must call it. Skipping
    //   it on some ranks leaves a stray send/recv that the next collective
    //   mis-matches → MPI_ERR_TRUNCATE.
    //
    //   Symptoms that misled earlier diagnosis:
    //     • Error fires after [Block Schur] Initialized → looked like
    //       Trilinos-internal AMG setup issue.
    //     • Adding MPI_Barriers around each op rearranged the collision
    //       point → looked like a Trilinos race condition.
    //     • Both led to the (incorrect) "Trilinos 12.x ML bug" hypothesis.
    //
    // WRONG hypotheses we ruled out (do not re-investigate):
    //   ✗ p_owned partition mismatch — Epetra map probes proved partitions
    //     agree exactly (all maps 34560 global, 17280 per rank).
    //   ✗ deal.II 64-bit indices Allreduce mismatch — local builds use
    //     32-bit; the existing MPI_UNSIGNED Allreduce is correct.
    //   ✗ AMG ML internal MPI race — replacing AMG with Jacobi or ILU
    //     for the Schur preconditioner did NOT change the failure mode.
    //   ✗ alternating IndexSets in Vector ctor — the bug fires regardless
    //     of which/how-many vectors are constructed.
    //
    // FIX: Move compress(insert) outside the pinning if-block so every
    // rank calls it — see vmult below.
    //
    // OPEN ISSUE (separate from this MPI bug): FGMRES convergence is
    // slow at default tolerances (often hits 1500-iteration cap and
    // falls back to direct). That is a preconditioner-tuning problem,
    // pre-existing and not introduced by this fix; it just wasn't
    // visible before because the MPI crash hid it.
    // ========================================================================

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
    // CRITICAL FIX: Build velocity block using Trilinos directly
    //
    // deal.II's SparsityPattern::reinit() requires LinearMap (contiguous GIDs).
    // With separate ux/uy DoFHandlers, vel_owned_ is typically non-contiguous.
    // Solution: Use Epetra_Map directly which supports non-contiguous GIDs.
    // ========================================================================

    // FIX: Use the communicator from the system matrix to avoid dangling reference
    const Epetra_Comm& epetra_comm = epetra_mat.Comm();

    // Create Epetra_Map with our (possibly non-contiguous) velocity GIDs
    const int n_my_vel = static_cast<int>(my_vel_gids.size());
    Epetra_Map vel_row_map(static_cast<long long>(n_vel_), n_my_vel,
                           my_vel_gids.data(), 0, epetra_comm);

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

    // Create CrsMatrix with estimated row sizes
    auto vel_crs = std::make_unique<Epetra_CrsMatrix>(Copy, vel_row_map, entries_per_row.data(), true);

    // Second pass: fill matrix values
    //
    // NOTE: Epetra_Map vel_row_map is built with 64-bit GIDs (long long), so
    // we MUST call InsertGlobalValues with long long row/col indices, not int.
    // Calling the int overload on a 64-bit map raises a non-std exception
    // ("Unknown exception" at top-level). This was the pre-existing bug.
    for (int local_row = 0; local_row < num_my_rows; ++local_row)
    {
        const long long ns_row = epetra_mat.GRID64(local_row);
        if (ns_row < 0 || static_cast<dealii::types::global_dof_index>(ns_row) >= n_total_)
            continue;

        const int vel_row_int = global_to_vel_[static_cast<dealii::types::global_dof_index>(ns_row)];
        if (vel_row_int < 0)
            continue;
        const long long vel_row = static_cast<long long>(vel_row_int);

        auto it = gid_to_local.find(vel_row);
        if (it == gid_to_local.end())
            continue;

        int num_entries = 0;
        double* values = nullptr;
        int* col_indices = nullptr;
        epetra_mat.ExtractMyRowView(local_row, num_entries, values, col_indices);

        // Collect velocity entries for this row (64-bit GIDs)
        std::vector<long long> col_gids;
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
                col_gids.push_back(static_cast<long long>(vel_col));
                col_vals.push_back(values[k]);
            }
        }

        if (!col_gids.empty())
        {
            int err = vel_crs->InsertGlobalValues(vel_row,
                                                   static_cast<int>(col_gids.size()),
                                                   col_vals.data(),
                                                   col_gids.data());
            if (err != 0 && rank_ == 0)
            {
                std::cerr << "[BSP] InsertGlobalValues err=" << err
                          << " row=" << vel_row << " ncols=" << col_gids.size() << "\n";
            }
        }
    }

    // Finalize the matrix
    vel_crs->FillComplete(vel_row_map, vel_row_map);

    // Wrap in deal.II matrix (vel_crs released automatically)
    velocity_block_.reinit(*vel_crs);

    // ========================================================================
    // [NEW 2026-05-05] Build B (divergence/coupling block) and L_p = B Q^-1 B^T
    //
    // For unsteady NS, the Schur complement at large α = ν + 1/dt is closer
    // to (B Q^-1 B^T)/α than to α M_p (the original code's approximation).
    // Here Q ≈ diag(A) approximates the velocity mass+stiffness.
    //
    // Steps:
    //   1. Extract B from system_matrix: rows = pressure DoFs (in p-space
    //      numbering), cols = velocity DoFs (in vel-space numbering).
    //   2. Compute Q_inv = 1 / diag(velocity_block_).
    //   3. Scale columns of B by Q_inv → B_scaled.
    //   4. L_p = B_scaled * B^T  via EpetraExt::MatrixMatrix::Multiply.
    //   5. AMG-precondition L_p.
    // ========================================================================
    if (use_lsc_)
    {
        // ---- Step 1: Extract B (n_p × n_vel) from the joint NS matrix ----
        // For each locally-owned NS row that maps to a pressure DoF, scan
        // its column entries; the velocity-mapped columns form B's row.

        // Build p row map (NS-aligned)
        std::vector<long long> my_p_gids;
        my_p_gids.reserve(n_p_);
        for (int local_row = 0; local_row < num_my_rows; ++local_row)
        {
            const long long ns_row = epetra_mat.GRID64(local_row);
            if (ns_row < 0 || static_cast<dealii::types::global_dof_index>(ns_row) >= n_total_)
                continue;
            const int p_row = global_to_p_[static_cast<dealii::types::global_dof_index>(ns_row)];
            if (p_row >= 0)
                my_p_gids.push_back(static_cast<long long>(p_row));
        }
        std::sort(my_p_gids.begin(), my_p_gids.end());
        my_p_gids.erase(std::unique(my_p_gids.begin(), my_p_gids.end()), my_p_gids.end());
        const int n_my_p = static_cast<int>(my_p_gids.size());

        Epetra_Map p_row_map(static_cast<long long>(n_p_), n_my_p,
                             my_p_gids.data(), 0, epetra_comm);

        // First pass: count B's nnz per row
        std::vector<int> B_entries_per_row(n_my_p, 0);
        std::map<long long, int> p_gid_to_local;
        for (int i = 0; i < n_my_p; ++i)
            p_gid_to_local[my_p_gids[i]] = i;

        for (int local_row = 0; local_row < num_my_rows; ++local_row)
        {
            const long long ns_row = epetra_mat.GRID64(local_row);
            if (ns_row < 0 || static_cast<dealii::types::global_dof_index>(ns_row) >= n_total_)
                continue;
            const int p_row = global_to_p_[static_cast<dealii::types::global_dof_index>(ns_row)];
            if (p_row < 0) continue;
            auto pit = p_gid_to_local.find(p_row);
            if (pit == p_gid_to_local.end()) continue;

            int num_entries = 0;
            double* values = nullptr;
            int* col_indices = nullptr;
            epetra_mat.ExtractMyRowView(local_row, num_entries, values, col_indices);

            for (int k = 0; k < num_entries; ++k)
            {
                const long long ns_col = epetra_mat.GCID64(col_indices[k]);
                if (ns_col < 0 || static_cast<dealii::types::global_dof_index>(ns_col) >= n_total_)
                    continue;
                if (global_to_vel_[static_cast<dealii::types::global_dof_index>(ns_col)] >= 0)
                    B_entries_per_row[pit->second]++;
            }
        }

        // Build B with column map = velocity space
        // Note: we use the same vel_row_map as the column map of B (n_vel cols).
        auto B_crs = std::make_unique<Epetra_CrsMatrix>(
            Copy, p_row_map, B_entries_per_row.data(), true);

        // Second pass: fill B entries
        for (int local_row = 0; local_row < num_my_rows; ++local_row)
        {
            const long long ns_row = epetra_mat.GRID64(local_row);
            if (ns_row < 0 || static_cast<dealii::types::global_dof_index>(ns_row) >= n_total_)
                continue;
            const int p_row_int = global_to_p_[static_cast<dealii::types::global_dof_index>(ns_row)];
            if (p_row_int < 0) continue;
            auto pit = p_gid_to_local.find(p_row_int);
            if (pit == p_gid_to_local.end()) continue;
            const long long p_row = static_cast<long long>(p_row_int);

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
                const long long ns_col = epetra_mat.GCID64(col_indices[k]);
                if (ns_col < 0 || static_cast<dealii::types::global_dof_index>(ns_col) >= n_total_)
                    continue;
                const int vel_col = global_to_vel_[static_cast<dealii::types::global_dof_index>(ns_col)];
                if (vel_col >= 0)
                {
                    col_gids.push_back(static_cast<long long>(vel_col));
                    col_vals.push_back(values[k]);
                }
            }
            if (!col_gids.empty())
            {
                B_crs->InsertGlobalValues(p_row,
                                          static_cast<int>(col_gids.size()),
                                          col_vals.data(),
                                          col_gids.data());
            }
        }
        B_crs->FillComplete(vel_row_map, p_row_map);
        B_block_.reinit(*B_crs);

        // ---- Step 2: Compute Q_inv = 1 / diag(velocity_block_) ----
        // Use Epetra_Vector on velocity row map.
        Epetra_Vector Q_inv(vel_row_map);
        velocity_block_.trilinos_matrix().ExtractDiagonalCopy(Q_inv);
        for (int i = 0; i < Q_inv.MyLength(); ++i)
        {
            const double d = Q_inv[i];
            // Guard against zero diagonal
            Q_inv[i] = (std::abs(d) > 1e-30) ? 1.0 / d : 0.0;
        }

        // ---- Step 3: Scale B's columns by Q_inv ----
        // Make a copy first so the original B is preserved for L_p formula.
        auto B_scaled_crs = std::make_unique<Epetra_CrsMatrix>(*B_crs);
        B_scaled_crs->RightScale(Q_inv);

        // ---- Step 4: L_p = B_scaled * B^T  via EpetraExt ----
        // Output map: pressure rows (p_row_map). The output graph is
        // computed by EpetraExt automatically.
        auto Lp_crs = std::make_unique<Epetra_CrsMatrix>(Copy, p_row_map, 0);
        const int err_mm = EpetraExt::MatrixMatrix::Multiply(
            *B_scaled_crs, /*transposeA=*/false,
            *B_crs,        /*transposeB=*/true,
            *Lp_crs,       /*call_FillComplete=*/true);
        if (err_mm != 0 && rank_ == 0)
        {
            std::cerr << "[BSP] EpetraExt MatrixMatrix err=" << err_mm
                      << " — falling back to pure-mass S preconditioner\n";
            use_lsc_ = false;
        }
        else
        {
            // L_p built successfully. Wrap in deal.II SparseMatrix and AMG it.
            // try/catch is defensive: on Trilinos exception, fall back to
            // pure-mass Schur (safer than aborting).
            try {
                Lp_block_.reinit(*Lp_crs);

                dealii::TrilinosWrappers::PreconditionAMG::AdditionalData amg_data_Lp;
                amg_data_Lp.elliptic = true;
                amg_data_Lp.higher_order_elements = false;
                amg_data_Lp.smoother_sweeps = 1;
                amg_data_Lp.aggregation_threshold = 0.02;
                amg_data_Lp.output_details = false;
                Lp_preconditioner_.initialize(Lp_block_, amg_data_Lp);

                if (rank_ == 0)
                    std::cout << "[Block Schur LSC] L_p ready (n_p=" << n_p_
                              << ", nnz_global=" << Lp_crs->NumGlobalNonzeros64()
                              << ")\n";
            } catch (const std::exception& e) {
                if (rank_ == 0)
                    std::cerr << "[BSP-LSC] L_p init failed (" << e.what()
                              << "); falling back to pure-mass S\n";
                use_lsc_ = false;
            } catch (...) {
                if (rank_ == 0)
                    std::cerr << "[BSP-LSC] L_p init failed (unknown);"
                              << " falling back to pure-mass S\n";
                use_lsc_ = false;
            }
        }
    }

    // ========================================================================
    // Initialize AMG preconditioners
    // ========================================================================

    // AMG for velocity block
    dealii::TrilinosWrappers::PreconditionAMG::AdditionalData amg_data_A;
    amg_data_A.elliptic = false;
    amg_data_A.higher_order_elements = true;
    amg_data_A.smoother_sweeps = 2;
    amg_data_A.aggregation_threshold = 0.02;
    amg_data_A.output_details = false;
    A_preconditioner_.initialize(velocity_block_, amg_data_A);

    // AMG for pressure mass (Schur complement approximation)
    dealii::TrilinosWrappers::PreconditionAMG::AdditionalData amg_data_S;
    amg_data_S.elliptic = true;
    amg_data_S.higher_order_elements = false;
    amg_data_S.smoother_sweeps = 1;
    amg_data_S.aggregation_threshold = 0.02;
    amg_data_S.output_details = false;
    S_preconditioner_.initialize(*pressure_mass_ptr_, amg_data_S);

    if (rank_ == 0)
    {
        std::cout << "[Block Schur] Initialized: "
                  << "A = " << n_vel_ << "x" << n_vel_
                  << " (owned: " << vel_owned_.n_elements() << ")"
                  << ", S = " << n_p_ << "x" << n_p_
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

    // Enforce pinned pressure mode consistency (p-space DoF 0).
    //
    // BUGFIX (2026-05-05): compress(insert) is COLLECTIVE — every rank
    // must call it. Previously only the owning rank entered the `if`,
    // leaving other ranks out of the collective → MPI_ERR_TRUNCATE.
    // The write itself is rank-local; the compress is symmetric.
    if (pinned_p_local_ >= 0)
        r_p[0] = 0.0;
    r_p.compress(dealii::VectorOperation::insert);

    // Step 1: Solve for pressure correction (Schur application).
    //
    // [LSC mode, default]   z_p = -α · L_p⁻¹ r_p   (CG against L_p with AMG)
    // [pure-mass fallback]  z_p = -α · M_p⁻¹ r_p   (CG against M_p with AMG)
    //
    // Sign convention: z_p = -S⁻¹ r_p, with S⁻¹ ≈ α · L_p⁻¹ (LSC) or α · M_p⁻¹.
    dealii::TrilinosWrappers::MPI::Vector z_p(p_owned_, mpi_comm_);
    z_p = 0;

    {
        const double p_rhs_norm = r_p.l2_norm();
        const double tol = inner_tolerance * std::max(p_rhs_norm, 1e-30);

        dealii::SolverControl solver_control(max_inner_iterations, tol, false, false);
        dealii::SolverCG<dealii::TrilinosWrappers::MPI::Vector> cg(solver_control);

        try
        {
            if (use_lsc_)
                cg.solve(Lp_block_, z_p, r_p, Lp_preconditioner_);
            else
                cg.solve(*pressure_mass_ptr_, z_p, r_p, S_preconditioner_);
        }
        catch (dealii::SolverControl::NoConvergence&)
        {
            // Fine for preconditioning
        }

        n_iterations_S += solver_control.last_step();

        // BUGFIX (2026-05-05): compress is collective — see vmult above.
        if (pinned_p_local_ >= 0)
            z_p[0] = 0.0;
        z_p.compress(dealii::VectorOperation::insert);

        // Schur sign + scaling.
        //
        // [LSC]   S ≈ B Q⁻¹ B^T = L_p directly (Q = diag(A) already
        //         absorbs the 1/dt factor through A's mass term). So
        //         S⁻¹ ≈ L_p⁻¹ and z_p = -L_p⁻¹ r_p (no extra scaling).
        //
        // [pure-mass]   S ≈ M_p / α  ⇒  S⁻¹ ≈ α M_p⁻¹.  z_p = -α M_p⁻¹ r_p.
        if (use_lsc_)
            z_p *= -1.0;
        else
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
            gmres.solve(velocity_block_, z_vel, rhs_vel, A_preconditioner_);
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