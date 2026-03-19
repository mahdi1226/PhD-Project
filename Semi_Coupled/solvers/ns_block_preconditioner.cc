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
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <Epetra_CrsMatrix.h>
#include <EpetraExt_MatrixMatrix.h>

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
    , inner_tolerance(1e-2)
    , max_inner_iterations(50)
    , system_matrix_ptr_(&system_matrix)
    , pressure_mass_ptr_(&pressure_mass)
    , ux_map_(ux_to_ns_map)
    , uy_map_(uy_to_ns_map)
    , p_map_(p_to_ns_map)
    , ns_owned_(ns_owned)
    , p_owned_(p_owned)
    , mpi_comm_(mpi_comm)
    , rank_(0)
    , use_bfbt_(false)  // BFBt stagnates for DG Q1 pressure; use pressure mass matrix
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

    for (int local_row = 0; local_row < num_my_rows; ++local_row)
    {
        const long long ns_idx = epetra_mat.GRID64(local_row);
        if (ns_idx >= 0 && static_cast<dealii::types::global_dof_index>(ns_idx) < n_total_)
        {
            const int vel_idx = global_to_vel_[static_cast<dealii::types::global_dof_index>(ns_idx)];
            if (vel_idx >= 0)
                vel_owned_.add_index(static_cast<dealii::types::global_dof_index>(vel_idx));
        }
    }
    vel_owned_.compress();

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
    // Build velocity block using deal.II's own SparsityPattern + SparseMatrix
    //
    // Previous approach used raw Epetra_Map + Epetra_CrsMatrix with 32-bit
    // GIDs, but deal.II vectors use IndexSet-derived maps (potentially 64-bit).
    // This mismatch caused MPI_ERR_TRUNCATE on 2+ ranks during Apply().
    //
    // Fix: use deal.II's TrilinosWrappers API throughout so all Epetra maps
    // are created consistently from the same IndexSet (vel_owned_).
    // ========================================================================

    // First pass: build sparsity pattern
    dealii::TrilinosWrappers::SparsityPattern vel_sp;
    vel_sp.reinit(vel_owned_, vel_owned_, mpi_comm_);

    for (int local_row = 0; local_row < num_my_rows; ++local_row)
    {
        const long long ns_row = epetra_mat.GRID64(local_row);
        if (ns_row < 0 || static_cast<dealii::types::global_dof_index>(ns_row) >= n_total_)
            continue;

        const int vel_row = global_to_vel_[static_cast<dealii::types::global_dof_index>(ns_row)];
        if (vel_row < 0)
            continue;
        if (!vel_owned_.is_element(static_cast<dealii::types::global_dof_index>(vel_row)))
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

            const int vel_col = global_to_vel_[static_cast<dealii::types::global_dof_index>(ns_col)];
            if (vel_col >= 0)
                vel_sp.add(static_cast<dealii::types::global_dof_index>(vel_row),
                           static_cast<dealii::types::global_dof_index>(vel_col));
        }
    }
    vel_sp.compress();

    // Create matrix from sparsity pattern and fill values
    velocity_block_.reinit(vel_sp);

    for (int local_row = 0; local_row < num_my_rows; ++local_row)
    {
        const long long ns_row = epetra_mat.GRID64(local_row);
        if (ns_row < 0 || static_cast<dealii::types::global_dof_index>(ns_row) >= n_total_)
            continue;

        const int vel_row = global_to_vel_[static_cast<dealii::types::global_dof_index>(ns_row)];
        if (vel_row < 0)
            continue;
        if (!vel_owned_.is_element(static_cast<dealii::types::global_dof_index>(vel_row)))
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

            const int vel_col = global_to_vel_[static_cast<dealii::types::global_dof_index>(ns_col)];
            if (vel_col >= 0)
                velocity_block_.set(static_cast<dealii::types::global_dof_index>(vel_row),
                                    static_cast<dealii::types::global_dof_index>(vel_col),
                                    values[k]);
        }
    }
    velocity_block_.compress(dealii::VectorOperation::insert);

    // ========================================================================
    // Build BFBt Schur complement approximation
    //
    // For DG Q1 pressure, the pressure mass matrix M_p is block-diagonal
    // (each cell's pressure DoFs are independent). But the actual Schur
    // complement S = B A^{-1} B^T has inter-cell coupling through the
    // CG velocity basis functions. Using M_p as S_approx fails because
    // it misses this coupling entirely.
    //
    // BFBt: S_approx = B * diag(A)^{-1} * B^T
    // This naturally has the correct sparsity pattern and captures both
    // the temporal (1/dt) and spatial (ν/h²) scaling of the Schur complement.
    // Reference: Elman, Silvester, Wathen - "Finite Elements and Fast
    //            Iterative Solvers", Oxford University Press.
    // ========================================================================
    if (use_bfbt_)
    {
        // Step 1: Get diagonal of velocity block (inverse)
        dealii::TrilinosWrappers::MPI::Vector A_diag_inv(vel_owned_, mpi_comm_);
        for (auto idx = vel_owned_.begin(); idx != vel_owned_.end(); ++idx)
        {
            const double diag_val = velocity_block_(*idx, *idx);
            A_diag_inv[*idx] = (std::abs(diag_val) > 1e-30) ? (1.0 / diag_val) : 0.0;
        }
        A_diag_inv.compress(dealii::VectorOperation::insert);

        // Step 2: Build B matrix (n_p x n_vel) from system matrix
        // B_{p_idx, vel_idx} = system_matrix(p_ns_idx, vel_ns_idx)
        dealii::TrilinosWrappers::SparsityPattern B_sp;
        B_sp.reinit(p_owned_, vel_owned_, mpi_comm_);

        for (int local_row = 0; local_row < num_my_rows; ++local_row)
        {
            const long long ns_row = epetra_mat.GRID64(local_row);
            if (ns_row < 0 || static_cast<dealii::types::global_dof_index>(ns_row) >= n_total_)
                continue;

            const int p_row = global_to_p_[static_cast<dealii::types::global_dof_index>(ns_row)];
            if (p_row < 0)
                continue;
            if (!p_owned_.is_element(static_cast<dealii::types::global_dof_index>(p_row)))
                continue;

            int n_entries = 0;
            double* vals = nullptr;
            int* cols = nullptr;
            epetra_mat.ExtractMyRowView(local_row, n_entries, vals, cols);

            for (int k = 0; k < n_entries; ++k)
            {
                const long long ns_col = epetra_mat.GCID64(cols[k]);
                if (ns_col < 0 || static_cast<dealii::types::global_dof_index>(ns_col) >= n_total_)
                    continue;

                const int v_col = global_to_vel_[static_cast<dealii::types::global_dof_index>(ns_col)];
                if (v_col >= 0)
                    B_sp.add(static_cast<dealii::types::global_dof_index>(p_row),
                             static_cast<dealii::types::global_dof_index>(v_col));
            }
        }
        B_sp.compress();

        dealii::TrilinosWrappers::SparseMatrix B_mat;
        B_mat.reinit(B_sp);

        for (int local_row = 0; local_row < num_my_rows; ++local_row)
        {
            const long long ns_row = epetra_mat.GRID64(local_row);
            if (ns_row < 0 || static_cast<dealii::types::global_dof_index>(ns_row) >= n_total_)
                continue;

            const int p_row = global_to_p_[static_cast<dealii::types::global_dof_index>(ns_row)];
            if (p_row < 0)
                continue;
            if (!p_owned_.is_element(static_cast<dealii::types::global_dof_index>(p_row)))
                continue;

            int n_entries = 0;
            double* vals = nullptr;
            int* cols = nullptr;
            epetra_mat.ExtractMyRowView(local_row, n_entries, vals, cols);

            for (int k = 0; k < n_entries; ++k)
            {
                const long long ns_col = epetra_mat.GCID64(cols[k]);
                if (ns_col < 0 || static_cast<dealii::types::global_dof_index>(ns_col) >= n_total_)
                    continue;

                const int v_col = global_to_vel_[static_cast<dealii::types::global_dof_index>(ns_col)];
                if (v_col >= 0)
                    B_mat.set(static_cast<dealii::types::global_dof_index>(p_row),
                              static_cast<dealii::types::global_dof_index>(v_col),
                              vals[k]);
            }
        }
        B_mat.compress(dealii::VectorOperation::insert);

        // Step 3: Scale columns of B by diag(A)^{-1} in-place
        // B_scaled(i,j) = B(i,j) * A_diag_inv(j)
        // Need ghosted A_diag_inv to access off-rank velocity entries
        {
            const Epetra_CrsMatrix& B_epetra = B_mat.trilinos_matrix();
            const int B_num_my_rows = B_epetra.NumMyRows();

            dealii::IndexSet vel_relevant(n_vel_);
            vel_relevant = vel_owned_;
            for (int lr = 0; lr < B_num_my_rows; ++lr)
            {
                int ne = 0; double* v = nullptr; int* c = nullptr;
                B_epetra.ExtractMyRowView(lr, ne, v, c);
                for (int kk = 0; kk < ne; ++kk)
                {
                    const long long gcol = B_epetra.GCID64(c[kk]);
                    if (gcol >= 0 && static_cast<dealii::types::global_dof_index>(gcol) < n_vel_)
                        vel_relevant.add_index(static_cast<dealii::types::global_dof_index>(gcol));
                }
            }
            vel_relevant.compress();

            dealii::TrilinosWrappers::MPI::Vector A_diag_inv_ghosted(
                vel_owned_, vel_relevant, mpi_comm_);
            A_diag_inv_ghosted = A_diag_inv;

            // Scale B in-place: B_mat becomes B * diag(A)^{-1}
            Epetra_CrsMatrix& B_mut =
                const_cast<Epetra_CrsMatrix&>(B_mat.trilinos_matrix());
            for (int lr = 0; lr < B_num_my_rows; ++lr)
            {
                int ne = 0; double* v = nullptr; int* c = nullptr;
                B_mut.ExtractMyRowView(lr, ne, v, c);
                for (int kk = 0; kk < ne; ++kk)
                {
                    const long long gcol = B_epetra.GCID64(c[kk]);
                    if (gcol >= 0)
                    {
                        const auto vel_idx = static_cast<dealii::types::global_dof_index>(gcol);
                        v[kk] *= A_diag_inv_ghosted[vel_idx];
                    }
                }
            }
        }

        // Step 4: Compute S_BFBt = B_scaled * B_orig^T using EpetraExt
        // Since we scaled B_mat in-place, we need B_orig for B^T.
        // But B_scaled * B^T = B * D^{-1} * B^T. We can compute this as
        // B_scaled * B_scaled^T * D, but that's more complex.
        // Instead, rebuild B_orig or use the fact that for BFBt we need
        // (B D^{-1}) * B^T. We'll build a fresh B for transpose.
        //
        // Actually: EpetraExt can multiply B_scaled * B_orig^T directly.
        // But we modified B_mat in-place. Let's rebuild B_orig.
        // Alternative: compute B_scaled * B_scaled^T and rescale. But simpler
        // to just build B^T as a separate matrix.
        //
        // Simplest correct approach: build B_orig separately, then use EpetraExt.
        // But that doubles memory. Instead, let's undo the scaling after multiply.
        //
        // Better: We'll compute BFBt using EpetraExt::MatrixMatrix::Multiply
        // with B_scaled (which is B*D^{-1}) and B^T (from the original system).
        // We need an unscaled B for B^T. Let's extract it fresh from system_matrix.
        {
            // Build unscaled B for transpose (needed for B_scaled * B_orig^T)
            dealii::TrilinosWrappers::SparseMatrix B_orig;
            B_orig.reinit(B_sp);

            for (int local_row = 0; local_row < num_my_rows; ++local_row)
            {
                const long long ns_row = epetra_mat.GRID64(local_row);
                if (ns_row < 0 || static_cast<dealii::types::global_dof_index>(ns_row) >= n_total_)
                    continue;
                const int p_row = global_to_p_[static_cast<dealii::types::global_dof_index>(ns_row)];
                if (p_row < 0) continue;
                if (!p_owned_.is_element(static_cast<dealii::types::global_dof_index>(p_row)))
                    continue;

                int n_entries = 0; double* vals = nullptr; int* cols = nullptr;
                epetra_mat.ExtractMyRowView(local_row, n_entries, vals, cols);
                for (int k = 0; k < n_entries; ++k)
                {
                    const long long ns_col = epetra_mat.GCID64(cols[k]);
                    if (ns_col < 0 || static_cast<dealii::types::global_dof_index>(ns_col) >= n_total_)
                        continue;
                    const int v_col = global_to_vel_[static_cast<dealii::types::global_dof_index>(ns_col)];
                    if (v_col >= 0)
                        B_orig.set(static_cast<dealii::types::global_dof_index>(p_row),
                                   static_cast<dealii::types::global_dof_index>(v_col), vals[k]);
                }
            }
            B_orig.compress(dealii::VectorOperation::insert);

            // Use EpetraExt to compute BFBt = B_scaled * B_orig^T
            // This handles parallel communication correctly (gathers off-rank columns).
            // EpetraExt::Multiply takes a pre-created output matrix (fills sparsity+values).
            // Create output with the pressure row map from B_mat.
            const Epetra_CrsMatrix& B_scaled_ep = B_mat.trilinos_matrix();
            Epetra_CrsMatrix bfbt_raw(Copy, B_scaled_ep.RowMap(), 0);

            const int mm_err = EpetraExt::MatrixMatrix::Multiply(
                B_scaled_ep, false,                    // B_scaled (= B * D^{-1})
                B_orig.trilinos_matrix(), true,        // B_orig^T
                bfbt_raw, true);                       // call_FillComplete=true

            if (mm_err != 0)
            {
                if (rank_ == 0)
                    std::cerr << "[Block Schur] EpetraExt::Multiply failed! err=" << mm_err << "\n";
                use_bfbt_ = false;
            }
            else
            {
                // Copy EpetraExt result into deal.II SparseMatrix (schur_approx_)
                dealii::TrilinosWrappers::SparsityPattern bfbt_sp;
                bfbt_sp.reinit(p_owned_, p_owned_, mpi_comm_);

                for (int lr = 0; lr < bfbt_raw.NumMyRows(); ++lr)
                {
                    const long long grow = bfbt_raw.GRID64(lr);
                    if (grow < 0) continue;
                    int ne = 0; double* v = nullptr; int* c = nullptr;
                    bfbt_raw.ExtractMyRowView(lr, ne, v, c);
                    for (int kk = 0; kk < ne; ++kk)
                    {
                        const long long gcol = bfbt_raw.GCID64(c[kk]);
                        if (gcol >= 0)
                            bfbt_sp.add(static_cast<dealii::types::global_dof_index>(grow),
                                        static_cast<dealii::types::global_dof_index>(gcol));
                    }
                }
                bfbt_sp.compress();

                schur_approx_.reinit(bfbt_sp);

                for (int lr = 0; lr < bfbt_raw.NumMyRows(); ++lr)
                {
                    const long long grow = bfbt_raw.GRID64(lr);
                    if (grow < 0) continue;
                    int ne = 0; double* v = nullptr; int* c = nullptr;
                    bfbt_raw.ExtractMyRowView(lr, ne, v, c);
                    for (int kk = 0; kk < ne; ++kk)
                    {
                        const long long gcol = bfbt_raw.GCID64(c[kk]);
                        if (gcol >= 0)
                            schur_approx_.set(
                                static_cast<dealii::types::global_dof_index>(grow),
                                static_cast<dealii::types::global_dof_index>(gcol),
                                v[kk]);
                    }
                }
                schur_approx_.compress(dealii::VectorOperation::insert);
            }
        }

        // Pin pressure DoF 0 to remove constant null space:
        // Set row 0 and col 0 of BFBt to identity
        if (p_owned_.is_element(0))
        {
            // Zero out row 0 and set diagonal to 1
            // We must iterate the full row of the original BFBt to clear entries
            const Epetra_CrsMatrix& bfbt_ep = schur_approx_.trilinos_matrix();
            const int local_row0 = bfbt_ep.LRID(0);
            if (local_row0 >= 0)
            {
                int ne = 0; double* v = nullptr; int* c = nullptr;
                const_cast<Epetra_CrsMatrix&>(bfbt_ep).ExtractMyRowView(local_row0, ne, v, c);
                for (int kk = 0; kk < ne; ++kk)
                {
                    const long long gcol = bfbt_ep.GCID64(c[kk]);
                    if (gcol == 0)
                        v[kk] = 1.0;  // diagonal
                    else
                        v[kk] = 0.0;  // off-diagonal
                }
            }
        }
        // Zero out column 0 (set all off-diagonal entries in column 0 to zero)
        // Iterate all local rows and zero entries that map to global column 0
        {
            Epetra_CrsMatrix& bfbt_ep =
                const_cast<Epetra_CrsMatrix&>(schur_approx_.trilinos_matrix());
            const int bnr = bfbt_ep.NumMyRows();
            for (int lr = 0; lr < bnr; ++lr)
            {
                const long long grow = bfbt_ep.GRID64(lr);
                if (grow == 0) continue;  // row 0 already handled
                int ne = 0; double* v = nullptr; int* c = nullptr;
                bfbt_ep.ExtractMyRowView(lr, ne, v, c);
                for (int kk = 0; kk < ne; ++kk)
                {
                    if (bfbt_ep.GCID64(c[kk]) == 0)
                        v[kk] = 0.0;
                }
            }
        }

        if (rank_ == 0)
            std::cout << "[Block Schur] BFBt Schur complement built: "
                      << schur_approx_.m() << "x" << schur_approx_.n()
                      << ", nnz=" << schur_approx_.n_nonzero_elements() << "\n";
    }

    // ========================================================================
    // Initialize preconditioners (AMG or ILU depending on use_ilu flag)
    // ========================================================================

    if (use_ilu)
    {
        // ILU preconditioner — works on HPC without ML/MueLu
        auto A_ilu = std::make_unique<dealii::TrilinosWrappers::PreconditionILU>();
        dealii::TrilinosWrappers::PreconditionILU::AdditionalData ilu_data_A;
        ilu_data_A.ilu_fill = 2;
        ilu_data_A.ilu_atol = 0.0;
        ilu_data_A.ilu_rtol = 1.0;
        A_ilu->initialize(velocity_block_, ilu_data_A);
        A_preconditioner_ = std::move(A_ilu);

        const auto& S_matrix = use_bfbt_ ? schur_approx_ : *pressure_mass_ptr_;
        if (use_bfbt_)
        {
            // BFBt is SPD → AMG. ILU hits zero-pivot on DG Q1 pressure.
            auto S_amg = std::make_unique<dealii::TrilinosWrappers::PreconditionAMG>();
            dealii::TrilinosWrappers::PreconditionAMG::AdditionalData amg_data_S;
            amg_data_S.elliptic = true;
            amg_data_S.higher_order_elements = false;
            amg_data_S.smoother_sweeps = 2;
            amg_data_S.aggregation_threshold = 0.02;
            amg_data_S.output_details = false;
            S_amg->initialize(S_matrix, amg_data_S);
            S_preconditioner_ = std::move(S_amg);
        }
        else
        {
            auto S_ilu = std::make_unique<dealii::TrilinosWrappers::PreconditionILU>();
            dealii::TrilinosWrappers::PreconditionILU::AdditionalData ilu_data_S;
            ilu_data_S.ilu_fill = 0;
            ilu_data_S.ilu_atol = 0.0;
            ilu_data_S.ilu_rtol = 1.0;
            S_ilu->initialize(S_matrix, ilu_data_S);
            S_preconditioner_ = std::move(S_ilu);
        }
    }
    else
    {
        // AMG preconditioner — default (requires ML or MueLu in Trilinos)
        // Velocity block has [ux | uy] DoFs. AMG needs near-null space
        // (constant modes) to respect the 2-component block structure.
        auto A_amg = std::make_unique<dealii::TrilinosWrappers::PreconditionAMG>();
        dealii::TrilinosWrappers::PreconditionAMG::AdditionalData amg_data_A;
        amg_data_A.elliptic = false;
        amg_data_A.higher_order_elements = true;
        amg_data_A.smoother_sweeps = 2;
        amg_data_A.aggregation_threshold = 0.02;
        amg_data_A.output_details = false;

        // Build constant modes: mode[0] = 1 for ux, 0 for uy
        //                       mode[1] = 0 for ux, 1 for uy
        // FIX #4: AMG needs LOCAL-sized modes, not global-sized.
        // Each rank only sees its owned velocity DoFs.
        {
            const auto n_local_vel = vel_owned_.n_elements();
            std::vector<std::vector<bool>> constant_modes(2, std::vector<bool>(n_local_vel, false));
            unsigned int local_idx = 0;
            for (auto it = vel_owned_.begin(); it != vel_owned_.end(); ++it, ++local_idx)
            {
                const auto global_vel_idx = *it;
                if (global_vel_idx < n_ux_)
                    constant_modes[0][local_idx] = true;   // ux component
                else
                    constant_modes[1][local_idx] = true;   // uy component
            }
            amg_data_A.constant_modes = constant_modes;
        }

        A_amg->initialize(velocity_block_, amg_data_A);
        A_preconditioner_ = std::move(A_amg);

        const auto& S_matrix = use_bfbt_ ? schur_approx_ : *pressure_mass_ptr_;
        {
            // BFBt = B*diag(A)^{-1}*B^T is SPD → AMG works well.
            // ILU hits zero-pivot on DG Q1 pressure sparsity → NaN.
            // AMG is the standard HPC choice for SPD Schur complements.
            auto S_amg = std::make_unique<dealii::TrilinosWrappers::PreconditionAMG>();
            dealii::TrilinosWrappers::PreconditionAMG::AdditionalData amg_data_S;
            amg_data_S.elliptic = true;
            amg_data_S.higher_order_elements = false;
            amg_data_S.smoother_sweeps = 2;
            amg_data_S.aggregation_threshold = 0.02;
            amg_data_S.output_details = false;
            S_amg->initialize(S_matrix, amg_data_S);
            S_preconditioner_ = std::move(S_amg);
        }
    }

    // Note: inner_tolerance and max_inner_iterations may be overridden
    // after construction (they are public members). This log shows constructor defaults.
    if (rank_ == 0)
    {
        std::cout << "[Block Schur] Initialized: "
                  << "A = " << n_vel_ << "x" << n_vel_
                  << " (owned: " << vel_owned_.n_elements() << ")"
                  << ", S = " << n_p_ << "x" << n_p_
                  << ", precond = " << (use_ilu ? "ILU" : "AMG")
                  << ", alpha = " << std::scientific << std::setprecision(2) << schur_alpha_
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
    // Lazy-init cached temporaries (allocated once, reused across vmult calls)
    if (!tmp_initialized_)
    {
        r_vel_.reinit(vel_owned_, mpi_comm_);
        r_p_.reinit(p_owned_, mpi_comm_);
        z_p_.reinit(p_owned_, mpi_comm_);
        Bt_zp_.reinit(vel_owned_, mpi_comm_);
        rhs_vel_.reinit(vel_owned_, mpi_comm_);
        z_vel_.reinit(vel_owned_, mpi_comm_);

        p_relevant_cached_.set_size(n_p_);
        p_relevant_cached_ = p_owned_;
        p_relevant_cached_.add_indices(pressure_indices_for_BT_);
        p_relevant_cached_.compress();

        z_p_ghosted_.reinit(p_owned_, p_relevant_cached_, mpi_comm_);
        tmp_initialized_ = true;
    }

    extract_velocity(src, r_vel_);
    extract_pressure(src, r_p_);

    // Enforce pinned pressure mode consistency (p-space DoF 0)
    if (pinned_p_local_ >= 0)
        r_p_[0] = 0.0;

    // Step 1: Solve for pressure using Schur complement approximation
    z_p_ = 0;

    {
        const double p_rhs_norm = r_p_.l2_norm();
        const double tol = inner_tolerance * std::max(p_rhs_norm, 1e-30);

        const auto& S_matrix = use_bfbt_ ? schur_approx_ : *pressure_mass_ptr_;

        dealii::SolverControl solver_control(max_inner_iterations, tol, false, false);

        {
            // BFBt = B*diag(A)^{-1}*B^T is SPD by construction → CG is optimal.
            // For pressure mass matrix fallback, also SPD → CG works.
            dealii::SolverCG<dealii::TrilinosWrappers::MPI::Vector> cg(solver_control);
            try
            {
                cg.solve(S_matrix, z_p_, r_p_, *S_preconditioner_);
            }
            catch (dealii::SolverControl::NoConvergence&) {}
        }

        if (verbose_ && rank_ == 0)
            std::cout << "  [Schur vmult] S solve: " << solver_control.last_step()
                      << " its, res=" << std::scientific << std::setprecision(2)
                      << solver_control.last_value()
                      << ", rhs=" << p_rhs_norm << std::defaultfloat << "\n";

        n_iterations_S += solver_control.last_step();

        if (pinned_p_local_ >= 0)
            z_p_[0] = 0.0;

        // BFBt solves S_approx * z_p = r_p, but the preconditioner needs
        // z_p = -alpha * S_approx^{-1} * r_p where alpha = nu + 1/dt.
        // This scaling is always needed regardless of Schur approximation.
        z_p_ *= -schur_alpha_;
    }

    // Step 2: Update velocity RHS: rhs_vel = r_vel + B^T * z_p
    z_p_ghosted_ = z_p_;
    apply_BT(z_p_ghosted_, Bt_zp_);

    rhs_vel_ = r_vel_;
    rhs_vel_ += Bt_zp_;

    // Step 3: Solve for velocity: A * z_vel = rhs_vel
    z_vel_ = 0;

    {
        const double vel_rhs_norm = rhs_vel_.l2_norm();
        const double tol = inner_tolerance * std::max(vel_rhs_norm, 1e-30);

        dealii::SolverControl solver_control(max_inner_iterations, tol, false, false);

        dealii::SolverGMRES<dealii::TrilinosWrappers::MPI::Vector>::AdditionalData gmres_data;
        gmres_data.max_n_tmp_vectors = 50;
        dealii::SolverGMRES<dealii::TrilinosWrappers::MPI::Vector> gmres(solver_control, gmres_data);

        try
        {
            gmres.solve(velocity_block_, z_vel_, rhs_vel_, *A_preconditioner_);
        }
        catch (dealii::SolverControl::NoConvergence&)
        {
            // Fine for preconditioning
        }

        if (verbose_ && rank_ == 0)
            std::cout << "  [Schur vmult] A solve: " << solver_control.last_step()
                      << " its, res=" << std::scientific << std::setprecision(2)
                      << solver_control.last_value()
                      << ", rhs=" << vel_rhs_norm << std::defaultfloat << "\n";

        n_iterations_A += solver_control.last_step();
    }

    // Step 4: Assemble output
    dst = 0;
    insert_velocity(z_vel_, dst);
    insert_pressure(z_p_, dst);
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

    // p is now a ghosted vector (created in vmult) — safe to read off-rank entries
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
                sum += values[k] * p[static_cast<dealii::types::global_dof_index>(p_idx)];
        }

        vel[static_cast<dealii::types::global_dof_index>(vel_idx)] = sum;
    }

    vel.compress(dealii::VectorOperation::insert);
}
