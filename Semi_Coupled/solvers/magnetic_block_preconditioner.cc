// ============================================================================
// solvers/magnetic_block_preconditioner.cc - Block Preconditioner for M+φ
//
// With component_wise renumbering, DoFs are contiguous:
//   [Mx(0..n_Mx-1) | My(n_Mx..n_M-1) | φ(n_M..n_total-1)]
//
// Block extraction uses simple index ranges — much simpler than NS
// where velocity GIDs are non-contiguous.
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "solvers/magnetic_block_preconditioner.h"

#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <Epetra_CrsMatrix.h>

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <memory>

// ============================================================================
// Constructor
// ============================================================================
MagneticBlockPreconditioner::MagneticBlockPreconditioner(
    const dealii::TrilinosWrappers::SparseMatrix& system_matrix,
    const dealii::IndexSet& mag_owned,
    dealii::types::global_dof_index n_M_dofs,
    dealii::types::global_dof_index n_phi_dofs,
    MPI_Comm mpi_comm,
    bool use_ilu)
    : n_iterations_M(0)
    , n_iterations_phi(0)
    , inner_tolerance(1e-1)
    , max_inner_iterations(20)
    , system_matrix_ptr_(&system_matrix)
    , n_M_(n_M_dofs)
    , n_phi_(n_phi_dofs)
    , n_total_(n_M_dofs + n_phi_dofs)
    , mag_owned_(mag_owned)
    , mpi_comm_(mpi_comm)
    , rank_(0)
    , tmp_initialized_(false)
{
    MPI_Comm_rank(mpi_comm_, &rank_);

    // ========================================================================
    // Step 1: Build owned index sets for M and φ blocks
    //
    // With component_wise renumbering:
    //   M DoFs:   [0, n_M)
    //   φ DoFs:   [n_M, n_total)
    // ========================================================================
    M_owned_.set_size(n_M_);
    phi_owned_.set_size(n_phi_);

    for (auto idx = mag_owned_.begin(); idx != mag_owned_.end(); ++idx)
    {
        if (*idx < n_M_)
            M_owned_.add_index(*idx);
        else
            phi_owned_.add_index(*idx - n_M_);
    }
    M_owned_.compress();
    phi_owned_.compress();

    // ========================================================================
    // Step 2: Extract diagonal blocks from system matrix
    //
    // We iterate over locally owned rows and split entries by column range.
    // M-block: rows [0, n_M), cols [0, n_M)      → A_M
    // φ-block: rows [n_M, n_total), cols [n_M, n_total) → A_φ
    // ========================================================================
    const Epetra_CrsMatrix& epetra_mat = system_matrix_ptr_->trilinos_matrix();
    const int num_my_rows = epetra_mat.NumMyRows();

    // --- Build block matrices using deal.II API (consistent maps) ---
    // First pass: build sparsity patterns
    dealii::TrilinosWrappers::SparsityPattern M_sp, phi_sp;
    M_sp.reinit(M_owned_, M_owned_, mpi_comm_);
    phi_sp.reinit(phi_owned_, phi_owned_, mpi_comm_);

    for (int local_row = 0; local_row < num_my_rows; ++local_row)
    {
        const long long gid = epetra_mat.GRID64(local_row);
        if (gid < 0)
            continue;

        int num_entries = 0;
        double* values = nullptr;
        int* col_indices = nullptr;
        epetra_mat.ExtractMyRowView(local_row, num_entries, values, col_indices);

        if (static_cast<dealii::types::global_dof_index>(gid) < n_M_)
        {
            if (!M_owned_.is_element(static_cast<dealii::types::global_dof_index>(gid)))
                continue;
            for (int k = 0; k < num_entries; ++k)
            {
                const long long col = epetra_mat.GCID64(col_indices[k]);
                if (col >= 0 &&
                    static_cast<dealii::types::global_dof_index>(col) < n_M_)
                    M_sp.add(static_cast<dealii::types::global_dof_index>(gid),
                             static_cast<dealii::types::global_dof_index>(col));
            }
        }
        else if (static_cast<dealii::types::global_dof_index>(gid) < n_total_)
        {
            const auto phi_row = static_cast<dealii::types::global_dof_index>(gid - n_M_);
            if (!phi_owned_.is_element(phi_row))
                continue;
            for (int k = 0; k < num_entries; ++k)
            {
                const long long col = epetra_mat.GCID64(col_indices[k]);
                if (col >= static_cast<long long>(n_M_) &&
                    static_cast<dealii::types::global_dof_index>(col) < n_total_)
                    phi_sp.add(phi_row,
                               static_cast<dealii::types::global_dof_index>(col - n_M_));
            }
        }
    }
    M_sp.compress();
    phi_sp.compress();

    // Create matrices and fill values
    M_block_.reinit(M_sp);
    phi_block_.reinit(phi_sp);

    for (int local_row = 0; local_row < num_my_rows; ++local_row)
    {
        const long long gid = epetra_mat.GRID64(local_row);
        if (gid < 0)
            continue;

        int num_entries = 0;
        double* values = nullptr;
        int* col_indices = nullptr;
        epetra_mat.ExtractMyRowView(local_row, num_entries, values, col_indices);

        if (static_cast<dealii::types::global_dof_index>(gid) < n_M_)
        {
            if (!M_owned_.is_element(static_cast<dealii::types::global_dof_index>(gid)))
                continue;
            for (int k = 0; k < num_entries; ++k)
            {
                const long long col = epetra_mat.GCID64(col_indices[k]);
                if (col >= 0 &&
                    static_cast<dealii::types::global_dof_index>(col) < n_M_)
                    M_block_.set(static_cast<dealii::types::global_dof_index>(gid),
                                 static_cast<dealii::types::global_dof_index>(col),
                                 values[k]);
            }
        }
        else if (static_cast<dealii::types::global_dof_index>(gid) < n_total_)
        {
            const auto phi_row = static_cast<dealii::types::global_dof_index>(gid - n_M_);
            if (!phi_owned_.is_element(phi_row))
                continue;
            for (int k = 0; k < num_entries; ++k)
            {
                const long long col = epetra_mat.GCID64(col_indices[k]);
                if (col >= static_cast<long long>(n_M_) &&
                    static_cast<dealii::types::global_dof_index>(col) < n_total_)
                    phi_block_.set(phi_row,
                                   static_cast<dealii::types::global_dof_index>(col - n_M_),
                                   values[k]);
            }
        }
    }
    M_block_.compress(dealii::VectorOperation::insert);
    phi_block_.compress(dealii::VectorOperation::insert);

    // ========================================================================
    // Step 2b: Precompute ghost phi indices for apply_C_M_phi
    //
    // For each locally owned M row, find all phi column indices in the
    // off-diagonal C_Mφ block. These may be on other MPI ranks.
    // We build an IndexSet that includes both owned and ghosted phi indices
    // so we can create a ghosted vector for the matvec product.
    // ========================================================================
    phi_relevant_for_coupling_ = phi_owned_;  // Start with owned

    for (int local_row = 0; local_row < num_my_rows; ++local_row)
    {
        const long long gid = epetra_mat.GRID64(local_row);
        if (gid < 0 || static_cast<dealii::types::global_dof_index>(gid) >= n_M_)
            continue;

        int num_entries = 0;
        double* values = nullptr;
        int* col_indices = nullptr;
        epetra_mat.ExtractMyRowView(local_row, num_entries, values, col_indices);

        for (int k = 0; k < num_entries; ++k)
        {
            const long long col = epetra_mat.GCID64(col_indices[k]);
            if (col >= static_cast<long long>(n_M_) &&
                static_cast<dealii::types::global_dof_index>(col) < n_total_)
            {
                phi_relevant_for_coupling_.add_index(
                    static_cast<dealii::types::global_dof_index>(col - n_M_));
            }
        }
    }
    phi_relevant_for_coupling_.compress();

    // ========================================================================
    // Step 3: Initialize preconditioners
    // ========================================================================
    if (use_ilu)
    {
        // ILU — HPC fallback (no ML/MueLu needed)
        auto M_ilu = std::make_unique<dealii::TrilinosWrappers::PreconditionILU>();
        dealii::TrilinosWrappers::PreconditionILU::AdditionalData ilu_data_M;
        ilu_data_M.ilu_fill = 0;  // ILU(0): DG mass-dominated, cheap
        M_ilu->initialize(M_block_, ilu_data_M);
        M_preconditioner_ = std::move(M_ilu);

        auto phi_ilu = std::make_unique<dealii::TrilinosWrappers::PreconditionILU>();
        dealii::TrilinosWrappers::PreconditionILU::AdditionalData ilu_data_phi;
        ilu_data_phi.ilu_fill = 1;  // ILU(1): Laplacian needs more fill
        phi_ilu->initialize(phi_block_, ilu_data_phi);
        phi_preconditioner_ = std::move(phi_ilu);
    }
    else
    {
        // AMG — preferred (requires ML or MueLu)
        // M block: DG mass + transport, non-elliptic
        auto M_amg = std::make_unique<dealii::TrilinosWrappers::PreconditionAMG>();
        dealii::TrilinosWrappers::PreconditionAMG::AdditionalData amg_data_M;
        amg_data_M.elliptic = false;               // DG, non-symmetric
        amg_data_M.higher_order_elements = false;   // Q1 DG
        amg_data_M.smoother_sweeps = 2;
        amg_data_M.aggregation_threshold = 0.02;
        amg_data_M.output_details = false;
        M_amg->initialize(M_block_, amg_data_M);
        M_preconditioner_ = std::move(M_amg);

        // φ block: Laplacian, SPD, elliptic
        auto phi_amg = std::make_unique<dealii::TrilinosWrappers::PreconditionAMG>();
        dealii::TrilinosWrappers::PreconditionAMG::AdditionalData amg_data_phi;
        amg_data_phi.elliptic = true;               // Laplacian is SPD
        amg_data_phi.higher_order_elements = true;   // Q2 CG
        amg_data_phi.smoother_sweeps = 2;
        amg_data_phi.aggregation_threshold = 0.02;
        amg_data_phi.output_details = false;
        phi_amg->initialize(phi_block_, amg_data_phi);
        phi_preconditioner_ = std::move(phi_amg);
    }

    if (rank_ == 0)
    {
        std::cout << "[Mag Block PC] Initialized: "
                  << "A_M = " << n_M_ << "x" << n_M_
                  << " (owned: " << M_owned_.n_elements() << ")"
                  << ", A_phi = " << n_phi_ << "x" << n_phi_
                  << " (owned: " << phi_owned_.n_elements() << ")"
                  << ", precond = " << (use_ilu ? "ILU" : "AMG")
                  << ", inner_tol = " << inner_tolerance
                  << ", max_inner = " << max_inner_iterations
                  << "\n";
    }
}

// ============================================================================
// vmult - Apply block-triangular preconditioner
//
// P^{-1} via back-substitution:
//   1. Solve A_φ · z_φ = r_φ         (CG + AMG)
//   2. rhs_M = r_M - C_Mφ · z_φ
//   3. Solve A_M · z_M = rhs_M       (GMRES + ILU/AMG)
// ============================================================================
void MagneticBlockPreconditioner::vmult(
    dealii::TrilinosWrappers::MPI::Vector& dst,
    const dealii::TrilinosWrappers::MPI::Vector& src) const
{
    // Lazy-initialize cached temporary vectors (first call only)
    if (!tmp_initialized_)
    {
        tmp_r_M_.reinit(M_owned_, mpi_comm_);
        tmp_r_phi_.reinit(phi_owned_, mpi_comm_);
        tmp_z_M_.reinit(M_owned_, mpi_comm_);
        tmp_z_phi_.reinit(phi_owned_, mpi_comm_);
        tmp_C_zphi_.reinit(M_owned_, mpi_comm_);
        tmp_rhs_M_.reinit(M_owned_, mpi_comm_);
        tmp_initialized_ = true;
    }

    // Extract sub-vectors
    extract_M(src, tmp_r_M_);
    extract_phi(src, tmp_r_phi_);

    // ====================================================================
    // Step 1: Solve A_φ · z_φ = r_φ
    // φ block is Laplacian (SPD) → CG + AMG
    // ====================================================================
    tmp_z_phi_ = 0;

    {
        const double phi_rhs_norm = tmp_r_phi_.l2_norm();
        if (phi_rhs_norm > 1e-30)
        {
            const double tol = inner_tolerance * phi_rhs_norm;
            dealii::SolverControl solver_control(max_inner_iterations, tol,
                                                  false, false);
            dealii::SolverCG<dealii::TrilinosWrappers::MPI::Vector> cg(solver_control);

            try
            {
                cg.solve(phi_block_, tmp_z_phi_, tmp_r_phi_, *phi_preconditioner_);
            }
            catch (dealii::SolverControl::NoConvergence&)
            {
                // Fine for preconditioning — partial convergence is OK
            }

            n_iterations_phi += solver_control.last_step();
        }
    }

    // ====================================================================
    // Step 2: Update M RHS: rhs_M = r_M - C_Mφ · z_φ
    // C_Mφ is the off-diagonal block coupling M rows to φ columns
    // ====================================================================
    apply_C_M_phi(tmp_z_phi_, tmp_C_zphi_);

    tmp_rhs_M_ = tmp_r_M_;
    tmp_rhs_M_ -= tmp_C_zphi_;

    // ====================================================================
    // Step 3: Solve A_M · z_M = rhs_M
    // M block is DG mass + transport (non-symmetric) → GMRES + AMG/ILU
    // ====================================================================
    tmp_z_M_ = 0;

    {
        const double M_rhs_norm = tmp_rhs_M_.l2_norm();
        if (M_rhs_norm > 1e-30)
        {
            const double tol = inner_tolerance * M_rhs_norm;
            dealii::SolverControl solver_control(max_inner_iterations, tol,
                                                  false, false);

            dealii::SolverGMRES<dealii::TrilinosWrappers::MPI::Vector>::AdditionalData gmres_data;
            gmres_data.max_n_tmp_vectors = 30;
            dealii::SolverGMRES<dealii::TrilinosWrappers::MPI::Vector> gmres(
                solver_control, gmres_data);

            try
            {
                gmres.solve(M_block_, tmp_z_M_, tmp_rhs_M_, *M_preconditioner_);
            }
            catch (dealii::SolverControl::NoConvergence&)
            {
                // Fine for preconditioning
            }

            n_iterations_M += solver_control.last_step();
        }
    }

    // ====================================================================
    // Step 4: Assemble output
    // ====================================================================
    dst = 0;
    insert_M(tmp_z_M_, dst);
    insert_phi(tmp_z_phi_, dst);
    dst.compress(dealii::VectorOperation::insert);
}

// ============================================================================
// Helper methods — simple index-range extraction/insertion
//
// With component_wise renumbering:
//   M DoFs:   global indices [0, n_M)       → sub-block index = global
//   φ DoFs:   global indices [n_M, n_total) → sub-block index = global - n_M
// ============================================================================

void MagneticBlockPreconditioner::extract_M(
    const dealii::TrilinosWrappers::MPI::Vector& src,
    dealii::TrilinosWrappers::MPI::Vector& M_vec) const
{
    M_vec = 0;
    for (auto idx = M_owned_.begin(); idx != M_owned_.end(); ++idx)
        M_vec[*idx] = src[*idx];

    M_vec.compress(dealii::VectorOperation::insert);
}

void MagneticBlockPreconditioner::extract_phi(
    const dealii::TrilinosWrappers::MPI::Vector& src,
    dealii::TrilinosWrappers::MPI::Vector& phi_vec) const
{
    phi_vec = 0;
    for (auto idx = phi_owned_.begin(); idx != phi_owned_.end(); ++idx)
    {
        const auto global_idx = *idx + n_M_;
        phi_vec[*idx] = src[global_idx];
    }

    phi_vec.compress(dealii::VectorOperation::insert);
}

void MagneticBlockPreconditioner::insert_M(
    const dealii::TrilinosWrappers::MPI::Vector& M_vec,
    dealii::TrilinosWrappers::MPI::Vector& dst) const
{
    for (auto idx = M_owned_.begin(); idx != M_owned_.end(); ++idx)
        dst[*idx] = M_vec[*idx];
}

void MagneticBlockPreconditioner::insert_phi(
    const dealii::TrilinosWrappers::MPI::Vector& phi_vec,
    dealii::TrilinosWrappers::MPI::Vector& dst) const
{
    for (auto idx = phi_owned_.begin(); idx != phi_owned_.end(); ++idx)
        dst[*idx + n_M_] = phi_vec[*idx];
}

// ============================================================================
// apply_C_M_phi: Compute C_Mφ · phi_vec
//
// C_Mφ is the off-diagonal block: rows in M range [0, n_M), columns in
// φ range [n_M, n_total). We iterate over M-owned rows and multiply
// only the φ-column entries.
//
// phi_vec is non-ghosted (locally owned only). The coupling stencil may
// reference phi DoFs on other MPI ranks. ReadWriteVector::import_elements
// handles MPI communication to fetch off-rank values.
// ============================================================================
void MagneticBlockPreconditioner::apply_C_M_phi(
    const dealii::TrilinosWrappers::MPI::Vector& phi_vec,
    dealii::TrilinosWrappers::MPI::Vector& M_vec) const
{
    M_vec = 0;

    // Create ghosted phi vector for off-rank access
    dealii::TrilinosWrappers::MPI::Vector phi_ghosted(
        phi_owned_, phi_relevant_for_coupling_, mpi_comm_);
    phi_ghosted = phi_vec;

    const Epetra_CrsMatrix& epetra_mat = system_matrix_ptr_->trilinos_matrix();
    const int num_my_rows = epetra_mat.NumMyRows();

    for (int local_row = 0; local_row < num_my_rows; ++local_row)
    {
        const long long gid = epetra_mat.GRID64(local_row);
        if (gid < 0)
            continue;

        // Only M rows
        if (static_cast<dealii::types::global_dof_index>(gid) >= n_M_)
            continue;

        // Only locally owned M DoFs
        if (!M_owned_.is_element(static_cast<dealii::types::global_dof_index>(gid)))
            continue;

        int num_entries = 0;
        double* values = nullptr;
        int* col_indices = nullptr;
        epetra_mat.ExtractMyRowView(local_row, num_entries, values, col_indices);

        double sum = 0.0;
        for (int k = 0; k < num_entries; ++k)
        {
            const long long col = epetra_mat.GCID64(col_indices[k]);
            if (col < static_cast<long long>(n_M_) || col < 0)
                continue;

            const auto phi_idx = static_cast<dealii::types::global_dof_index>(col - n_M_);
            if (phi_idx < n_phi_)
                sum += values[k] * phi_ghosted[phi_idx];
        }

        M_vec[static_cast<dealii::types::global_dof_index>(gid)] = sum;
    }

    M_vec.compress(dealii::VectorOperation::insert);
}
