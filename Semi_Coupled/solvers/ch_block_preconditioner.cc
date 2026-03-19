// ============================================================================
// solvers/ch_block_preconditioner.cc - Block Preconditioner for CH (θ+ψ)
//
// With CH setup, DoFs are contiguous:
//   [θ(0..n_theta-1) | ψ(n_theta..n_total-1)]
//
// Block extraction uses simple index ranges — same pattern as Mag block PC.
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "solvers/ch_block_preconditioner.h"

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
CHBlockPreconditioner::CHBlockPreconditioner(
    const dealii::TrilinosWrappers::SparseMatrix& system_matrix,
    const dealii::IndexSet& ch_owned,
    dealii::types::global_dof_index n_theta_dofs,
    dealii::types::global_dof_index n_psi_dofs,
    MPI_Comm mpi_comm,
    bool use_ilu)
    : n_iterations_theta(0)
    , n_iterations_psi(0)
    , inner_tolerance(1e-2)
    , max_inner_iterations(50)
    , system_matrix_ptr_(&system_matrix)
    , n_theta_(n_theta_dofs)
    , n_psi_(n_psi_dofs)
    , n_total_(n_theta_dofs + n_psi_dofs)
    , ch_owned_(ch_owned)
    , mpi_comm_(mpi_comm)
    , rank_(0)
    , tmp_initialized_(false)
{
    MPI_Comm_rank(mpi_comm_, &rank_);

    // ========================================================================
    // Step 1: Build owned index sets for θ and ψ blocks
    //
    // With CH setup:
    //   θ DoFs:   [0, n_theta)
    //   ψ DoFs:   [n_theta, n_total)
    // ========================================================================
    theta_owned_.set_size(n_theta_);
    psi_owned_.set_size(n_psi_);

    for (auto idx = ch_owned_.begin(); idx != ch_owned_.end(); ++idx)
    {
        if (*idx < n_theta_)
            theta_owned_.add_index(*idx);
        else
            psi_owned_.add_index(*idx - n_theta_);
    }
    theta_owned_.compress();
    psi_owned_.compress();

    // ========================================================================
    // Step 2: Extract diagonal blocks from system matrix
    //
    // θ-block: rows [0, n_theta), cols [0, n_theta)         → A_θθ
    // ψ-block: rows [n_theta, n_total), cols [n_theta, n_total) → A_ψψ
    // ========================================================================
    const Epetra_CrsMatrix& epetra_mat = system_matrix_ptr_->trilinos_matrix();
    const int num_my_rows = epetra_mat.NumMyRows();

    // Build sparsity patterns
    dealii::TrilinosWrappers::SparsityPattern theta_sp, psi_sp;
    theta_sp.reinit(theta_owned_, theta_owned_, mpi_comm_);
    psi_sp.reinit(psi_owned_, psi_owned_, mpi_comm_);

    for (int local_row = 0; local_row < num_my_rows; ++local_row)
    {
        const long long gid = epetra_mat.GRID64(local_row);
        if (gid < 0)
            continue;

        int num_entries = 0;
        double* values = nullptr;
        int* col_indices = nullptr;
        epetra_mat.ExtractMyRowView(local_row, num_entries, values, col_indices);

        if (static_cast<dealii::types::global_dof_index>(gid) < n_theta_)
        {
            if (!theta_owned_.is_element(static_cast<dealii::types::global_dof_index>(gid)))
                continue;
            for (int k = 0; k < num_entries; ++k)
            {
                const long long col = epetra_mat.GCID64(col_indices[k]);
                if (col >= 0 &&
                    static_cast<dealii::types::global_dof_index>(col) < n_theta_)
                    theta_sp.add(static_cast<dealii::types::global_dof_index>(gid),
                                 static_cast<dealii::types::global_dof_index>(col));
            }
        }
        else if (static_cast<dealii::types::global_dof_index>(gid) < n_total_)
        {
            const auto psi_row = static_cast<dealii::types::global_dof_index>(gid - n_theta_);
            if (!psi_owned_.is_element(psi_row))
                continue;
            for (int k = 0; k < num_entries; ++k)
            {
                const long long col = epetra_mat.GCID64(col_indices[k]);
                if (col >= static_cast<long long>(n_theta_) &&
                    static_cast<dealii::types::global_dof_index>(col) < n_total_)
                    psi_sp.add(psi_row,
                               static_cast<dealii::types::global_dof_index>(col - n_theta_));
            }
        }
    }
    theta_sp.compress();
    psi_sp.compress();

    // Create matrices and fill values
    theta_block_.reinit(theta_sp);
    psi_block_.reinit(psi_sp);

    for (int local_row = 0; local_row < num_my_rows; ++local_row)
    {
        const long long gid = epetra_mat.GRID64(local_row);
        if (gid < 0)
            continue;

        int num_entries = 0;
        double* values = nullptr;
        int* col_indices = nullptr;
        epetra_mat.ExtractMyRowView(local_row, num_entries, values, col_indices);

        if (static_cast<dealii::types::global_dof_index>(gid) < n_theta_)
        {
            if (!theta_owned_.is_element(static_cast<dealii::types::global_dof_index>(gid)))
                continue;
            for (int k = 0; k < num_entries; ++k)
            {
                const long long col = epetra_mat.GCID64(col_indices[k]);
                if (col >= 0 &&
                    static_cast<dealii::types::global_dof_index>(col) < n_theta_)
                    theta_block_.set(static_cast<dealii::types::global_dof_index>(gid),
                                     static_cast<dealii::types::global_dof_index>(col),
                                     values[k]);
            }
        }
        else if (static_cast<dealii::types::global_dof_index>(gid) < n_total_)
        {
            const auto psi_row = static_cast<dealii::types::global_dof_index>(gid - n_theta_);
            if (!psi_owned_.is_element(psi_row))
                continue;
            for (int k = 0; k < num_entries; ++k)
            {
                const long long col = epetra_mat.GCID64(col_indices[k]);
                if (col >= static_cast<long long>(n_theta_) &&
                    static_cast<dealii::types::global_dof_index>(col) < n_total_)
                    psi_block_.set(psi_row,
                                   static_cast<dealii::types::global_dof_index>(col - n_theta_),
                                   values[k]);
            }
        }
    }
    theta_block_.compress(dealii::VectorOperation::insert);
    psi_block_.compress(dealii::VectorOperation::insert);

    // ========================================================================
    // Step 2b: Precompute ghost ψ indices for apply_A_theta_psi
    // ========================================================================
    psi_relevant_for_coupling_ = psi_owned_;

    for (int local_row = 0; local_row < num_my_rows; ++local_row)
    {
        const long long gid = epetra_mat.GRID64(local_row);
        if (gid < 0 || static_cast<dealii::types::global_dof_index>(gid) >= n_theta_)
            continue;

        int num_entries = 0;
        double* values = nullptr;
        int* col_indices = nullptr;
        epetra_mat.ExtractMyRowView(local_row, num_entries, values, col_indices);

        for (int k = 0; k < num_entries; ++k)
        {
            const long long col = epetra_mat.GCID64(col_indices[k]);
            if (col >= static_cast<long long>(n_theta_) &&
                static_cast<dealii::types::global_dof_index>(col) < n_total_)
            {
                psi_relevant_for_coupling_.add_index(
                    static_cast<dealii::types::global_dof_index>(col - n_theta_));
            }
        }
    }
    psi_relevant_for_coupling_.compress();

    // ========================================================================
    // Step 3: Initialize preconditioners
    // ========================================================================
    if (use_ilu)
    {
        // θ block: mass + convection (non-symmetric) → ILU(2)
        auto theta_ilu = std::make_unique<dealii::TrilinosWrappers::PreconditionILU>();
        dealii::TrilinosWrappers::PreconditionILU::AdditionalData ilu_data_theta;
        ilu_data_theta.ilu_fill = 2;
        theta_ilu->initialize(theta_block_, ilu_data_theta);
        theta_preconditioner_ = std::move(theta_ilu);

        // ψ block: mass matrix (SPD, well-conditioned) → ILU(0) sufficient
        auto psi_ilu = std::make_unique<dealii::TrilinosWrappers::PreconditionILU>();
        dealii::TrilinosWrappers::PreconditionILU::AdditionalData ilu_data_psi;
        ilu_data_psi.ilu_fill = 0;
        psi_ilu->initialize(psi_block_, ilu_data_psi);
        psi_preconditioner_ = std::move(psi_ilu);
    }
    else
    {
        // θ block: mass + convection, non-symmetric → AMG with non-elliptic settings
        auto theta_amg = std::make_unique<dealii::TrilinosWrappers::PreconditionAMG>();
        dealii::TrilinosWrappers::PreconditionAMG::AdditionalData amg_data_theta;
        amg_data_theta.elliptic = false;
        amg_data_theta.higher_order_elements = true;   // Q2 CG
        amg_data_theta.smoother_sweeps = 2;
        amg_data_theta.aggregation_threshold = 0.02;
        amg_data_theta.output_details = false;
        theta_amg->initialize(theta_block_, amg_data_theta);
        theta_preconditioner_ = std::move(theta_amg);

        // ψ block: mass matrix (SPD) → Jacobi is sufficient
        auto psi_jacobi = std::make_unique<dealii::TrilinosWrappers::PreconditionJacobi>();
        psi_jacobi->initialize(psi_block_);
        psi_preconditioner_ = std::move(psi_jacobi);
    }

    if (rank_ == 0)
    {
        std::cout << "[CH Block PC] Initialized: "
                  << "A_theta = " << n_theta_ << "x" << n_theta_
                  << " (owned: " << theta_owned_.n_elements() << ")"
                  << ", A_psi = " << n_psi_ << "x" << n_psi_
                  << " (owned: " << psi_owned_.n_elements() << ")"
                  << ", precond = " << (use_ilu ? "ILU" : "AMG+Jacobi")
                  << ", inner_tol = " << inner_tolerance
                  << ", max_inner = " << max_inner_iterations
                  << "\n";
    }
}

// ============================================================================
// vmult - Apply block-triangular preconditioner
//
// P^{-1} via forward substitution:
//   1. Solve A_ψψ · z_ψ = r_ψ         (CG + Jacobi, mass matrix)
//   2. rhs_θ = r_θ - A_θψ · z_ψ
//   3. Solve A_θθ · z_θ = rhs_θ       (GMRES + AMG/ILU)
// ============================================================================
void CHBlockPreconditioner::vmult(
    dealii::TrilinosWrappers::MPI::Vector& dst,
    const dealii::TrilinosWrappers::MPI::Vector& src) const
{
    // Lazy-initialize cached temporary vectors
    if (!tmp_initialized_)
    {
        tmp_r_theta_.reinit(theta_owned_, mpi_comm_);
        tmp_r_psi_.reinit(psi_owned_, mpi_comm_);
        tmp_z_theta_.reinit(theta_owned_, mpi_comm_);
        tmp_z_psi_.reinit(psi_owned_, mpi_comm_);
        tmp_C_zpsi_.reinit(theta_owned_, mpi_comm_);
        tmp_rhs_theta_.reinit(theta_owned_, mpi_comm_);
        tmp_psi_ghosted_.reinit(psi_owned_, psi_relevant_for_coupling_, mpi_comm_);
        tmp_initialized_ = true;
    }

    // Extract sub-vectors
    extract_theta(src, tmp_r_theta_);
    extract_psi(src, tmp_r_psi_);

    // ====================================================================
    // Step 1: Solve A_ψψ · z_ψ = r_ψ
    // ψ block is mass matrix (SPD) → CG + Jacobi (~1-2 iterations)
    // ====================================================================
    tmp_z_psi_ = 0;

    {
        const double psi_rhs_norm = tmp_r_psi_.l2_norm();
        if (psi_rhs_norm > 1e-30)
        {
            const double tol = inner_tolerance * psi_rhs_norm;
            dealii::SolverControl solver_control(max_inner_iterations, tol,
                                                  false, false);
            dealii::SolverCG<dealii::TrilinosWrappers::MPI::Vector> cg(solver_control);

            try
            {
                cg.solve(psi_block_, tmp_z_psi_, tmp_r_psi_, *psi_preconditioner_);
            }
            catch (dealii::SolverControl::NoConvergence&)
            {
                // Fine for preconditioning — partial convergence is OK
            }

            n_iterations_psi += solver_control.last_step();
        }
    }

    // ====================================================================
    // Step 2: Update θ RHS: rhs_θ = r_θ - A_θψ · z_ψ
    // A_θψ is the off-diagonal block coupling θ rows to ψ columns
    // ====================================================================
    apply_A_theta_psi(tmp_z_psi_, tmp_C_zpsi_);

    tmp_rhs_theta_ = tmp_r_theta_;
    tmp_rhs_theta_ -= tmp_C_zpsi_;

    // ====================================================================
    // Step 3: Solve A_θθ · z_θ = rhs_θ
    // θ block is mass + convection (non-symmetric) → GMRES + AMG/ILU
    // ====================================================================
    tmp_z_theta_ = 0;

    {
        const double theta_rhs_norm = tmp_rhs_theta_.l2_norm();
        if (theta_rhs_norm > 1e-30)
        {
            const double tol = inner_tolerance * theta_rhs_norm;
            dealii::SolverControl solver_control(max_inner_iterations, tol,
                                                  false, false);

            dealii::SolverGMRES<dealii::TrilinosWrappers::MPI::Vector>::AdditionalData gmres_data;
            gmres_data.max_n_tmp_vectors = 30;
            dealii::SolverGMRES<dealii::TrilinosWrappers::MPI::Vector> gmres(
                solver_control, gmres_data);

            try
            {
                gmres.solve(theta_block_, tmp_z_theta_, tmp_rhs_theta_, *theta_preconditioner_);
            }
            catch (dealii::SolverControl::NoConvergence&)
            {
                // Fine for preconditioning
            }

            n_iterations_theta += solver_control.last_step();
        }
    }

    // ====================================================================
    // Step 4: Assemble output
    // ====================================================================
    dst = 0;
    insert_theta(tmp_z_theta_, dst);
    insert_psi(tmp_z_psi_, dst);
    dst.compress(dealii::VectorOperation::insert);
}

// ============================================================================
// Helper methods — simple index-range extraction/insertion
//
// With CH setup:
//   θ DoFs:   global indices [0, n_theta)       → sub-block index = global
//   ψ DoFs:   global indices [n_theta, n_total) → sub-block index = global - n_theta
// ============================================================================

void CHBlockPreconditioner::extract_theta(
    const dealii::TrilinosWrappers::MPI::Vector& src,
    dealii::TrilinosWrappers::MPI::Vector& theta_vec) const
{
    theta_vec = 0;
    for (auto idx = theta_owned_.begin(); idx != theta_owned_.end(); ++idx)
        theta_vec[*idx] = src[*idx];

    theta_vec.compress(dealii::VectorOperation::insert);
}

void CHBlockPreconditioner::extract_psi(
    const dealii::TrilinosWrappers::MPI::Vector& src,
    dealii::TrilinosWrappers::MPI::Vector& psi_vec) const
{
    psi_vec = 0;
    for (auto idx = psi_owned_.begin(); idx != psi_owned_.end(); ++idx)
    {
        const auto global_idx = *idx + n_theta_;
        psi_vec[*idx] = src[global_idx];
    }

    psi_vec.compress(dealii::VectorOperation::insert);
}

void CHBlockPreconditioner::insert_theta(
    const dealii::TrilinosWrappers::MPI::Vector& theta_vec,
    dealii::TrilinosWrappers::MPI::Vector& dst) const
{
    for (auto idx = theta_owned_.begin(); idx != theta_owned_.end(); ++idx)
        dst[*idx] = theta_vec[*idx];
}

void CHBlockPreconditioner::insert_psi(
    const dealii::TrilinosWrappers::MPI::Vector& psi_vec,
    dealii::TrilinosWrappers::MPI::Vector& dst) const
{
    for (auto idx = psi_owned_.begin(); idx != psi_owned_.end(); ++idx)
        dst[*idx + n_theta_] = psi_vec[*idx];
}

// ============================================================================
// apply_A_theta_psi: Compute A_θψ · psi_vec
//
// A_θψ is the off-diagonal block: rows in θ range [0, n_theta), columns in
// ψ range [n_theta, n_total).
// ============================================================================
void CHBlockPreconditioner::apply_A_theta_psi(
    const dealii::TrilinosWrappers::MPI::Vector& psi_vec,
    dealii::TrilinosWrappers::MPI::Vector& theta_vec) const
{
    theta_vec = 0;

    // Use cached ghosted ψ vector
    tmp_psi_ghosted_ = psi_vec;

    const Epetra_CrsMatrix& epetra_mat = system_matrix_ptr_->trilinos_matrix();
    const int num_my_rows = epetra_mat.NumMyRows();

    for (int local_row = 0; local_row < num_my_rows; ++local_row)
    {
        const long long gid = epetra_mat.GRID64(local_row);
        if (gid < 0)
            continue;

        // Only θ rows
        if (static_cast<dealii::types::global_dof_index>(gid) >= n_theta_)
            continue;

        // Only locally owned θ DoFs
        if (!theta_owned_.is_element(static_cast<dealii::types::global_dof_index>(gid)))
            continue;

        int num_entries = 0;
        double* values = nullptr;
        int* col_indices = nullptr;
        epetra_mat.ExtractMyRowView(local_row, num_entries, values, col_indices);

        double sum = 0.0;
        for (int k = 0; k < num_entries; ++k)
        {
            const long long col = epetra_mat.GCID64(col_indices[k]);
            if (col < static_cast<long long>(n_theta_) || col < 0)
                continue;

            const auto psi_idx = static_cast<dealii::types::global_dof_index>(col - n_theta_);
            if (psi_idx < n_psi_)
                sum += values[k] * tmp_psi_ghosted_[psi_idx];
        }

        theta_vec[static_cast<dealii::types::global_dof_index>(gid)] = sum;
    }

    theta_vec.compress(dealii::VectorOperation::insert);
}
