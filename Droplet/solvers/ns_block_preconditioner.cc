// ============================================================================
// solvers/ns_block_preconditioner.cc - Schur Complement Preconditioner
//
// Implementation following deal.II step-22/56 pattern.
// ============================================================================

#include "solvers/ns_block_preconditioner.h"

#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <iostream>

// ============================================================================
// BlockSchurPreconditioner implementation
// ============================================================================
BlockSchurPreconditioner::BlockSchurPreconditioner(
    const dealii::SparseMatrix<double>& system_matrix,
    const dealii::SparseMatrix<double>& pressure_mass,
    const std::vector<dealii::types::global_dof_index>& ux_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& uy_to_ns_map,
    const std::vector<dealii::types::global_dof_index>& p_to_ns_map,
    bool do_solve_A)
    : n_iterations_A(0)
    , n_iterations_S(0)
    , system_matrix_ptr_(&system_matrix)
    , pressure_mass_ptr_(&pressure_mass)
    , ux_map_(ux_to_ns_map)      // Copy maps
    , uy_map_(uy_to_ns_map)
    , p_map_(p_to_ns_map)
    , n_ux_(ux_to_ns_map.size())
    , n_uy_(uy_to_ns_map.size())
    , n_p_(p_to_ns_map.size())
    , n_vel_(ux_to_ns_map.size() + uy_to_ns_map.size())
    , n_total_(ux_to_ns_map.size() + uy_to_ns_map.size() + p_to_ns_map.size())
    , do_solve_A_(do_solve_A)
{
    // Build reverse mappings for O(1) lookups
    global_to_vel_.assign(n_total_, -1);
    global_to_p_.assign(n_total_, -1);

    for (unsigned int i = 0; i < n_ux_; ++i)
        global_to_vel_[ux_map_[i]] = i;
    for (unsigned int i = 0; i < n_uy_; ++i)
        global_to_vel_[uy_map_[i]] = n_ux_ + i;
    for (unsigned int i = 0; i < n_p_; ++i)
        global_to_p_[p_map_[i]] = i;

    // Build velocity block sparsity pattern
    dealii::DynamicSparsityPattern dsp(n_vel_, n_vel_);

    for (unsigned int i = 0; i < n_ux_; ++i)
    {
        const unsigned int row = ux_map_[i];
        for (auto it = system_matrix_ptr_->begin(row); it != system_matrix_ptr_->end(row); ++it)
        {
            const int col_local = global_to_vel_[it->column()];
            if (col_local >= 0)
                dsp.add(i, col_local);
        }
    }
    for (unsigned int i = 0; i < n_uy_; ++i)
    {
        const unsigned int row = uy_map_[i];
        for (auto it = system_matrix_ptr_->begin(row); it != system_matrix_ptr_->end(row); ++it)
        {
            const int col_local = global_to_vel_[it->column()];
            if (col_local >= 0)
                dsp.add(n_ux_ + i, col_local);
        }
    }

    velocity_sparsity_.copy_from(dsp);
    velocity_block_.reinit(velocity_sparsity_);

    // Extract values and initialize preconditioners
    extract_velocity_block();
    A_preconditioner_.initialize(velocity_block_);

    // Verify maps don't exceed n_total_
    for (unsigned int i = 0; i < n_ux_; ++i)
        AssertIndexRange(ux_map_[i], n_total_);
    for (unsigned int i = 0; i < n_uy_; ++i)
        AssertIndexRange(uy_map_[i], n_total_);
    for (unsigned int i = 0; i < n_p_; ++i)
        AssertIndexRange(p_map_[i], n_total_);

    S_preconditioner_.initialize(*pressure_mass_ptr_);

    std::cout << "[Schur Preconditioner] Initialized: "
              << "A = " << n_vel_ << "x" << n_vel_
              << ", S = " << n_p_ << "x" << n_p_ << "\n";
}

void BlockSchurPreconditioner::update(
    const dealii::SparseMatrix<double>& system_matrix,
    const dealii::SparseMatrix<double>& pressure_mass)
{
    // Update pointers to current matrices
    system_matrix_ptr_ = &system_matrix;
    pressure_mass_ptr_ = &pressure_mass;

    // Re-extract velocity block values (sparsity unchanged)
    extract_velocity_block();

    // Rebuild ILU preconditioners with new values
    A_preconditioner_.initialize(velocity_block_);
    S_preconditioner_.initialize(*pressure_mass_ptr_);
}

void BlockSchurPreconditioner::extract_velocity_block()
{
    velocity_block_ = 0;

    for (unsigned int i = 0; i < n_ux_; ++i)
    {
        const unsigned int row = ux_map_[i];
        for (auto it = system_matrix_ptr_->begin(row); it != system_matrix_ptr_->end(row); ++it)
        {
            const int col_local = global_to_vel_[it->column()];
            if (col_local >= 0)
                velocity_block_.set(i, col_local, it->value());
        }
    }
    for (unsigned int i = 0; i < n_uy_; ++i)
    {
        const unsigned int row = uy_map_[i];
        for (auto it = system_matrix_ptr_->begin(row); it != system_matrix_ptr_->end(row); ++it)
        {
            const int col_local = global_to_vel_[it->column()];
            if (col_local >= 0)
                velocity_block_.set(n_ux_ + i, col_local, it->value());
        }
    }
}

void BlockSchurPreconditioner::vmult(
    dealii::Vector<double>& dst,
    const dealii::Vector<double>& src) const
{
    Assert(src.size() == n_total_,
           dealii::ExcDimensionMismatch(src.size(), n_total_));
    Assert(dst.size() == n_total_,
           dealii::ExcDimensionMismatch(dst.size(), n_total_));

    dealii::Vector<double> r_vel(n_vel_), r_p(n_p_);
    extract_velocity(src, r_vel);
    extract_pressure(src, r_p);

    // ========================================================================
    // Step 1: Solve for pressure  S̃ * z_p = r_p
    // ========================================================================
    dealii::Vector<double> z_p(n_p_);
    {
        const double p_rhs_norm = r_p.l2_norm();
        dealii::SolverControl solver_control(1000, 1e-4 * std::max(p_rhs_norm, 1e-30));
        dealii::SolverCG<dealii::Vector<double>> cg(solver_control);

        z_p = 0;
        cg.solve(*pressure_mass_ptr_, z_p, r_p, S_preconditioner_);
        n_iterations_S += solver_control.last_step();

        z_p *= -1.0;
    }

    // ========================================================================
    // Step 2: Update velocity RHS:  r_vel' = r_vel - B^T * z_p
    // ========================================================================
    dealii::Vector<double> Bt_zp(n_vel_);
    apply_BT(z_p, Bt_zp);

    dealii::Vector<double> rhs_vel(n_vel_);
    for (unsigned int i = 0; i < n_vel_; ++i)
        rhs_vel[i] = r_vel[i] + Bt_zp[i];

    // ========================================================================
    // Step 3: Solve for velocity  A * z_u = rhs_vel
    // ========================================================================
    dealii::Vector<double> z_vel(n_vel_);

    if (do_solve_A_)
    {
        const double vel_rhs_norm = rhs_vel.l2_norm();
        dealii::SolverControl solver_control(1000, 1e-4 * std::max(vel_rhs_norm, 1e-30));
        dealii::SolverCG<dealii::Vector<double>> cg(solver_control);

        z_vel = 0;
        cg.solve(velocity_block_, z_vel, rhs_vel, A_preconditioner_);
        n_iterations_A += solver_control.last_step();
    }
    else
    {
        A_preconditioner_.vmult(z_vel, rhs_vel);
        n_iterations_A += 1;
    }

    // ========================================================================
    // Step 4: Assemble solution
    // ========================================================================
    dst = 0;
    insert_velocity(z_vel, dst);
    insert_pressure(z_p, dst);
}

void BlockSchurPreconditioner::extract_velocity(
    const dealii::Vector<double>& src,
    dealii::Vector<double>& vel) const
{
    for (unsigned int i = 0; i < n_ux_; ++i)
        vel[i] = src[ux_map_[i]];
    for (unsigned int i = 0; i < n_uy_; ++i)
        vel[n_ux_ + i] = src[uy_map_[i]];
}

void BlockSchurPreconditioner::extract_pressure(
    const dealii::Vector<double>& src,
    dealii::Vector<double>& p) const
{
    for (unsigned int i = 0; i < n_p_; ++i)
        p[i] = src[p_map_[i]];
}

void BlockSchurPreconditioner::insert_velocity(
    const dealii::Vector<double>& vel,
    dealii::Vector<double>& dst) const
{
    for (unsigned int i = 0; i < n_ux_; ++i)
        dst[ux_map_[i]] = vel[i];
    for (unsigned int i = 0; i < n_uy_; ++i)
        dst[uy_map_[i]] = vel[n_ux_ + i];
}

void BlockSchurPreconditioner::insert_pressure(
    const dealii::Vector<double>& p,
    dealii::Vector<double>& dst) const
{
    for (unsigned int i = 0; i < n_p_; ++i)
        dst[p_map_[i]] = p[i];
}

void BlockSchurPreconditioner::apply_BT(
    const dealii::Vector<double>& p,
    dealii::Vector<double>& vel) const
{
    vel = 0;

    for (unsigned int i = 0; i < n_ux_; ++i)
    {
        const unsigned int row = ux_map_[i];
        double sum = 0.0;
        for (auto it = system_matrix_ptr_->begin(row); it != system_matrix_ptr_->end(row); ++it)
        {
            const int j = global_to_p_[it->column()];
            if (j >= 0)
                sum += it->value() * p[j];
        }
        vel[i] = sum;
    }

    for (unsigned int i = 0; i < n_uy_; ++i)
    {
        const unsigned int row = uy_map_[i];
        double sum = 0.0;
        for (auto it = system_matrix_ptr_->begin(row); it != system_matrix_ptr_->end(row); ++it)
        {
            const int j = global_to_p_[it->column()];
            if (j >= 0)
                sum += it->value() * p[j];
        }
        vel[n_ux_ + i] = sum;
    }
}