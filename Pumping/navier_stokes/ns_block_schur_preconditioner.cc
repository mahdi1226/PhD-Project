// ============================================================================
// navier_stokes/ns_block_schur_preconditioner.cc - Block-Schur Implementation
//
// vmult() applies the right preconditioner P^{-1} to the monolithic vector:
//
//   Given src = [f_u; f_p] in the coupled [ux|uy|p] layout:
//
//   Step 1: Solve  S * p_tmp = -f_p  (Schur complement approximation)
//           S ≈ (1/ν_eff) M_p  →  p_tmp = -ν_eff * M_p^{-1} f_p
//
//   Step 2: Compute  u_rhs = f_u - B^T * p_tmp
//
//   Step 3: Solve  A * u_dst = u_rhs  (separate ux and uy via AMG+CG)
//
//   Step 4: Assemble dst = [u_dst; p_tmp]
//
// Reference: Elman, Silvester & Wathen, "Finite Elements and Fast Iterative
//            Solvers", 2nd ed., Chapter 8
// ============================================================================

#include "navier_stokes/ns_block_schur_preconditioner.h"

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>

template <int dim>
NSBlockSchurPreconditioner<dim>::NSBlockSchurPreconditioner(
    const dealii::TrilinosWrappers::SparseMatrix& A_ux_ux,
    const dealii::TrilinosWrappers::SparseMatrix& A_uy_uy,
    const dealii::TrilinosWrappers::SparseMatrix& Bt_ux,
    const dealii::TrilinosWrappers::SparseMatrix& Bt_uy,
    const dealii::TrilinosWrappers::SparseMatrix& B_ux,
    const dealii::TrilinosWrappers::SparseMatrix& B_uy,
    const dealii::TrilinosWrappers::SparseMatrix& M_p,
    double nu_eff,
    const dealii::IndexSet& ux_owned,
    const dealii::IndexSet& uy_owned,
    const dealii::IndexSet& p_owned,
    dealii::types::global_dof_index n_ux,
    dealii::types::global_dof_index n_uy,
    const LinearSolverParams& params,
    MPI_Comm mpi_comm)
    : A_ux_ux_(A_ux_ux)
    , A_uy_uy_(A_uy_uy)
    , Bt_ux_(Bt_ux)
    , Bt_uy_(Bt_uy)
    , B_ux_(B_ux)
    , B_uy_(B_uy)
    , M_p_(M_p)
    , nu_eff_(nu_eff)
    , ux_owned_(ux_owned)
    , uy_owned_(uy_owned)
    , p_owned_(p_owned)
    , n_ux_(n_ux)
    , n_uy_(n_uy)
    , params_(params)
    , mpi_comm_(mpi_comm)
{
}

template <int dim>
void NSBlockSchurPreconditioner<dim>::initialize()
{
    // AMG for velocity diagonal blocks
    dealii::TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
    amg_data.elliptic = true;
    amg_data.higher_order_elements = true;  // Q2 velocity
    amg_data.smoother_sweeps = 2;
    amg_data.aggregation_threshold = 0.02;

    amg_ux_.initialize(A_ux_ux_, amg_data);
    amg_uy_.initialize(A_uy_uy_, amg_data);

    // Jacobi for pressure mass matrix
    jacobi_p_.initialize(M_p_);

    // Allocate workspace vectors
    tmp_ux_.reinit(ux_owned_, mpi_comm_);
    tmp_uy_.reinit(uy_owned_, mpi_comm_);
    tmp_p_.reinit(p_owned_, mpi_comm_);
    rhs_ux_.reinit(ux_owned_, mpi_comm_);
    rhs_uy_.reinit(uy_owned_, mpi_comm_);
    rhs_p_.reinit(p_owned_, mpi_comm_);

    initialized_ = true;
}

template <int dim>
void NSBlockSchurPreconditioner<dim>::vmult(
    dealii::TrilinosWrappers::MPI::Vector& dst,
    const dealii::TrilinosWrappers::MPI::Vector& src) const
{
    Assert(initialized_, dealii::ExcMessage("Preconditioner not initialized"));

    // ================================================================
    // Extract components from monolithic src vector
    // Layout: [ux(0..n_ux) | uy(n_ux..n_ux+n_uy) | p(n_ux+n_uy..end)]
    // ================================================================
    for (auto it = ux_owned_.begin(); it != ux_owned_.end(); ++it)
        rhs_ux_[*it] = src[*it];
    rhs_ux_.compress(dealii::VectorOperation::insert);

    for (auto it = uy_owned_.begin(); it != uy_owned_.end(); ++it)
        rhs_uy_[*it] = src[n_ux_ + *it];
    rhs_uy_.compress(dealii::VectorOperation::insert);

    for (auto it = p_owned_.begin(); it != p_owned_.end(); ++it)
        rhs_p_[*it] = src[n_ux_ + n_uy_ + *it];
    rhs_p_.compress(dealii::VectorOperation::insert);

    // ================================================================
    // Step 1: Solve the Schur complement system
    //   S ≈ (1/ν_eff) M_p
    //   S * p_tmp = -f_p
    //   → M_p * p_tmp = -ν_eff * f_p
    // ================================================================
    // Scale RHS: rhs_p_ = -ν_eff * f_p
    rhs_p_ *= -nu_eff_;

    tmp_p_ = 0.0;
    {
        dealii::SolverControl solver_control(
            params_.schur_max_inner_iters,
            params_.schur_inner_tolerance * rhs_p_.l2_norm());

        dealii::SolverCG<dealii::TrilinosWrappers::MPI::Vector>
            cg(solver_control);

        try
        {
            cg.solve(M_p_, tmp_p_, rhs_p_, jacobi_p_);
        }
        catch (const dealii::SolverControl::NoConvergence&)
        {
            // Accept partial convergence for inner solve
        }
    }

    // ================================================================
    // Step 2: Compute velocity RHS = f_u - B^T * p_tmp
    //   rhs_ux -= Bt_ux * p_tmp
    //   rhs_uy -= Bt_uy * p_tmp
    // ================================================================
    // Restore original rhs_ux, rhs_uy (they were not modified)
    // rhs_ux_ and rhs_uy_ still hold the extracted values

    // But we need to revert rhs_p_ scaling for any future use - not needed here
    // Actually rhs_ux/uy were set in the extraction step and not modified, so they're fine

    dealii::TrilinosWrappers::MPI::Vector Bt_p_ux(ux_owned_, mpi_comm_);
    dealii::TrilinosWrappers::MPI::Vector Bt_p_uy(uy_owned_, mpi_comm_);

    Bt_ux_.vmult(Bt_p_ux, tmp_p_);
    Bt_uy_.vmult(Bt_p_uy, tmp_p_);

    // Restore original rhs from src (rhs_ux was extracted from src, not scaled)
    for (auto it = ux_owned_.begin(); it != ux_owned_.end(); ++it)
        rhs_ux_[*it] = src[*it];
    rhs_ux_.compress(dealii::VectorOperation::insert);

    for (auto it = uy_owned_.begin(); it != uy_owned_.end(); ++it)
        rhs_uy_[*it] = src[n_ux_ + *it];
    rhs_uy_.compress(dealii::VectorOperation::insert);

    rhs_ux_ -= Bt_p_ux;
    rhs_uy_ -= Bt_p_uy;

    // ================================================================
    // Step 3: Solve A * u_dst = u_rhs (AMG + CG, separate components)
    // ================================================================
    tmp_ux_ = 0.0;
    tmp_uy_ = 0.0;

    {
        dealii::SolverControl solver_control(
            params_.schur_max_inner_iters,
            params_.schur_inner_tolerance * rhs_ux_.l2_norm());

        dealii::SolverCG<dealii::TrilinosWrappers::MPI::Vector>
            cg(solver_control);

        try
        {
            cg.solve(A_ux_ux_, tmp_ux_, rhs_ux_, amg_ux_);
        }
        catch (const dealii::SolverControl::NoConvergence&)
        {
            // Accept partial convergence
        }
    }

    {
        dealii::SolverControl solver_control(
            params_.schur_max_inner_iters,
            params_.schur_inner_tolerance * rhs_uy_.l2_norm());

        dealii::SolverCG<dealii::TrilinosWrappers::MPI::Vector>
            cg(solver_control);

        try
        {
            cg.solve(A_uy_uy_, tmp_uy_, rhs_uy_, amg_uy_);
        }
        catch (const dealii::SolverControl::NoConvergence&)
        {
            // Accept partial convergence
        }
    }

    // ================================================================
    // Step 4: Insert components into monolithic dst
    // ================================================================
    dst = 0.0;

    for (auto it = ux_owned_.begin(); it != ux_owned_.end(); ++it)
        dst[*it] = tmp_ux_[*it];

    for (auto it = uy_owned_.begin(); it != uy_owned_.end(); ++it)
        dst[n_ux_ + *it] = tmp_uy_[*it];

    for (auto it = p_owned_.begin(); it != p_owned_.end(); ++it)
        dst[n_ux_ + n_uy_ + *it] = tmp_p_[*it];

    dst.compress(dealii::VectorOperation::insert);
}

// Explicit instantiations
template class NSBlockSchurPreconditioner<2>;
template class NSBlockSchurPreconditioner<3>;
