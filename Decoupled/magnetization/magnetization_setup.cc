// ============================================================================
// magnetization/magnetization_setup.cc — CG DoF Distribution, Constraints,
//                                         Sparsity, Vectors
//
// Private methods called by MagnetizationSubsystem::setup():
//   1. distribute_dofs()       — CG Q1 DoF distribution + CM renumbering
//   2. build_constraints()     — Hanging node constraints (AMR)
//   3. build_sparsity_pattern() — Standard CG sparsity (no face coupling)
//   4. allocate_vectors()       — Owned + ghosted vectors for Mx, My
//
// CG SPECIFICS (Zhang Eq 3.6: N_h ∈ C⁰(Ω)):
//   - Hanging-node constraints for AMR compatibility
//   - Standard sparsity (no face coupling needed)
//   - Single matrix shared by Mx and My (same transport operator)
//
// Reference: Zhang, He & Yang, SIAM J. Sci. Comput. 43(1) (2021) B167-B193
// ============================================================================

#include "magnetization/magnetization.h"

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

using namespace dealii;

// ============================================================================
// distribute_dofs() — Attach FE_Q(1) and extract parallel index sets
// ============================================================================
template <int dim>
void MagnetizationSubsystem<dim>::distribute_dofs()
{
    dof_handler_.distribute_dofs(fe_);

    // Cuthill-McKee renumbering (before extracting index sets)
    if (params_.renumber_dofs)
        DoFRenumbering::Cuthill_McKee(dof_handler_);

    locally_owned_dofs_    = dof_handler_.locally_owned_dofs();
    locally_relevant_dofs_ = DoFTools::extract_locally_relevant_dofs(dof_handler_);

    pcout_ << "[Magnetization Setup] DoFs distributed: "
           << dof_handler_.n_dofs() << " total, "
           << locally_owned_dofs_.n_elements() << " local"
           << std::endl;
}

// ============================================================================
// build_constraints() — Hanging node constraints for CG
//
// CG elements require hanging-node constraints for AMR compatibility.
// No Dirichlet or pinning constraints for magnetization.
// ============================================================================
template <int dim>
void MagnetizationSubsystem<dim>::build_constraints()
{
    constraints_.clear();
    constraints_.reinit(locally_owned_dofs_, locally_relevant_dofs_);

    DoFTools::make_hanging_node_constraints(dof_handler_, constraints_);

    constraints_.close();
}

// ============================================================================
// build_sparsity_pattern() — Standard CG sparsity (no face coupling)
//
// CG does not need face coupling (continuity is built into the FE space).
// Constraints are respected in the sparsity pattern.
// ============================================================================
template <int dim>
void MagnetizationSubsystem<dim>::build_sparsity_pattern()
{
    TrilinosWrappers::SparsityPattern trilinos_sp(
        locally_owned_dofs_,
        locally_owned_dofs_,
        locally_relevant_dofs_,
        mpi_comm_);

    DoFTools::make_sparsity_pattern(
        dof_handler_, trilinos_sp, constraints_,
        /*keep_constrained_dofs=*/false);

    trilinos_sp.compress();

    system_matrix_.reinit(trilinos_sp);

    pcout_ << "[Magnetization Setup] Sparsity: nnz = "
           << trilinos_sp.n_nonzero_elements() << " (CG)"
           << std::endl;
}

// ============================================================================
// allocate_vectors() — Owned (Mx, My solution + rhs) and ghosted (relevant)
// ============================================================================
template <int dim>
void MagnetizationSubsystem<dim>::allocate_vectors()
{
    // RHS vectors (owned only — assembled locally, no ghost writes)
    Mx_rhs_.reinit(locally_owned_dofs_, mpi_comm_);
    My_rhs_.reinit(locally_owned_dofs_, mpi_comm_);

    // Solution vectors (owned — solver output)
    Mx_solution_.reinit(locally_owned_dofs_, mpi_comm_);
    My_solution_.reinit(locally_owned_dofs_, mpi_comm_);

    // Ghosted vectors (for cross-subsystem reads)
    Mx_relevant_.reinit(locally_owned_dofs_, locally_relevant_dofs_, mpi_comm_);
    My_relevant_.reinit(locally_owned_dofs_, locally_relevant_dofs_, mpi_comm_);

    // Old-time ghosted vectors (for M^{n-1})
    Mx_old_relevant_.reinit(locally_owned_dofs_, locally_relevant_dofs_, mpi_comm_);
    My_old_relevant_.reinit(locally_owned_dofs_, locally_relevant_dofs_, mpi_comm_);

    // Spin-vorticity RHS cache (Zhang Eq 3.14 term: +½(∇×u × m^{n-1}, Z))
    spin_vort_rhs_x_.reinit(locally_owned_dofs_, mpi_comm_);
    spin_vort_rhs_y_.reinit(locally_owned_dofs_, mpi_comm_);

    // Explicit transport RHS cache (Zhang Eq 3.14 Step 5)
    explicit_transport_rhs_x_.reinit(locally_owned_dofs_, mpi_comm_);
    explicit_transport_rhs_y_.reinit(locally_owned_dofs_, mpi_comm_);

    pcout_ << "[Magnetization Setup] Vectors allocated (Mx+My: "
           << "2×solution, 2×rhs, 2×ghosted, 2×old_ghosted, 2×spin_vort, 2×transport)" << std::endl;
}

// ============================================================================
// Explicit instantiations
// ============================================================================
template void MagnetizationSubsystem<2>::distribute_dofs();
template void MagnetizationSubsystem<2>::build_constraints();
template void MagnetizationSubsystem<2>::build_sparsity_pattern();
template void MagnetizationSubsystem<2>::allocate_vectors();

template void MagnetizationSubsystem<3>::distribute_dofs();
template void MagnetizationSubsystem<3>::build_constraints();
template void MagnetizationSubsystem<3>::build_sparsity_pattern();
template void MagnetizationSubsystem<3>::allocate_vectors();
