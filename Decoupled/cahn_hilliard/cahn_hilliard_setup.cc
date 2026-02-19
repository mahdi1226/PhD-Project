// ============================================================================
// cahn_hilliard/cahn_hilliard_setup.cc - DoFs, Constraints, Sparsity, Vectors
//
// Two CG Q2 DoFHandlers (θ, ψ) on the same triangulation.
// Coupled system layout:
//   [0, n_theta)        = θ block
//   [n_theta, n_total)  = ψ block
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================

#include "cahn_hilliard/cahn_hilliard.h"

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>

// ============================================================================
// Step 1: Distribute DoFs for both θ and ψ
// ============================================================================
template <int dim>
void CahnHilliardSubsystem<dim>::distribute_dofs()
{
    theta_dof_handler_.distribute_dofs(fe_);
    psi_dof_handler_.distribute_dofs(fe_);

    Assert(theta_dof_handler_.n_dofs() == psi_dof_handler_.n_dofs(),
           dealii::ExcMessage("θ and ψ DoF counts must match (same FE, same mesh)"));

    // Individual field index sets
    theta_locally_owned_ = theta_dof_handler_.locally_owned_dofs();
    theta_locally_relevant_ =
        dealii::DoFTools::extract_locally_relevant_dofs(theta_dof_handler_);

    psi_locally_owned_ = psi_dof_handler_.locally_owned_dofs();
    psi_locally_relevant_ =
        dealii::DoFTools::extract_locally_relevant_dofs(psi_dof_handler_);

    // Coupled system index sets
    const unsigned int n_theta = theta_dof_handler_.n_dofs();
    const unsigned int n_total = 2 * n_theta;

    // Owned: shift ψ indices by n_theta
    ch_locally_owned_.set_size(n_total);
    ch_locally_owned_.add_indices(theta_locally_owned_);

    dealii::IndexSet psi_owned_shifted(n_total);
    for (auto idx = psi_locally_owned_.begin(); idx != psi_locally_owned_.end(); ++idx)
        psi_owned_shifted.add_index(*idx + n_theta);
    ch_locally_owned_.add_indices(psi_owned_shifted);

    // Relevant: shift ψ indices by n_theta
    ch_locally_relevant_.set_size(n_total);
    ch_locally_relevant_.add_indices(theta_locally_relevant_);

    dealii::IndexSet psi_relevant_shifted(n_total);
    for (auto idx = psi_locally_relevant_.begin(); idx != psi_locally_relevant_.end(); ++idx)
        psi_relevant_shifted.add_index(*idx + n_theta);
    ch_locally_relevant_.add_indices(psi_relevant_shifted);

    pcout_ << "[CH Setup] DoFs distributed: "
           << n_theta << " per field, "
           << n_total << " coupled, "
           << ch_locally_owned_.n_elements() << " locally owned\n";
}

// ============================================================================
// Step 2: Build constraints (hanging nodes only for production)
// ============================================================================
template <int dim>
void CahnHilliardSubsystem<dim>::build_constraints()
{
    // θ constraints: hanging nodes only (Neumann BCs are natural)
    theta_constraints_.clear();
    theta_constraints_.reinit(theta_locally_owned_, theta_locally_relevant_);
    dealii::DoFTools::make_hanging_node_constraints(theta_dof_handler_,
                                                     theta_constraints_);
    theta_constraints_.close();

    // ψ constraints: hanging nodes only
    psi_constraints_.clear();
    psi_constraints_.reinit(psi_locally_owned_, psi_locally_relevant_);
    dealii::DoFTools::make_hanging_node_constraints(psi_dof_handler_,
                                                     psi_constraints_);
    psi_constraints_.close();

    // Build coupled constraints from individual constraints
    rebuild_coupled_constraints();
}

// ============================================================================
// Step 3: Build index maps (field DoF → coupled system index)
// ============================================================================
template <int dim>
void CahnHilliardSubsystem<dim>::build_index_maps()
{
    const unsigned int n_theta = theta_dof_handler_.n_dofs();
    const unsigned int n_psi = psi_dof_handler_.n_dofs();

    theta_to_ch_map_.resize(n_theta);
    psi_to_ch_map_.resize(n_psi);

    for (unsigned int i = 0; i < n_theta; ++i)
        theta_to_ch_map_[i] = i;

    for (unsigned int i = 0; i < n_psi; ++i)
        psi_to_ch_map_[i] = n_theta + i;
}

// ============================================================================
// Step 4: Build coupled sparsity pattern
//
// Block structure:
//   [θ-θ  θ-ψ]
//   [ψ-θ  ψ-ψ]
//
// Since θ and ψ use the same FE on the same mesh, all 4 blocks have
// identical sparsity structure. Build once from θ, replicate to all blocks.
// ============================================================================
template <int dim>
void CahnHilliardSubsystem<dim>::build_coupled_sparsity()
{
    const unsigned int n_theta = theta_dof_handler_.n_dofs();
    const unsigned int n_total = 2 * n_theta;

    // Base sparsity for single field (θ-θ block)
    dealii::DynamicSparsityPattern base_dsp(n_theta, n_theta,
                                             theta_locally_relevant_);
    dealii::DoFTools::make_sparsity_pattern(theta_dof_handler_,
                                             base_dsp,
                                             theta_constraints_,
                                             /*keep_constrained_dofs=*/true);

    // Coupled sparsity: replicate to all 4 blocks
    dealii::DynamicSparsityPattern dsp(n_total, n_total, ch_locally_relevant_);

    for (auto idx = theta_locally_relevant_.begin();
         idx != theta_locally_relevant_.end(); ++idx)
    {
        const unsigned int i = *idx;
        for (auto j = base_dsp.begin(i); j != base_dsp.end(i); ++j)
        {
            const unsigned int col = j->column();
            // θ-θ block
            dsp.add(theta_to_ch_map_[i], theta_to_ch_map_[col]);
            // θ-ψ block
            dsp.add(theta_to_ch_map_[i], psi_to_ch_map_[col]);
            // ψ-θ block
            dsp.add(psi_to_ch_map_[i], theta_to_ch_map_[col]);
            // ψ-ψ block
            dsp.add(psi_to_ch_map_[i], psi_to_ch_map_[col]);
        }
    }

    // Distribute across MPI ranks
    dealii::SparsityTools::distribute_sparsity_pattern(
        dsp, ch_locally_owned_, mpi_comm_, ch_locally_relevant_);

    // Initialize Trilinos matrix
    system_matrix_.reinit(ch_locally_owned_, ch_locally_owned_,
                          dsp, mpi_comm_);

    pcout_ << "[CH Setup] Sparsity: " << n_total << " x " << n_total
           << ", nnz = " << system_matrix_.n_nonzero_elements() << "\n";
}

// ============================================================================
// Step 5: Allocate vectors
// ============================================================================
template <int dim>
void CahnHilliardSubsystem<dim>::allocate_vectors()
{
    // Coupled RHS
    system_rhs_.reinit(ch_locally_owned_, mpi_comm_);

    // Individual field solutions (owned)
    theta_solution_.reinit(theta_locally_owned_, mpi_comm_);
    psi_solution_.reinit(psi_locally_owned_, mpi_comm_);

    // Individual field solutions (ghosted, for cross-subsystem reads)
    theta_relevant_.reinit(theta_locally_owned_, theta_locally_relevant_, mpi_comm_);
    psi_relevant_.reinit(psi_locally_owned_, psi_locally_relevant_, mpi_comm_);
}

// ============================================================================
// Rebuild coupled constraints from individual θ and ψ constraints
//
// Called during initial setup and after apply_dirichlet_boundary().
// Maps individual-field constraint entries to coupled system indices.
// ============================================================================
template <int dim>
void CahnHilliardSubsystem<dim>::rebuild_coupled_constraints()
{
    ch_constraints_.clear();
    ch_constraints_.reinit(ch_locally_owned_, ch_locally_relevant_);

    // Map θ constraints → coupled system
    for (auto idx = theta_locally_relevant_.begin();
         idx != theta_locally_relevant_.end(); ++idx)
    {
        const unsigned int i = *idx;
        if (theta_constraints_.is_constrained(i))
        {
            const auto* entries = theta_constraints_.get_constraint_entries(i);
            const double inhomogeneity = theta_constraints_.get_inhomogeneity(i);
            const auto coupled_i = theta_to_ch_map_[i];

            ch_constraints_.add_line(coupled_i);

            if (entries != nullptr && !entries->empty())
            {
                std::vector<std::pair<dealii::types::global_dof_index, double>> coupled_entries;
                for (const auto& entry : *entries)
                    coupled_entries.emplace_back(theta_to_ch_map_[entry.first],
                                                 entry.second);
                ch_constraints_.add_entries(coupled_i, coupled_entries);
            }

            ch_constraints_.set_inhomogeneity(coupled_i, inhomogeneity);
        }
    }

    // Map ψ constraints → coupled system
    for (auto idx = psi_locally_relevant_.begin();
         idx != psi_locally_relevant_.end(); ++idx)
    {
        const unsigned int i = *idx;
        if (psi_constraints_.is_constrained(i))
        {
            const auto* entries = psi_constraints_.get_constraint_entries(i);
            const double inhomogeneity = psi_constraints_.get_inhomogeneity(i);
            const auto coupled_i = psi_to_ch_map_[i];

            ch_constraints_.add_line(coupled_i);

            if (entries != nullptr && !entries->empty())
            {
                std::vector<std::pair<dealii::types::global_dof_index, double>> coupled_entries;
                for (const auto& entry : *entries)
                    coupled_entries.emplace_back(psi_to_ch_map_[entry.first],
                                                 entry.second);
                ch_constraints_.add_entries(coupled_i, coupled_entries);
            }

            ch_constraints_.set_inhomogeneity(coupled_i, inhomogeneity);
        }
    }

    ch_constraints_.close();
}

// ============================================================================
// Explicit instantiations
// ============================================================================
template class CahnHilliardSubsystem<2>;
template class CahnHilliardSubsystem<3>;
