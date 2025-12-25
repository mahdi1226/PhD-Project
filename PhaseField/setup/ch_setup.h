// ============================================================================
// setup/ch_setup.h - Cahn-Hilliard Coupled System Setup
//
// Free function to build coupled θ-ψ system:
//   - Index maps (θ → coupled, ψ → coupled)
//   - Combined constraints (hanging nodes + BCs mapped to coupled indices)
//   - Sparsity pattern (2×2 block structure)
//
// This is extracted from phase_field_setup.cc to:
//   1. Keep setup code modular and reusable
//   2. Establish pattern for NS, Poisson, Magnetization setups
//   3. Avoid monolithic setup files
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef CH_SETUP_H
#define CH_SETUP_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <vector>

/**
 * @brief Set up the coupled Cahn-Hilliard system
 *
 * Creates the data structures needed for the coupled θ-ψ system:
 *   - Index maps: field DoF → coupled system index
 *   - Combined constraints: hanging nodes + BCs mapped to coupled indices
 *   - Sparsity pattern: 2×2 block structure [θ-θ, θ-ψ; ψ-θ, ψ-ψ]
 *
 * Data layout in coupled system:
 *   θ occupies indices [0, n_theta)
 *   ψ occupies indices [n_theta, n_theta + n_psi)
 *
 * @param theta_dof_handler    DoFHandler for phase field θ
 * @param psi_dof_handler      DoFHandler for chemical potential ψ
 * @param theta_constraints    Individual constraints for θ (hanging nodes + BCs)
 * @param psi_constraints      Individual constraints for ψ (hanging nodes + BCs)
 * @param theta_to_ch_map      [OUT] Index map: θ DoF → coupled index
 * @param psi_to_ch_map        [OUT] Index map: ψ DoF → coupled index
 * @param ch_combined_constraints [OUT] Combined constraints for coupled system
 * @param ch_sparsity          [OUT] Sparsity pattern for coupled system
 * @param verbose              Print setup info
 */
template <int dim>
void setup_ch_coupled_system(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::DoFHandler<dim>& psi_dof_handler,
    const dealii::AffineConstraints<double>& theta_constraints,
    const dealii::AffineConstraints<double>& psi_constraints,
    std::vector<dealii::types::global_dof_index>& theta_to_ch_map,
    std::vector<dealii::types::global_dof_index>& psi_to_ch_map,
    dealii::AffineConstraints<double>& ch_combined_constraints,
    dealii::SparsityPattern& ch_sparsity,
    bool verbose = false);

#endif // CH_SETUP_H