// ============================================================================
// setup/ns_setup.h - Navier-Stokes Coupled System Setup
//
// Sets up the 3-field NS system (ux, uy, p):
//   - Index maps (ux → NS, uy → NS, p → NS)
//   - Combined constraints (hanging nodes + no-slip BCs)
//   - Sparsity pattern (3×3 block structure)
//
// Block structure:
//   [ A_uxux  A_uxuy  B_uxp ]   [ ux ]
//   [ A_uyux  A_uyuy  B_uyp ] × [ uy ]
//   [ B_pux   B_puy   0     ]   [ p  ]
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef NS_SETUP_H
#define NS_SETUP_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <vector>

/**
 * @brief Set up the coupled Navier-Stokes system
 *
 * Creates data structures for the coupled ux-uy-p system:
 *   - Index maps: field DoF → coupled system index
 *   - Combined constraints: hanging nodes + BCs mapped to coupled indices
 *   - Sparsity pattern: 3×3 block structure
 *
 * Data layout in coupled system:
 *   ux occupies indices [0, n_ux)
 *   uy occupies indices [n_ux, n_ux + n_uy)
 *   p  occupies indices [n_ux + n_uy, n_ux + n_uy + n_p)
 *
 * @param ux_dof_handler      DoFHandler for velocity x-component (Q2)
 * @param uy_dof_handler      DoFHandler for velocity y-component (Q2)
 * @param p_dof_handler       DoFHandler for pressure (Q1)
 * @param ux_constraints      Individual constraints for ux
 * @param uy_constraints      Individual constraints for uy
 * @param p_constraints       Individual constraints for p
 * @param ux_to_ns_map        [OUT] Index map: ux DoF → coupled index
 * @param uy_to_ns_map        [OUT] Index map: uy DoF → coupled index
 * @param p_to_ns_map         [OUT] Index map: p DoF → coupled index
 * @param ns_combined_constraints [OUT] Combined constraints for coupled system
 * @param ns_sparsity         [OUT] Sparsity pattern for coupled system
 * @param verbose             Print setup info
 */
template <int dim>
void setup_ns_coupled_system(
    const dealii::DoFHandler<dim>& ux_dof_handler,
    const dealii::DoFHandler<dim>& uy_dof_handler,
    const dealii::DoFHandler<dim>& p_dof_handler,
    const dealii::AffineConstraints<double>& ux_constraints,
    const dealii::AffineConstraints<double>& uy_constraints,
    const dealii::AffineConstraints<double>& p_constraints,
    std::vector<dealii::types::global_dof_index>& ux_to_ns_map,
    std::vector<dealii::types::global_dof_index>& uy_to_ns_map,
    std::vector<dealii::types::global_dof_index>& p_to_ns_map,
    dealii::AffineConstraints<double>& ns_combined_constraints,
    dealii::SparsityPattern& ns_sparsity,
    bool verbose = false);

#endif // NS_SETUP_H