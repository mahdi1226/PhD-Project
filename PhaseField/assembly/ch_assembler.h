// ============================================================================
// assembly/ch_assembler.h - Cahn-Hilliard System Assembler
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Equations 42a-42b (discrete scheme), p.505
// ============================================================================
#ifndef CH_ASSEMBLER_H
#define CH_ASSEMBLER_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <vector>

// Forward declaration
struct Parameters;

/**
 * @brief Assemble the coupled Cahn-Hilliard system (Paper Eq. 42a-42b)
 *
 * @param theta_dof_handler  DoFHandler for phase field θ
 * @param psi_dof_handler    DoFHandler for chemical potential ψ
 * @param theta_old          Previous time step θ^{k-1}
 * @param ux_solution        Velocity x-component U^k_x
 * @param uy_solution        Velocity y-component U^k_y
 * @param params             Simulation parameters
 * @param dt                 Time step size τ
 * @param current_time       Current simulation time (for MMS)
 * @param theta_to_ch_map    Index mapping: θ DoF → coupled system index
 * @param psi_to_ch_map      Index mapping: ψ DoF → coupled system index
 * @param matrix             Output: assembled system matrix
 * @param rhs                Output: assembled RHS vector
 */
template <int dim>
void assemble_ch_system(
    const dealii::DoFHandler<dim>& theta_dof_handler,
    const dealii::DoFHandler<dim>& psi_dof_handler,
    const dealii::Vector<double>& theta_old,
    const dealii::Vector<double>& ux_solution,
    const dealii::Vector<double>& uy_solution,
    const Parameters& params,
    double dt,
    double current_time,
    const std::vector<dealii::types::global_dof_index>& theta_to_ch_map,
    const std::vector<dealii::types::global_dof_index>& psi_to_ch_map,
    dealii::SparseMatrix<double>& matrix,
    dealii::Vector<double>& rhs);

#endif // CH_ASSEMBLER_H