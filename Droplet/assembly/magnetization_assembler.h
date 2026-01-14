// ============================================================================
// assemblers/magnetization_assembler.h - DG Magnetization Equation Assembler
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Eq. 42c: Magnetization transport with relaxation
//
// Discrete equation (per component):
//   (1/τ + 1/T)(M^k, Z) - B_h^m(U^k, Z, M^k) = (1/T)(χ_θ H^k, Z) + (1/τ)(M^{k-1}, Z)
//
// Key insight from paper:
//   - M_x and M_y are DECOUPLED (same matrix, different RHS)
//   - Solve two SCALAR DG equations, not a coupled vector system
//
// Assembly strategy:
//   - Assemble ONE scalar system matrix (reuse for both components)
//   - Assemble TWO RHS vectors (one per component)
//   - Solve TWO scalar systems
//
// ============================================================================
#ifndef MAGNETIZATION_ASSEMBLER_H
#define MAGNETIZATION_ASSEMBLER_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include "utilities/parameters.h"

/**
 * @brief Assembles the scalar DG magnetization system (Eq. 42c)
 *
 * One matrix, two RHS vectors (Mx and My share the same operator).
 *
 * Matrix: (1/τ + 1/T)M - A_transport
 *   - Mass term: (1/τ + 1/T) ∫ φ_i φ_j
 *   - Transport cell: -[(U·∇φ_i)φ_j + (1/2)(∇·U)(φ_i φ_j)]
 *   - Transport face: +[[φ_i]]·{φ_j}·(U·n)  [Eq. 57 with -B_h^m]
 *
 * RHS_x: (1/T)(χ_θ H_x, φ_i) + (1/τ)(Mx_old, φ_i)
 * RHS_y: (1/T)(χ_θ H_y, φ_i) + (1/τ)(My_old, φ_i)
 */
template <int dim>
class MagnetizationAssembler
{
public:
    /**
     * @brief Constructor
     *
     * @param params       Simulation parameters
     * @param M_dof        DoFHandler for M (DG, scalar - same for Mx and My)
     * @param U_dof        DoFHandler for velocity components (CG)
     * @param phi_dof      DoFHandler for potential φ (CG)
     * @param theta_dof    DoFHandler for phase field θ (CG)
     */
    MagnetizationAssembler(const Parameters& params,
                           const dealii::DoFHandler<dim>& M_dof,
                           const dealii::DoFHandler<dim>& U_dof,
                           const dealii::DoFHandler<dim>& phi_dof,
                           const dealii::DoFHandler<dim>& theta_dof);

    /**
     * @brief Create sparsity pattern with face coupling
     *
     * DG transport requires neighbor coupling through faces.
     * Must call this BEFORE assemble().
     */
    void create_sparsity_pattern(dealii::SparsityPattern& sparsity) const;

    /**
     * @brief Assemble system matrix and both RHS vectors
     *
     * @param system_matrix  [OUT] Scalar system matrix
     * @param rhs_x          [OUT] RHS for Mx component
     * @param rhs_y          [OUT] RHS for My component
     * @param Ux, Uy         Velocity components (CG)
     * @param phi            Magnetic potential (CG)
     * @param theta          Phase field (CG)
     * @param Mx_old, My_old Previous magnetization (DG)
     * @param dt             Time step τ
     */
    void assemble(dealii::SparseMatrix<double>& system_matrix,
                  dealii::Vector<double>& rhs_x,
                  dealii::Vector<double>& rhs_y,
                  const dealii::Vector<double>& Ux,
                  const dealii::Vector<double>& Uy,
                  const dealii::Vector<double>& phi,
                  const dealii::Vector<double>& theta,
                  const dealii::Vector<double>& Mx_old,
                  const dealii::Vector<double>& My_old,
                  double dt) const;

    /**
     * @brief Assemble only RHS vectors (when matrix unchanged)
     *
     * Use when U hasn't changed significantly.
     */
    void assemble_rhs_only(dealii::Vector<double>& rhs_x,
                           dealii::Vector<double>& rhs_y,
                           const dealii::Vector<double>& phi,
                           const dealii::Vector<double>& theta,
                           const dealii::Vector<double>& Mx_old,
                           const dealii::Vector<double>& My_old,
                           double dt) const;

private:
    /**
     * @brief Compute χ(θ, H) with optional Langevin saturation
     *
     * Linear:   χ = χ₀ H(θ/ε)
     * Langevin: χ = 3χ₀ L(ξ)/ξ H(θ/ε) where ξ = |H|/H_c
     *
     * @param theta_val  Phase field value
     * @param H_mag      Magnetic field magnitude |H|
     */
    double chi(double theta_val, double H_mag) const;

    const Parameters& params_;
    const dealii::DoFHandler<dim>& M_dof_handler_;
    const dealii::DoFHandler<dim>& U_dof_handler_;
    const dealii::DoFHandler<dim>& phi_dof_handler_;
    const dealii::DoFHandler<dim>& theta_dof_handler_;
};

#endif // MAGNETIZATION_ASSEMBLER_H