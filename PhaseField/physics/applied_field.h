// ============================================================================
// physics/applied_field.h - Applied Magnetic Field from Dipoles
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Equations 96-98, p.519; Section 6.2, p.522
// ============================================================================
#ifndef APPLIED_FIELD_H
#define APPLIED_FIELD_H

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <vector>

/**
 * @brief Computes the applied magnetic field h_a from point dipoles
 *
 * 2D Dipole potential (Eq. 97, p.519):
 *   φ_s(x) = d · (x_s - x) / |x_s - x|²
 *
 * 3D Dipole potential (Eq. 96, p.519):
 *   φ_s(x) = d · (x_s - x) / |x_s - x|³
 *
 * Applied field (Eq. 98):
 *   h_a = Σ_s α_s ∇φ_s
 *
 * NOTE: No 1/(2π) factor - absorbed into intensity α_s
 *
 * Dipole configuration (Section 6.2, p.522):
 *   Positions: (-0.5, -1.5), (0, -1.5), (0.5, -1.5), (1, -1.5), (1.5, -1.5)
 *   Direction: d = (0, 1)^T (upward)
 *   Intensity ramp: α_s(t) = 6000 * min(t/1.6, 1)
 */
template <int dim>
class AppliedField
{
public:
    /**
     * @brief Constructor with dipole parameters
     * @param positions Dipole locations x_s
     * @param direction Dipole direction d (unit vector)
     * @param intensity_max Maximum intensity (6000)
     * @param ramp_time Time to reach max intensity (1.6)
     */
    AppliedField(const std::vector<dealii::Point<dim>>& positions,
                 const dealii::Tensor<1, dim>& direction,
                 double intensity_max,
                 double ramp_time);
    
    /**
     * @brief Compute dipole potential φ_s at point x
     * 
     * 2D (Eq. 97): φ_s = d · (x_s - x) / |x_s - x|²
     * 3D (Eq. 96): φ_s = d · (x_s - x) / |x_s - x|³
     */
    double compute_potential(const dealii::Point<dim>& x, double time) const;
    
    /**
     * @brief Compute applied field h_a = Σ_s α_s ∇φ_s at point x
     */
    dealii::Tensor<1, dim> compute_field(const dealii::Point<dim>& x, double time) const;
    
    /**
     * @brief Compute intensity α_s(t) with ramping
     * 
     * α_s(t) = intensity_max * min(t / ramp_time, 1.0)
     */
    double get_intensity(double time) const;

private:
    std::vector<dealii::Point<dim>> positions_;
    dealii::Tensor<1, dim> direction_;
    double intensity_max_;
    double ramp_time_;
};

#endif // APPLIED_FIELD_H
