// ============================================================================
// diagnostics/spikes.h - Rosensweig Spike Detection
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Section 6.2, p.522: "5 peaks form inside the domain"
// ============================================================================
#ifndef SPIKES_H
#define SPIKES_H

#include "core/phase_field.h"
#include <vector>

/**
 * @brief Detects and analyzes Rosensweig instability spikes
 *
 * Expected results (Section 6.2, p.522):
 *   - 5 peaks form (theory predicts 4 from Eq. 103)
 *   - Peak spacing ≈ ℓ_c = 0.25
 *   - Interesting dynamics: t ∈ [0.7, 1.3]
 */
template <int dim>
class SpikeDetector
{
public:
    explicit SpikeDetector(const PhaseFieldProblem<dim>& problem);
    
    /**
     * @brief Find interface height at each x position
     * 
     * Interface defined as θ = 0 contour
     */
    std::vector<std::pair<double, double>> trace_interface() const;
    
    /**
     * @brief Count number of peaks
     */
    unsigned int count_peaks() const;
    
    /**
     * @brief Compute peak positions and heights
     */
    std::vector<std::pair<double, double>> find_peaks() const;
    
    /**
     * @brief Compute average peak spacing
     * 
     * Should be ≈ ℓ_c = 0.25 (Eq. 103, p.522)
     */
    double compute_peak_spacing() const;
    
    /**
     * @brief Print spike analysis summary
     */
    void print_analysis() const;

private:
    const PhaseFieldProblem<dim>& problem_;
};

#endif // SPIKES_H
