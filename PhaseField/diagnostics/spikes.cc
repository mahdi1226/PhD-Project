// ============================================================================
// diagnostics/spikes.cc - Spike Detection Implementation
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Section 6.2, p.522
// ============================================================================

#include "spikes.h"
#include "output/logger.h"

template <int dim>
SpikeDetector<dim>::SpikeDetector(const PhaseFieldProblem<dim>& problem)
    : problem_(problem)
{
    Logger::info("SpikeDetector constructor");
}

template <int dim>
std::vector<std::pair<double, double>> SpikeDetector<dim>::trace_interface() const
{
    Logger::info("  SpikeDetector::trace_interface() [skeleton]");
    
    // TODO: Trace the θ = 0 contour
    // For each x in [0, 1], find y where θ(x, y) = 0
    
    return {};
}

template <int dim>
unsigned int SpikeDetector<dim>::count_peaks() const
{
    Logger::info("  SpikeDetector::count_peaks() [skeleton]");
    
    // TODO: Count local maxima in interface height
    // Expected: 5 peaks (p.522)
    
    return 0;
}

template <int dim>
std::vector<std::pair<double, double>> SpikeDetector<dim>::find_peaks() const
{
    Logger::info("  SpikeDetector::find_peaks() [skeleton]");
    
    // TODO: Find (x, height) of each peak
    
    return {};
}

template <int dim>
double SpikeDetector<dim>::compute_peak_spacing() const
{
    Logger::info("  SpikeDetector::compute_peak_spacing() [skeleton]");
    
    // TODO: Compute average distance between adjacent peaks
    // Expected: ℓ_c ≈ 0.25 (Eq. 103, p.522)
    
    return 0.0;
}

template <int dim>
void SpikeDetector<dim>::print_analysis() const
{
    Logger::info("  SpikeDetector::print_analysis()");
    Logger::info("    Number of peaks: " + std::to_string(count_peaks()));
    Logger::info("    Peak spacing: " + std::to_string(compute_peak_spacing()));
}

template class SpikeDetector<2>;
template class SpikeDetector<3>;
