// ============================================================================
// output/vtk_writer.h - VTK Output Writer
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef VTK_WRITER_H
#define VTK_WRITER_H

#include "core/phase_field.h"
#include <string>

/**
 * @brief Writes solution data to VTK files for visualization
 *
 * Output fields:
 *   - theta: Phase field θ
 *   - psi: Chemical potential ψ
 *   - mx, my: Magnetization components
 *   - phi: Magnetic potential φ
 *   - ux, uy: Velocity components
 *   - p: Pressure
 */
template <int dim>
class VTKWriter
{
public:
    explicit VTKWriter(const PhaseFieldProblem<dim>& problem);
    
    /**
     * @brief Write solution at given time step
     * @param step Time step number
     */
    void write(unsigned int step) const;
    
    /**
     * @brief Set output directory
     */
    void set_output_directory(const std::string& dir);

private:
    const PhaseFieldProblem<dim>& problem_;
    std::string output_dir_;
};

#endif // VTK_WRITER_H
