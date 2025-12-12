// ============================================================================
// utilities/block_structure.h - Block Structure for Coupled Systems
// ============================================================================
#ifndef BLOCK_STRUCTURE_H
#define BLOCK_STRUCTURE_H

/**
 * @brief Block indices for coupled systems
 *
 * Cahn-Hilliard system (2 blocks):
 *   Block 0: θ (phase field)
 *   Block 1: ψ (chemical potential)
 *
 * Navier-Stokes system (3 blocks in 2D):
 *   Block 0: u_x
 *   Block 1: u_y
 *   Block 2: p
 *
 * Magnetization system (2 blocks in 2D):
 *   Block 0: m_x
 *   Block 1: m_y
 */
namespace BlockIndices
{
    // Cahn-Hilliard blocks
    namespace CH
    {
        constexpr unsigned int theta = 0;
        constexpr unsigned int psi   = 1;
        constexpr unsigned int n_blocks = 2;
    }
    
    // Navier-Stokes blocks (2D)
    namespace NS
    {
        constexpr unsigned int u_x = 0;
        constexpr unsigned int u_y = 1;
        constexpr unsigned int p   = 2;
        constexpr unsigned int n_blocks = 3;
    }
    
    // Magnetization blocks (2D)
    namespace Mag
    {
        constexpr unsigned int m_x = 0;
        constexpr unsigned int m_y = 1;
        constexpr unsigned int n_blocks = 2;
    }
    
} // namespace BlockIndices

#endif // BLOCK_STRUCTURE_H
