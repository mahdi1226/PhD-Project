// ============================================================================
// block_structure.h - Block indices for coupled NS-CH system
// ============================================================================
#ifndef NSCH_BLOCK_STRUCTURE_H
#define NSCH_BLOCK_STRUCTURE_H

/**
 * @brief Block structure for staggered (segregated) approach
 * 
 * We solve two separate systems:
 *   NS system: [velocity (dim components), pressure (1 component)]
 *   CH system: [concentration c, chemical potential μ]
 * 
 * This allows reusing the existing NS and CH solvers.
 */
struct NSBlocks
{
    static constexpr unsigned int velocity = 0;
    static constexpr unsigned int pressure = 1;
    static constexpr unsigned int n_blocks = 2;
};

struct CHBlocks
{
    static constexpr unsigned int c        = 0;
    static constexpr unsigned int mu       = 1;
    static constexpr unsigned int n_blocks = 2;
};

/**
 * @brief Block structure for monolithic approach (future)
 * 
 * Single system: [velocity, pressure, c, μ]
 */
struct NSCHBlocksMonolithic
{
    static constexpr unsigned int velocity = 0;
    static constexpr unsigned int pressure = 1;
    static constexpr unsigned int c        = 2;
    static constexpr unsigned int mu       = 3;
    static constexpr unsigned int n_blocks = 4;
};

#endif // NSCH_BLOCK_STRUCTURE_H