// ============================================================================
// physics/boundary_conditions.cc - Boundary Conditions Implementation
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// Equation 15, p.501
//
// Most boundary condition classes are header-only (simple zero functions).
// This file exists for explicit instantiations if needed.
// ============================================================================

#include "boundary_conditions.h"

// Explicit instantiations (if needed)
template class NoSlipBoundary<2>;
template class NoSlipBoundary<3>;
template class ZeroFunction<2>;
template class ZeroFunction<3>;
