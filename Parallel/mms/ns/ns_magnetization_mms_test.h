// ============================================================================
// mms/ns/ns_magnetization_mms_test.h - NS with Magnetic Force MMS Test Header
//
// Tests: F_mag = μ₀(M·∇)H in Navier-Stokes equations
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef NS_MAGNETIZATION_MMS_TEST_H
#define NS_MAGNETIZATION_MMS_TEST_H

#include "utilities/parameters.h"
#include "mms/mms_verification.h"  // For MMSConvergenceResult
#include <vector>

// Forward declaration
struct NSMagMMSResult;
struct NSMagMMSConvergenceResult;

// ============================================================================
// Main test function
// ============================================================================
NSMagMMSConvergenceResult run_ns_magnetization_mms(
    const std::vector<unsigned int>& refinements,
    const Parameters& params,
    unsigned int n_time_steps = 10);

// ============================================================================
// Wrapper for mms_verification.cc integration
// ============================================================================
MMSConvergenceResult run_ns_magnetization_standalone(
    const std::vector<unsigned int>& refinements,
    const Parameters& params,
    unsigned int n_time_steps = 10);

#endif // NS_MAGNETIZATION_MMS_TEST_H