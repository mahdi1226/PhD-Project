// ============================================================================
// mms/ns/ns_variable_nu_mms_test.h - NS Variable Viscosity MMS Test Interface
//
// Tests variable viscosity ν(θ) in Navier-Stokes.
// Uses PRODUCTION code paths with prescribed analytical θ.
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef NS_VARIABLE_NU_MMS_TEST_H
#define NS_VARIABLE_NU_MMS_TEST_H

#include "utilities/parameters.h"
#include "mms/mms_verification.h"  // For MMSConvergenceResult
#include <vector>

// ============================================================================
// Main test function - runs convergence study
//
// Validates that variable viscosity ν(θ) is correctly handled:
//   ν(θ) = ν_water·(1-θ)/2 + ν_ferro·(1+θ)/2
//
// Uses prescribed θ = cos(πx)cos(πy/L_y) to get known ν(x,y).
// If convergence rates match NS_STANDALONE, variable viscosity is correct.
// ============================================================================
MMSConvergenceResult run_ns_variable_nu_standalone(
    const std::vector<unsigned int>& refinements,
    const Parameters& params,
    unsigned int n_time_steps = 10);

#endif // NS_VARIABLE_NU_MMS_TEST_H