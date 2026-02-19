# ============================================================================
# navier_stokes/CMakeLists.txt — Navier-Stokes Subsystem Build
#
# Targets:
#   navier_stokes_lib          — Static library (facade + internals + VTK output)
#   test_navier_stokes_mms     — MMS convergence test (4-phase, pass/fail)
#
# Usage:
#   mpirun -np 4 ./test_navier_stokes_mms --phase A B C D --ref 2 3 4 5
#
# Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) Eq. 42e-42f
# ============================================================================

cmake_minimum_required(VERSION 3.13)
find_package(deal.II 9.4 REQUIRED HINTS ${DEAL_II_DIR} ${deal.II_DIR} $ENV{DEAL_II_DIR})
deal_ii_initialize_cached_variables()

project(navier_stokes_subsystem)

# SRC_ROOT = parent directory (src/) so includes like "navier_stokes/navier_stokes.h" work
set(SRC_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
include_directories(${SRC_ROOT})

# ============================================================================
# Static library: facade + setup + assembly + solver + VTK output
# ============================================================================
add_library(navier_stokes_lib STATIC
        navier_stokes.cc
        navier_stokes_setup.cc
        navier_stokes_assemble.cc
        navier_stokes_solve.cc
        navier_stokes_output.cc
        ${SRC_ROOT}/utilities/parameters.cc
)
deal_ii_setup_target(navier_stokes_lib)

# ============================================================================
# MMS convergence test (4-phase: Steady/Unsteady Stokes/NS)
#
# Usage:
#   mpirun -np 4 ./test_navier_stokes_mms --phase A B C D --ref 2 3 4 5
# ============================================================================
add_executable(test_navier_stokes_mms
        tests/navier_stokes_mms_test.cc
)
target_link_libraries(test_navier_stokes_mms navier_stokes_lib)
deal_ii_setup_target(test_navier_stokes_mms)

# ============================================================================
# Standalone driver (4-mode: mms, 2d, 3d, temporal)
#
# Usage:
#   mpirun -np 4 ./navier_stokes_main --mode mms --ref 2 3 4 5
# ============================================================================
add_executable(navier_stokes_main
        navier_stokes_main.cc
)
target_link_libraries(navier_stokes_main navier_stokes_lib)
deal_ii_setup_target(navier_stokes_main)
