# ============================================================================
# cahn_hilliard/CMakeLists.txt - Cahn-Hilliard Subsystem Build
#
# Targets:
#   cahn_hilliard_lib        — Static library (facade + internals)
#   test_cahn_hilliard_mms   — MMS convergence study
#   cahn_hilliard_main       — Standalone production driver (2D/3D VTK)
#
# Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) Eq. 42a-42b
# ============================================================================

cmake_minimum_required(VERSION 3.13)
find_package(deal.II 9.4 REQUIRED HINTS ${DEAL_II_DIR} ${deal.II_DIR} $ENV{DEAL_II_DIR})
deal_ii_initialize_cached_variables()

project(cahn_hilliard_subsystem)

# SRC_ROOT = parent directory (src/) so includes like "cahn_hilliard/cahn_hilliard.h" work
set(SRC_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
include_directories(${SRC_ROOT})

# Cahn-Hilliard static library
add_library(cahn_hilliard_lib STATIC
        cahn_hilliard.cc
        cahn_hilliard_setup.cc
        cahn_hilliard_assemble.cc
        cahn_hilliard_solve.cc
        cahn_hilliard_output.cc
        ${SRC_ROOT}/utilities/parameters.cc
)
deal_ii_setup_target(cahn_hilliard_lib)

# Standalone MMS convergence test
add_executable(test_cahn_hilliard_mms
        tests/cahn_hilliard_mms_test.cc
)
target_link_libraries(test_cahn_hilliard_mms cahn_hilliard_lib)
deal_ii_setup_target(test_cahn_hilliard_mms)

# Standalone production driver (spinodal decomposition + VTK)
add_executable(cahn_hilliard_main
        cahn_hilliard_main.cc
)
target_link_libraries(cahn_hilliard_main cahn_hilliard_lib)
deal_ii_setup_target(cahn_hilliard_main)