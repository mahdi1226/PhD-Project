# ============================================================================
# FHD/CMakeLists.txt — Unified Build for Ferrohydrodynamics Solver
#
# Phase A: Reproduce Nochetto, Salgado & Tomas, arXiv:1511.04381 (2015)
#   Single-phase ferrofluid: NS + Angular Momentum + Magnetization + Poisson
#
# Phase B: Two-phase extension with Cahn-Hilliard (future)
#
# Usage:
#   cd FHD && mkdir build && cd build
#   cmake .. -DDEAL_II_DIR=/path/to/dealii && make -j$(nproc)
# ============================================================================

cmake_minimum_required(VERSION 3.13)
project(fhd_solver)

find_package(deal.II 9.4 REQUIRED
    HINTS ${DEAL_II_DIR} ${deal.II_DIR} $ENV{DEAL_II_DIR})
deal_ii_initialize_cached_variables()

# SRC_ROOT for the entire project tree
set(SRC_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${SRC_ROOT})

# ============================================================================
# Shared utilities (compiled once, linked everywhere)
# ============================================================================
add_library(utilities_lib STATIC
    utilities/parameters.cc
)
deal_ii_setup_target(utilities_lib)

# ============================================================================
# Poisson subsystem library
# ============================================================================
add_library(poisson_lib STATIC
    poisson/poisson.cc
    poisson/poisson_setup.cc
    poisson/poisson_assemble.cc
    poisson/poisson_solve.cc
    poisson/poisson_output.cc
)
target_link_libraries(poisson_lib utilities_lib)
deal_ii_setup_target(poisson_lib)

# ============================================================================
# Magnetization subsystem library (DG interior penalty)
# ============================================================================
add_library(magnetization_lib STATIC
    magnetization/magnetization.cc
    magnetization/magnetization_setup.cc
    magnetization/magnetization_assemble.cc
    magnetization/magnetization_solve.cc
    magnetization/magnetization_output.cc
)
target_link_libraries(magnetization_lib utilities_lib)
deal_ii_setup_target(magnetization_lib)

# ============================================================================
# Navier-Stokes subsystem library (micropolar)
# ============================================================================
add_library(navier_stokes_lib STATIC
    navier_stokes/navier_stokes.cc
    navier_stokes/navier_stokes_setup.cc
    navier_stokes/navier_stokes_assemble.cc
    navier_stokes/navier_stokes_solve.cc
    navier_stokes/navier_stokes_output.cc
)
target_link_libraries(navier_stokes_lib utilities_lib)
deal_ii_setup_target(navier_stokes_lib)

# ============================================================================
# Angular Momentum subsystem library
# ============================================================================
add_library(angular_momentum_lib STATIC
    angular_momentum/angular_momentum.cc
    angular_momentum/angular_momentum_setup.cc
    angular_momentum/angular_momentum_assemble.cc
    angular_momentum/angular_momentum_solve.cc
    angular_momentum/angular_momentum_output.cc
)
target_link_libraries(angular_momentum_lib utilities_lib)
deal_ii_setup_target(angular_momentum_lib)

# ============================================================================
# Passive scalar subsystem library (Section 7.3, Eq. 104)
# ============================================================================
add_library(passive_scalar_lib STATIC
    passive_scalar/passive_scalar.cc
    passive_scalar/passive_scalar_setup.cc
    passive_scalar/passive_scalar_assemble.cc
    passive_scalar/passive_scalar_solve.cc
    passive_scalar/passive_scalar_output.cc
)
target_link_libraries(passive_scalar_lib utilities_lib)
deal_ii_setup_target(passive_scalar_lib)

# ============================================================================
# Subdirectories (add standalone MMS tests + coupled tests + drivers)
# ============================================================================
add_subdirectory(poisson)
add_subdirectory(magnetization)
add_subdirectory(navier_stokes)
add_subdirectory(angular_momentum)
add_subdirectory(mms_tests)
add_subdirectory(drivers)
