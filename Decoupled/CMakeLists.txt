# ============================================================================
# Decoupled/CMakeLists.txt — Unified Build for Decoupled Ferrofluid Solver
#
# Builds all 4 subsystems + coupled MMS tests + production drivers
# in a single build tree. Each subdirectory remains independently buildable.
#
# Usage:
#   cd Decoupled && mkdir build && cd build
#   cmake .. -DDEAL_II_DIR=/path/to/dealii && make -j$(nproc)
#
# Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) Eq. 42a-42f
# ============================================================================

cmake_minimum_required(VERSION 3.13)
project(ferrofluid_decoupled_solver)

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
# Magnetization subsystem library
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
# Cahn-Hilliard subsystem library
# ============================================================================
add_library(cahn_hilliard_lib STATIC
    cahn_hilliard/cahn_hilliard.cc
    cahn_hilliard/cahn_hilliard_setup.cc
    cahn_hilliard/cahn_hilliard_assemble.cc
    cahn_hilliard/cahn_hilliard_solve.cc
    cahn_hilliard/cahn_hilliard_output.cc
)
target_link_libraries(cahn_hilliard_lib utilities_lib)
deal_ii_setup_target(cahn_hilliard_lib)

# ============================================================================
# Navier-Stokes subsystem library
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
# Subdirectories (libraries already defined above, subdirs add executables)
# ============================================================================
add_subdirectory(poisson)
add_subdirectory(magnetization)
add_subdirectory(cahn_hilliard)
add_subdirectory(navier_stokes)
add_subdirectory(mms_tests)
add_subdirectory(drivers)
