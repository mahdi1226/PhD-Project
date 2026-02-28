# ============================================================================
# drivers/CMakeLists.txt — Production Ferrofluid Coupled Drivers
#
# Targets:
#   ferrofluid_decoupled   — Strategy A: Fully decoupled (Gauss-Seidel)
#
# All 4 subsystems (Poisson, Magnetization, CH, NS) are compiled into
# static libraries and linked against each driver executable.
#
# Usage:
#   cd drivers && mkdir build && cd build
#   cmake .. -DDEAL_II_DIR=/path/to/dealii && make -j$(nproc)
#   mpirun -np 4 ./ferrofluid_decoupled --rosensweig --refinement 5
#
# Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) Eq. 42a-42f
# ============================================================================

cmake_minimum_required(VERSION 3.13)
find_package(deal.II 9.4 REQUIRED HINTS ${DEAL_II_DIR} ${deal.II_DIR} $ENV{DEAL_II_DIR})
deal_ii_initialize_cached_variables()

project(ferrofluid_drivers)

set(SRC_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
include_directories(${SRC_ROOT})

# ============================================================================
# Shared utilities — guarded for top-level unified build
# ============================================================================
if(NOT TARGET utilities_lib)
    add_library(utilities_lib STATIC
        ${SRC_ROOT}/utilities/parameters.cc
    )
    deal_ii_setup_target(utilities_lib)
endif()

# ============================================================================
# Poisson subsystem library — guarded for top-level unified build
# ============================================================================
if(NOT TARGET poisson_lib)
    add_library(poisson_lib STATIC
        ${SRC_ROOT}/poisson/poisson.cc
        ${SRC_ROOT}/poisson/poisson_setup.cc
        ${SRC_ROOT}/poisson/poisson_assemble.cc
        ${SRC_ROOT}/poisson/poisson_solve.cc
        ${SRC_ROOT}/poisson/poisson_output.cc
    )
    target_link_libraries(poisson_lib utilities_lib)
    deal_ii_setup_target(poisson_lib)
endif()

# ============================================================================
# Magnetization subsystem library — guarded for top-level unified build
# ============================================================================
if(NOT TARGET magnetization_lib)
    add_library(magnetization_lib STATIC
        ${SRC_ROOT}/magnetization/magnetization.cc
        ${SRC_ROOT}/magnetization/magnetization_setup.cc
        ${SRC_ROOT}/magnetization/magnetization_assemble.cc
        ${SRC_ROOT}/magnetization/magnetization_solve.cc
        ${SRC_ROOT}/magnetization/magnetization_output.cc
    )
    target_link_libraries(magnetization_lib utilities_lib)
    deal_ii_setup_target(magnetization_lib)
endif()

# ============================================================================
# Cahn-Hilliard subsystem library — guarded for top-level unified build
# ============================================================================
if(NOT TARGET cahn_hilliard_lib)
    add_library(cahn_hilliard_lib STATIC
        ${SRC_ROOT}/cahn_hilliard/cahn_hilliard.cc
        ${SRC_ROOT}/cahn_hilliard/cahn_hilliard_setup.cc
        ${SRC_ROOT}/cahn_hilliard/cahn_hilliard_assemble.cc
        ${SRC_ROOT}/cahn_hilliard/cahn_hilliard_solve.cc
        ${SRC_ROOT}/cahn_hilliard/cahn_hilliard_output.cc
    )
    target_link_libraries(cahn_hilliard_lib utilities_lib)
    deal_ii_setup_target(cahn_hilliard_lib)
endif()

# ============================================================================
# Navier-Stokes subsystem library — guarded for top-level unified build
# ============================================================================
if(NOT TARGET navier_stokes_lib)
    add_library(navier_stokes_lib STATIC
        ${SRC_ROOT}/navier_stokes/navier_stokes.cc
        ${SRC_ROOT}/navier_stokes/navier_stokes_setup.cc
        ${SRC_ROOT}/navier_stokes/navier_stokes_assemble.cc
        ${SRC_ROOT}/navier_stokes/navier_stokes_solve.cc
        ${SRC_ROOT}/navier_stokes/navier_stokes_output.cc
    )
    target_link_libraries(navier_stokes_lib utilities_lib)
    deal_ii_setup_target(navier_stokes_lib)
endif()

# ============================================================================
# Strategy A: Fully Decoupled Driver (Gauss-Seidel Splitting)
# ============================================================================
add_executable(ferrofluid_decoupled decoupled_driver.cc)
target_link_libraries(ferrofluid_decoupled
    poisson_lib magnetization_lib cahn_hilliard_lib navier_stokes_lib utilities_lib)
deal_ii_setup_target(ferrofluid_decoupled)
