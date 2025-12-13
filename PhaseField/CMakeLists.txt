## ============================================================================
## CMakeLists.txt - Cahn-Hilliard Standalone Test
##
## Build:
##   mkdir build && cd build
##   cmake ..
##   make
##
## Run:
##   ./ferrofluid --ic_type 0 --t-final 0.1 --output_frequency 10
## ============================================================================

cmake_minimum_required(VERSION 3.13)
project(ferrofluid CXX)

# Find deal.II
find_package(deal.II 9.5 REQUIRED
        HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR})
deal_ii_initialize_cached_variables()

# Include directories
include_directories(${CMAKE_SOURCE_DIR})

# Source files
set(SOURCES
        main.cc
        core/phase_field.cc
        core/phase_field_setup.cc
        assembly/ch_assembler.cc
        assembly/poisson_assembler.cc
        assembly/ns_assembler.cc
        solvers/ch_solver.cc
        solvers/poisson_solver.cc
        solvers/ns_solver.cc
        setup/ch_setup.cc
        setup/poisson_setup.cc
        setup/ns_setup.cc
        diagnostics/ch_mms.cc
        utilities/parameters.cc
)

# Create executable
add_executable(ferrofluid ${SOURCES})

# Link deal.II
deal_ii_setup_target(ferrofluid)

# Enable warnings
target_compile_options(ferrofluid PRIVATE -Wall -Wextra)