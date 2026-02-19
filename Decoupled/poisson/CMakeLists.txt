cmake_minimum_required(VERSION 3.13)
find_package(deal.II 9.4 REQUIRED HINTS ${DEAL_II_DIR} ${deal.II_DIR} $ENV{DEAL_II_DIR})
deal_ii_initialize_cached_variables()

project(poisson_subsystem)

# SRC_ROOT = parent directory (src/) so includes like "poisson/poisson.h" work
set(SRC_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
include_directories(${SRC_ROOT})

# Poisson static library
add_library(poisson_lib STATIC
        poisson.cc
        poisson_setup.cc
        poisson_assemble.cc
        poisson_solve.cc
        poisson_output.cc
        ${SRC_ROOT}/utilities/parameters.cc
)
deal_ii_setup_target(poisson_lib)

# Standalone MMS convergence test
add_executable(test_poisson_mms
        tests/poisson_mms_test.cc
)
target_link_libraries(test_poisson_mms poisson_lib)
deal_ii_setup_target(test_poisson_mms)

# Standalone Poisson driver with VTK output
add_executable(poisson_main
        poisson_main.cc
)
target_link_libraries(poisson_main poisson_lib)
deal_ii_setup_target(poisson_main)