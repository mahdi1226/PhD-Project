cmake_minimum_required(VERSION 3.13)
find_package(deal.II 9.4 REQUIRED HINTS ${DEAL_II_DIR} ${deal.II_DIR} $ENV{DEAL_II_DIR})
deal_ii_initialize_cached_variables()

project(magnetization_subsystem)

# SRC_ROOT = parent directory (src/) so includes like "magnetization/magnetization.h" work
set(SRC_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
include_directories(${SRC_ROOT})

# Magnetization static library
add_library(magnetization_lib STATIC
        magnetization.cc
        magnetization_setup.cc
        magnetization_assemble.cc
        magnetization_solve.cc
        magnetization_output.cc
        ${SRC_ROOT}/utilities/parameters.cc
)
deal_ii_setup_target(magnetization_lib)

# Standalone MMS convergence test
add_executable(test_magnetization_mms
        tests/magnetization_mms_test.cc
)
target_link_libraries(test_magnetization_mms magnetization_lib)
deal_ii_setup_target(test_magnetization_mms)

# Standalone production driver (relaxation + VTK output)
add_executable(magnetization_main
        magnetization_main.cc
)
target_link_libraries(magnetization_main magnetization_lib)
deal_ii_setup_target(magnetization_main)