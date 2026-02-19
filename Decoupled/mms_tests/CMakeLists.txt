cmake_minimum_required(VERSION 3.13)
find_package(deal.II 9.4 REQUIRED HINTS ${DEAL_II_DIR} ${deal.II_DIR} $ENV{DEAL_II_DIR})
deal_ii_initialize_cached_variables()

project(mms_tests)

# SRC_ROOT = parent directory (Decoupled/) so includes like "poisson/poisson.h" work
set(SRC_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
include_directories(${SRC_ROOT})

# Poisson subsystem
add_library(poisson_lib STATIC
        ${SRC_ROOT}/poisson/poisson.cc
        ${SRC_ROOT}/poisson/poisson_setup.cc
        ${SRC_ROOT}/poisson/poisson_assemble.cc
        ${SRC_ROOT}/poisson/poisson_solve.cc
        ${SRC_ROOT}/utilities/parameters.cc
)
deal_ii_setup_target(poisson_lib)

# Magnetization subsystem
add_library(magnetization_lib STATIC
        ${SRC_ROOT}/magnetization/magnetization.cc
        ${SRC_ROOT}/magnetization/magnetization_setup.cc
        ${SRC_ROOT}/magnetization/magnetization_assemble.cc
        ${SRC_ROOT}/magnetization/magnetization_solve.cc
)
deal_ii_setup_target(magnetization_lib)

# Coupled Poisson + Magnetization MMS test
add_executable(test_poisson_mag_mms
        poisson_mag_mms_test.cc
)
target_link_libraries(test_poisson_mag_mms poisson_lib magnetization_lib)
deal_ii_setup_target(test_poisson_mag_mms)