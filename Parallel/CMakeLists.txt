## ============================================================================
## CMakeLists.txt - Parallel Ferrofluid Solver + MMS Tests
##
## Targets:
##   ferrofluid          - Main parallel solver
##   parallel_test_mms   - MMS verification tests (standalone + coupled)
## ============================================================================

cmake_minimum_required(VERSION 3.13)
project(ferrofluid_parallel LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(deal.II 9.5 REQUIRED
        HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR}
)
deal_ii_initialize_cached_variables()

# =============================================================================
# Main Ferrofluid Solver (PARALLEL)
# =============================================================================
set(FERROFLUID_SOURCES
        main.cc
        core/phase_field.cc
        core/phase_field_setup.cc

        assembly/ch_assembler.cc
        assembly/poisson_assembler.cc
        assembly/magnetization_assembler.cc
        assembly/ns_assembler.cc

        setup/ch_setup.cc
        setup/poisson_setup.cc
        setup/magnetization_setup.cc
        setup/ns_setup.cc

        solvers/ch_solver.cc
        solvers/poisson_solver.cc
        solvers/magnetization_solver.cc
        solvers/ns_solver.cc
        solvers/ns_block_preconditioner.cc

        utilities/parameters.cc
)

add_executable(ferrofluid ${FERROFLUID_SOURCES})
target_include_directories(ferrofluid PRIVATE ${CMAKE_SOURCE_DIR})
deal_ii_setup_target(ferrofluid)
target_compile_options(ferrofluid PRIVATE
        -Wall -Wextra -Wpedantic
        $<$<CONFIG:Debug>:-g -O0>
        $<$<CONFIG:Release>:-O3 -march=native>
)

# =============================================================================
# MMS Test executable (PARALLEL) - Standalone + Coupled Tests
# =============================================================================
set(MMS_TEST_SOURCES
        # Test driver
        mms/mms_core/test_mms.cc
        mms/mms_core/mms_verification.cc

        # Standalone subsystem tests
        mms/ch/ch_mms_test.cc
        mms/poisson/poisson_mms_test.cc
        mms/ns/ns_mms_test.cc
        mms/magnetization/magnetization_mms_test.cc

        # Coupled MMS tests (NEW)
        mms/coupled/poisson_mag_mms_test.cc
        mms/coupled/ch_ns_mms_test.cc

        # Production assembly (used by MMS tests)
        assembly/ch_assembler.cc
        assembly/poisson_assembler.cc
        assembly/magnetization_assembler.cc
        assembly/ns_assembler.cc

        # Production setup (used by MMS tests)
        setup/ch_setup.cc
        setup/poisson_setup.cc
        setup/magnetization_setup.cc
        setup/ns_setup.cc

        # Production solvers (used by MMS tests)
        solvers/ch_solver.cc
        solvers/poisson_solver.cc
        solvers/magnetization_solver.cc
        solvers/ns_solver.cc
        solvers/ns_block_preconditioner.cc

        utilities/parameters.cc
)

add_executable(parallel_test_mms ${MMS_TEST_SOURCES})
target_include_directories(parallel_test_mms PRIVATE ${CMAKE_SOURCE_DIR})
deal_ii_setup_target(parallel_test_mms)
target_compile_options(parallel_test_mms PRIVATE
        -Wall -Wextra -Wpedantic
        $<$<CONFIG:Debug>:-g -O0>
        $<$<CONFIG:Release>:-O3 -march=native>
)

# =============================================================================
# NS Phase Test executable (PARALLEL)
# =============================================================================
set(NS_TEST_SOURCES
        mms/ns/test_ns_phases.cc
        setup/ns_setup.cc
        assembly/ns_assembler.cc
        solvers/ns_solver.cc
        solvers/ns_block_preconditioner.cc
)

add_executable(parallel_test_ns_phases ${NS_TEST_SOURCES})
target_include_directories(parallel_test_ns_phases PRIVATE
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/tests/ns
)
deal_ii_setup_target(parallel_test_ns_phases)
target_compile_options(parallel_test_ns_phases PRIVATE
        -Wall -Wextra -Wpedantic
        $<$<CONFIG:Debug>:-g -O0>
        $<$<CONFIG:Release>:-O3 -march=native>
)

# =============================================================================
# Usage instructions
# =============================================================================
message(STATUS "")
message(STATUS "=== Parallel Ferrofluid Build ===")
message(STATUS "")
message(STATUS "Targets:")
message(STATUS "  ferrofluid              - Main parallel solver")
message(STATUS "  parallel_test_mms       - MMS verification tests")
message(STATUS "  parallel_test_ns_phases - NS phase A-D tests")
message(STATUS "")
message(STATUS "Build:")
message(STATUS "  mkdir build && cd build")
message(STATUS "  cmake -DCMAKE_BUILD_TYPE=Release ..")
message(STATUS "  cmake --build . --target ferrofluid -j4")
message(STATUS "")
message(STATUS "Run solver:")
message(STATUS "  mpirun -np 4 ./ferrofluid")
message(STATUS "  mpirun -np 4 ./ferrofluid --magnetic --ns")
message(STATUS "  mpirun -np 4 ./ferrofluid --help")
message(STATUS "")
message(STATUS "Run MMS tests (standalone):")
message(STATUS "  mpirun -np 4 ./parallel_test_mms --level CH_STANDALONE --refs 3 4 5")
message(STATUS "  mpirun -np 4 ./parallel_test_mms --level POISSON_STANDALONE --refs 3 4 5")
message(STATUS "  mpirun -np 4 ./parallel_test_mms --level NS_STANDALONE --refs 3 4 5")
message(STATUS "  mpirun -np 4 ./parallel_test_mms --level MAGNETIZATION_STANDALONE --refs 3 4 5")
message(STATUS "")
message(STATUS "Run MMS tests (coupled):")
message(STATUS "  mpirun -np 4 ./parallel_test_mms --level POISSON_MAGNETIZATION --refs 3 4 5 --steps 50")
message(STATUS "  mpirun -np 4 ./parallel_test_mms --level CH_NS --refs 3 4 5 --steps 50")
message(STATUS "")