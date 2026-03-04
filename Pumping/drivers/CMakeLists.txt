# ============================================================================
# drivers/CMakeLists.txt — Production simulation drivers
# ============================================================================

add_executable(fhd_driver fhd_driver.cc)
target_link_libraries(fhd_driver
    navier_stokes_lib angular_momentum_lib passive_scalar_lib
    poisson_lib magnetization_lib utilities_lib)
deal_ii_setup_target(fhd_driver)

# Phase B: CH + NS benchmark driver (droplet + square)
add_executable(ch_benchmark ch_benchmark_driver.cc)
target_link_libraries(ch_benchmark
    navier_stokes_lib cahn_hilliard_lib utilities_lib)
deal_ii_setup_target(ch_benchmark)

# Phase B: Full FHD + CH two-phase coupled driver (all 6 subsystems)
add_executable(fhd_ch_driver fhd_ch_driver.cc)
target_link_libraries(fhd_ch_driver
    navier_stokes_lib angular_momentum_lib magnetization_lib
    poisson_lib cahn_hilliard_lib utilities_lib)
deal_ii_setup_target(fhd_ch_driver)
