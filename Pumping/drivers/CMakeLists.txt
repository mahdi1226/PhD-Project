# ============================================================================
# drivers/CMakeLists.txt — Production simulation drivers
# ============================================================================

add_executable(fhd_driver fhd_driver.cc)
target_link_libraries(fhd_driver
    navier_stokes_lib angular_momentum_lib passive_scalar_lib
    poisson_lib magnetization_lib utilities_lib)
deal_ii_setup_target(fhd_driver)
