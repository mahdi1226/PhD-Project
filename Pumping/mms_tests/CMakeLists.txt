# ============================================================================
# mms_tests/CMakeLists.txt — Coupled MMS convergence tests
# ============================================================================

# Poisson ↔ Magnetization coupled test
add_executable(test_poisson_mag_mms
    poisson_mag_mms_test.cc
)
target_link_libraries(test_poisson_mag_mms
    poisson_lib
    magnetization_lib
    utilities_lib
)
deal_ii_setup_target(test_poisson_mag_mms)

# NS ↔ Angular Momentum coupled test (micropolar coupling)
add_executable(test_ns_angmom_mms
    ns_angmom_mms_test.cc
)
target_link_libraries(test_ns_angmom_mms
    navier_stokes_lib
    angular_momentum_lib
    utilities_lib
)
deal_ii_setup_target(test_ns_angmom_mms)

# Full 4-system coupled test (Nochetto Algorithm 42)
add_executable(test_full_coupled_mms
    full_coupled_mms_test.cc
)
target_link_libraries(test_full_coupled_mms
    navier_stokes_lib
    angular_momentum_lib
    poisson_lib
    magnetization_lib
    utilities_lib
)
deal_ii_setup_target(test_full_coupled_mms)

# CH ↔ NS coupled test (capillary force + convection, Phase B)
add_executable(test_ch_ns_mms
    ch_ns_mms_test.cc
)
target_link_libraries(test_ch_ns_mms
    navier_stokes_lib
    cahn_hilliard_lib
    utilities_lib
)
deal_ii_setup_target(test_ch_ns_mms)

# Full 6-subsystem coupled test (FHD + CH with chi(phi), nu(phi))
add_executable(test_full_ch_fhd_mms
    full_ch_fhd_mms_test.cc
)
target_link_libraries(test_full_ch_fhd_mms
    navier_stokes_lib
    angular_momentum_lib
    poisson_lib
    magnetization_lib
    cahn_hilliard_lib
    utilities_lib
)
deal_ii_setup_target(test_full_ch_fhd_mms)
