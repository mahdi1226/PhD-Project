# Cahn-Hilliard standalone MMS convergence test
add_executable(test_cahn_hilliard_mms
    tests/cahn_hilliard_mms_test.cc
)
target_link_libraries(test_cahn_hilliard_mms
    cahn_hilliard_lib
    utilities_lib
)
deal_ii_setup_target(test_cahn_hilliard_mms)
