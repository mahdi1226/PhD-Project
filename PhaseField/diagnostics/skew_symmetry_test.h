#ifndef SKEW_SYMMETRY_TEST_H
#define SKEW_SYMMETRY_TEST_H

#include "physics/skew_forms.h"
#include <deal.II/base/tensor.h>
#include <random>
#include <iostream>
#include <cassert>

template <int dim>
void test_skew_symmetry()
{
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    
    // Random vectors
    dealii::Tensor<1, dim> U, V, W;
    dealii::Tensor<2, dim> grad_U, grad_V, grad_W;
    
    for (unsigned int d = 0; d < dim; ++d)
    {
        U[d] = dis(gen);
        V[d] = dis(gen);
        W[d] = dis(gen);
        for (unsigned int e = 0; e < dim; ++e)
        {
            grad_U[d][e] = dis(gen);
            grad_V[d][e] = dis(gen);
            grad_W[d][e] = dis(gen);
        }
    }
    
    double B_UVW = skew_magnetic_cell_value<dim>(U, grad_U, V, grad_V, W);
    double B_UWV = skew_magnetic_cell_value<dim>(U, grad_U, W, grad_W, V);
    
    double test = B_UVW + B_UWV;
    
    std::cout << "[TEST] Skew-symmetry: B(U,V,W) + B(U,W,V) = " << test 
              << (std::abs(test) < 1e-10 ? " ✓ PASS" : " ✗ FAIL") << "\n";
    
    assert(std::abs(test) < 1e-10);
}

#endif