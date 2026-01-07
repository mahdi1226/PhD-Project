// ============================================================================
// assembly/poisson_assembler.h - Magnetostatic Poisson Assembly
//
// PAPER EQUATION 42d (corrected interpretation):
//   (∇φ, ∇χ) = (-M^k, ∇χ)  ∀χ ∈ X_h
//
// This solves for the demagnetizing potential φ where h_d = ∇φ.
// The applied field h_a is handled in the magnetization equation.
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef POISSON_ASSEMBLER_H
#define POISSON_ASSEMBLER_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include "utilities/parameters.h"

// ============================================================================
// Assemble magnetostatic Poisson system (Paper Eq. 42d)
//
//   (∇φ, ∇χ) = (-M^k, ∇χ)  ∀χ ∈ X_h
//
// Solves: -Δφ = ∇·M  (demagnetizing field equation)
// ============================================================================
template <int dim>
void assemble_poisson_system(
    const dealii::DoFHandler<dim>& phi_dof_handler,
    const dealii::DoFHandler<dim>& M_dof_handler,
    const dealii::Vector<double>& mx_solution,
    const dealii::Vector<double>& my_solution,
    const Parameters& params,
    double current_time,
    dealii::SparseMatrix<double>& phi_matrix,
    dealii::Vector<double>& phi_rhs,
    const dealii::AffineConstraints<double>& phi_constraints);

#endif // POISSON_ASSEMBLER_H