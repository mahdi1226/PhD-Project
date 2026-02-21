# Ferrofluid Solver - Technical Notes from Paper Review

> **HISTORICAL** -- These technical notes from December 2025 review Nochetto's formulation.
> The "CRITICAL CORRECTIONS NEEDED" below have been addressed in Sessions 1-3 of the
> Decoupled solver. See `session_handoff.md` in this directory for details.

**Date:** December 12, 2025
**Source:** Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531

---

## CRITICAL CORRECTIONS NEEDED (ALL RESOLVED)

### 1. Poisson Equation - WRONG RHS AND BC [DONE]

> **STATUS: DONE** -- Rewritten in the Decoupled solver (`poisson/poisson_assemble.cc`).
> Uses correct weak form `((1+chi)*grad(phi), grad(X)) = ((1-chi)*h_a, grad(X))` with Neumann BC and DoF-0 pinning for uniqueness.

**Our current implementation (OLD monolithic solver):**
```
-∇·(μ(θ)∇φ) = 0           in Ω       ← WRONG (RHS should not be 0)
φ = φ_dipole               on ∂Ω     ← WRONG (should be Neumann)
```

**Paper (Eq. 14d):**
```
∇·(μ_θ ∇φ) = ∇·(μ₀ χ_θ h_a)    in Ω
∂_n(φ) = (h_a - m)·n            on ∂Ω
```

Where:
- h_a = applied field from dipoles = -∇φ_s (sum of dipole potentials)
- m = κ_θ h = χ₀ H(θ/ε) h (magnetization)
- μ_θ = 1 + χ_θ (permeability)

**ACTION:** Rewrite poisson_assembler.cc with correct RHS and Neumann BC.

### 2. Capillary Force Formula [DONE]

> **STATUS: DONE** -- Resolved in Session 3 (Bug #3). Correct form is `theta * grad(psi)`
> per Zhang Eq 2.6 (Phi * grad(W)). The SAV variable psi already contains lambda,
> so no extra lambda factor. Fixed in `navier_stokes/navier_stokes_assemble.cc`.

**Paper says:**
```
f_c = (λ/ε) ∇ψ
```
(After simplification - modifies pressure, see Remark 2.1)

**We had (OLD monolithic solver):**
```
F_cap = (λ/ε) θ ∇ψ
```

**Resolution:** The correct form from Zhang Eq 2.6 is `theta * grad(psi)` where psi = W = lambda*(-epsilon*Laplacian(theta) + (1/epsilon)*f(theta)). The SAV CH solve already includes lambda in psi, so no extra lambda/(epsilon) factor is needed.

---

## BOUNDARY CONDITIONS (CONFIRMED)

| Field | Boundary | Condition |
|-------|----------|-----------|
| θ | All ∂Ω | ∂_n(θ) = 0 (Neumann) |
| ψ | All ∂Ω | ∂_n(ψ) = 0 (Neumann) |
| u | All ∂Ω (walls) | u = 0 (no-slip) |
| φ | All ∂Ω | ∂_n(φ) = (h_a - m)·n (Neumann) |

**Note:** Top is also no-slip because it's a wall (ferrofluid deforms inside fixed domain).

---

## FORMULAS (CONFIRMED)

### Dipole Potential (Eq. 97)
```
φ_s(x) = d · (x_s - x) / |x_s - x|^p

where p = dim (p=2 for 2D, p=3 for 3D)
```

For 2D: φ_s = d·r / |r|² ✓ (we have this correct)

grad(φ_s) = 0 outside singularities (harmonic)

### Material Properties (Eq. 17)
```
ν_θ = ν_w + (ν_f - ν_w) H(θ/ε)    ✓ correct
κ_θ = κ₀ H(θ/ε)                    ✓ correct
μ_θ = 1 + κ_θ                      ✓ correct
```

Where H(x) = 1/(1 + e^{-x}) sigmoid function.

### Kelvin Force (Eq. 14f)
```
f_mag = μ₀ (m·∇)h

where m = κ_θ h (quasi-static)
      h = -∇φ (total field)
```

### Gravity/Buoyancy (Eq. 19 - Boussinesq)
```
f_g = (1 + r H(θ/ε)) g

where r = |ρ_f - ρ_w| / min(ρ_f, ρ_w) = 0.1 (density ratio)
      g = (0, -30000)^T
```

**Note:** Density is "unitary" (non-dimensional due to Reynolds scaling).

### Critical Wavelength (analytical)
```
l_c = 2π √(σ / (g Δρ))

With Δρ = 0.1, l_c ≈ 0.25
```
This predicts ~4 peaks in unit domain, hence g ~ 10⁴ order.

---

## ROSENSWEIG PARAMETERS (Section 6.2, p.520-522)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Domain | [0,1] × [0,0.6] | |
| ε | 0.01 | Interface thickness |
| λ | 0.05 | Capillary, λ ≈ σε |
| γ | 0.0002 | Mobility |
| ν_w | 1.0 | Water viscosity |
| ν_f | 2.0 | Ferrofluid viscosity |
| κ₀ | 0.5 | Susceptibility |
| μ₀ | 1.0 | Permeability |
| r | 0.1 | Density ratio |
| g | (0, -30000) | Gravity |
| Pool depth | 0.2 | 20% of height |
| Dipoles | 5 at y = -15 | x = -0.5, 0, 0.5, 1, 1.5 |
| d | (0, 1) | Dipole direction (up) |
| α(t) | 0 → 6000 | Linear ramp t ∈ [0, 1.6], constant after |
| dt | ~5e-4 | t_F/4000 |
| t_final | 2.0 | |

---

## HEDGEHOG PARAMETERS (Section 6.3)

| Parameter | Value | Notes |
|-----------|-------|-------|
| ε | 0.005 | Thinner interface |
| λ | 0.025 | Smaller capillary |
| κ₀ | 0.9 | Higher susceptibility |
| Dipoles | Multiple rows | y = -0.5, -0.75, 1, 14 dipoles per row |
| α(t) | 0 → 4.3 | Linear ramp t ∈ [0, 4.3], constant to t = 6 |

---

## EQUATIONS SUMMARY

### Cahn-Hilliard (Eq. 14a-14b)
```
θ_t + u·∇θ - γΔψ = 0
ψ = -εΔθ + (1/ε)(θ³ - θ)
```

### Magnetostatics (Eq. 14c-14d)
```
m = κ_θ h                           (quasi-static, T → 0)
∇·(μ_θ ∇φ) = ∇·(μ₀ κ_θ h_a)        (Poisson with RHS!)
∂_n(φ) = (h_a - m)·n                (Neumann BC)
```

### Navier-Stokes (Eq. 14e-14f)
```
u_t + (u·∇)u - ∇·(ν_θ ∇u) + ∇p = f_c + f_mag + f_g
∇·u = 0
```

---

## PRIORITY FIXES FOR NEXT SESSION (ALL RESOLVED)

1. [DONE] **HIGH - Poisson RHS:** Rewritten in Decoupled solver with correct weak form
2. [DONE] **HIGH - Poisson BC:** Neumann BC with DoF-0 pinning in Decoupled solver
3. [DONE] **HIGH - Pressure uniqueness:** DoF-0 pinning implemented
4. [DONE] **MEDIUM - Verify capillary force:** Resolved as theta*grad(psi) per Zhang Eq 2.6 (Session 3, Bug #3)
5. [DONE] **MEDIUM - CSV diagnostics:** Step-level diagnostics in decoupled_driver.cc (theta range, energy, |U|, |H|, SAV r)

---

## USER NOTES (VERBATIM)

"these are interesting we should improve the model first by implementing diagnostics and preconditioners/solvers then we need to focus on these details"

**Agreed approach:**
1. First: Diagnostics (CSV output)
2. Second: Better solvers/preconditioners
3. Third: Fix Poisson equation
4. Fourth: Verify all formulas against paper

---

*End of Technical Notes*
