# Ferrofluid Solver - Technical Notes from Paper Review

**Date:** December 12, 2025
**Source:** Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531

---

## CRITICAL CORRECTIONS NEEDED

### 1. Poisson Equation - WRONG RHS AND BC

**Our current implementation:**
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

### 2. Capillary Force Formula

**Paper says:**
```
f_c = (λ/ε) ∇ψ
```
(After simplification - modifies pressure, see Remark 2.1)

**We have:**
```
F_cap = (λ/ε) θ ∇ψ
```

**QUESTION:** Is the θ factor correct or not? Need to verify Eq. 10 carefully.

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

## PRIORITY FIXES FOR NEXT SESSION

1. **HIGH - Poisson RHS:** Add ∇·(μ₀ κ_θ h_a) source term
2. **HIGH - Poisson BC:** Change from Dirichlet to Neumann ∂_n(φ) = (h_a - m)·n
3. **HIGH - Pressure uniqueness:** Add mean-zero constraint (pure Neumann φ problem)
4. **MEDIUM - Verify capillary force:** Is it (λ/ε)∇ψ or (λ/ε)θ∇ψ?
5. **MEDIUM - CSV diagnostics:** Track mass, energy, forces, CFL

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
