# Diagnostics: Issues Encountered and Resolutions

## Resolved Issues

### 1. NS Viscous Term Factor Error (Standalone NS)

**Symptom**: NS MMS test giving U_L2 rate ~2 instead of expected 3.

**Root Cause**: The strain-rate form nu(D(u), D(v)) where D(u) = 1/2(grad u + grad u^T) produces a strong-form Laplacian of -(nu/2) Delta u, NOT -nu Delta u. The MMS source was using -nu*lap instead of -(nu/2)*lap.

**Fix**: Changed MMS source to use `-(nu_eff/2.0) * lap[d]` in all NS source functions. The factor 1/2 arises because (D(u):D(v)) = 1/4 (grad u + grad u^T):(grad v + grad v^T), and when v = basis function, the assembly integrates nu * D(u):D(v), which equals nu/2 * grad u : grad v for the symmetric case.

**Key Lesson**: Weak form nu(D(u),D(v)) maps to strong form -(nu/2) Delta u.

### 2. NS Pressure Pin (Standalone NS)

**Symptom**: Solver failing or giving nonsensical pressure values.

**Root Cause**: The DG P1 pressure space requires a mean-zero constraint to make the saddle-point system non-singular. The pressure pin was not being properly applied in the monolithic system.

**Fix**: Set the first pressure DoF constraint explicitly, and ensure the (p,p)=0 diagonal block has a proper value (1.0) for the constrained DoF so `distribute_local_to_global` works correctly.

### 3. MMS Source 1/tau Amplification (Full Coupled MMS)

**Symptom**: Full coupled MMS test showing O(1) errors in M, phi, p, w despite correct standalone and pairwise tests. Errors did not decrease with mesh refinement.

**Root Cause**: MMS sources used ANALYTICAL old-time values (M*(t_old), u*(t_old), w*(t_old)) while the assembly uses DISCRETE values from the previous time step. The mismatch between analytical and discrete old values is O(h^2), but after division by tau in the time derivative, it becomes O(h^2/tau) = O(1) for fixed dt.

**Fix**: Updated all MMS callback signatures to pass discrete old values from assembly. The source now computes `(M*_new - M_old_disc)/tau` instead of `(M*_new - M*_old)/tau`, eliminating the 1/tau amplification.

**Files Modified**: 14 files across all subsystems (typedefs, assembly call sites, source functions, test lambdas).

### 4. DG Transport Face Velocity Not Evaluated (Full Coupled MMS)

**Symptom**: After fixing the 1/tau issue, M_L2 errors stalled at ~2e-3 (not converging with refinement).

**Root Cause**: In `magnetization_assemble.cc`, the velocity at face quadrature points (`U_dot_n`) was declared as 0.0 but never computed. The `fe_face_U` object was allocated but never used.

**Fix**: Added velocity evaluation at face quadrature points using `fe_face_U->reinit()` and `get_function_values()`.

### 5. DG Face Double-Counting (Full Coupled MMS)

**Symptom**: After fixing U_dot_n, M_L2 converged but at rate ~2.4 instead of expected 3.0, and the rate DECREASED with finer meshes.

**Root Cause**: Each internal DG face was processed by BOTH adjacent cells, causing the face flux contributions to be assembled TWICE. This breaks the skew-symmetry property.

**Fix**: Added face deduplication — each same-level face is processed only from the cell with the smaller CellId.

### 6. Poisson-Mag Transport Skew Form (Full Coupled MMS)

**Symptom**: MMS source for magnetization didn't account for the 1/2(div U) M term in the skew-symmetric transport.

**Root Cause**: Analytically div(u*) = 0, but the DISCRETE velocity has nonzero divergence, and the assembly applies the full skew form including the 1/2(div U) M term.

**Fix**: MMS source now includes `+ 0.5 * div_U_disc * M*_new` to match the assembly.

### 7. H = ∇φ Double-Counting Bug (Section 7.3)

**Symptom**: All three assemblers (magnetization, NS, angular momentum) were computing the total magnetic field as H = h_a + ∇φ, which double-counts h_a.

**Root Cause**: The Poisson equation (∇φ,∇χ) = (h_a − M,∇χ) already encodes h_a into φ. The paper states on p.8: "use that h = ∇φ." Therefore ∇φ IS the total field h. Adding h_a separately double-counts the applied field.

**Fix**: Changed all assemblers to use H = ∇φ only:
- `magnetization_assemble.cc`: Removed h_a addition to grad_phi
- `navier_stokes_assemble.cc`: Kelvin force uses grad_phi directly
- `angular_momentum_assemble.cc`: Torque uses grad_phi directly
- `applied_field.h`: Updated comments documenting this convention

**Key Lesson**: h = ∇φ is the TOTAL magnetic field. Never add h_a to ∇φ.

### 8. NS Convection MMS Source (Production Readiness)

**Symptom**: Need to enable NS nonlinear convection for production runs while keeping MMS rates.

**Root Cause**: The skew-symmetric convection form `(U_old·∇)u + ½(∇·U_old)u` requires a matching MMS source that uses discrete old velocity, not analytical.

**Fix**: Extended MmsSourceFunction to accept `div_U_old_disc` and `include_convection` flag. Source includes `(U_old_disc·∇)u*(t_new) + ½ div_U_old_disc · u*(t_new)` when convection is enabled. All coupled MMS rates preserved.

### 9. Production Driver Static Method Bug

**Symptom**: `parse_command_line` called as instance method caused compilation error.

**Root Cause**: `parse_command_line` is a static method in Parameters.

**Fix**: Use `Parameters::parse_command_line(argc, argv)` instead of instance call.

## Open Issues

### 10. Approach 2 Velocity Mesh-Dependence (Section 7.3) — UNDER INVESTIGATION

**Symptom**: Velocity U_max decreases monotonically with mesh refinement while the total
Kelvin force converges. The paper's Figure 18 shows U_max = 4.33.

**Kelvin force diagnostics** (step 20, t=0.2, with face integrals):

| Metric | Ref 5 (32²) | Ref 6 (64²) | Ref 7 (128²) | Trend |
|--------|-------------|-------------|---------------|-------|
| U_max | **4.09** | **1.03** | **0.51** | ↓ halving per refinement |
| divU_L2 | **21.4** | **3.87** | **0.59** | ↓ improving |
| kelvin_cell_L2 | 3.58e4 | 3.70e4 | 3.73e4 | ✅ converged |
| kelvin_face_L2 | 203 | 50 | 10 | → 0 (expected) |
| kelvin_Fy | −7515 | −7638 | −7676 | ✅ converged |
| p range | ±7k | ±8.5k | ±9k | growing with h |
| E_kin | 0.082 | 0.019 | 0.015 | stabilizing? |

**Key finding**: The Kelvin force (both cell and face contributions, and total resultant)
fully converges across all mesh resolutions. The divergence is catastrophic at ref 5
(divU_L2/U_max ≈ 5) and progressively improves with refinement. This rules out the
dipole singularity as the root cause.

**Diagnosis: Pressure robustness issue.** The Kelvin force has a large irrotational
(gradient) component near the boundary where dipoles are concentrated. Standard mixed
FE methods (including Q2/DG-P1) have a well-known limitation: when the body force is
dominantly irrotational, the velocity error is polluted by the pressure error because
the discrete pressure space cannot perfectly cancel the gradient component. At coarse
meshes, the pressure badly approximates this cancellation, leading to large spurious
"compressible" velocity. As the mesh refines, the pressure better cancels the
irrotational force, and velocity decreases.

**Paper match at ref 5**: Our ref 5 gives U_max = 4.31 (at t=2), which matches the
paper's 4.3364866 from Figure 18 almost exactly. The paper likely uses a similar coarse
mesh (~32×32) and Section 7.3 is a proof of concept — the text says "We make no attempt
to use realistic scalings" and "The main goal is to provide a proof of concept."

**Previous hypotheses ruled out**:
- Dipole singularity: RULED OUT (force converges)
- Kelvin face integrals: Added correctly, MMS passes, but doesn't fix mesh-dependence
- Simplified model (h:=h_a): Unlikely (other Nochetto paper shows full model is critical)

**Next steps**:
- Determine paper's actual mesh for Section 7.3
- Investigate pressure-robust formulations as diagnostic
- Try DG P2 or higher pressure degree to test if pressure resolution is the bottleneck

### 11. Passive Scalar Overshoot (Section 7.3, Approach 2)

**Symptom**: At ref 5 y=-0.1, passive scalar c goes to [-0.37, 2.7] (severe undershoot/overshoot). At ref 6, c stays in [0, 1.28] — much better but still overshooting.

**Root Cause**: CG Q2 with SUPG does not enforce a maximum principle. The strong convection at approach 2 velocity magnitudes creates Gibbs-type oscillations near the sharp concentration front.

**Mitigation**: The paper does not discuss limiter strategies. At ref 6 the undershoot is eliminated (c_min >= 0) and overshoot is moderate (28%). At ref 7 this should improve further. Approach 1 at ref 5 has c in [0, 1.001] — essentially perfect.

### 12. Poisson CG Solver Failure (Section 7.3, Approach 2) — RESOLVED

**Symptom**: At ref 5 y=-0.1, the Poisson CG solver reported "loss of precision" with
AztecOO. Fall back to direct solver.

**Fix**: Switched Poisson to deal.II native CG + AMG (SolverCG + TrilinosWrappers::PreconditionAMG).
This eliminates the AztecOO precision issue. Works reliably at all refinement levels.

### 13. Missing Spin-Magnetization Coupling (M × W) — RESOLVED

**Found during**: Paper audit (full comparison of Eq. 52d against assembler).

**Paper Eq. 52d** (p.11, magnetization equation LHS):
```
(δM^k/τ, Z) − b_h^m(U^k, Z, M^k) + σ a_h(M^k, Z) + (M^k × W^k, Z) + (1/T)(M^k, Z) = (κ₀/T)(H^k, Z)
```

**Fix**: Treated explicitly (M^{k-1} instead of M^k) to preserve the shared-matrix approach where Mx and My use the same system matrix. The term moves to the RHS as `-(M_old × W, Z)`:
- Mx RHS contribution: `-w * My_old`
- My RHS contribution: `+w * Mx_old`

**Implementation**:
1. Extended `magnetization.h` `assemble()` with optional `w_relevant` + `w_dof_handler`
2. CG-to-DG evaluation: w (CG Q2) evaluated at DG quadrature points via `cell->as_dof_handler_iterator()`
3. Extended MmsSourceFunction typedef to include `w_disc` parameter
4. Updated all 6 MMS source functions: standalone/pairwise (w ignored), full coupled (w used)
5. Full coupled MMS source: `f[0] += w_disc * My_old_disc`, `f[1] -= w_disc * Mx_old_disc`
6. Driver passes `w_old_rel` and `&am.get_dof_handler()` to `mag.assemble()`
7. All MMS convergence rates preserved (M_L2=2.92, all others unchanged)

### 14. Angular Momentum Convection Disabled — RESOLVED

**Found during**: Paper audit (Eq. 52c vs driver).

**Paper Eq. 52c** (p.11): `j(δW^k/τ, X) + j b_h(U^k, W^k, X) + c₁(∇W^k, ∇X) + ... = ...`

**Fix**: Changed `include_convection` from `false` to `true` for angular momentum in both driver and full coupled MMS test. Extended AngMom MMS source functions (4 files) to accept `U_old_disc`, `div_U_old_disc`, and `include_convection` flag. Convection source uses:
```
j * [(U_old_disc · ∇)w*(t_new) + 0.5 * div_U_old_disc * w*(t_new)]
```
All MMS convergence rates preserved (w_L2=3.09, w_H1=2.06)

## Potential Future Issues

### Picard Convergence
Current settings: omega=0.35, max_iter=50, tol=1e-4 (production). Spinning magnet: 2-5 iters. Pumping: 10 iters. Stirring approach 2: 5 iters. No convergence issues observed for Section 7 experiments.

### AMR Face Handling
The current face dedup uses same-level CellId comparison. For AMR (hanging nodes), the face processing needs additional logic for subface iteration. The `cell->neighbor_is_coarser(f)` skip is in place but the coarser-cell processing path (with subfaces) is not implemented.

### Temporal Order
Currently first-order backward Euler. The paper uses first-order for Algorithm 42. Higher-order BDF2 would require storing two old solutions and modified MMS sources.

### Ref 7 Performance
With **block-Schur preconditioner** (`--block-schur`): ~10s/step at ref 7 vs ~258s/step
with direct solver = **26× speedup**. Approach 2 ref 7: 20 steps in 250s (12.5s/step).
Enhanced stirring ref 7: 400 steps in 12829s (3.6h, ~32s/step with 4 Picard iters).
