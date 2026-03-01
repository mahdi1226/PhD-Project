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

### 10. Approach 2 Incompressibility Violation (Section 7.3) — CRITICAL

**Symptom**: Approach 2 at y=-0.1 (ref 6) gives divU_L2 = 4.0, compared to 0.006 for approach 1. Velocity U_max = 1.75, far from paper's 4.33.

**Diagnostics comparison**:

| Metric | Approach 1 (ref5, y=-0.4) | Approach 2 (ref5, y=-0.4) | Approach 2 (ref6, y=-0.1) |
|--------|---------------------------|---------------------------|---------------------------|
| divU_L2 | 0.006 | 0.017 | **4.02** |
| p range | [-50, +100] | [-449, +152] | **[-8698, +8354]** |
| U_max | 0.016 | 0.013 | 1.75 |
| H_max | 15.3 | 31 | 203 |
| Picard | 3 | 7 | 5 |
| CFL | 0.004 | 0.003 | 0.75 |

**Root Cause**: At y=-0.1 the 8 dipoles create a near-singular 1/r^5 Kelvin force. The Q2/P1 discretization on a 64x64 mesh cannot resolve the sharp force gradient. The pressure (which must balance the body force to enforce div(u)=0) shows enormous values [-8698, +8354], indicating the solver struggles to maintain incompressibility.

**Non-monotonic mesh convergence**: ref 5 (32x32) overshoots at U=5.9, ref 6 (64x64) undershoots at U=1.7. The paper uses ~100x100 and gets U=4.33. This is characteristic of under-resolved near-singular forcing.

**Expected Resolution**: Run at ref 7 (128x128) to match paper's mesh resolution ("100 elements in each space direction"). The paper does not use any special treatment — it simply uses a fine enough mesh.

### 11. Passive Scalar Overshoot (Section 7.3, Approach 2)

**Symptom**: At ref 5 y=-0.1, passive scalar c goes to [-0.37, 2.7] (severe undershoot/overshoot). At ref 6, c stays in [0, 1.28] — much better but still overshooting.

**Root Cause**: CG Q2 with SUPG does not enforce a maximum principle. The strong convection at approach 2 velocity magnitudes creates Gibbs-type oscillations near the sharp concentration front.

**Mitigation**: The paper does not discuss limiter strategies. At ref 6 the undershoot is eliminated (c_min >= 0) and overshoot is moderate (28%). At ref 7 this should improve further. Approach 1 at ref 5 has c in [0, 1.001] — essentially perfect.

### 12. Poisson CG Solver Failure (Section 7.3, Approach 2)

**Symptom**: At ref 5 y=-0.1, the Poisson CG solver reports "loss of precision" and falls back to direct solver.

**Root Cause**: The strong near-singular applied field creates a poorly conditioned Poisson system. The iterative solver cannot converge.

**Impact**: Negligible — the direct solver produces correct results, just slower. At ref 6 the conditioning improves and CG works. At ref 7 this should not be an issue.

### 13. Missing Spin-Magnetization Coupling (M × W) — CRITICAL, UNRESOLVED

**Found during**: Paper audit (full comparison of Eq. 52d against assembler).

**Paper Eq. 52d** (p.11, magnetization equation LHS):
```
(δM^k/τ, Z) − b_h^m(U^k, Z, M^k) + σ a_h(M^k, Z) + (M^k × W^k, Z) + (1/T)(M^k, Z) = (κ₀/T)(H^k, Z)
```

**Missing term**: `(M^k × W^k, Z)` — the spin-magnetization coupling.

This comes from the continuous equation (1d): `m_t + (u·∇)m − σΔm = w × m − (1/T)(m − κ₀h)`.

In 2D, `M × W = w·(My, −Mx)` where w is the scalar angular velocity (pseudoscalar in 3D).

**Our assembler**: Only has mass + transport + diffusion + relaxation. The `w` field is never passed to or used in `magnetization_assemble.cc`.

**Why MMS didn't catch it**: The MMS source function also omits this term — so both assembler and source are consistently missing it, and convergence rates pass by construction.

**Physical meaning**: Angular velocity rotates the local magnetization. Without this term, the spin-magnetization feedback loop is broken — AngMom receives torque from M×H but M doesn't get rotated by w.

**Impact estimate**: For Section 7 experiments with T_relax = 10^{-4}, the relaxation term `(1/T)m ≈ 10^4 m` dominates the LHS. The `M × W` term is `O(w·M)`. For approach 1 (w ≈ 0.01), this is `O(0.01·M)` vs `O(10^4·M)` from relaxation — negligible. For approach 2 with larger w, still likely small relative to relaxation. But this must be verified.

**Resolution needed**:
1. Add `w_relevant` as input to `magnetization_assemble.cc`
2. In cell loop: evaluate `w` at DG quadrature points via CG FEValues
3. Add LHS contribution: `+w_q * My_j * Zx_i − w_q * Mx_j * Zy_i` (implicit in M^k)
   — OR RHS contribution: `+w_q * My_old * Zx_i − w_q * Mx_old * Zy_i` (explicit in M^{k-1})
4. Paper uses M^k (implicit): the term is on the LHS of Eq. 52d, meaning it multiplies the unknown M^k. Since Mx and My share one matrix, this couples them — the term `w·(My, −Mx)` mixes Mx and My. This means the single shared matrix approach may need modification (separate Mx/My systems or a coupled 2-component system).
5. Update MMS source to include the w × m contribution
6. Re-verify coupled MMS convergence rates

### 14. Angular Momentum Convection Disabled — MODERATE, UNRESOLVED

**Found during**: Paper audit (Eq. 52c vs driver).

**Paper Eq. 52c** (p.11): `j(δW^k/τ, X) + j b_h(U^k, W^k, X) + c₁(∇W^k, ∇X) + ... = ...`

Both schemes (52) and (78) include the angular momentum convection `j b_h(U^k, W^k, X)`.

**Our code**: `include_convection = false` in fhd_driver.cc (line 600).

**Impact**: For approach 1 (U≈0.01), negligible. For approach 2 (U≈1-5), potentially significant.

**Resolution needed**:
1. Change `include_convection` to `true` for angular momentum in the driver
2. The infrastructure already exists (the assembler supports it)
3. Update MMS source if needed (it already supports convection flag)
4. Re-verify coupled MMS convergence rates

## Potential Future Issues

### Picard Convergence
Current settings: omega=0.35, max_iter=50, tol=1e-4 (production). Spinning magnet: 2-5 iters. Pumping: 10 iters. Stirring approach 2: 5 iters. No convergence issues observed for Section 7 experiments.

### AMR Face Handling
The current face dedup uses same-level CellId comparison. For AMR (hanging nodes), the face processing needs additional logic for subface iteration. The `cell->neighbor_is_coarser(f)` skip is in place but the coarser-cell processing path (with subfaces) is not implemented.

### Temporal Order
Currently first-order backward Euler. The paper uses first-order for Algorithm 42. Higher-order BDF2 would require storing two old solutions and modified MMS sources.

### Ref 7 Performance
Ref 7 (128x128) will have ~4x more cells than ref 6 (64x64), leading to ~4x longer wall time. Approach 2 at ref 6 takes ~35s/step, so ref 7 would be ~140s/step. A 200-step run would take ~8 hours. The direct solver for the monolithic NS system is the bottleneck.
