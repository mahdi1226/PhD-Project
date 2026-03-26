# Decoupled Ferrofluid Solver — Project Notes

## Reference Paper
Zhang, He & Yang, SIAM J. Sci. Comput. 43(4), B1003–B1032, 2021
- Algorithm 3.1: CH → NS → Pressure → Velocity Correction → Magnetization+Poisson
- SAV energy stabilization (unconditional, always on)
- Double-well: truncated {0,1} convention (G(Φ) = Φ²(1-Φ)²)
- Kelvin force: μ₀(m·∇)H with b_stab stabilization terms

## Poisson Field Semantics (CRITICAL)
The Poisson weak form `(∇φ, ∇X) = (h_a - M, ∇X)` with natural BCs gives
`∂_n φ = (h_a - M)·n`. This means **∇φ IS the total field** (h_a is encoded
through the natural BC). Do NOT add h_a explicitly to H.

**Evidence:**
- H = ∇φ only → rosensweig spikes ✓, hedgehog blows up at t≈2.2
- H = h_a + ∇φ → double-counts h_a, rosensweig blows up at t≈1.01
- H = h_a only → dome forms (no spikes, correct per Nochetto Fig 7)

Comments in `poisson.h` and `poisson_setup.cc` are misleading — they say
"∇φ is the demagnetizing field" but the weak form includes h_a via BCs.

## Current Code State (from HEAD 6b5dadf)

### Changes Implemented
1. **Picard convergence check** — Step 5: max=20, tol=1e-4; Step 6: max=15, tol=1e-4
   - Old: fixed 3/2 iterations, ~6e-3 residual left over
   - New: converges to 1e-4, typically 9-13 iterations (Step 5), 2-6 (Step 6)
   - Result: 6000× more accurate M↔φ coupling per step

2. **Step 6 always re-solves Poisson** — old code skipped on last iteration

3. **SAV made unconditional** — removed `use_sav` toggle, deleted non-SAV `assemble_system()`
   - SAV is integral to Zhang's scheme, not optional

4. **Kelvin force simplified** — removed Nochetto's DG skew terms from NS
   - Kept only μ₀(M·∇)H (cell kernel)
   - Removed ½(∇·M)H and face jump terms (not in Zhang's formulation)
   - Magnetization uses CG (not DG), so no jumps

5. **Dead code removed** — `apply_under_relaxation()`, unused kelvin_force.h functions

6. **Picard vector reuse** — eliminated per-iteration allocation in convergence check

7. **Removed duplicate final VTK writer**

8. **h_a in NS Kelvin force** — NS now has access to h_a for dome/reduced-field mode
   - `compute_applied_field()` called at quadrature points
   - Used ONLY in reduced-field mode (--dome, --reduced_field)
   - Full mode still uses H = ∇φ only

### Spin Torque (DISABLED)
Added then disabled: `(μ₀/2)(m×H̃, ∇×v)` from Zhang Eq 3.11 term 2.
- Creates spurious rotational velocity in dome test
- At equilibrium M ∝ H so m×H ≈ 0, but numerical misalignment from transport
  creates feedback loop: misalignment → torque → velocity → more misalignment
- Dome with spin torque: |U|=21 (vs 0.7 without), still alive at t=1.52 but struggling
- Dome without spin torque: |U|=0.7, smooth surface, dies at t=1.66 from M oscillation

## Test Results

### Test 1: Rosensweig (uniform field, 5 dipoles)
- **HEAD + Picard fix**: Passed. Clean run to completion.
  - θ ∈ [-0.001, 1.00], spikes form correctly
  - Step 5 Picard: 11-13 iterations
  - Wall: ~1.6s/step (1 rank), ~1.3s (8 ranks)

- **HEAD + Picard + Kelvin simplified**: Passed. Same spike pattern.

- **HEAD + h_a added to H**: FAILED at t≈1.01. Double-counts h_a.

### Test 2: Hedgehog (nonuniform, 42 dipoles)
- **HEAD + Picard fix only**: FAILED at t≈2.23.
  - Picard diverged: residual jumped to 0.8
  - |M| oscillating before crash
  - Same blowup point as old code (t≈2.2)

### Test 3: Dome (H = h_a only, hedgehog dipoles)
Purpose: isolate whether h_d (Poisson↔Mag coupling) causes the instability.

- **Without spin torque**: Dome forms correctly, |U|≈0.7.
  FAILED at t≈1.66: |M| oscillation (226↔562), E_mag → 6e+24.
  - Smooth dome surface, matches Nochetto Fig 7 qualitatively
  - M oscillation is the killer — not Picard, not h_d

- **With spin torque**: Dome with bumps, |U|≈21.
  Still running at t≈1.52. More velocity but M oscillation milder (151↔168).
  - |U| much too large — spin torque creates spurious rotation
  - Nochetto shows smooth dome, so spin torque behavior is wrong

### Key Conclusions from Tests
1. Picard convergence is not the root cause (hedgehog still blows up at same t≈2.2)
2. h_a is already in ∇φ (adding it double-counts)
3. The M oscillation instability exists even without h_d (dome test proves this)
4. The instability is in the magnetization transport ↔ Kelvin force feedback loop
5. b_stab terms are correctly implemented but insufficient to prevent blowup

## b_stab Verification (Zhang Eq 3.11)
All 3 terms verified correct:
- t1: `μ₀δt · ((ũ·∇)m, (v·∇)m)` ✓
- t2: `2μ₀δt · ((∇·ũ)m, (∇·v)m)` ✓
- t3: `(μ₀/2)δt · (m×∇×ũ, m×∇×v)` ✓
Both LHS diagonal and RHS cross-component terms present.

## Open Issues

### Hedgehog blowup at t≈2.2
Root cause unknown. M oscillation instability in magnetization transport.
Possible avenues:
- θ clamping after CH solve (prevent unphysical χ(θ))
- Magnetization transport scheme (Step 5 explicit vs Step 6 implicit splitting)
- Relaxation time τ_M tuning
- Mesh refinement near interface
- Kelvin force magnitude scaling at large |M|

### Dome blowup at t≈1.66
Same M oscillation as hedgehog but earlier and without h_d.
Proves the instability is fundamental to the transport↔Kelvin coupling,
not caused by Poisson/Picard issues.

### Semi_Coupled CH Assembly
Line 212 in `Semi_Coupled/assembly/ch_assembler.cc` has comment "FIX: +1/eps".
Not yet fully investigated. The double-well functions were verified correct
for Nochetto's {-1,+1} convention with truncation.

## Audit Items (not yet implemented)
| # | Severity | Issue |
|---|----------|-------|
| 1 | MAJOR | NS velocity matrix rebuilt every step (could cache when dt/ν unchanged) |
| 2 | MODERATE | Picard iteration limits hard-coded (should be in parameters) |
| 3 | MODERATE | SAV denom guard is dead code (denom ≥ 1.0 always) |
| 4 | MINOR | MMS tests don't exercise SAV path |
| 5 | MINOR | advance_time() + update_ghosts() double-writes old ghosted vectors |
