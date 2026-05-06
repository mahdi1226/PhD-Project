# Scheme II — Hand-off note for the T1/T2/T3 implementation

**Audience.** A fresh Claude conversation picking up the implementation of
"Scheme II" (the full Shliomis model with all three nonlinear terms), starting
from the current `Semi_Coupled/` codebase which implements only the Nochetto
2016 base scheme.

**Prerequisite reading (in order).**
1. This document — full briefing.
2. `Proofs/scheme_II_standalone.pdf` — the math and theorems being implemented.
3. `Proofs/STATUS.md` — current code state at session-close (May 5 evening).

The companion `MEMORY.md` rule still applies: this work is in `Semi_Coupled/`
and references **only** Nochetto CMAME 2016 + the Scheme II extension paper.
No cross-pollination from `Decoupled/`.

---

## 1. What Scheme II is

The Nochetto, Salgado & Tomas (CMAME 309, 2016) base scheme — which is what
`Semi_Coupled/` currently implements — drops three nonlinear terms from the
full Shliomis ferrofluid model:

| Term | Symbol | Physical meaning |
|---|---|---|
| **T1** | `−β M × (M × H)` | Landau–Lifshitz damping |
| **T2** | `½ (∇×U) × M` | Spin–vorticity coupling |
| **T3** | `(μ₀/2) ∇ × (M × H)` | Antisymmetric magnetic stress (in NS) |

**Scheme II** is a single-step, mixed CG–DG, fully discrete FEM scheme that
restores all three terms, building on the energy-stable framework of Nochetto.
It is proved, in `scheme_II_standalone.pdf`:

- **Theorem 4.5** — *unconditional* discrete energy stability. The β-term
  contributes the genuine dissipation `+2μ₀βτ ‖M^{k-1} × H^k‖²` on the LHS.
- **Theorem 5.4** — existence + conditional uniqueness, via a Brouwer
  fixed-point, under two parabolic-type CFL conditions
  `τ ≤ τ_* ~ h^d / R²` (existence) and `τ ≤ τ_0 ~ h^{2/3} / R^{2/3}` (uniqueness).
- **Theorem 6.6** — weak convergence to a weak solution of the continuous
  Shliomis system as `h, τ → 0`, under a uniform `L⁴_{t,x}` bound on the
  discrete magnetization (Assumption 6.1).

These are the publication targets. The implementation must respect the
algebraic structure that makes the proofs go through (cf. §3 below).

---

## 2. Where we are now (May 5 evening)

**Implemented and validated** (Nochetto base scheme, Scheme II with β = 0,
no T2 in M-equation, no T3 in NS):

- CH (Q2 / Q2), DG-Q1 magnetization, monolithic [DG-Q1 M | CG-Q1 φ] block,
  Q2/DQ1 saddle-point NS.
- Iterative solvers: CH GMRES+AMG, magnetic block-precond (ILU on M, AMG on φ),
  NS LSC Schur (`B Q⁻¹ B^T`), all with direct fallback.
- AMR every 5 steps, cross-AMR mag preconditioner caching with adaptive
  staleness rebuild trigger.
- MMS framework: spatial rates at optimum (θ_L2=3, θ_H1=2, M_L2=2, M_H1=1,
  φ_H1=2), temporal rates measurable with `--mms-analytical` (NS U=1, p=1,
  M≈1 with `--tau-M` override, φ≈1).
- Hedgehog L5 iter producing physically correct Rosensweig pattern (5 spikes,
  λ≈0.20, 88–86% match to corrected theory).

**NOT implemented** — exactly the three things this hand-off is about:

- **T1** Landau–Lifshitz damping `−β M^{k-1} × (M^{k-1} × H^k)` in eq. (15a).
- **T2** Spin–vorticity coupling `½(∇×U^k) × M^k` in eq. (15a).
- **T3** Antisymmetric magnetic stress `(μ₀/2)(M^k × H^k, ∇ × V)` in eq. (15b).

Also missing:
- Time-level discipline change in NS-mag coupling: the magnetization transport
  must use `U^k` (current Picard iterate), not `U^{k-1}` — see Remark 3.2 of
  the Scheme II paper. This is a one-line scheme change, but it requires the
  Picard iteration loop in `core/phase_field.cc::run()` to be reorganized so
  the magnetic solve sees the current NS velocity.
- Picard iteration on the residual of the coupled `(M, U)` block (Theorem 5.4
  step). The current code does block-Gauss-Seidel `CH → Mag → NS` once per
  step and doesn't iterate. For Scheme II to actually realize the proven
  unconditional stability, the inner Picard loop must converge each step.

---

## 3. The three terms — exact discrete forms

**Notation** (matches paper §3 and §4):
- `Θ^k, Ψ^k, Φ^k, M^k, U^k, P^k` are the discrete unknowns at time level k.
- `H^k := ∇Φ^k`.
- `B_h^m(U, V, W)` is the DG skew-symmetric trilinear form (paper eq. 13);
  already implemented in `assembly/magnetic_assembler.cc`.
- `χ_Θ := χ(Θ^{k-1})`, `ν_Θ := ν(Θ^{k-1})` are phase-dependent coefficients,
  evaluated at the lagged phase field.
- In 2D, all cross products use the `(a₁, a₂) ↦ (a₁, a₂, 0)` embedding of
  paper eq. (17). Specifically, for 2D vectors `a, b`: `a × b = (0, 0, a₁b₂ − a₂b₁)`,
  and `∇ × U = ∂₁U₂ − ∂₂U₁` is a scalar.

### 3.1 T1 — Landau–Lifshitz damping

**Continuous:** added to the M-equation: `+ β M × (M × H)` on the LHS, i.e.
`∂_t M + (U·∇)M + (1/T)(M − χ̃H) − β M × (M × H) + ½(∇×U)×M = 0`.

**Discrete (paper eq. 15a):**

```
−β (M^{k-1} × (M^{k-1} × H^k), Z)
```

i.e. **fully explicit** — uses `M^{k-1}` for both factors, only `H^k` is
implicit. Tested with `Z = 2τμ₀ M^k`, the term combines with the Young
inequality (`a·(b × c) ≤ |a||b||c|`) and is bounded by

```
2μ₀τβ ‖M^{k-1}‖² · ‖M^{k-1} × H^k‖
  ≤ μ₀τβ ‖M^{k-1} × (M^{k-1} × H^k)‖² + μ₀τβ ‖M^k‖²    (paper eq. 32, Young)
```

The first term is non-positive (from BAC–CAB), the second is absorbed into
the Shliomis dissipation `(μ₀τ/T)‖M^k‖²` under the hypothesis `βT ≤ 1`
(consumes half of it).

The genuine dissipation `+2μ₀τβ ‖M^{k-1} × H^k‖²` comes from a *different*
test — testing with `Z = 2τμ₀ H^k` (paper eq. 34), which is valid because
`H^k = ∇Φ^k ∈ ∇𝒳_h ⊂ ℳ_h` (the inclusion `∇𝒳_h ⊂ ℳ_h`, paper eq. 12, is
exactly the structural hypothesis that makes T1's dissipation visible).

**Implementation:**
- File: `assembly/magnetic_assembler.cc`, M-equation block, RHS.
- Form: at each quadrature point, compute `M_old × (M_old × H_q)` where
  `M_old` is the previous-step M evaluated at the quadrature point,
  `H_q = ∇Φ_q` from the **current** Picard iterate (or from the lagged
  Poisson solve, if we don't iterate).
- Add `−β · (M_old × (M_old × H_q)) · Z_q · |J| dx` to the RHS for each
  M-test-function `Z`.
- New parameter: `params.physics.beta_LL` (double, default 0.0). When 0,
  the assembler must produce results bitwise identical to the current code
  — this is the regression baseline.

### 3.2 T2 — Spin–vorticity coupling

**Discrete (paper eq. 15a):**

```
+ ½ ((∇×U^k) × M^k, Z)
```

**Crucial:** uses `U^k` (current NS iterate), not `U^{k-1}`. This is what
makes T2 cancel against T3 in the energy proof (paper Step 3a' / Step 2',
Cross-cancellation, eqs. 35–38). The cancellation is purely algebraic via
the scalar triple product `(a × b) · c = (b × c) · a`.

In 2D with the embedding (eq. 17): `(∇×U) × M = (∂₁U₂ − ∂₂U₁)·(−M₂, M₁, 0)`,
so the 2D component contribution is `½ ((∂₁U₂ − ∂₂U₁)·(−M₂ Z₁ + M₁ Z₂))`.

**Implementation:**
- File: `assembly/magnetic_assembler.cc`, M-equation block, **LHS** (because
  it's a linear function of the unknown `M^k`, when `U^k` is treated as
  Picard-lagged).
- Inside the inner Picard loop, `∇×U^k` is computed once per loop iteration
  from the current `U^k` solution.
- Add the bilinear form contribution to the M-block: at each quadrature
  point, `½ · (curl_U) · (−M̂_j₂ φ_i₁ + M̂_j₁ φ_i₂)` where `(φ_i₁, φ_i₂)` are
  the M-test-function components and `M̂_j` is the M-trial-function.

### 3.3 T3 — Antisymmetric magnetic stress

**Discrete (paper eq. 15b, Remark 3.1):** **Use the weak form, not the strong
form:**

```
+ μ₀/2 · (M^k × H^k, ∇ × V)
```

**NOT:** `(μ₀/2)(∇ × (M^k × H^k), V)` — this would require integration by
parts on `M^k × H^k`, which sits in `[L²]^d` (DG, no H(curl) regularity), so
distributional IBP would generate uncontrollable face-jump terms.

The weak form **is** the discretization. It is a standard `L²` inner product,
well-defined for DG `M^k`, requires no regularity beyond `L²`. This avoids
the H(curl) regularity issue that the strong form would introduce.

**Implementation:**
- File: `assembly/ns_assembler.cc`, NS RHS (since `M^k`, `H^k` are
  Picard-lagged from the prior magnetic solve in the loop).
- In 2D: `M × H = (0, 0, M₁H₂ − M₂H₁)`, `∇ × V = ∂₁V₂ − ∂₂V₁`, so the term is
  `(μ₀/2) · (M₁H₂ − M₂H₁) · (∂₁V₂ − ∂₂V₁)`.
- Add `(μ₀/2) · (M_q₁ H_q₂ − M_q₂ H_q₁) · (∂₁ V_test₂ − ∂₂ V_test₁) · |J| dx`
  to NS RHS for each velocity test function.
- New parameter: not needed; `μ₀` is already in `params.physics`.

### 3.4 Velocity time-level fix (Remark 3.2)

The magnetic transport `B_h^m(U^k, Z, M^k)` and T2 both use `U^k`. The
current Nochetto-base implementation uses `U^{k-1}` because the NS hasn't
been solved yet at the point Mag is solved (block-Gauss-Seidel order).

**To fix:** wrap CH–Mag–NS in a Picard loop, on each iterate use the latest
available `U`. Convergence under the CFL `τ ≤ τ_*` is proved in Theorem 5.4.

```
# Before (current — Nochetto base):
solve_ch()         # uses U^{k-1}
solve_magnetics()  # uses U^{k-1}  ← will become U^k after fix
solve_ns()         # uses M^k, U^{k-1} for convection

# After (Scheme II):
solve_ch()         # uses U^{k-1} (unchanged; CH is decoupled from U upstream)
for picard_iter in range(MAX):
    solve_magnetics()  # uses U^{k-1}_picard (latest from prev iter, init = U^{k-1})
    solve_ns()         # uses M^k_picard
    if residual < tol: break
```

The first iterate (`picard_iter = 0`) reduces to the current code with
`U^{k-1}` in transport. So a Picard convergence in one iter is the regression
test for β = 0.

---

## 4. Recommended implementation order

Each step has a clean validation checkpoint. Do not skip them.

### Step A — `β` parameter wiring + regression baseline (~2 h)

1. Add `physics.beta_LL = 0.0` to `utilities/parameters.{h,cc}`. Add
   `--beta-LL VALUE` CLI flag.
2. **Do not** add T1 to the assembler yet. Just plumb the parameter through.
3. Run hedgehog L5 with `--beta-LL 0` and confirm the diagnostics.csv is
   bitwise identical to the current `Reports/hedgehog_L5_iter_final/`.
4. **Checkpoint**: regression baseline locked. Now we know β = 0 reproduces
   the Nochetto base exactly.

### Step B — T1 in the M-equation (~half day)

1. Add the T1 contribution to `assembly/magnetic_assembler.cc`, RHS, guarded
   by `if (params.physics.beta_LL > 0)` so β = 0 still bypasses the entire
   code path.
2. **Unit test**: extend `mms/magnetic/magnetization_mms.h` to add a T1
   source term. The MMS exact `M_x = sin(πx)cos(πy)·t²`, `M_y = ...` already
   in place; just append `+β M_exact × (M_exact × H_exact)` to the manufactured
   source.
3. Run the magnetic MMS standalone with `--mms-analytical --beta-LL 0.1`. M
   spatial rates should still hit DG-Q1 = 2 for L2.
4. Run hedgehog L5 with `--beta-LL 0.1` for 1000 steps. Check:
   - Energy decay slope steeper than β=0 baseline (T1 is dissipative).
   - Spike pattern still forms.
   - No θ-bound violations beyond baseline.
5. **Checkpoint**: T1 working. Energy law verified by inspection of CSV.

### Step C — T2 in the M-equation + U^k discipline (~1 day)

This is the hardest step because it requires the Picard loop.

1. Refactor `core/phase_field.cc::run()` to add a Picard inner loop around
   `solve_magnetics + solve_ns`. Use a residual check on `U^k − U^k_prev`.
   Default `MAX_PICARD = 5`, residual tolerance `1e-6` (relative).
2. Add T2 (LHS) to `magnetic_assembler.cc`. Use `U^k_picard`.
3. Extend MMS magnetic source with T2 contribution (exact `∇×U_exact` is a
   closed-form scalar from the inline NS exact velocity).
4. Run MMS coupled `MAGNETIC_NS` test (currently a stub) — implement it now.
   Spatial rates must still be at optimum.
5. Run hedgehog L5 with both T1 and T2 — pattern should still form, with
   T2 contributing rotational coupling that can shift spike timing slightly
   (paper notes T2 doesn't add dissipation, just rotation).
6. **Checkpoint**: M-equation has all Shliomis terms. Picard loop converging
   at most 2–3 iters/step.

### Step D — T3 in the NS RHS (~half day)

1. Add T3 (weak form, RHS) to `assembly/ns_assembler.cc`, inside the inner
   Picard iter so it sees the current `M^k`, `H^k`.
2. Verify the T2/T3 cross-cancellation holds at the discrete level by
   computing both sides of paper eq. (38) from the assembled vectors.
   They must agree to round-off.
3. Extend NS MMS source with T3 contribution.
4. Run MMS coupled `FULL_SYSTEM` test (currently a stub) — implement it now.
5. Run hedgehog L5 with T1+T2+T3 + Picard. Compare energy decay against the
   β=0 baseline; the difference is a direct measurement of the T1 dissipation.
6. **Checkpoint**: full Scheme II realised in code. Now the publication runs.

### Step E — Validation cascade (~2 days)

1. MMS spatial: re-run all cases at refs 3,4,5. Rates must still be optimal.
2. MMS temporal: with `--mms-analytical`, all rates must still ≈ 1 (BE).
3. Energy-law sanity check (`scripts/check_energy_law.py` to be written):
   load `diagnostics.csv`, evaluate eq. (26) numerically, confirm LHS ≤ RHS
   at every step.
4. Hedgehog L5 → L6 (HPC) with the full scheme.
5. Compare full-Scheme-II hedgehog pattern vs Nochetto-base hedgehog pattern
   on identical IC. Document any qualitative differences (likely: T1 narrows
   spikes slightly, T2 introduces minor rotation in the pattern).

---

## 5. File-by-file touch list

| File | Change |
|---|---|
| `utilities/parameters.{h,cc}` | Add `physics.beta_LL`, `--beta-LL`, `--max-picard`, `--picard-tol` |
| `core/phase_field.{h,cc}` | Picard loop in `run()`; cache `U_picard_prev_` |
| `assembly/magnetic_assembler.cc` | T1 (RHS, guarded by `beta_LL > 0`), T2 (LHS, always when scheme_II) |
| `assembly/ns_assembler.cc` | T3 (RHS, weak form `(M×H, ∇×V)`) |
| `mms/magnetic/magnetization_mms.h` | Add T1 + T2 to manufactured source |
| `mms/ns/ns_mms.h` | Add T3 to NS manufactured source |
| `mms/coupled/full_system_mms_test.cc` | Implement (currently stub) — exercises all three |
| `mms/coupled/magnetic_ns_mms_test.cc` | Implement (currently stub) — exercises T2 + T3 |
| `output/metrics_logger.{h,cc}` | Add `picard_iters` column to `diagnostics.csv` |
| `analysis/check_energy_law.py` | NEW — verifies eq. (26) holds on diagnostics.csv |

Each `.cc` change includes corresponding header instantiation updates.

---

## 6. Branch strategy

```bash
cd /Users/mahdi/Projects/git/PhD-Project/Semi_Coupled
git checkout main
git pull
git checkout -b scheme-II
# work here; main stays at the Nochetto-base + Scheme-I joint paper state.
```

Rationale (decided in this session, May 5):
- A folder fork (e.g. `Semi_Coupled_II/`) was rejected because it would
  duplicate ~30k LOC and force every fix to be made twice.
- A branch keeps the diff visible (`git diff main`) and lets us back-port
  bug fixes either direction.
- The joint paper (`ferrofluid_proofs_v20_2.pdf`) already covers both
  schemes, so the paper deliverable is unaffected by branch choice.

When Scheme II is publication-ready, merge `scheme-II → main` (or keep them
separate — TBD based on whether the user wants the base scheme to remain
the "default" build).

---

## 7. Known gotchas (don't waste time on these)

1. **Don't IBP the T3 term.** The strong form `∇×(M×H)` is NOT in `H(curl)`
   for DG M. Stay in the weak form `(M×H, ∇×V)`. See Remark 3.1.
2. **Don't use `U^{k-1}` in T2.** The cancellation with T3 only works for
   matched velocity argument. See Remark 3.2.
3. **`χ₀ < 4` is mandatory.** The energy law requires the Shliomis dissipation
   coefficient `μ₀/T · (1 − χ₀/4) > 0`. The current code uses `χ₀ = 1` for
   physical reasons; just don't push it past 4.
4. **`βT ≤ 1` is mandatory.** Theorem 4.5 hypothesis. Default `T = 1`,
   so `β ≤ 1`. Easy to check at parameter-load time.
5. **2D vs 3D cross product.** Always use the eq. (17) embedding. The
   `physics/` utilities don't currently have a 2D-cross helper — write
   one in `physics/vector_calc.h` (NEW) early in Step A.
6. **MMS regression** when changing assemblers: spatial rates can shift in
   absolute magnitude when the source is reformulated (e.g. analytical d/dt
   adopted), but rates themselves (slope of log-error vs log-h) must stay
   constant. The earlier session diagnosed this in detail; see STATUS.md.
7. **Picard convergence at large `β`**: Theorem 5.4 says `τ ≤ τ_* ~ h^d/R²`
   for existence. At hedgehog L5, `h ~ 0.02`, so for d=2, `τ_* ~ 4e-4 / R²`.
   With the production `τ = 1e-4`, this allows `R ~ 2` — fine for typical
   `‖M‖_∞ ≤ 250`. But if `‖M‖_∞` blows up, Picard will stall. Use the
   adaptive iter-rebuild trick from Plan A.

---

## 8. Quick-start once in the new conversation

```bash
cd /Users/mahdi/Projects/git/PhD-Project/Semi_Coupled
git status                                 # confirm clean tree
git checkout -b scheme-II                  # new branch
# Step A.1: add the parameter
$EDITOR utilities/parameters.h utilities/parameters.cc
# Build:
cmake --build cmake-build-release -j 8 --target ferrofluid
# Step A.3: regression baseline
mpirun -np 8 ./cmake-build-release/ferrofluid --hedgehog -r 5 \
    --iterative_mag --max_steps 100 --beta-LL 0 \
    --run_name scheme_II_step_A_baseline
# Diff against the previous frozen run:
diff Results/scheme_II_step_A_baseline/diagnostics.csv \
     Results/<previous_hedge_L5>/diagnostics.csv | head
```

If `diff` produces zero output, Step A is locked. Move to Step B.

---

## 9. Reference paper map (`scheme_II_standalone.pdf`)

| Page(s) | Section | What's there |
|---|---|---|
| 2 | §2.1 | Continuous Shliomis equations with all three terms |
| 3 | §2.2 | Continuous energy law — eq. (7) is the target |
| 5 | §3 "The Scheme: T1, T2, T3" | **Discrete equations 15a, 15b — implement these** |
| 5 | Remark 3.1 | Why T3 is in weak form, not strong form |
| 6 | Remark 3.2 | Why U^k (not U^{k-1}) in transport — implementation-critical |
| 7 | §4 | Energy-stability proof — read for sign conventions |
| 9 | Theorem 4.5 | The headline stability theorem (eq. 26) |
| 10 | Step 3a' / 5' / cross-cancellation | T2/T3 algebraic cancellation, eqs. 33–38 |
| 11–14 | §5 | Existence/uniqueness — drives the Picard CFL |
| 15–20 | §6 | Convergence — drives Assumption 6.1 (uniform L⁴) |

---

*Prepared 2026-05-05 as a hand-off briefing. The implementation work begins
in a fresh conversation against branch `scheme-II`.*
