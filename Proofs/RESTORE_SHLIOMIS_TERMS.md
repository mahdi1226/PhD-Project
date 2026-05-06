# Restore the Shliomis terms — incremental T1, T2, T3 implementation

**Audience.** A fresh Claude conversation picking up the work of restoring
the three Shliomis-model terms that the Nochetto, Salgado & Tomás
CMAME 2016 base scheme drops. The end goal is the **complete Shliomis
form**, reached by adding the three terms **one at a time** with
validation between each.

**No "Scheme I" / "Scheme II" framing.** The earlier session drafted
hand-offs around the paper's named variants (Scheme I = T1+T2 conditional;
Scheme II = T1+T2+T3 unconditional). User direction (2026-05-06): we don't
want the intermediate named variants — we want the **final complete
Shliomis equations**, reached step-by-step:

```
  base (no T1/T2/T3)  →  +T1  →  +T1+T2  →  +T1+T2+T3 (= complete Shliomis)
```

Each `+Tn` is its own validation milestone. The intermediate states are
debugging waypoints, not publication targets.

**Mathematical reference.** Use `Proofs/scheme_II_standalone.pdf` for the
discrete formulation — it has all three terms, the energy proof, and the
existence/uniqueness analysis. **Cite the equations**, not the scheme name.

`Proofs/scheme_I_standalone.pdf` (T1+T2 only, conditional stability) and
the joint `ferrofluid_proofs_v20_2.pdf` are useful as reference but not as
implementation targets.

**Prerequisite reading.**
1. This document — full plan.
2. `Proofs/scheme_II_standalone.pdf` §3 — discrete equations 15a–b for
   M-equation and NS-equation with all three terms; Remarks 3.1, 3.2.
3. `Proofs/STATUS.md` and `Proofs/AUDIT_FINDINGS_2026-05-05.md` — current
   state of the codebase, what's clean, what's deferred.

The companion `MEMORY.md` rule still applies: this work is in
`Semi_Coupled/`, references **only** Nochetto + the three-term extension
math. No cross-pollination from `Decoupled/`.

---

## 1. Where we are now (2026-05-06 evening)

The Nochetto base scheme is implemented, audited, and validated:

- CH (Q2/Q2), DG-Q1 magnetization, monolithic `[DG-Q1 M | CG-Q1 φ]` block,
  Q2/DQ1 saddle-point NS.
- Iterative solver paths shipped: CH GMRES+AMG, magnetic block-precond
  (ILU on M, AMG on φ), NS LSC Schur (`B Q⁻¹ B^T`); cross-AMR cache for
  all three; direct fallback wired.
- Hedgehog L5 ran to t=6.0 cleanly (~50 h of compute), 2-spike pattern,
  λ=0.20, mass conserved to 8.8×10⁻⁴, Rosensweig validation matches
  theory after the Δρ fix.
- MMS framework exercises CH+NS+CH-Magnetic at refs 3-4 in `ctest`,
  all 4 tests pass in 88 s.
- Tree is clean — no orphans, no STALE files, no documented "kept for
  later" debt. The audit deferred list is fully drained.

**NOT yet implemented (this hand-off's scope):**

- T1: Landau–Lifshitz damping `−β M × (M × H)` in the M-equation.
- T2: Spin–vorticity coupling `½(∇×U) × M` in the M-equation.
- T3: Antisymmetric magnetic stress `(μ₀/2)(M × H, ∇ × V)` in the NS RHS.
- Velocity time-level discipline change: M-transport must use `U^k`
  (current Picard iterate) not `U^{k-1}` (see Remark 3.2 of the PDF).
  This is one-line in the assembler but requires a Picard inner loop
  in the time-step driver.

---

## 2. The three terms — exact discrete forms

**Notation** (matches the PDF §3 and §4):
- `Θ^k, Ψ^k, Φ^k, M^k, U^k, P^k` are the discrete unknowns at time k.
- `H^k := ∇Φ^k`.
- `B_h^m(U, V, W)` is the DG skew-symmetric trilinear form (PDF eq. 13);
  already implemented in `assembly/magnetic_assembler.cc`.
- 2D cross product uses the `(a₁, a₂) ↦ (a₁, a₂, 0)` embedding (PDF eq. 17).
- `χ_Θ := χ(Θ^{k-1})` etc. — phase-dependent coefficients lagged at the
  old phase field for energy stability.

### T1 — Landau–Lifshitz damping (M-equation, RHS, fully explicit)

```
−β (M^{k-1} × (M^{k-1} × H^k), Z)        ← PDF eq. 15a
```

Both factors at `k-1`; only `H^k` is implicit. Tested with `Z = 2τμ₀ M^k`
this term combines via Young's inequality (PDF eq. 32) and is absorbed
into the Shliomis dissipation under `βT ≤ 1` (consumes half).

The genuine dissipation `+2μ₀τβ ‖M^{k-1} × H^k‖²` shows up from a
**different** test — `Z = 2τμ₀ H^k` (PDF eq. 34). The inclusion
`∇𝒳_h ⊂ ℳ_h` (PDF eq. 12) is what makes that test admissible.

Implementation:
- File: `Semi_Coupled/assembly/magnetic_assembler.cc`, M-equation block, RHS.
- Per-quad-point: compute `M_old × (M_old × H_q)` with `M_old` from
  `mag_old_relevant_` and `H_q = ∇Φ_q`.
- Add `−β · (M_old × (M_old × H_q)) · Z_q · |J| dx` to the RHS.
- New parameter: `params.physics.beta_LL` (double, default 0.0). When 0,
  the assembler **must** produce results bitwise identical to the
  current code — that's the regression baseline.

### T2 — Spin–vorticity coupling (M-equation, LHS, uses U^k)

```
+ ½ ((∇×U^k) × M^k, Z)        ← PDF eq. 15a
```

**Crucial:** uses `U^k` (current Picard iterate), not `U^{k-1}`. This is
what makes T2 cancel against T3 algebraically in the energy proof
(PDF Step 3a' / 5', eqs. 35–38). Cancellation is via the scalar triple
product `(a × b)·c = (b × c)·a`.

In 2D embedding: `(∇×U) × M = (∂₁U₂ − ∂₂U₁) · (−M₂, M₁, 0)`, so the
assembled contribution is `½ (∂₁U₂ − ∂₂U₁) · (−M̂_j₂ φ_i₁ + M̂_j₁ φ_i₂)`.

Implementation:
- File: `Semi_Coupled/assembly/magnetic_assembler.cc`, M-equation block, **LHS**.
- Inside the inner Picard loop (added in Step C below), `∇×U^k` is
  computed once per loop iter from the latest `U^k`.

### T3 — Antisymmetric magnetic stress (NS RHS, weak form)

```
+ (μ₀/2) · (M^k × H^k, ∇ × V)        ← PDF eq. 15b
```

**Weak form, NOT strong form.** The strong `∇ × (M × H)` would require
`H(curl)` regularity on `M × H`, which DG `M` doesn't have; distributional
IBP would generate uncontrollable face-jump terms. The weak form is a
plain `L²` inner product, well-defined for DG `M` (PDF Remark 3.1).

In 2D: `M × H = (0, 0, M₁H₂ − M₂H₁)`, `∇ × V = ∂₁V₂ − ∂₂V₁`, so the
contribution is `(μ₀/2) · (M₁H₂ − M₂H₁) · (∂₁V₂ − ∂₂V₁)`.

Implementation:
- File: `Semi_Coupled/assembly/ns_assembler.cc`, NS RHS.
- `M^k`, `H^k` are Picard-lagged from the prior magnetic solve in the
  loop — well-defined references at NS assembly time.

### Velocity time-level fix (Remark 3.2)

Current Nochetto-base uses `U^{k-1}` in `B_h^m(U^{k-1}, Z, M^k)` because
NS hasn't been solved yet at the point Mag is solved (block-Gauss-Seidel
order). Restoring all three terms requires `U^k` here. Fix: wrap CH–Mag–NS
in a Picard inner loop, latest `U` on each iterate. Convergence under
`τ ≤ τ_* ~ h^d/R²` is proved in PDF Theorem 5.4.

```python
# Before (current Nochetto base):
solve_ch()         # CH uses U^{k-1} (decoupled from U upstream — unchanged)
solve_magnetics()  # uses U^{k-1}  ← becomes U^k_picard after fix
solve_ns()         # uses M^k, U^{k-1} for convection

# After (complete Shliomis):
solve_ch()
for picard_iter in range(MAX):
    solve_magnetics()  # uses U^{k-1}_picard (init = U^{k-1})
    solve_ns()         # uses M^k_picard
    if residual < tol: break
```

`picard_iter == 0` reduces to the current code with `U^{k-1}` in
transport — that's the regression test for Step C below.

---

## 3. Implementation order — five steps

Each step has a clean validation checkpoint. **Do not skip them.**

### Step A — `β_LL` parameter wiring + regression baseline (~2 h)

1. Add `physics.beta_LL = 0.0` to `Semi_Coupled/utilities/parameters.{h,cc}`.
2. Add `--beta-LL VALUE` CLI flag.
3. **Do not** add T1 to the assembler yet. Just plumb the parameter.
4. Validate hardcoded β=0 path: hedgehog L5 with `--beta-LL 0` for 100
   steps must produce `diagnostics.csv` bit-equal to a reference run on
   the prior binary.
5. **Checkpoint A**: regression baseline locked. β=0 reproduces Nochetto
   base exactly.

### Step B — T1 in the M-equation (~half day)

1. Add the T1 contribution to `assembly/magnetic_assembler.cc` (RHS),
   guarded by `if (params.physics.beta_LL > 0)` so β=0 still bypasses
   the entire code path.
2. Extend MMS magnetic source: append `+ β M_exact × (M_exact × H_exact)`
   to `mms/magnetic/magnetization_mms.h`. The MMS exact `M`, `H` are
   already in t² form.
3. Run `ctest -R mms_ch_magnetic` with `--beta-LL 0.1` (need to add a
   test variant; or run the binary directly).
4. Run hedgehog L5 with `--beta-LL 0.1` for ~1000 steps. Check:
   - Energy decay slope steeper than β=0 (T1 dissipates).
   - Spike pattern still forms.
   - No θ-bound violations beyond baseline.
   - Mass drift comparable to baseline.
5. **Checkpoint B**: T1 verified. Energy law quantitatively shows the
   `+2μ₀τβ ‖M^{k-1} × H^k‖²` term.

### Step C — T2 + Picard discipline (~1 day, hardest step)

1. Refactor `core/phase_field.cc::run()` to add a Picard loop around
   `solve_magnetics + solve_ns`. Residual on `‖U^k − U^k_prev‖`.
   Default `MAX_PICARD = 5`, `picard_tol = 1e-6`.
2. Add T2 (LHS) to `magnetic_assembler.cc`. Use the Picard-current `U^k`.
3. Extend MMS magnetic source with the T2 contribution (analytical
   `∇×U_exact` is closed-form from the inline NS exact velocity).
4. Implement `mms/coupled/magnetic_ns_mms_test.cc` (currently a stub
   per A3-1). Spatial rates must hit DG-Q1 = 2 for L2.
5. Run hedgehog L5 with T1+T2 — pattern still forms; T2 is rotational,
   no new dissipation, may shift spike timing slightly.
6. **Checkpoint C**: M-equation has the full Shliomis terms. Picard
   loop converges in ≤2-3 iters/step.

### Step D — T3 in the NS RHS (~half day)

1. Add T3 (weak form, RHS) to `assembly/ns_assembler.cc`. Inside the
   Picard inner iter so it sees the current `M^k`, `H^k`.
2. Verify the T2/T3 cross-cancellation at the discrete level: compute
   both sides of PDF eq. 38 from the assembled vectors. They must
   agree to round-off.
3. Extend NS MMS source with T3 contribution.
4. Implement `mms/coupled/full_system_mms_test.cc` end-to-end T1+T2+T3.
5. Run hedgehog L5 with all three — compare energy decay to β=0
   baseline; the difference quantifies the T1 dissipation.
6. **Checkpoint D**: full complete-Shliomis form realised. Production
   runs use this binary.

### Step E — Validation cascade (~2 days)

1. MMS spatial: refs 3,4,5 with all three terms active.
   Rates must remain optimal.
2. MMS temporal with `--mms-analytical`: rate ≈ 1 (BE).
3. Energy-law sanity check (`scripts/check_energy_law.py` to be written):
   load `diagnostics.csv`, evaluate PDF eq. 26 numerically, confirm
   `LHS ≤ RHS` at every step.
4. Hedgehog L5 with the full scheme — compare pattern vs Nochetto base.
   Document differences (T1 narrows spikes; T2 slight rotation).
5. Hedgehog L6 (HPC) when L5 is stable.

---

## 4. File-by-file touch list

| File | Change |
|---|---|
| `utilities/parameters.{h,cc}` | Add `physics.beta_LL`, `--beta-LL`, `--max-picard`, `--picard-tol` |
| `core/phase_field.{h,cc}` | Picard loop in `run()`; cache `U_picard_prev_` |
| `assembly/magnetic_assembler.cc` | T1 (RHS, `beta_LL>0`-guarded), T2 (LHS, Picard-`U^k`) |
| `assembly/ns_assembler.cc` | T3 (RHS, weak form `(M×H, ∇×V)`) |
| `mms/magnetic/magnetization_mms.h` | T1 + T2 in manufactured source |
| `mms/ns/ns_mms.h` | T3 in NS manufactured source |
| `mms/coupled/full_system_mms_test.cc` | Implement (currently stub) — exercises all three |
| `mms/coupled/magnetic_ns_mms_test.cc` | Implement (stub) — T2 + T3 only |
| `output/metrics_logger.{h,cc}` | Add `picard_iters` column to `diagnostics.csv` |
| `analysis/check_energy_law.py` | NEW — verify PDF eq. 26 holds on `diagnostics.csv` |
| `CMakeLists.txt` | Add `add_test()` for new MMS variants once they pass |

---

## 5. Branch strategy — open question, decide before starting

**`main` and `rosensweig-working` have diverged** as of 2026-05-06:
- `rosensweig-working`: 35 commits ahead of common ancestor — VTK output
  cadence, audit Rounds 1–4, deferred-list drain, CH AMG + NS Schur cache.
- `main`: 42 commits ahead of common ancestor — parallel work including
  Kelvin-force `h_a` fix, pressure-FE revert to DG Q1, more MMS bug fixes.

The two branches touch the same files. Three options for the new work:

| Option | Substrate | Trade-off |
|---|---|---|
| **`shliomis` off `rosensweig-working`** | clean audit-validated base | inherits caches + ctest infra; manually cherry-pick `main`'s math fixes |
| **`shliomis` off `main`** | `main`'s parallel improvements | loses audit work until cherry-picked back |
| **Reconcile first** | merge `main ↔ rosensweig-working` before branching | ~30 min of conflict resolution; produces the cleanest substrate |

**Recommended: reconcile first.** Spend the upfront time to merge them
(or rebase one onto the other) so the new branch starts from a single
authoritative base. Document each conflict resolution in the merge commit.

If reconciliation is deferred, branch off `rosensweig-working` and write
a "TO-CHERRY-PICK FROM MAIN" list with the commit hashes of `main`'s
math-relevant fixes (Kelvin h_a, pressure FE revert, etc.) so they
aren't forgotten.

```bash
cd /Users/mahdi/Projects/git/PhD-Project/Semi_Coupled
git fetch origin
git checkout rosensweig-working   # or main, after reconciliation
git pull
git checkout -b shliomis
```

Then start with **Step A**.

---

## 6. Known gotchas (don't waste time on these)

1. **Don't IBP the T3 term.** The strong form `∇×(M×H)` is NOT in
   `H(curl)` for DG `M`. Stay in the weak form `(M×H, ∇×V)`.
   See PDF Remark 3.1.
2. **Don't use `U^{k-1}` in T2.** The cancellation with T3 only works
   for matched velocity argument. See PDF Remark 3.2.
3. **`χ₀ < 4` is mandatory.** Energy law requires
   `μ₀/T · (1 − χ₀/4) > 0`. Production uses `χ₀ = 0.9`; safe.
4. **`βT ≤ 1` is mandatory.** PDF Theorem 4.5 hypothesis. Default
   `T = 1`, so `β ≤ 1`. Easy to assert at parameter-load time.
5. **2D vs 3D cross product.** Always use the PDF eq. 17 embedding.
   The `physics/` utilities don't currently have a 2D-cross helper —
   write one in `physics/vector_calc.h` (NEW) early in Step A.
6. **MMS regression** when changing assemblers: spatial rates can shift
   in absolute magnitude when the source is reformulated, but the
   slope of `log(error) vs log(h)` must stay constant. See
   `Proofs/AUDIT_FINDINGS_2026-05-05.md` for prior diagnosis.
7. **Picard convergence at large `β`**: PDF Theorem 5.4 says
   `τ ≤ τ_* ~ h^d/R²`. At hedgehog L5, `h ~ 0.02`, d=2, so
   `τ_* ~ 4e-4 / R²`. With `τ = 1e-4`, this allows `R ~ 2`.
   For `‖M‖_∞ ≤ 250` (typical) we're fine. If `‖M‖_∞` blows up, Picard
   stalls — use the adaptive iter-rebuild trick from the magnetic cache.
8. **Cache invalidation on assembler change.** When you change the
   M-equation matrix structure (T2 added to LHS), the cached AMG
   aggregations are still valid (sparsity unchanged), but iter counts
   may climb temporarily. The adaptive trigger in
   `magnetic_solver.cc::solve_iterative` handles this — leave it alone.

---

## 7. Quick-start once in the new conversation

```bash
cd /Users/mahdi/Projects/git/PhD-Project/Semi_Coupled
git status              # confirm clean tree
# Branch decision (see §5):
git checkout rosensweig-working  # or main, after reconciliation
git pull
git checkout -b shliomis

# Step A.1: parameter wiring
$EDITOR utilities/parameters.h utilities/parameters.cc
cmake --build cmake-build-release -j 8 --target ferrofluid

# Step A.4: regression baseline
mpirun -np 8 ./cmake-build-release/ferrofluid --hedgehog -r 5 \
    --iterative_mag --max_steps 100 --beta-LL 0 \
    --run_name shliomis_step_A_baseline

# Bit-equality check vs the just-completed reference hedgehog L5:
diff Results/<timestamp>_shliomis_step_A_baseline/diagnostics.csv \
     Results/20260430_235602_hedge_r3_direct_amr/diagnostics.csv | head
```

If the diff is empty (or only a column-order/format change), Step A is
locked. Move to Step B.

---

## 8. Reference paper map (`scheme_II_standalone.pdf`)

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

*Prepared 2026-05-06 evening. Renamed from `SCHEME_II_HANDOFF.md` per
user direction: no "Scheme I/II" naming — go straight to the complete
Shliomis form, one term at a time.*
