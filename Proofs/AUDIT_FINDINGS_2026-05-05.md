# Five-pillar audit findings — 2026-05-05 evening

Five agents ran in parallel. Reports captured here verbatim-summarized for
later triage. **Not fixing in this session** — current session refocused on
VTK-portion bugs only.

---

## Pillar A1 — Paper fidelity (Nochetto CMAME 2016)

**Net**: 16 of ~18 checked items match paper cleanly.

- **[A1-9 SMELL]** `assembly/ns_assembler.cc:431-583`: production NS Kelvin
  force uses gradient-identity form `(μ₀/2)|H|²∇χ + μ₀χ Hess(φ)·H`, NOT the
  paper's DG skew form `B_h^m(V, H, M)` (Eq 42e/57). Equivalent in continuum,
  not discretely identical. Paper's energy stability proof relies on skew form.
  Skew variant exists at line 595+ but only behind MMS/use_h_a_only paths.
- **[A1-13 SMELL]** `assembly/ns_assembler.cc:261,266`: ρ(θ)/dt mass coefficient
  may be missing the `½(ρ^n + ρ^{n+1})/τ` averaging that variable-density NS
  formulations use. Verify against paper's exact Eq 42e form.
- All other items (CH Eyre signs, B_h^m skew structure, mass coefficient
  `1/dt + 1/τ_M`, time-level discipline, χ/ν/ρ formulas, applied field, BGS
  loop) verified clean.

## Pillar A2 — Solver correctness

**Net**: 9 bugs, 4 smells, 1 nit, 7 verified-clean.

Most consequential:
- **[A2-1 BUG]** `solvers/ch_solver.cc:235`: `constraints.distribute()` called
  on owned-only vector; AffineConstraints requires ghosted entries when
  hanging-node constraints span ranks. Silent corruption near AMR boundaries
  in CH solution.
- **[A2-3 BUG]** `solvers/ch_solver.cc:172-181`: AMG preconditioner is a
  stack-local — rebuilt every CH solve, not cached cross-AMR. Hot perf miss.
- **[A2-9 BUG]** `core/phase_field.cc:918`: `last_mag_info_.converged = true`
  hardcoded after every magnetic solve. Magnetic failures invisible to
  diagnostics.csv.
- **[A2-16 BUG]** `solvers/ns_solver.cc:212-266`: `pin_pressure_dof` mutates
  the matrix in-place; iterative fallback after direct path corrupts LSC
  preconditioner (pinned p-row breaks incompressibility in `L_p`).
- **[A2-19 BUG]** `solvers/ch_solver.cc:115`: zero-RHS handling — `tol = rel·||rhs||`
  becomes 0 if RHS is exactly zero; spurious NoConvergence. Use
  `max(rel·norm, abs_tol)`.
- LSC map plumbing, MPI_Allreduce(MPI_LOR) symmetry, compress hoisting,
  magnetic cache invariants, direct-fallback wiring, rank-symmetric ctor —
  all verified clean.

## Pillar A3 — MMS framework

**Net**: 10 findings; biggest is a stub.

- **[A3-1 BUG]** `mms/magnetic/magnetic_mms_test.h:47-72`: `run_magnetic_mms_standalone`
  and `run_magnetic_mms_single` are STUBS. MAGNETIC_STANDALONE and MAG_TEMPORAL
  always return rate=0 / FAIL regardless of correctness. The `--mms-analytical`
  and `--tau-M` flags cannot be exercised through these levels.
- **[A3-2 BUG]** `mms/ns/ns_mms.h:594-614`: `compute_ns_kelvin_coupled_mms_source`
  doesn't forward `analytical_dt` (defaults false). Latent — currently unused.
- **[A3-3 BUG]** `mms/ch/ch_mms.h:271-307` (`CHSourcePsi`): Eyre term cannot
  be converted to analytical d/dt (structural), causing the documented ~0.65
  CH temporal rate. The anomaly is **not commented in code**, so CH_TEMPORAL
  silently FAILs the default 0.7 threshold.
- Cross-coupling (t² consistency), source-term completeness, BCs, tau_M
  override propagation, and rate-formula mechanics all verified clean.

## Pillar A4 — Performance (static analysis only, no benchmarks run)

**Net**: 1 HOT, 5 WARM, 2 COLD, 3 CLEAN. Top-3 ROI:

- **[A4-1 HOT]** `solvers/ns_solver.cc:76`: NS BlockSchurPreconditionerParallel
  rebuilt every step (2 AMG hierarchies + LSC). Magnetic side has cross-AMR
  cache; NS doesn't. **Estimated 15-30% of step time recoverable.**
- **[A4-2 HOT]** `diagnostics/ns_diagnostics.h:140-156`: manual `shape_value(i,q)*a_i`
  loops where `get_function_values()` would vectorize. **2-5% of step time.**
- **[A4-6 WARM]** `core/phase_field_amr.cc`: `ux_old_ghost`/`uy_old_ghost`
  held alive during full AMR step, doubling NS-vector memory peak.
  Could OOM on L6+. Scope-down to release before new-mesh allocations.
- Other WARMs: output_results allocates std::vector inside cell loop;
  BGS scratch reallocated each iter; magnetic_diagnostics has redundant
  second cell loop; AMR subface uses linear scan.
- ch_assembler quadrature loops, magnetic_assembler hoisting, magnetic
  solver caching all verified clean.

## Pillar A5 — Tree hygiene

**Net**: 9 actionable, 2 ignore. Tree is mostly clean.

- **[A5-1 REMOVE]** `Semi_Coupled/diagnostics_comparison.png`: stray PNG.
- **[A5-2 REMOVE]** `utilities/tools.h:27-29`: TEST_ID/TEST_H_FORMULA macros
  pinned to "A" with unused A/B/C selector. Dead config flag.
- **[A5-5 FIX]** `mms/mms_core/long_duration_mms.cc:675-705,711-735`:
  `run_ch_ns_long_duration_mms` and `run_full_long_duration_mms` are real
  stubs that print "Not yet implemented" and return empty results. Wired
  into dispatcher, advertised as TODO. Silent-success risk.
- **[A5-7 CLEAN-UP]** ~20 std::cout sites in solvers (ch, ns, ns_block_precond)
  duplicate output under MPI. Should be pcout_.
- **[A5-3 CONSOLIDATE]** Several files bypass `MPIUtils::rank/size` with
  direct `MPI_Comm_rank/size` calls. Easy fix.
- **[A5-9 CONSOLIDATE]** ConsoleLogger / MetricsLogger / TimingLogger
  overlap on StepData consumption. Could share a Sink interface.
- **[A5-10 IGNORE]** `mms/coupled/*` are NOT stubs (470-795 LOC each, in build).
  Earlier STATUS notes outdated — should be corrected.
- No backup files, no `#if 0` blocks, no orphans beyond the documented
  stale set.

---

## Triage priority (when we return)

**Tier 1 — silent correctness bugs**:
- A2-1 (CH constraint distribute on non-ghosted vector)
- A2-9 (mag converged flag hardcoded true)
- A2-16 (NS matrix-pinning leaks into iterative fallback)
- A1-9 (Kelvin force form vs paper)

**Tier 2 — perf wins**:
- A4-1 (NS preconditioner cache, ~15-30% step time)
- A2-3 (CH AMG cache)
- A4-2 (ns_diagnostics shape_value loops)

**Tier 3 — silent test stubs**:
- A3-1 (magnetic MMS stubs)
- A5-5 (long-duration MMS stubs)

**Tier 4 — hygiene**:
- A5-1, A5-2 (delete dead files/macros)
- A5-7, A5-8 (cout → pcout)
- A3-3 (CH temporal anomaly comment)

---

*Captured 2026-05-05. Source: five parallel audit agents (general-purpose).*

---

## ✅ Verification of Tier-1 Tier-1 BUGs (read directly from source)

| Bug | Status | Notes |
|---|---|---|
| **A2-9** mag converged hardcoded true | **VERIFIED** | `core/phase_field.cc:985` literal `last_mag_info_.converged = true;` regardless of solver outcome. |
| **A2-16** NS pin-leak into iterative fallback | **VERIFIED** | `solvers/ns_solver.cc:226-253` `pin_pressure_dof` does in-place `const_cast` writes; the unified solver at lines 639-654 calls iterative on the post-pin matrix. Real bug. |
| **A1-9** Kelvin gradient form vs paper's skew form | **DOWNGRADED** | `assembly/ns_assembler.cc:430-583` uses gradient identity form. Code comment (lines 444-446) explicitly justifies the choice (better-resolved ∇θ vs poorly-resolved Hess(φ) for Q2 elements). Engineering trade-off, documented. Leave as-is. |
| **A2-1** CH constraints.distribute on owned-only | **DOWNGRADED** | `solvers/ch_solver.cc:112` constructs owned-only vector; line 235 distributes. MMS rates at optimum (θ_L2=3.00) on AMR-refined meshes suggests deal.II's distribute() handles this. Defensive cleanup possible but not urgent. |

## ⏩ Pillar A6 — Diagnostics correctness

**Net**: 25 findings; six real correctness bugs.

- **[A6-2 BUG]** `diagnostics/ch_diagnostics.h:91-94`: CH energy uses
  coefficients `ε/2` (gradient) and `1/ε` (double-well). Paper Eq 13 says
  `λ/2` and `λ/ε²`. The surface-tension parameter `λ` is dropped. Energy
  drift in CSV is therefore in wrong units; absolute energy values useless.
- **[A6-15 BUG]** `diagnostics/validation_diagnostics.cc:354`: Rosensweig
  λ_c uses `density = 1.0 + r` (= ρ_ferro). Cowley-Rosensweig formula needs
  the contrast Δρ = r. λ_c is therefore biased. Same fix already applied to
  the Python analyzer; in-binary version still wrong.
- **[A6-9 BUG]** `diagnostics/system_diagnostics.h:131`: `interface_y_initial`
  baseline taken from `iface.y_max` (the maximum across x). For a flat IC,
  y_max = pool_baseline only by accident; for any real IC, drift metric is
  biased. Use `iface.y_mean` or `params.ic.pool_depth`.
- **[A6-24 BUG]** `diagnostics/system_diagnostics.h:140`: `spike_count`
  in CSV uses the broken `estimate_spike_count` (`domain_width / 0.2`,
  literal constant!). The proper FFT-style detector exists in
  `validation_diagnostics::count_peaks` but isn't wired in. Always-wrong
  CSV column.
- **[A6-10 BUG]** `diagnostics/ns_diagnostics.h:90`: `U_max` sampled at
  Gauss quadrature points only. For Q2 velocity, true L∞ is at vertices/extrema
  between Gauss points → systematic underestimate → CFL underestimate.
- **[A6-4 BUG]** `diagnostics/magnetic_diagnostics.h:142`: `M_max` reported
  in CSV is computed as `χ·H` (equilibrium estimate from φ + θ), NOT from
  the actual solved DG-Q1 M field. Misleading column name.
- Several SMELLs and 3 verified-clean.

## ⏩ Pillar A7 — Output / utilities / build hygiene

**Net**: 20 findings; mostly cleanup.

- **[A7-3 SMELL]** `parallel_diagnostics.csv` is a strict superset of
  `timing.csv` — redundant per-step I/O.
- **[A7-4 BUG]** `output/timing_logger.h:107`: `get_memory_usage_mb()` is
  `__linux__`-only — silently 0 on macOS. ParallelDiagnosticsLogger has
  Apple+Linux dual implementation; reuse it.
- **[A7-7 SMELL]** `convergence.csv` opened for every run, not just MMS.
  Empty file in every Results/ dir clutters downstream globs.
- **[A7-8 SMELL]** `utilities/tools.h:27-32`: `TEST_ID` / `TEST_H_FORMULA`
  preprocessor macros pinned to "A". Dead config flag; pollutes global
  macro namespace.
- **[A7-9 BUG]** `utilities/tools.cc:114-115`: `ensure_directory` shells
  out via `system("mkdir -p ...")`. Path-injection footgun on shared HPC.
  Replace with `std::filesystem::create_directories`.
- **[A7-10 SMELL]** `utilities/parameters.h:33`: `Parameters::current_time`
  field — set/read nowhere. Dead.
- **[A7-11 SMELL]** `utilities/parameters.h:76-77`: `ic.perturbation` and
  `ic.perturbation_modes` — no consumer. Project rule explicitly forbids
  initial perturbations (see MEMORY.md). Delete.
- **[A7-15 SMELL]** `mms/ns/ns_variable_nu_mms_test.cc` not in build, also
  not in the documented STALE list. Either add to STALE or delete.
- **[A7-16 BUG]** Top-level `CMakeLists.txt`: no `enable_testing()`, no
  `add_test()`. `ctest` does nothing. CI cannot run regressions.

## Updated triage priority

**Round 1 — verified Tier-1 correctness** (this session):
- A2-9 (mag converged) → trivial fix
- A2-16 (NS pin-leak) → drop direct→iterative fallback
- A6-2 (CH energy coefficients) → use λ instead of ε
- A6-15 (Rosensweig Δρ) → density contrast
- A6-9 (interface_y_initial baseline) → use y_mean
- A6-24 (spike_count) → wire in real detector

**Round 2 — diagnostic clarity** (this session):
- A6-4 (M_max rename to M_eq_max)
- A6-10 (U_max via support_points for true L∞)
- A6-22/23 (drop dead StepData fields: poisson_iterations, ns_inner_iterations)
- A7-4 (TimingLogger memory_mb on macOS)

**Round 3 — perf wins** (this session, static patches):
- A4-1 (NS Schur preconditioner cross-AMR cache)
- A4-2 (ns_diagnostics shape_value loops → get_function_values)

**Round 4 — hygiene** (this session):
- A5-1 (delete stray PNG)
- A7-8 (drop TEST_ID macros)
- A7-9 (filesystem::create_directories)
- A7-10 (drop Parameters::current_time)
- A7-11 (drop ic.perturbation*)
- A7-7 (gate convergence.csv on enable_mms)
- A7-15 (mark or delete ns_variable_nu_mms_test.cc)

**Round 5 — defer** (needs hedgehog to finish):
- Run benchmarks to confirm A4 perf estimates
- Verify A2-1 (CH distribute) with a targeted test on AMR-hanging-node case
- A2-19 (zero-RHS handling) and A2-3 (CH AMG cache) deferred

*Verified and consolidated 2026-05-05 evening. Plan: implement Rounds 1-4 in this session; commit per-tier.*

---

## ✅ Status — end of 2026-05-05 session

| Round | Commit | Status |
|---|---|---|
| Round 1 — Tier-1 correctness | `2ed5ae1` | ✅ shipped (6 bugs fixed, 2 downgraded after verification) |
| Round 2 — Diagnostic clarity | `e7b45b3` | ✅ shipped (M_max docstring, CFL via Lobatto, macOS memory) |
| Round 3 — Small perf wins | `b32fcec` | ✅ shipped (output allocs, magnetic dedup, AMR memory) |
| Round 4 — Tree hygiene | `674e2b2` | ✅ shipped (TEST_ID, dead params, ensure_directory, convergence.csv gate, stray PNG) |

VTK output cadence (`9c7a253`) shipped earlier in this session. All five
audit-related commits are on `rosensweig-working` ahead of origin by 5
commits.

## 🚧 Deferred to next session

### High-value perf
- **A4-1** NS Schur preconditioner cross-AMR cache. Estimated 15-30% step
  time recoverable. Requires moving `BlockSchurPreconditionerParallel` from
  a `solve_ns_system_schur_parallel` local to a `unique_ptr` member of
  `PhaseFieldProblem`, with a `needs_ns_preconditioner_rebuild_` flag set
  on AMR (mirroring the `needs_mag_preconditioner_rebuild_` pattern).
  Touches: `solvers/ns_solver.{cc,h}`, `solvers/ns_block_preconditioner.{cc,h}`,
  `core/phase_field.{cc,h}`. ~1 day of work + validation.
- **A2-3** CH solver AMG cache. Same shape as A4-1 but smaller. CH is
  faster than NS, less impact (~5-10% step time), but the same refactor
  pattern applies. Likely lump with A4-1.

### Silent stubs
- **A3-1** `mms/magnetic/magnetic_mms_test.h:47-72` magnetic standalone /
  single-level MMS runners are stubs. MAGNETIC_STANDALONE / MAG_TEMPORAL
  always FAIL even on correct code. Either implement or remove.
- **A5-5** `mms/mms_core/long_duration_mms.cc:675-735`
  `run_ch_ns_long_duration_mms` and `run_full_long_duration_mms` are stubs
  printing "Not yet implemented". Wired into dispatcher → silent-success
  risk if anyone runs them in CI.

### Build / test infra
- **A7-16** Top-level `CMakeLists.txt` has no `enable_testing()` /
  `add_test()`. CI cannot run regressions via `ctest`. Add explicit
  `add_test(NAME mms_ch_standalone COMMAND mpirun -np 2 …)` for each
  parallel_test_* target.
- **A7-15** `mms/ns/ns_variable_nu_mms_test.cc` — orphan source not in
  CMakeLists. The companion `mms/ns/STALE_ns_variable_nu.md` documents
  the variable-ν files as STALE; verify the `_test.cc` is included in the
  STALE list or delete it.

### Defensive correctness (low priority)
- **A2-1** CH solver `constraints.distribute(coupled_solution)` on an
  owned-only Trilinos vector. MMS rates at optimum suggest deal.II handles
  it correctly via internal ghost copy, but a defensive refactor (use
  ghosted intermediate) is safer for hanging-node-on-rank-boundary cases.
- **A2-19** CH solver: when `rhs.l2_norm() == 0` exactly, `tol = rel·norm`
  becomes 0, GMRES never converges → spurious NoConvergence and direct
  fallback fires. Use `max(rel·norm, abs_tol)`.
- **A6-7, A6-17** `interface_tracking.h` and `validation_diagnostics.cc`
  read DoFs without locally-owned check on non-ghosted vectors. Has not
  bitten yet but is fragile.

### Benchmarks (need hedgehog to finish)
- Confirm A4-1 / A4-2 / A4-3 perf estimates with real wall-time deltas.
- Compare LSC vs direct on hedgehog L5 (the original Plan A/B perf goal).
- Validate cache invariants by running `--iterative_mag` with vs without
  Plan A on a 200-step dome L5 and confirming matching diagnostics.csv.

### Documentation tasks
- **A6-12** `force_diagnostics.h:117-147` — header comment claims one
  formula, implementation computes a different (truncated) form. Either
  rename `F_mag_max` → `F_mag_interface_max` or include the missing
  `∇|H|²` term.
- **A6-22/23** Drop unused StepData fields (`poisson_iterations`,
  `ns_inner_iterations`) from CSV header — risks breaking Python
  analysis, so do alongside an analyzer update.
- **A4-1 / A2-3 etc.** all need to land before SCHEME_II branch starts;
  T1+T2+T3 work assumes a clean and fast Nochetto base.

---

*Last updated 2026-05-05 night. Total: 5 audit-related commits on
`rosensweig-working`. Tree is clean for current scope; deferred list
above guides the next cleanup pass.*
