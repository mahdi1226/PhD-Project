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
*Session paused on bigger-audit fixes; current focus is VTK-only bugs.*
