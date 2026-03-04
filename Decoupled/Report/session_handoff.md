# Decoupled Ferrofluid Solver -- Session Handoff

## Project Location
`/Users/mahdi/Projects/git/PhD-Project/Decoupled/`
Build: `cd /Users/mahdi/Projects/git/PhD-Project/Decoupled/build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j8`

## Reference Papers
- Zhang, He & Yang, SIAM J. Sci. Comput. 43(1), 2021 -- Parameters, validation, SAV scheme
  PDF: `/Users/mahdi/Desktop/droplet/GZhang_XMHe_XYang.pdf`
- Nochetto, Salgado & Tomas, CMAME 309, 2016 -- PDE formulation (simplified model)
  PDF: `/Users/mahdi/Desktop/RHNochetto_AJSalgado_ITomas.pdf`

---

## Current Status Summary

| Test | Status | Result |
|------|--------|--------|
| **Full 4-system MMS** | **PASS** | u_L2=3.0, p_L2=2.1, phi_L2=3.0, M_L2=2.0, theta_L2=3.0 |
| **Square** (CH-only, r=6) | PASS | 5000 steps, theta in [-0.992, 1.01] |
| **Droplet WITH field** (r=7) | PASS | 1500 steps, theta in [-1.00, 1.03] |
| **Droplet WITHOUT field** | PASS | 1500 steps, theta in [-1.00, 1.01], \|U\|=0 |
| **Rosensweig uniform** (r=4) | PASS | 2000 steps, theta in [-1.00, 1.00] |
| **Rosensweig uniform + AMR** (r=3) | **RUNNING** | 4 MPI, physics-based activation at step 63 |
| **Rosensweig nonuniform** (r=3, dt=2e-4) | **RUNNING** | 4 MPI, step ~4300/17500 |

---

## What Was Done Across All Sessions

### Sessions 1-3: SAV CH Assembly Fix + Rosensweig Fix + Equation Audit
- Removed Douglas-Dupont stabilization from CH assembly
- Fixed 10/14 Rosensweig parameters
- Fixed Rosensweig IC (flat at y=0.2, no perturbation)
- Fixed capillary force direction (theta*grad_psi, not psi*grad_theta)
- Fixed Kelvin force grad(h_a) missing from grad(H)
- Added `droplet_nofield` validation preset

### Session 4: Complete Zhang Deviation Audit + 7 Fixes
- Fixed viscous term 2x error (nu/2 -> nu/4)
- Fixed S1 auto-computation: lambda/(4*epsilon)
- Enabled full magnetization PDE (beta=1, tau=1e-4)
- Removed dipole regularization delta
- Sharp step IC for Rosensweig

### Sessions 5-8: MMS Verification + Critical DG Bug Fix
- Built incremental MMS strategy: standalone -> coupled -> full system
- Fixed DG face assembly `FEInterfaceValues` indexing (zero cross-cell coupling)
- Added upwind penalty for optimal O(h^2) DG convergence
- All standalone and coupled MMS tests PASS

### Sessions 9-10: Validation Suite + Nonuniform Rosensweig
- All validation tests rerun post-DG fix
- Nonuniform Rosensweig: two clean spikes form, instability at t~2.07
- Root cause: operator splitting instability under strong coupling (chi_0=0.9)

### Sessions 11-12: Paper Audit + Parameter Cleanup
- Reverted chi/nu to sigmoid interpolation (was wrongly set to linear)
- Added spin-vorticity term to magnetization equation
- Removed S2 and C0 (not in Zhang's paper)
- Paper is sole authority for parameters

### Session 13: AMR Implementation (CURRENT)

**Adaptive Mesh Refinement** for all 4 subsystems on shared p4est triangulation.

**New file:** `utilities/amr.h` -- header-only 14-step AMR algorithm:
1. Kelly error estimation on theta (interface field)
2. Mark cells with fixed-fraction (upper/lower)
3. Enforce level limits (max/min)
4. Interface protection (never coarsen where |theta| < threshold)
5. Prepare triangulation
6. Create SolutionTransfer for all subsystems (7 DoFHandlers)
7. Execute mesh refinement
8. Re-setup all subsystems
9. Interpolate solutions to new mesh
10. Apply constraints
11. Clamp theta to [-1, 1]
12. Recompute psi = theta^3 - theta nodally
13. Update all ghost vectors
14. Log diagnostics

**Critical bug found and fixed:** `ghosts_valid_` flag in CH subsystem not reset
by `setup()`. After AMR, `update_ghosts()` skipped the copy, leaving theta_relevant_
at zero. Fixed by adding `ghosts_valid_ = false` in `setup()` and belt-and-suspenders
`invalidate_ghosts()` calls in amr.h.

**Physics-based AMR activation gate:**
- AMR stays dormant until |U|_max exceeds `amr_activation_U` threshold (default 1e-3)
- Once activated, stays active for the rest of the simulation
- When NS is disabled, activates immediately (no velocity to gate on)
- `--amr-activation-U 0` for immediate activation

**AMR parameters (all CLI-configurable):**
```
--amr / --no-amr              Enable/disable (default: OFF)
--amr-interval N              Refine every N steps (default: 5)
--amr-max-level N             Max refinement level (default: 0 = no cap)
--amr-min-level N             Min refinement level (default: 0)
--amr-upper-fraction V        Top fraction to refine (default: 0.3)
--amr-lower-fraction V        Bottom fraction to coarsen (default: 0.10)
--amr-activation-U V          |U| threshold to start AMR (default: 1e-3)
```

**AMR test results:**
- CH-only: 200+ steps, mesh 960->420, energy decreasing
- Full 4-system: 370+ steps at r3, mesh 3840->1140 (70% reduction), all fields preserved
- Uniform Rosensweig with AMR: 100 steps, activation at step 63, mesh 3840->2112->11688
- Physics gate saves 37% wall time vs fixed-interval AMR
- Default path (no AMR): unchanged behavior, no regression

**VTK mesh_level field** added to `write_combined_vtu()` and `write_ch_only_vtu()`
for ParaView AMR visualization.

**Files modified for AMR:**
| File | Change |
|------|--------|
| `utilities/amr.h` | **NEW** -- 14-step AMR algorithm (header-only template) |
| `utilities/parameters.h` | AMR fields in Mesh struct, amr_activation_U |
| `utilities/parameters.cc` | AMR CLI parsing, help text |
| `cahn_hilliard/cahn_hilliard.h` | Mutable DoFHandler accessors |
| `cahn_hilliard/cahn_hilliard.cc` | ghosts_valid_ = false in setup() |
| `navier_stokes/navier_stokes.h` | Mutable solution + DoFHandler accessors |
| `poisson/poisson.h` | Mutable solution + DoFHandler accessor |
| `magnetization/magnetization.h` | Mutable solution + DoFHandler accessors |
| `drivers/decoupled_driver.cc` | AMR call in time loop + activation gate + mesh_level VTK |

---

## AMR Switchability

`use_amr = false` by default. The existing `refine_global(initial_refinement)` path
is unchanged. AMR only activates when `--amr` is passed on the CLI. Presets do NOT
enable AMR. Any existing test/preset works identically without modification.

---

## Key Technical Details

### Phase Field Convention
- Zhang uses Phi in {0,1}, code uses theta in {-1,+1}
- Equivalent via Phi = (theta+1)/2
- Material properties use SIGMOID interpolation (both Zhang and Nochetto):
  - chi(theta) = chi_0 * H(theta/epsilon)
  - nu(theta) = nu_w + (nu_f - nu_w) * H(theta/epsilon)
  - rho(theta) = 1 + r * H(theta/epsilon)

### SAV Stabilization (Zhang Eq 3.5-3.6)
- S1 = lambda/(4*epsilon), no S2 or C0 (paper only)
- S does NOT depend on dropped terms (verified from energy proof Eq 3.41)

### Coupling Strategy
```
FOR each timestep:
  1. CH solve (SAV, using U^{n-1})
  2. Picard loop:
     a. Poisson -> H
     b. Magnetization -> M (DG transport + relaxation)
     c. Under-relax M
  3. NS solve (Kelvin + capillary + gravity)
```

---

## Currently Running Tests

1. **Rosensweig nonuniform** (no AMR): 4 MPI, r3, 17500 steps, dt=2e-4
   - Directory: `Results/030326_181257_rosensweig_nonuniform_r3/`
   - Progress: step ~4300, t=0.86

2. **Rosensweig uniform + AMR**: 4 MPI, r3, 2000 steps, dt=1e-3
   - Directory: `Results/030326_212356_rosensweig_r3/`
   - AMR activated at step 63 (|U| > 1e-3)

---

## Pending Tasks (Priority Order)

1. **Monitor running tests** -- check AMR Rosensweig results and nonuniform completion
2. **Nonuniform Rosensweig stability** -- blows up at t~2.07 with dt=2e-4
3. **Full system MMS with AMR** -- verify convergence rates preserved with mesh adaptation
4. **Extension study** (Phase C.0+) -- parametric sweeps, see `extension.md`

---

## Git Status

**Uncommitted changes (Sessions 11-13):**
- AMR implementation (utilities/amr.h, all 4 subsystem headers, driver, parameters)
- Sigmoid chi/nu revert, spin-vorticity, S2/C0 removal (Sessions 11-12)
- MMS test updates

---

*Updated: March 3, 2026 (Session 13 -- AMR implementation with physics-based activation gate)*
