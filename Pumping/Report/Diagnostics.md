# Diagnostics: Issues Encountered and Resolutions

## Inherited from FHD Phase A

See `FHD/Report/Diagnostics.md` for full history of resolved issues from the
single-phase solver development. Key lessons:

1. NS viscous-MMS factor: nu(D(u),D(v)) -> strong form -(nu/2)*Delta(u)
2. MMS 1/tau amplification: sources must use discrete old values
3. DG face velocity evaluation: must use fe_face_U at quadrature points
4. DG face dedup: each face processed once (smaller CellId)
5. H = grad(phi) total field: never add h_a separately (double-counting)

## Phase B Issues

### Issue B1: DataComponentInterpretation include (resolved)
- **Problem**: `#include <deal.II/base/data_component_interpretation.h>` not found
- **Cause**: Not a separate header in deal.II
- **Fix**: Remove include; DataComponentInterpretation is available through `<deal.II/numerics/data_out.h>`

### Issue B2: FEValues get_function_values API (resolved)
- **Problem**: `fe_values[extractor].get_function_values(vec)[0]` doesn't compile
- **Cause**: Return value is not directly subscriptable
- **Fix**: Use buffer vector:
  ```cpp
  std::vector<double> phi_vals(n_q_points);
  fe_values[phi_extractor].get_function_values(ch_solution, phi_vals);
  double phi_q = phi_vals[q];
  ```

### Issue B3: Rosensweig instability -- no spike formation (ABANDONED)
- **Symptoms**: Multiple runs with H=10 and H=30, refinement levels 3-4, 500+ steps.
  Maximum velocity U_max=0.005-0.165 but no visible interface deformation.
- **Root cause**: Nochetto's fully coupled scheme over-damps the instability.
  Confirmed across 3+ projects using this scheme. Zhang's decoupled IEQ+ZEC scheme
  produces spikes; Nochetto's does not.
- **Resolution**: Abandoned Rosensweig benchmark. Not a convergence study in the paper.
  Moved to droplet deformation benchmark which validates the same physics (Kelvin + capillary).

### Issue B4: CH chemical potential -- magnetic energy term NOT needed (resolved)
- **Problem**: Initially proposed adding -0.5*mu_0*chi'(phi)*|H|^2 to CH chemical potential
- **Resolution**: After reviewing all 13 reference papers, confirmed NO paper does this.
  The magnetic coupling enters through chi(phi) in magnetization and Kelvin force in NS.
  The variational derivative of the magnetic energy is handled through the coupled
  system's energy estimate, not as an explicit term in W.
- **Impact**: Cancelled the planned modification. Current implementation is correct.

### Issue B5: Capillary force form equivalence (resolved)
- **Question**: Is sigma*mu*grad(phi) (our form) equivalent to (lambda/eps)*phi*grad(W)?
- **Answer**: Yes, modulo a pressure gradient term. Nochetto's Remark 2.1 shows these
  are related by pressure redefinition: p_new = p + (lambda/eps)*W. Both forms yield
  the same physics. Our form sigma*mu*grad(phi) is correct and widely used.

### Issue B6: Sigmoid material properties cause instability (resolved)
- **Problem**: Sigmoid interpolation for chi(phi) and nu(phi) caused solver blow-up
  in Rosensweig instability runs (commit c8666bf).
- **Root cause**: Isolated by testing combinations. Sigmoid (not spin-vorticity) was
  the sole culprit. Linear + spin-vorticity: stable 2000 steps.
- **Fix**: Use linear interpolation: chi(phi) = chi_ferro*(phi+1)/2,
  nu(phi) = nu_w*(1-phi)/2 + nu_f*(phi+1)/2. Density uses sigmoid (correct per Zhang).

### Issue B7: Grid-quantized interface tracking (resolved)
- **Problem**: Interface tracking using cell midpoints (|phi|<0.9) gave discrete AR values
  at ref 6. Three different H0 values (1.0, 1.5, 2.0) all showed identical AR=1.0571
  because the tracking resolution was limited to cell width h=1/64=0.0156.
- **Root cause**: Cell-center method detects which cells straddle the interface. The
  interface extent changes only when a new cell enters/exits the band, giving O(h) jumps.
- **Fix**: Sub-cell phi=0 contour tracking. For each cell edge, check if phi changes sign
  at the two vertices. If yes, linearly interpolate to find the phi=0 crossing point.
  This gives continuous AR values with sub-cell accuracy.
  ```cpp
  // QTrapezoid evaluates at vertices (0,0),(1,0),(0,1),(1,1)
  if (phi_v[v1] * phi_v[v2] < 0.0) {
      double t = phi_v[v1] / (phi_v[v1] - phi_v[v2]);
      double xc = p1[0] + t * (p2[0] - p1[0]);
      // track min/max of crossing points
  }
  ```
- **Verification**: Initial circle x_min=0.3001 (true: 0.3000) vs old method 0.2266.
  AR values now show clear separation between different H0 cases.

### Issue B8: High-Bo_m runs not equilibrated (in progress)
- **Problem**: First sweep used t_final=1.0. High-H0 cases (3,4,5) had significant
  residual velocity (U_max=0.02-0.07) and AR was still changing at t=1.0.
- **Cause**: Capillary relaxation timescale increases with deformation magnitude.
  Stronger fields produce larger deformations that take longer to equilibrate.
- **Fix**: Extended to t_final=3.0. All 6 runs re-launched with new sub-cell tracking.
  H0=1.0 appears equilibrated (U~1.5e-3, AR stable). Higher H0 cases still evolving.

### Issue B9: Deformation below linear theory (expected)
- **Observation**: D_numerical is 3-4x lower than the 2D linear theory D = Bo_m*chi/(2(2+chi)).
- **Explanation**: This is expected for two reasons:
  1. Linear theory valid only for D < 0.05; most of our Bo_m range is nonlinear
  2. Diffuse interface (CH, eps=0.02) adds stiffness vs sharp interface (VOF in Afkhami).
     The CH interface energy penalizes deformation beyond just surface tension sigma.
- **Status**: Qualitative behavior is correct (monotonic D vs Bo_m, nonlinear saturation).
  Quantitative match not expected between 2D CH and 3D axisymmetric VOF.
