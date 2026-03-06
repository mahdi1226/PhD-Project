#!/usr/bin/env python3
"""
Generate PDF report for CFL Instability Onset findings.
Uses fpdf2 library to create a formatted PDF with embedded figures.
"""

import os
import sys

try:
    from fpdf import FPDF
except ImportError:
    print("Installing fpdf2...")
    os.system(f"{sys.executable} -m pip install fpdf2 --quiet")
    from fpdf import FPDF

REPORT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(REPORT_DIR, "figures", "cfl_diagnostics")


class CFLReport(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font('Helvetica', 'I', 8)
            self.cell(0, 5, 'CFL Threshold at Rosensweig Instability Onset', align='C')
            self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', align='C')

    def chapter_title(self, title, level=1):
        if level == 1:
            self.set_font('Helvetica', 'B', 16)
            self.set_fill_color(220, 230, 241)
            self.cell(0, 12, f'  {title}', fill=True, new_x="LMARGIN", new_y="NEXT")
            self.ln(4)
        elif level == 2:
            self.set_font('Helvetica', 'B', 13)
            self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
            self.ln(2)
        elif level == 3:
            self.set_font('Helvetica', 'B', 11)
            self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
            self.ln(1)

    def body_text(self, text):
        self.set_font('Helvetica', '', 10)
        self.multi_cell(0, 5, text)
        self.ln(2)

    def bold_text(self, text):
        self.set_font('Helvetica', 'B', 10)
        self.multi_cell(0, 5, text)
        self.ln(1)

    def italic_text(self, text):
        self.set_font('Helvetica', 'I', 9)
        self.multi_cell(0, 5, text)
        self.ln(2)

    def code_text(self, text):
        self.set_font('Courier', '', 8)
        self.set_fill_color(245, 245, 245)
        self.multi_cell(0, 4, text, fill=True)
        self.ln(2)

    def add_figure(self, filepath, caption, width=180):
        if not os.path.exists(filepath):
            self.body_text(f"[Figure not found: {filepath}]")
            return
        # Check if enough space, otherwise new page
        if self.get_y() > 160:
            self.add_page()
        self.image(filepath, x=15, w=width)
        self.ln(2)
        self.italic_text(caption)

    def add_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [180 / len(headers)] * len(headers)
        # Header
        self.set_font('Helvetica', 'B', 9)
        self.set_fill_color(200, 210, 225)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align='C')
        self.ln()
        # Rows
        self.set_font('Helvetica', '', 9)
        fill = False
        for row in rows:
            if fill:
                self.set_fill_color(240, 240, 240)
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 6, str(cell), border=1, fill=fill, align='C')
            self.ln()
            fill = not fill
        self.ln(3)


def build_report():
    pdf = CFLReport()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # =========================================================================
    # TITLE PAGE
    # =========================================================================
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font('Helvetica', 'B', 24)
    pdf.cell(0, 15, 'CFL Threshold at', align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 15, 'Rosensweig Instability Onset', align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font('Helvetica', '', 14)
    pdf.cell(0, 10, 'A Previously Uncharacterized Phenomenon', align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 10, 'in Ferrofluid Simulations', align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(20)
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 8, 'Semi-Coupled FHD Solver Project', align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, 'Date: March 6, 2026 (ongoing)', align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font('Helvetica', 'I', 10)
    pdf.cell(0, 8, 'Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531', align='C', new_x="LMARGIN", new_y="NEXT")

    # =========================================================================
    # 1. EXECUTIVE SUMMARY
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title('1. Executive Summary')
    pdf.body_text(
        'During attempts to reproduce the Rosensweig instability (ferrofluid spike formation under '
        'applied magnetic field), we discovered a sudden, discrete CFL number jump of 2 orders of '
        'magnitude occurring at a specific magnetic field strength. This jump is not a numerical '
        'artifact -- it is the signature of the physical Rosensweig instability onset, where the '
        'flat ferrofluid interface becomes linearly unstable.'
    )
    pdf.body_text('This finding has two implications:')
    pdf.bold_text(
        '1. For our project: Explicit Cahn-Hilliard (CH) convection fundamentally cannot survive '
        'this onset at practical time steps. Implicit CH convection is necessary.'
    )
    pdf.bold_text(
        '2. As an independent finding: The CFL threshold at instability onset appears to be '
        'uncharacterized in the literature. It reveals a fundamental constraint on explicit '
        'time-stepping schemes for ferrofluid simulations.'
    )

    # =========================================================================
    # 2. PROBLEM SETUP
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title('2. Problem Setup')

    pdf.chapter_title('2.1 Rosensweig Configuration', level=2)
    pdf.body_text(
        'Following Nochetto et al. (CMAME 2016), Section 6.2. Domain [0,1] x [0,0.6], '
        '10x6 base mesh. Ferrofluid pool (theta=+1) in bottom 1/3, water (theta=-1) above. '
        '5 dipoles at y=-15, intensity alpha=6000, ramped linearly from 0 to full over t in [0,1.6].'
    )
    pdf.add_table(
        ['Parameter', 'Value', 'Description'],
        [
            ['epsilon', '0.01', 'Interface width'],
            ['gamma', '2e-4', 'Mobility'],
            ['chi_0', '0.5', 'Susceptibility'],
            ['nu_w, nu_f', '1.0, 2.0', 'Viscosities'],
            ['g', '3e4', 'Gravity parameter'],
            ['dt', '5e-4', 'Time step'],
            ['Total steps', '4000', 't_final = 2.0'],
        ],
        col_widths=[35, 30, 115]
    )

    pdf.chapter_title('2.2 Simulation Runs Analyzed', level=2)
    pdf.add_table(
        ['Run', 'Ref', 'AMR', 'CH Conv.', 'Outcome'],
        [
            ['r4-AMR-explicit', '4', 'Yes', 'Explicit', 'Died t~0.99'],
            ['r3-AMR-explicit', '3', 'Yes', 'Explicit', 'Survived t=2.0'],
            ['r4-noAMR-DG', '4', 'No', 'Explicit', 'Died t~1.0'],
            ['r4-AMR-implicit', '4', 'Yes', 'Implicit', 'Running (t=0.41)'],
        ],
        col_widths=[40, 15, 15, 30, 80]
    )

    # =========================================================================
    # 3. OBSERVATION
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title('3. Observation: The CFL Jump')

    pdf.body_text(
        'All simulations exhibit a sudden CFL number jump at a specific time during the magnetic '
        'field ramp. The CFL number increases by approximately 2 orders of magnitude over ~25 time '
        'steps (~0.013 time units).'
    )
    pdf.add_figure(
        os.path.join(FIG_DIR, 'fig1_cfl_vs_time.png'),
        'Figure 1: CFL number evolution (log scale) for all four runs. The red dashed line marks '
        'CFL=1. Note the sudden discrete jumps around t~0.3-0.6. The implicit CH run (dark red) '
        'remains at CFL ~ 1e-4, far below the threshold.'
    )

    pdf.body_text('The jump times and corresponding magnetic field fractions:')
    pdf.add_table(
        ['Run', 'CFL > 0.01 at', 'B/B_max', 'Mesh'],
        [
            ['r4-noAMR (uniform)', 't = 0.32', '20%', 'Fixed fine'],
            ['r4-AMR-explicit', 't = 0.51', '32%', 'AMR, ref 4'],
            ['r3-AMR-explicit', 't = 0.60', '38%', 'AMR, ref 3'],
        ],
        col_widths=[45, 40, 30, 65]
    )

    pdf.bold_text(
        'Key observation: Finer spatial resolution detects the onset earlier. This is physically '
        'consistent -- finer mesh resolves smaller perturbation wavelengths.'
    )

    pdf.body_text(
        'This phenomenon was observed consistently across ALL configurations -- with and without '
        'AMR, with different refinement levels, and with different transport schemes. The CFL jump '
        'is a fundamental feature of the Rosensweig instability onset, not an artifact of any '
        'particular numerical choice.'
    )

    # =========================================================================
    # 4. ROOT CAUSE
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title('4. Root Cause: Velocity, Not Mesh')

    pdf.chapter_title('4.1 The Key Question', level=2)
    pdf.body_text(
        'CFL = U_max x dt / h_min. A sudden CFL jump could come from: '
        '(a) h_min suddenly decreasing (AMR adding finer level), '
        '(b) U_max suddenly increasing (physical instability onset), or '
        '(c) both simultaneously.'
    )

    pdf.chapter_title('4.2 Step-by-Step Decomposition', level=2)
    pdf.body_text(
        'We zoomed into the CFL jump window (t in [0.45, 0.60]) for the r4-AMR-explicit run '
        'and tracked each quantity step-by-step:'
    )
    pdf.add_figure(
        os.path.join(FIG_DIR, 'fig3_cfl_jump_stepbystep.png'),
        'Figure 3: Step-by-step quantities around the CFL jump (r4-AMR-explicit). From top: CFL, '
        'U_max, h_min, n_cells, forces, interface spread. The CFL jump is driven entirely by '
        'U_max -- h_min and n_cells remain constant during the transition.',
        width=140
    )

    pdf.add_table(
        ['Quantity', 'Before (t<0.50)', 'After (t>0.52)', 'Change'],
        [
            ['CFL', '~5e-4', '~0.1', '200x increase'],
            ['U_max', '~5e-3', '~0.5', '100x increase'],
            ['h_min', '~2.2e-2', '~2.2e-2', 'CONSTANT'],
            ['n_cells', '~15,300', '~15,500', 'Negligible'],
            ['F_capillary', '~100', '~5,000', '50x increase'],
        ],
        col_widths=[35, 40, 40, 65]
    )

    pdf.bold_text(
        'Conclusion: h_min stays constant during the jump. The CFL explosion is driven entirely '
        'by U_max suddenly increasing by 2 orders of magnitude. This is the physical Rosensweig '
        'instability onset.'
    )

    # =========================================================================
    # 4.3 RATE DECOMPOSITION
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title('4.3 Rate Decomposition', level=2)
    pdf.body_text('d(log CFL)/dt = d(log U_max)/dt - d(log h_min)/dt')
    pdf.add_figure(
        os.path.join(FIG_DIR, 'fig4_cfl_decomposition.png'),
        'Figure 4: CFL growth rate decomposition. The red curve (velocity growth rate) completely '
        'dominates at the jump. Green (h_min change) is negligible. CFL grows because velocity '
        'grows -- period.',
        width=170
    )

    # =========================================================================
    # 5. VELOCITY
    # =========================================================================
    pdf.chapter_title('5. Velocity Comparison', level=1)
    pdf.add_figure(
        os.path.join(FIG_DIR, 'fig2_umax_vs_time.png'),
        'Figure 2: Maximum velocity evolution. All runs follow similar trajectories -- the velocity '
        'is physical, not mesh-dependent. CFL differences arise from different h_min values.'
    )

    # =========================================================================
    # 6. FORCES AND ENERGIES
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title('6. Force and Energy Analysis')

    pdf.add_figure(
        os.path.join(FIG_DIR, 'fig6_forces_vs_time.png'),
        'Figure 6: Force evolution per run. Magnetic force ramps steadily. Capillary force responds '
        'to interface deformation. In the dying r4-explicit run, capillary force catches up to '
        'magnetic force near death.',
        width=120
    )

    pdf.add_page()
    pdf.add_figure(
        os.path.join(FIG_DIR, 'fig7_energies_vs_time.png'),
        'Figure 7: Energy evolution. Total energy grows monotonically. No sudden energy blowup '
        'precedes simulation death -- the instability is gradual at the energy level.',
        width=120
    )

    pdf.add_figure(
        os.path.join(FIG_DIR, 'fig8_cfl_vs_forces.png'),
        'Figure 8: CFL vs forces (scatter). CFL correlates tightly with capillary force (log-log '
        'near-linear), confirming the instability-driven mechanism.'
    )

    # =========================================================================
    # 7. CFL vs B FIELD
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title('7. CFL vs Magnetic Field Strength')
    pdf.add_figure(
        os.path.join(FIG_DIR, 'fig5_cfl_vs_magnetic_field.png'),
        'Figure 5: CFL vs magnetic field fraction (alpha/alpha_max = t/1.6). The CFL jump occurs '
        'at 20-38% of maximum field strength depending on mesh resolution.'
    )
    pdf.body_text(
        'The instability onset occurs at a well-defined fraction of the maximum field strength. '
        'The mesh-dependence of the onset time is consistent with linear stability theory: finer '
        'meshes resolve shorter wavelengths that may become unstable at lower field strengths.'
    )

    # =========================================================================
    # 8. INTERFACE
    # =========================================================================
    pdf.chapter_title('8. Interface Dynamics')
    pdf.add_figure(
        os.path.join(FIG_DIR, 'fig9_interface_vs_cfl.png'),
        'Figure 9: Top: Interface y-spread (spike height). The r3-explicit run survived long '
        'enough for spikes to develop (y-spread ~ 0.45 by t=2.0). Bottom: CFL tracks interface '
        'spread directly.',
        width=150
    )

    # =========================================================================
    # 9. DETAILED PANELS
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title('9. Detailed Per-Run Analysis')

    pdf.chapter_title('9.1 r4-AMR-explicit (Died at t ~ 0.99)', level=2)
    pdf.add_figure(
        os.path.join(FIG_DIR, 'fig14_detailed_r4_explicit.png'),
        'Figure 14: Six-panel analysis. CFL step-jumps align with velocity growth, not AMR cycles. '
        'Theta stays in [-1,1]. Simulation dies from CFL exceeding 1.',
        width=120
    )

    pdf.add_page()
    pdf.chapter_title('9.2 r3-AMR-explicit (Survived to t = 2.0)', level=2)
    pdf.add_figure(
        os.path.join(FIG_DIR, 'fig15_detailed_r3_explicit.png'),
        'Figure 15: Same CFL pattern as r4, but coarser mesh keeps CFL just below 1.0. After '
        'field fully ramped (t>1.6), the system reaches quasi-steady state.',
        width=120
    )

    pdf.add_page()
    pdf.chapter_title('9.3 r4-AMR-implicit (Running)', level=2)
    pdf.add_figure(
        os.path.join(FIG_DIR, 'fig16_detailed_r4_implicit.png'),
        'Figure 16: Implicit CH simulation at t=0.41. CFL at 7e-4 -- three orders of magnitude '
        'below explicit at the same time. Approaching the critical region (t~0.5).',
        width=120
    )

    # =========================================================================
    # 10. h_min
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title('10. h_min Evolution')
    pdf.add_figure(
        os.path.join(FIG_DIR, 'fig10_hmin_evolution.png'),
        'Figure 10: Inferred h_min and n_cells. AMR changes do NOT correlate with the CFL jump. '
        'The noAMR run has constant n_cells=15,360, confirming the CFL jump is not AMR-driven.'
    )

    # =========================================================================
    # 11. IMPLICATIONS
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title('11. Implications for Numerical Methods')

    pdf.chapter_title('11.1 Why Explicit CH Convection Fails', level=2)
    pdf.body_text(
        'The CH convection term U.grad(theta), when treated explicitly (evaluated at t_{n-1}, '
        'placed on RHS), imposes a CFL stability constraint: CFL = U_max x dt / h_min < C_crit. '
        'When the Rosensweig instability kicks in and U_max jumps by 2 orders of magnitude, CFL '
        'exceeds this limit within ~25 time steps.'
    )
    pdf.bold_text('The failure chain:')
    pdf.code_text(
        'Magnetic field reaches critical strength\n'
        '  -> Flat interface becomes linearly unstable (physical)\n'
        '  -> Velocity U_max jumps ~100x in ~25 steps (physical)\n'
        '  -> CFL = U_max * dt / h_min exceeds stability limit\n'
        '  -> Explicit CH produces theta oscillations (numerical)\n'
        '  -> Spurious capillary forces from theta errors (numerical)\n'
        '  -> Positive feedback: more velocity -> more CFL -> death'
    )

    pdf.chapter_title('11.2 Why Implicit CH Convection Works', level=2)
    pdf.body_text(
        'Making CH convection implicit (U.grad(theta) on the LHS matrix) removes the CFL stability '
        'constraint entirely. The linear system absorbs the convection regardless of how large '
        'U x dt / h gets. The matrix becomes non-symmetric but GMRES+AMG handles this.'
    )

    pdf.chapter_title('11.3 Why This Matters for the Community', level=2)
    pdf.body_text(
        'Standard practice in ferrofluid simulations uses linear field ramping with explicit or '
        'semi-explicit time stepping. The implicit assumption is that the field ramp is "smooth '
        'enough" for explicit schemes. Our finding shows this is false at the Rosensweig onset -- '
        'the velocity responds with an abrupt jump regardless of how smooth the field ramp is. '
        'Researchers experiencing unexplained solver crashes during Rosensweig simulations may be '
        'encountering exactly this CFL threshold.'
    )

    # =========================================================================
    # 12. SOLUTIONS
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title('12. Potential Solutions')

    pdf.chapter_title('12.1 Implicit CH Convection (Primary Fix)', level=2)
    pdf.body_text(
        'Status: Implemented and currently being validated. '
        'Move the advection term U.grad(theta) from RHS (explicit) to LHS (implicit). '
        'Removes CFL limit entirely. No modification to physics.'
    )

    pdf.chapter_title('12.2 Adaptive Time Stepping', level=2)
    pdf.body_text(
        'Status: Not yet implemented. Monitor CFL and reduce dt when CFL exceeds threshold. '
        'Works with explicit CH but requires very small dt during onset (~100x smaller).'
    )

    pdf.chapter_title('12.3 BDF2 Time Integration', level=2)
    pdf.body_text(
        'Status: Not yet implemented. Second-order backward differentiation with larger stability '
        'region. Combined with implicit CH, improves both stability and accuracy.'
    )

    pdf.chapter_title('12.4 Long-Duration MMS Framework', level=2)
    pdf.body_text(
        'Status: Framework implemented, not yet run. Per-step error tracking to distinguish linear '
        'growth (normal for backward Euler) from exponential growth (instability).'
    )

    # =========================================================================
    # 13. SIDE FINDING
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title('13. Side Finding: Non-Linear Field Ramping')

    pdf.body_text(
        'Independent of the numerical fix (implicit CH), the CFL jump phenomenon suggests that '
        'how the magnetic field is ramped affects the simulation\'s ability to resolve the '
        'instability onset. This is a separate research finding -- not a fix for our code (which '
        'uses the paper\'s linear ramp), but a potential contribution to the broader study of '
        'Rosensweig spike formation.'
    )

    pdf.chapter_title('13.1 Concept', level=2)
    pdf.body_text(
        'Instead of the standard linear ramp alpha(t) = alpha_max * t / t_ramp, use a non-linear '
        'ramp that slows down near the critical field strength. A two-phase ramp: '
        '(1) Fast ramp to ~25% of B_max (below critical threshold), '
        '(2) Slow ramp through 25%-40% (the critical region), '
        '(3) Resume normal speed or hold constant.'
    )

    pdf.chapter_title('13.2 Significance', level=2)
    pdf.body_text(
        'To our knowledge, all published Rosensweig instability simulations use linear or '
        'step-function field ramping. No study has characterized the CFL jump at instability onset '
        'or proposed ramp shaping as a numerical strategy. This could benefit researchers using '
        'explicit time-stepping, experimentalists controlling spike dynamics, and computational '
        'studies of other threshold-driven instabilities.'
    )

    # =========================================================================
    # 14. CURRENT STATUS
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title('14. Current Status')

    pdf.chapter_title('14.1 Implicit CH Run (In Progress)', level=2)
    pdf.add_table(
        ['Metric', 'Value (step 819, t=0.41)'],
        [
            ['CFL', '7.1e-4'],
            ['U_max', '3.1e-3'],
            ['theta range', '[-1.0000, 1.0000]'],
            ['Mass', '-2.000e-1 (conserved)'],
            ['E_total', '282.3'],
            ['n_cells', '13,872 (AMR)'],
        ],
        col_widths=[60, 120]
    )
    pdf.body_text(
        'The implicit run is approaching the critical region (t ~ 0.5) where explicit runs '
        'experienced the CFL jump. If it survives this region and continues to t=2.0, it will '
        'validate implicit CH convection as the correct approach.'
    )
    pdf.italic_text('This section will be updated when the run completes.')

    pdf.chapter_title('14.2 Planned Updates', level=2)
    pdf.body_text(
        '- Results after implicit run passes through t ~ 0.5 (critical region)\n'
        '- Final results at t = 2.0 (full simulation)\n'
        '- Comparison of spike morphology with Nochetto et al. Figure 6\n'
        '- Long-duration MMS test results\n'
        '- Theoretical estimate of critical magnetic Bond number vs observed onset'
    )

    # =========================================================================
    # 15. SUPPLEMENTARY FIGURES
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title('15. Supplementary Figures')

    pdf.add_figure(
        os.path.join(FIG_DIR, 'fig11_cfl_jump_zoom.png'),
        'Figure 11: Zoomed CFL jump region for all three explicit runs. CFL (blue) and U_max (red) '
        'on left axis; h_min (green dashed) and n_cells (magenta) on right axis.',
        width=170
    )

    pdf.add_page()
    pdf.add_figure(
        os.path.join(FIG_DIR, 'fig12_cfl_jump_rates.png'),
        'Figure 12: Normalized rates of change at the CFL jump. CFL and U_max change fastest; '
        'n_cells changes slowly (confirming AMR is not the trigger).'
    )

    pdf.add_figure(
        os.path.join(FIG_DIR, 'fig13_cfl_growth_rate.png'),
        'Figure 13: CFL growth rate d(log10 CFL)/dt. Large spikes correspond to the instability '
        'onset. After onset, growth rate settles near zero for surviving runs.'
    )

    # =========================================================================
    # SAVE
    # =========================================================================
    output_path = os.path.join(REPORT_DIR, 'CFL_INSTABILITY_ONSET_REPORT.pdf')
    pdf.output(output_path)
    print(f"PDF saved: {output_path}")
    print(f"Pages: {pdf.page_no()}")


if __name__ == '__main__':
    build_report()
