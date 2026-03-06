#!/usr/bin/env python3
"""
Generate a PDF document for the C.0 Parametric Study of Rosensweig Instability.

Uses a cursor-based layout to prevent text/equation/line overlaps.
Split across 9 pages with generous spacing.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import textwrap

# ============================================================================
# Configuration
# ============================================================================
OUTPUT = "parametric_study_DOE.pdf"

TITLE_SIZE   = 16
SECTION_SIZE = 12.5
SUBSEC_SIZE  = 11
BODY_SIZE    = 9
EQ_SIZE      = 10.5
SMALL_SIZE   = 8
TABLE_FONT   = 8

COLOR_TITLE    = "#1a1a2e"
COLOR_SECTION  = "#16213e"
COLOR_SUBSEC   = "#0f3460"
COLOR_BODY     = "#222222"
COLOR_EQ_BG    = "#f0f4ff"
COLOR_HIGHLIGHT = "#e63946"
COLOR_BASELINE = "#2a9d8f"

LEFT   = 0.07
RIGHT  = 0.93
BOTTOM = 0.06
TOP    = 0.92
PAGE_W = RIGHT - LEFT

# ============================================================================
# Cursor-based layout helpers
# ============================================================================
# All heights are in axes-fraction units (0 to 1).
# Font size in pts → axes height: h = (size_pt * 1.3 / 72) / (PAGE_H_inches)
# PAGE_H_inches = (TOP - BOTTOM) * 11 = 0.86 * 11 = 9.46
# So 12.5pt → 0.024,  11pt → 0.021,  10.5pt → 0.020,  9pt → 0.017,  8pt → 0.015

H_SECTION = 0.026     # height of section text
H_SUBSEC  = 0.022     # height of subsec text
H_BODY    = 0.018     # height of body text
H_SMALL   = 0.016     # height of small text
H_EQ_BOX  = 0.035     # height of equation shaded box
GAP_S     = 0.008     # small gap
GAP_M     = 0.014     # medium gap
GAP_L     = 0.022     # large gap


class Cursor:
    """Tracks vertical position, descending from top."""
    def __init__(self, start=0.98):
        self.y = start

    def advance(self, amount):
        self.y -= amount

    @property
    def pos(self):
        return self.y


def new_page(pdf, figsize=(8.5, 11)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([LEFT, BOTTOM, PAGE_W, TOP - BOTTOM])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def put(ax, y, text, size=BODY_SIZE, color=COLOR_BODY, weight="normal",
        ha="left", x=0.0, style="normal", fontfamily="serif"):
    ax.text(x, y, text, fontsize=size, color=color, fontweight=weight,
            ha=ha, va="top", style=style, fontfamily=fontfamily,
            transform=ax.transAxes)


def add_title(ax, cur, text, subtitle=None, credit=None):
    """Page title with optional subtitle and credit line."""
    put(ax, cur.pos, text, size=TITLE_SIZE, color=COLOR_TITLE,
        weight="bold", x=0.5, ha="center")
    cur.advance(H_SECTION + GAP_S)
    if subtitle:
        put(ax, cur.pos, subtitle, size=SUBSEC_SIZE, color="#555555",
            x=0.5, ha="center", style="italic")
        cur.advance(H_SUBSEC + GAP_S)
    if credit:
        put(ax, cur.pos, credit, size=SMALL_SIZE, color="#777777",
            x=0.5, ha="center")
        cur.advance(H_SMALL + GAP_S)
    # Decorative line
    ax.plot([0.15, 0.85], [cur.pos + 0.002, cur.pos + 0.002],
            color=COLOR_TITLE, linewidth=1.2, transform=ax.transAxes)
    cur.advance(GAP_M)


def add_section(ax, cur, text):
    """Section header with underline below the text."""
    put(ax, cur.pos, text, size=SECTION_SIZE, color=COLOR_SECTION, weight="bold")
    cur.advance(H_SECTION + GAP_S)
    # Line sits below the text
    ax.plot([0, 0.98], [cur.pos + 0.003, cur.pos + 0.003],
            color=COLOR_SECTION, linewidth=0.8, transform=ax.transAxes,
            clip_on=False)
    cur.advance(GAP_M)


def add_subsec(ax, cur, text):
    """Subsection header."""
    put(ax, cur.pos, text, size=SUBSEC_SIZE, color=COLOR_SUBSEC, weight="bold")
    cur.advance(H_SUBSEC + GAP_M)


def add_body(ax, cur, text, **kw):
    sz = kw.pop("size", BODY_SIZE)
    put(ax, cur.pos, text, size=sz, **kw)
    h = H_SMALL if sz <= SMALL_SIZE else H_BODY
    cur.advance(h + GAP_S)


def add_body_wrap(ax, cur, text, width=100, **kw):
    lines = textwrap.wrap(text, width=width)
    for line in lines:
        add_body(ax, cur, line, **kw)


def add_eq(ax, cur, eq_text, size=EQ_SIZE, align="center", x=0.05):
    """Equation in a shaded box, with proper spacing."""
    box_top = cur.pos
    box_bot = box_top - H_EQ_BOX
    ax.axhspan(box_bot, box_top, xmin=0.03, xmax=0.97,
               color=COLOR_EQ_BG, zorder=0)
    text_y = (box_top + box_bot) / 2
    ha = align
    tx = 0.5 if align == "center" else x
    ax.text(tx, text_y, eq_text, fontsize=size, color=COLOR_BODY,
            ha=ha, va="center", fontfamily="serif", transform=ax.transAxes)
    cur.advance(H_EQ_BOX + GAP_M)


def add_table(ax, cur, headers, rows, col_x, header_color="#2b2d42",
              baseline_row=None, font_size=TABLE_FONT, row_h=0.026):
    y = cur.pos
    for j, h in enumerate(headers):
        ax.text(col_x[j], y, h, fontsize=font_size, fontweight="bold",
                color="white", ha="left", va="top", transform=ax.transAxes,
                bbox=dict(boxstyle="square,pad=0.15", fc=header_color, ec="none"))
    y -= 0.030
    ax.plot([col_x[0] - 0.01, 0.97], [y + 0.006, y + 0.006],
            color="#cccccc", linewidth=0.5, transform=ax.transAxes)

    for i, row in enumerate(rows):
        is_base = (baseline_row is not None and i == baseline_row)
        bg = "#e8f5e9" if is_base else None
        fw = "bold" if is_base else "normal"
        clr = COLOR_BASELINE if is_base else COLOR_BODY
        for j, val in enumerate(row):
            kw = dict(fontsize=font_size, fontweight=fw, color=clr,
                      ha="left", va="top", transform=ax.transAxes)
            if bg:
                kw["bbox"] = dict(boxstyle="square,pad=0.1", fc=bg, ec="none")
            ax.text(col_x[j], y, str(val), **kw)
        y -= row_h
        if i < len(rows) - 1:
            ax.plot([col_x[0] - 0.01, 0.97], [y + 0.006, y + 0.006],
                    color="#eeeeee", linewidth=0.3, transform=ax.transAxes)

    cur.y = y
    cur.advance(GAP_S)


def add_matrix(ax, cur, title, row_label, col_label, row_vals, col_vals,
               baseline_r=None, baseline_c=None, font_size=SMALL_SIZE):
    add_subsec(ax, cur, title)

    x0 = 0.22
    dx = 0.12
    dy_row = 0.026

    # Col header label
    ax.text(0.5, cur.pos, col_label, fontsize=font_size, fontweight="bold",
            color=COLOR_SUBSEC, ha="center", va="top", transform=ax.transAxes)
    cur.advance(H_SMALL + GAP_S)

    # Col values
    for j, cv in enumerate(col_vals):
        fw = "bold" if baseline_c is not None and j == baseline_c else "normal"
        clr = COLOR_BASELINE if baseline_c is not None and j == baseline_c else COLOR_BODY
        ax.text(x0 + j * dx, cur.pos, str(cv), fontsize=font_size, fontweight=fw,
                color=clr, ha="center", va="top", transform=ax.transAxes)
    cur.advance(H_SMALL + GAP_M)

    # Row label
    rows_height = len(row_vals) * dy_row
    ax.text(0.02, cur.pos - rows_height / 2, row_label,
            fontsize=font_size, fontweight="bold", color=COLOR_SUBSEC,
            ha="left", va="center", rotation=0, transform=ax.transAxes)

    for i, rv in enumerate(row_vals):
        fw = "bold" if baseline_r is not None and i == baseline_r else "normal"
        clr = COLOR_BASELINE if baseline_r is not None and i == baseline_r else COLOR_BODY
        ax.text(0.14, cur.pos, str(rv), fontsize=font_size, fontweight=fw,
                color=clr, ha="right", va="top", transform=ax.transAxes)
        for j in range(len(col_vals)):
            is_base = (baseline_r is not None and i == baseline_r and
                       baseline_c is not None and j == baseline_c)
            fc = COLOR_BASELINE if is_base else "white"
            ec = COLOR_BASELINE if is_base else "#888888"
            ms = 5 if is_base else 4
            ax.plot(x0 + j * dx, cur.pos - 0.006, "o", markersize=ms,
                    markerfacecolor=fc, markeredgecolor=ec, markeredgewidth=0.8,
                    transform=ax.transAxes)
        cur.advance(dy_row)

    count = len(row_vals) * len(col_vals)
    ax.text(0.95, cur.pos + 0.012, f"{count} runs", fontsize=SMALL_SIZE,
            color="#666666", ha="right", va="top", transform=ax.transAxes)
    cur.advance(GAP_S)


# ============================================================================
# Build the PDF
# ============================================================================
with PdfPages(OUTPUT) as pdf:

    # ====================================================================
    # PAGE 1 — Governing Equations
    # ====================================================================
    fig, ax = new_page(pdf)
    c = Cursor(0.98)

    add_title(ax, c, "C.0 — Parametric Study of Rosensweig Instability",
              subtitle="Design of Experiments & Governing Equations",
              credit="Zhang, He & Yang, SIAM J. Sci. Comput. 43(1), B167-B193, 2021")

    add_section(ax, c, "1. Governing Equations (4-System Decoupled Scheme)")

    # 1.1 CH
    add_subsec(ax, c, "1.1  Cahn-Hilliard (Phase Field)")
    add_eq(ax, c,
           r"$\frac{\theta^n - \theta^{n-1}}{\delta t} "
           r"= M \, \Delta \psi^n "
           r"+ \nabla \cdot (u^{n-1} \theta^{n-1})$"
           r"$\qquad\qquad$"
           r"$\psi^n = -\lambda\,\varepsilon\,\Delta\theta^n "
           r"+ \frac{\lambda}{\varepsilon}\,f(\theta^{n-1}) "
           r"+ S_1(\theta^n - \theta^{n-1})$")
    add_body(ax, c,
         r"Parameters: $\lambda$ (surface tension), "
         r"$\varepsilon$ (interface width), "
         r"$M$ (mobility), $S_1$ (SAV stabilization)",
         size=SMALL_SIZE, color="#444444")
    add_body(ax, c,
         r"$f(\theta) = (\theta^3 - \theta)/4$;  "
         r"$\sigma_{\mathrm{eff}} = \lambda\sqrt{2}/3$  (effective surface tension)",
         size=SMALL_SIZE, color="#444444")
    c.advance(GAP_S)

    # 1.2 NS
    add_subsec(ax, c, "1.2  Navier-Stokes (Momentum + Incompressibility)")
    add_eq(ax, c,
           r"$\frac{\tilde{u}^n - u^{n-1}}{\delta t} "
           r"+ B(u^{n-1};\, \tilde{u}^n) "
           r"- \nabla\cdot[\nu(\theta^n)\, D(\tilde{u}^n)] "
           r"+ \nabla p^n "
           r"= \rho(\theta^n)\, g "
           r"+ \theta^{n-1}\nabla\psi^n "
           r"+ \mu_0 F_K$")
    add_body(ax, c,
         r"$B(u;\,v) = (u\cdot\nabla)v + \frac{1}{2}(\nabla\cdot u)\,v$  "
         r"(skew-symmetric);   "
         r"$\nabla\cdot \tilde{u}^n = 0$  (incompressibility)",
         size=SMALL_SIZE, color="#444444")
    add_body(ax, c,
         r"$\nu(\theta) = \nu_w + (\nu_f - \nu_w)\,H(\theta/\varepsilon)$;  "
         r"$\rho(\theta) = 1 + r\,H(\theta/\varepsilon)$;  "
         r"$F_K = (m\cdot\nabla)h + \frac{1}{2}(\nabla\cdot m)\,h$",
         size=SMALL_SIZE, color="#444444")
    c.advance(GAP_S)

    # 1.3 Poisson
    add_subsec(ax, c, "1.3  Magnetostatic Poisson")
    add_eq(ax, c,
           r"$\left((1 + \chi(\theta^n))\,\nabla\varphi^n,\, \nabla X\right) "
           r"= (h_a,\, \nabla X)$"
           r"$\qquad$"
           r"$h = \nabla\varphi$  (total field)")
    add_body(ax, c,
         r"$\chi(\theta) = \chi_0\,H(\theta/\varepsilon)$  "
         r"(phase-dependent susceptibility);  "
         r"$h_a$ = applied field from dipoles with intensity $\alpha$",
         size=SMALL_SIZE, color="#444444")
    c.advance(GAP_S)

    # 1.4 Magnetization
    add_subsec(ax, c, "1.4  Magnetization Transport (DG)")
    add_eq(ax, c,
           r"$\left(\frac{1}{\delta t} + \frac{1}{\tau}\right)(m^n, z) "
           r"+ B_h^m(u;\, m^n,\, z) "
           r"= \frac{1}{\tau}\,(\chi\, h^n,\, z) "
           r"+ \frac{1}{\delta t}\,(m^{n-1},\, z) "
           r"+ \frac{1}{2}(\nabla\times u \times m^{n-1},\, z)$")
    add_body(ax, c,
         r"$B_h^m$: DG upwind transport;  "
         r"$\tau$ = relaxation time;  "
         r"$\beta\, m\times(m\times h)$ (Landau-Lifshitz, omitted for brevity)",
         size=SMALL_SIZE, color="#444444")

    pdf.savefig(fig); plt.close(fig)

    # ====================================================================
    # PAGE 2 — Analytical Predictions + Baseline
    # ====================================================================
    fig, ax = new_page(pdf)
    c = Cursor(0.98)

    add_title(ax, c, "Analytical Predictions & Baseline")

    add_section(ax, c, "2. Analytical Predictions (Linear Stability Theory)")

    add_subsec(ax, c, "2.1  Dispersion Relation")
    add_eq(ax, c,
           r"$\omega^2(k) = -\Delta\rho\, g\, k "
           r"\;-\; \sigma\, k^3 "
           r"\;+\; \mu_0\, f(\chi_0)\, H_0^2\, k^2$"
           r"$\qquad$"
           r"$f(\chi_0) = \frac{\chi_0^2}{2 + \chi_0}$")

    add_subsec(ax, c, "2.2  Critical Conditions (onset of instability)")
    add_eq(ax, c,
           r"$k_c = \sqrt{\frac{\Delta\rho\, g}{\sigma}}$"
           r"$\qquad\quad$"
           r"$\lambda_c = \frac{2\pi}{k_c} = 2\pi\sqrt{\frac{\sigma}{\Delta\rho\, g}}$"
           r"$\qquad\quad$"
           r"$H_c^2 = \frac{2(2+\chi_0)}{\mu_0\,\chi_0^2}"
           r"\sqrt{\Delta\rho\, g\, \sigma}$")

    add_subsec(ax, c, "2.3  Parameter-to-Observable Map")

    map_lines = [
        (r"$\chi_0 \uparrow$",            "spike count unchanged, spikes taller, onset earlier"),
        (r"$\lambda \uparrow$",            "fewer spikes (wider spacing), onset later"),
        (r"$L_x \uparrow$",               r"more spikes (linear in $L_x$), onset unchanged"),
        (r"$\alpha_{\max} \uparrow$",      "stronger field, supercritical, taller spikes"),
        (r"$y_{\mathrm{int}} \uparrow$",   "deeper pool, closer to semi-infinite theory"),
        (r"$\varepsilon \uparrow$",        "wider interface, weaker capillary, easier onset"),
    ]
    for sym, desc in map_lines:
        ax.text(0.06, c.pos, sym, fontsize=SMALL_SIZE, color=COLOR_HIGHLIGHT,
                fontweight="bold", ha="left", va="top", transform=ax.transAxes)
        ax.text(0.24, c.pos, desc, fontsize=SMALL_SIZE, color=COLOR_BODY,
                ha="left", va="top", transform=ax.transAxes)
        c.advance(0.028)

    c.advance(GAP_L)

    # Baseline box
    box_h = 0.070
    ax.add_patch(plt.Rectangle((0.03, c.pos - box_h), 0.94, box_h,
                                transform=ax.transAxes, fill=True,
                                facecolor="#f8f9fa", edgecolor="#cccccc",
                                linewidth=0.5))
    put(ax, c.pos - 0.008, "Baseline Parameters (Zhang Section 4.3)",
        size=SUBSEC_SIZE, color=COLOR_SUBSEC, weight="bold", x=0.05)
    put(ax, c.pos - 0.032,
        r"$\chi_0=0.5$,  $\alpha_{\max}=8000$,  $\lambda=1.0$,  "
        r"$\varepsilon=5\times10^{-3}$,  $y_{\mathrm{int}}=0.2$,  "
        r"$\nu_f=2$, $\nu_w=1$,  $r=0.1$,  $g=6\times10^4$",
        size=SMALL_SIZE, x=0.05)
    put(ax, c.pos - 0.052,
        r"Domain $[0,1]\times[0,0.6]$,  5 dipoles at $y=-15$,  "
        r"$\delta t=10^{-3}$,  2000 steps,  refinement $=4$  "
        r"($\approx 128\times80$ cells)",
        size=SMALL_SIZE, x=0.05)
    c.advance(box_h + GAP_L)

    pdf.savefig(fig); plt.close(fig)

    # ====================================================================
    # PAGE 3 — Tier 1: chi_0 and alpha_max
    # ====================================================================
    fig, ax = new_page(pdf)
    c = Cursor(0.98)

    add_title(ax, c, "Tier 1 — Single-Parameter Screening",
              subtitle="31 runs  |  One parameter varied at a time, all others at baseline")

    # 1A-1: chi_0
    add_section(ax, c, r"1A-1.  Susceptibility  $\chi_0$")
    add_body(ax, c,
         r"Appears in:  Poisson $(1+\chi)\nabla\varphi$,  "
         r"Magnetization $m_{eq}=\chi\, h$,  "
         r"Kelvin force $\sim\chi_0^2/(2+\chi_0)$",
         size=SMALL_SIZE, color="#444444", style="italic")
    add_body_wrap(ax, c,
        "Increasing chi_0 strengthens the magnetic body force without changing the "
        "capillary length, so instability onset occurs at a lower applied field. "
        "Spike count should remain constant while spike height grows. "
        "This sweep maps the uncharted chi_0 > 1 regime.",
        width=100)
    c.advance(GAP_S)
    add_table(ax, c,
              ["ID", "chi_0", "H_c ratio", "Expected effect"],
              [["R01", "0.10", "2.21x baseline", "Sub-threshold: no spikes or very weak"],
               ["R02", "0.25", "1.36x baseline", "Delayed onset, shorter spikes"],
               ["R03", "0.50", "1.00 (baseline)", "Reference: ~5 spikes"],
               ["R04", "0.75", "0.81x baseline", "Earlier onset, taller spikes"],
               ["R05", "1.00", "0.67x baseline", "Significantly earlier onset"],
               ["R06", "1.50", "0.50x baseline", "Strong Kelvin force, tall spikes"],
               ["R07", "2.00", "0.40x baseline", "Highest force; possible nonlinear regime"]],
              [0.05, 0.14, 0.30, 0.52],
              baseline_row=2)
    c.advance(GAP_L)

    # 1A-2: alpha_max
    add_section(ax, c, r"1A-2.  Field Strength  $\alpha_{\max}$")
    add_body(ax, c,
         r"Appears in:  Applied field $h_a = \alpha(t)\cdot$dipole formula;  "
         r"ramp $\alpha(t) = \min(\alpha_{\max}/1.6 \cdot t,\; \alpha_{\max})$",
         size=SMALL_SIZE, color="#444444", style="italic")
    add_body_wrap(ax, c,
        "The applied field intensity directly controls the magnetic Bond number. "
        "Below a critical alpha the interface stays flat; above it spikes form. "
        "This sweep identifies the critical threshold and maps the supercritical "
        "regime where spike height grows with field strength.",
        width=100)
    c.advance(GAP_S)
    add_table(ax, c,
              ["ID", "alpha_max", "Ramp slope", "Expected effect"],
              [["R08", "5000", "3125", "Possibly sub-critical: flat or very weak spikes"],
               ["R09", "7000", "4375", "Near onset: marginal instability"],
               ["R10", "8000", "5000 (baseline)", "Reference: clear spikes"],
               ["R11", "9000", "5625", "Above baseline: taller spikes"],
               ["R12", "10000", "6250", "Well above onset: tallest spikes"]],
              [0.05, 0.18, 0.36, 0.56],
              baseline_row=2)

    pdf.savefig(fig); plt.close(fig)

    # ====================================================================
    # PAGE 4 — Tier 1: lambda and y_interface
    # ====================================================================
    fig, ax = new_page(pdf)
    c = Cursor(0.98)

    add_title(ax, c, "Tier 1 — Single-Parameter Screening (cont.)")

    # 1A-3: lambda
    add_section(ax, c, r"1A-3.  Surface Tension  $\lambda$")
    add_body(ax, c,
         r"Appears in:  CH $\psi = -\lambda\varepsilon\Delta\theta + (\lambda/\varepsilon)f$;  "
         r"NS capillary force $\theta\nabla\psi$;  "
         r"$\sigma = \lambda\sqrt{2}/3$",
         size=SMALL_SIZE, color="#444444", style="italic")
    add_body_wrap(ax, c,
        "Surface tension opposes interface deformation. Higher lambda raises H_c "
        "(harder to trigger instability) and increases the capillary length "
        "(fewer, wider-spaced spikes). This is the primary control on spike count "
        "and spacing, independent of chi_0.",
        width=100)
    c.advance(GAP_S)
    add_table(ax, c,
              ["ID", "lambda", "N/N_base", "H_c/H_c,base", "Expected effect"],
              [["R13", "0.50", "1.19x", "0.92x", "More spikes, easier onset"],
               ["R14", "0.75", "1.07x", "0.96x", "Slightly more spikes"],
               ["R15", "0.90", "1.02x", "0.99x", "Near baseline"],
               ["R16", "1.00", "1.00", "1.00", "Reference (baseline)"],
               ["R17", "1.10", "0.98x", "1.01x", "Near baseline"],
               ["R18", "1.25", "0.95x", "1.03x", "Slightly fewer spikes"],
               ["R19", "1.50", "0.91x", "1.06x", "Fewer spikes, harder onset"],
               ["R20", "2.00", "0.84x", "1.10x", "Noticeably fewer, wider spikes"]],
              [0.05, 0.16, 0.30, 0.44, 0.60],
              baseline_row=3)
    c.advance(GAP_L)

    # 1A-4: y_interface
    add_section(ax, c, r"1A-4.  Pool Depth  $y_{\mathrm{interface}}$")
    add_body(ax, c,
         r"Appears in:  Initial condition $\theta_0(y) = +1$ if $y \leq y_{\mathrm{int}}$, "
         r"$-1$ otherwise;  controls ferrofluid volume fraction",
         size=SMALL_SIZE, color="#444444", style="italic")
    add_body_wrap(ax, c,
        "Linear stability theory assumes a semi-infinite pool. Finite depth "
        "modifies the dispersion relation through a tanh(k*d) factor that "
        "suppresses long-wavelength modes. Shallow pools (y_int=0.05) should "
        "raise H_c and may suppress the instability entirely.",
        width=100)
    c.advance(GAP_S)
    add_table(ax, c,
              ["ID", "y_interface", "Depth/lambda_c", "Expected effect"],
              [["R21", "0.05", "~0.25", "Very shallow: instability may be suppressed"],
               ["R22", "0.10", "~0.50", "Finite-depth correction; fewer modes"],
               ["R23", "0.20", "1.00 (baseline)", "Reference: standard onset"],
               ["R24", "0.40", "~2.00", "Approaches semi-infinite; minimal change"]],
              [0.05, 0.20, 0.40, 0.60],
              baseline_row=2)

    pdf.savefig(fig); plt.close(fig)

    # ====================================================================
    # PAGE 5 — Tier 1: epsilon + geometric + demagnetizing
    # ====================================================================
    fig, ax = new_page(pdf)
    c = Cursor(0.98)

    add_title(ax, c, "Tier 1 — Single-Parameter Screening (cont.)")

    # 1A-5: epsilon
    add_section(ax, c, r"1A-5.  Interface Width  $\varepsilon$")
    add_body(ax, c,
         r"Appears in:  CH gradient energy $\lambda\varepsilon|\nabla\theta|^2/2$,  "
         r"bulk energy $(\lambda/\varepsilon)F(\theta)$,  "
         r"$\chi(\theta) = \chi_0\,H(\theta/\varepsilon)$,  $\nu(\theta)$",
         size=SMALL_SIZE, color="#444444", style="italic")
    add_body_wrap(ax, c,
        "Epsilon controls the diffuse interface thickness (~5.66*epsilon). "
        "The effective surface tension is sigma = lambda*sqrt(2)/3, which does "
        "NOT depend on epsilon. However, epsilon affects the interfacial "
        "structure, the smoothness of material property transitions, "
        "and the numerical resolution requirement (need ~3 cells across interface).",
        width=100)
    c.advance(GAP_S)
    add_table(ax, c,
              ["ID", "epsilon", "Width (5.66*eps)", "Cells across (r=4)", "Expected effect"],
              [["R25", "3e-3", "0.017", "~2.2", "Marginally resolved; sharpest interface"],
               ["R26", "5e-3", "0.028", "~3.6 (baseline)", "Reference"],
               ["R27", "7e-3", "0.040", "~5.1", "Well-resolved; wider transition"],
               ["R28", "1e-2", "0.057", "~7.3", "Very wide interface; diffuse spikes"]],
              [0.05, 0.17, 0.32, 0.50, 0.68],
              baseline_row=1)
    c.advance(GAP_L)

    # 1B: Geometric
    add_section(ax, c, "1B.  Geometric Parameters")

    add_subsec(ax, c, r"1B-1.  Domain Width  $L_x$")
    add_body(ax, c,
         r"Appears in:  Quantized wavenumbers $k_n = 2\pi n / L_x$;  "
         r"spike count $N \approx L_x / \lambda_c$  (linear scaling)",
         size=SMALL_SIZE, color="#444444", style="italic")
    add_body_wrap(ax, c,
        "The domain width selects which Fourier modes are allowed. Spike count "
        "should scale linearly with L_x. Below a critical width L_x < lambda_c, "
        "no unstable mode fits and spikes are completely suppressed.",
        width=100)
    c.advance(GAP_S)
    add_table(ax, c,
              ["ID", "L_x", "N (predicted)", "Expected effect"],
              [["R29a", "0.50", "~2-3", "Mode suppression possible"],
               ["R29b", "0.75", "~3-4", "Mode competition"],
               ["R29c", "1.00", "~5 (baseline)", "Reference"],
               ["R29d", "1.50", "~7-8", "Proportional increase"],
               ["R29e", "2.00", "~10", "Verify N linear in L_x"]],
              [0.05, 0.18, 0.35, 0.58],
              baseline_row=2)
    c.advance(GAP_L)

    add_subsec(ax, c, r"1B-2.  Domain Height  $L_y$")
    add_body(ax, c,
         "Appears in:  Upper boundary distance from interface. "
         "Affects air-phase volume and far-field boundary influence.",
         size=SMALL_SIZE, color="#444444", style="italic")
    add_body_wrap(ax, c,
        "Should have minimal effect as long as L_y >> y_interface. "
        "Tests whether the upper boundary influences spike growth.",
        width=100)
    c.advance(GAP_S)
    add_table(ax, c,
              ["ID", "L_y", "Air gap", "Expected effect"],
              [["R30a", "0.40", "0.20", "Tight: boundary may clip spikes"],
               ["R30b", "0.60", "0.40 (baseline)", "Reference"],
               ["R30c", "1.00", "0.80", "Generous: no boundary effect"]],
              [0.05, 0.18, 0.35, 0.58],
              baseline_row=1)

    pdf.savefig(fig); plt.close(fig)

    # ====================================================================
    # PAGE 6 — 1C Demagnetizing + Tier 2 Matrices A-B
    # ====================================================================
    fig, ax = new_page(pdf)
    c = Cursor(0.98)

    add_title(ax, c, "Tier 1C + Tier 2 — Interaction Matrices")

    # 1C
    add_section(ax, c, "1C.  Demagnetizing Field Comparison")
    add_body(ax, c,
         r"Appears in:  Poisson: $h = h_a + h_d$ (full) vs $h = h_a$ (reduced, Nochetto)",
         size=SMALL_SIZE, color="#444444", style="italic")
    add_body_wrap(ax, c,
        "The demagnetizing field h_d opposes the applied field inside the ferrofluid, "
        "reducing the effective force. Removing it (reduced model) should over-predict "
        "spike height and under-estimate the critical field threshold.",
        width=100)
    c.advance(GAP_S)
    add_table(ax, c,
              ["ID", "Field model", "CLI flag", "Expected effect"],
              [["R31a", "Full (h_a + h_d)", "(default)", "Reference: self-consistent field"],
               ["R31b", "Reduced (h_a only)", "--reduced_field", "Taller spikes, earlier onset"]],
              [0.05, 0.22, 0.45, 0.65],
              baseline_row=0)
    c.advance(GAP_L)

    # Tier 2
    add_section(ax, c, "Tier 2 — 2D Parameter Interaction Matrices")
    add_body(ax, c, "91 runs  |  Cross the most influential parameters pairwise",
             size=SMALL_SIZE, color="#777777")
    c.advance(GAP_M)

    add_matrix(ax, c,
        r"Matrix A:  $\chi_0 \times \alpha_{\max}$  — Onset Boundary Map",
        r"$\chi_0$", r"$\alpha_{\max}$",
        ["0.10", "0.25", "0.50", "0.75", "1.00", "1.50", "2.00"],
        ["5000", "7000", "8000", "9000", "10000"],
        baseline_r=2, baseline_c=2)
    add_body(ax, c,
         "Maps the critical onset boundary in the most important 2D subspace. "
         "At high chi_0, onset should shift to lower alpha.",
         size=SMALL_SIZE, color="#444444", style="italic")
    c.advance(GAP_L)

    add_matrix(ax, c,
        r"Matrix B:  $\lambda \times \alpha_{\max}$  — Surface Tension vs Field",
        r"$\lambda$", r"$\alpha_{\max}$",
        ["0.50", "0.75", "1.00", "1.25", "1.50", "2.00"],
        ["5000", "7000", "8000", "9000", "10000"],
        baseline_r=2, baseline_c=2)
    add_body(ax, c,
         "Tests whether lambda merely shifts onset or also changes spike morphology.",
         size=SMALL_SIZE, color="#444444", style="italic")

    pdf.savefig(fig); plt.close(fig)

    # ====================================================================
    # PAGE 7 — Tier 2 Matrices C-E
    # ====================================================================
    fig, ax = new_page(pdf)
    c = Cursor(0.98)

    add_title(ax, c, "Tier 2 — Interaction Matrices (cont.)")

    add_matrix(ax, c,
        r"Matrix C:  $\chi_0 \times y_{\mathrm{int}}$  — Susceptibility vs Pool Depth",
        r"$\chi_0$", r"$y_{\mathrm{int}}$",
        ["0.10", "0.50", "1.00", "1.50", "2.00"],
        ["0.05", "0.10", "0.20", "0.40"],
        baseline_r=1, baseline_c=2)
    add_body(ax, c,
         "Does finite pool depth change the instability threshold at high chi_0? "
         "Shallow pools should stabilize even at large susceptibility.",
         size=SMALL_SIZE, color="#444444", style="italic")
    c.advance(GAP_L)

    add_matrix(ax, c,
        r"Matrix D:  Dipole Curvature $\times$ $\chi_0$  — Geometry vs Susceptibility",
        r"$\chi_0$", "curvature",
        ["0.50", "1.00", "1.50", "2.00"],
        ["concave(5)", "flat(inf)", "convex(5)"],
        baseline_r=0, baseline_c=1)
    add_body(ax, c,
         "Does higher susceptibility amplify nonuniformity from curved magnets? "
         "Concave should produce taller central spikes.",
         size=SMALL_SIZE, color="#444444", style="italic")
    c.advance(GAP_L)

    add_matrix(ax, c,
        r"Matrix E:  Dipole Curvature $\times$ $\lambda$  — Geometry vs Surface Tension",
        r"$\lambda$", "curvature",
        ["0.50", "1.00", "1.50"],
        ["concave(5)", "flat(inf)", "convex(5)"],
        baseline_r=1, baseline_c=1)
    add_body(ax, c,
         "Does surface tension smooth out asymmetric spike patterns from "
         "curved magnets? Higher lambda should regularize nonuniform patterns.",
         size=SMALL_SIZE, color="#444444", style="italic")

    pdf.savefig(fig); plt.close(fig)

    # ====================================================================
    # PAGE 8 — Tier 3: Three-Way Tensor + Dipole Geometry
    # ====================================================================
    fig, ax = new_page(pdf)
    c = Cursor(0.98)

    add_title(ax, c, "Tier 3 — Three-Way Tensor & Dipole Geometry",
              subtitle="38 runs  |  Only if Tier 2 reveals parameter interactions")

    # 3A
    add_section(ax, c,
        r"3A.  Three-Way Tensor:  $\chi_0 \times \alpha_{\max} \times \lambda$  (27 runs)")
    add_body(ax, c,
         "Does the critical field depend on surface tension differently at high chi_0?",
         size=SMALL_SIZE, color="#444444", style="italic")
    add_eq(ax, c,
           r"$\chi_0 \in \{0.50,\; 1.50,\; 2.00\}$"
           r"$\;\times\;$"
           r"$\alpha_{\max} \in \{5000,\; 8000,\; 10000\}$"
           r"$\;\times\;$"
           r"$\lambda \in \{0.50,\; 1.00,\; 2.00\}$"
           r"$\;=\; 27\;\mathrm{runs}$",
           size=9, align="left")
    add_body_wrap(ax, c,
        "This tensor resolves whether the H_c(chi_0) scaling changes at different "
        "lambda. Linear theory predicts separable effects; any deviation reveals "
        "nonlinear coupling between susceptibility and capillarity.",
        width=100)
    c.advance(GAP_L)

    # 3B
    add_section(ax, c, "3B.  Dipole Geometry Sweeps  (11 runs)")
    add_body(ax, c,
         "Curved magnet arrangements break translational symmetry, creating "
         "spatially varying field across the interface.",
         size=SMALL_SIZE, color="#444444", style="italic")
    c.advance(GAP_S)
    add_table(ax, c,
              ["Config", "Geometry", "Parameters", "Expected result"],
              [["Flat", "y_i = -15 (all)", "baseline", "5 equal-height spikes"],
               ["Concave R=5", "y_i = -15 + R - sqrt(R^2-..)", "R=5", "Central spikes taller"],
               ["Concave R=10", "", "R=10", "Milder central focusing"],
               ["Concave R=15", "", "R=15", "Weak central enhancement"],
               ["Concave R=50", "", "R=50", "Nearly flat"],
               ["Convex R=5", "y_i = -15 - R + sqrt(R^2-..)", "R=5", "Edge spikes taller"],
               ["Convex R=10", "", "R=10", "Milder edge focusing"],
               ["Convex R=15", "", "R=15", "Weak edge enhancement"],
               ["Convex R=50", "", "R=50", "Nearly flat"],
               ["Deep-V s=5", "y_i = -15 + s*|x_i-0.5|", "slope=5", "Single dominant central spike"],
               ["Deep-V s=10", "", "slope=10", "Stronger central isolation"]],
              [0.05, 0.22, 0.52, 0.68],
              baseline_row=0,
              font_size=7.5,
              row_h=0.024)

    pdf.savefig(fig); plt.close(fig)

    # ====================================================================
    # PAGE 9 — Summary + Diagnostics + CLI
    # ====================================================================
    fig, ax = new_page(pdf)
    c = Cursor(0.98)

    add_title(ax, c, "Summary — All Tiers")

    add_section(ax, c, "Run Count and Compute Budget")
    add_table(ax, c,
              ["Tier", "Description", "Runs", "Core-hours", "Prerequisite"],
              [["1", "Single-parameter screening", "31", "124", "CLI flags"],
               ["2", "2D interaction matrices (A-E)", "91", "364", "Tier 1 done"],
               ["3", "3-way tensor + dipole geometry", "38", "152", "Tier 2 done"],
               ["Total", "", "160", "640", ""]],
              [0.05, 0.14, 0.48, 0.58, 0.72],
              baseline_row=3)
    add_body(ax, c,
         "On HPC (4 cores each, all runs within a tier in parallel):  "
         "~4 hours wall time per tier.",
         size=SMALL_SIZE, color="#444444")
    c.advance(GAP_L)

    # Measured quantities
    add_section(ax, c, "Measured Quantities (per run)")
    metrics = [
        ("Spike count",     "Count peaks in theta along y = y_int at final time"),
        ("Max spike height", "Maximum y where theta > 0 at final time"),
        ("Onset time",       "First time E_CH exceeds 1.05 x E_CH(0)"),
        ("Wavelength",       "FFT of theta along x at interface height"),
        ("Max velocity",     "Peak |U| over entire simulation"),
        ("Steady state",     "Is d(E_CH)/dt < tolerance at t_final?"),
    ]
    for name, desc in metrics:
        ax.text(0.06, c.pos, name, fontsize=9, fontweight="bold",
                color=COLOR_SUBSEC, ha="left", va="top", transform=ax.transAxes)
        ax.text(0.28, c.pos, desc, fontsize=9, color=COLOR_BODY,
                ha="left", va="top", transform=ax.transAxes)
        c.advance(0.030)
    c.advance(GAP_L)

    # Per-step diagnostics
    add_section(ax, c, "Per-Step Diagnostics")
    diags = [
        ("theta range",     r"$[\min \theta,\; \max \theta]$ — expect close to $[-1, +1]$"),
        ("Mass",            r"$\int \theta\, dx$ — should be conserved (SAV property)"),
        ("CH energy",       r"$E_{CH} = \int \frac{\lambda\varepsilon}{2}|\nabla\theta|^2 "
                            r"+ \frac{\lambda}{\varepsilon}F(\theta)\, dx$"),
        ("div u",           r"$\| \nabla\cdot u \|_{L^2}$ — incompressibility check"),
        ("Picard residual", "Residual of Poisson-Magnetization sub-iteration"),
    ]
    for name, desc in diags:
        ax.text(0.06, c.pos, name, fontsize=9, fontweight="bold",
                color=COLOR_SUBSEC, ha="left", va="top", transform=ax.transAxes)
        ax.text(0.28, c.pos, desc, fontsize=9, color=COLOR_BODY,
                ha="left", va="top", transform=ax.transAxes)
        c.advance(0.030)
    c.advance(GAP_L)

    # CLI flags
    add_section(ax, c, "CLI Flags for Parameter Sweeps")
    flags = [
        ("--chi0 VALUE",        "Override susceptibility chi_0"),
        ("--alpha_max VALUE",   "Override max field intensity"),
        ("--lambda VALUE",      "Override surface tension lambda"),
        ("--y_interface VALUE", "Override pool depth"),
        ("--epsilon VALUE",     "Override interface width"),
        ("--Lx VALUE",          "Override domain width"),
        ("--Ly VALUE",          "Override domain height"),
        ("--reduced_field",     "Use reduced model (no demagnetizing field)"),
        ("--dipole_curvature V", "Set dipole array curvature radius"),
    ]
    for flag, desc in flags:
        ax.text(0.06, c.pos, flag, fontsize=8, fontweight="bold",
                color="#333333", ha="left", va="top", transform=ax.transAxes,
                fontfamily="monospace")
        ax.text(0.35, c.pos, desc, fontsize=8, color=COLOR_BODY,
                ha="left", va="top", transform=ax.transAxes)
        c.advance(0.025)

    pdf.savefig(fig); plt.close(fig)

print(f"PDF generated: {OUTPUT}")
