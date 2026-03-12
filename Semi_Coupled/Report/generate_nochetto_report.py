#!/usr/bin/env python3
"""
Generate a comprehensive PDF report reproducing the mathematical formulation
from Nochetto, Salgado & Tomas (CMAME 2016):
"A diffuse interface model for two-phase ferrofluid flows"

This is the paper behind the Semi_Coupled / Archived_Nochetto code.
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
    KeepTogether
)
from reportlab.lib import colors
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "Nochetto_CMAME2016_Formulation.pdf")


def build_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name='DocTitle',
        parent=styles['Title'],
        fontSize=16,
        leading=20,
        spaceAfter=6,
        alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        name='DocSubtitle',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        spaceAfter=4,
        alignment=TA_CENTER,
        textColor=colors.grey,
    ))
    styles.add(ParagraphStyle(
        name='SectionHead',
        parent=styles['Heading1'],
        fontSize=14,
        leading=18,
        spaceBefore=18,
        spaceAfter=8,
        textColor=colors.HexColor('#1a1a6c'),
    ))
    styles.add(ParagraphStyle(
        name='SubHead',
        parent=styles['Heading2'],
        fontSize=12,
        leading=15,
        spaceBefore=12,
        spaceAfter=6,
        textColor=colors.HexColor('#2d2d8c'),
    ))
    styles.add(ParagraphStyle(
        name='SubSubHead',
        parent=styles['Heading3'],
        fontSize=11,
        leading=14,
        spaceBefore=8,
        spaceAfter=4,
        textColor=colors.HexColor('#3a3aaa'),
    ))
    styles.add(ParagraphStyle(
        name='Body',
        parent=styles['Normal'],
        fontSize=10,
        leading=13,
        spaceAfter=6,
        alignment=TA_JUSTIFY,
    ))
    styles.add(ParagraphStyle(
        name='Eq',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        spaceBefore=4,
        spaceAfter=4,
        leftIndent=36,
        alignment=TA_LEFT,
        fontName='Courier',
    ))
    styles.add(ParagraphStyle(
        name='EqLabel',
        parent=styles['Normal'],
        fontSize=9,
        leading=12,
        spaceBefore=2,
        spaceAfter=6,
        leftIndent=36,
        alignment=TA_LEFT,
        fontName='Courier',
        textColor=colors.HexColor('#555555'),
    ))
    styles.add(ParagraphStyle(
        name='Remark',
        parent=styles['Normal'],
        fontSize=9.5,
        leading=12,
        spaceBefore=6,
        spaceAfter=6,
        leftIndent=18,
        rightIndent=18,
        backColor=colors.HexColor('#f0f0ff'),
        borderColor=colors.HexColor('#9999cc'),
        borderWidth=0.5,
        borderPadding=6,
        alignment=TA_JUSTIFY,
    ))
    styles.add(ParagraphStyle(
        name='TableCell',
        parent=styles['Normal'],
        fontSize=8.5,
        leading=11,
    ))
    return styles


def eq(text):
    """Wrap text in Courier for equation display."""
    return f'<font face="Courier" size="10">{text}</font>'


def eqn(label, text):
    """Equation with label on the right."""
    return f'<font face="Courier" size="10">{text}</font>  <font face="Courier" size="8" color="#666666">({label})</font>'


def bold(text):
    return f'<b>{text}</b>'


def ital(text):
    return f'<i>{text}</i>'


def build_document():
    styles = build_styles()
    story = []

    # ====================================================================
    # TITLE PAGE
    # ====================================================================
    story.append(Spacer(1, 1.5 * inch))
    story.append(Paragraph(
        "Nochetto, Salgado &amp; Tomas (CMAME 2016)<br/>"
        "Complete Mathematical Formulation",
        styles['DocTitle']
    ))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        '"A diffuse interface model for two-phase ferrofluid flows"<br/>'
        'Comput. Methods Appl. Mech. Engrg. 309 (2016) 497-531',
        styles['DocSubtitle']
    ))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "Reference document for the Semi_Coupled / Archived_Nochetto codebase.<br/>"
        "All equation numbers refer to the original paper.",
        styles['DocSubtitle']
    ))
    story.append(Spacer(1, 24))
    story.append(Paragraph(
        bold("Paper scope:") + " A simplified two-phase FHD model assembled from "
        "the Shliomis model, Cahn-Hilliard phase field, and magnetostatics. "
        "Energy-stable numerical scheme with DG magnetization and Picard-like "
        "Block-Gauss-Seidel iteration. Convergence proof for simplified model.",
        styles['Body']
    ))
    story.append(PageBreak())

    # ====================================================================
    # TABLE OF CONTENTS (manual)
    # ====================================================================
    story.append(Paragraph("Contents", styles['SectionHead']))
    toc_items = [
        "1. Two-Phase Model (Paper Section 2)",
        "   1.1 Phase-Field: Cahn-Hilliard",
        "   1.2 Simplified Ferrohydrodynamics: Shliomis Model",
        "   1.3 Capillary Force",
        "   1.4 Magnetostatics and Poisson Equation",
        "   1.5 Complete Two-Phase System (Eq 14)",
        "   1.6 Boundary Conditions and Material Functions",
        "   1.7 Gravity (Boussinesq)",
        "2. Energy Estimates (Paper Section 3)",
        "   2.1 Trilinear Forms and Key Identity",
        "   2.2 Energy Functional and Dissipation",
        "   2.3 Energy Estimate (Proposition 3.1)",
        "3. Numerical Scheme (Paper Section 4)",
        "   3.1 Notation and Discrete Norms",
        "   3.2 FE Spaces (CG + DG)",
        "   3.3 Discrete Trilinear Forms",
        "   3.4 Energy-Stable Scheme (Eq 42)",
        "   3.5 Discrete Energy Stability (Proposition 4.1)",
        "   3.6 Block-Gauss-Seidel Iteration (Picard)",
        "4. Simplified Model (Paper Section 5)",
        "   4.1 h = h_a Simplification",
        "   4.2 Convergent Scheme (Eq 65)",
        "5. Practical Space Discretization",
        "   5.1 FE Space Choices",
        "   5.2 Upwind Stabilization",
        "   5.3 Finite Element Pairs",
        "6. Numerical Experiments (Paper Section 6)",
        "   6.1 Rosensweig Instability (Uniform Field)",
        "   6.2 Ferrofluid Hedgehog (Non-Uniform Field)",
        "   6.3 Parameter Tables",
        "7. Code Correspondence Table",
    ]
    for item in toc_items:
        indent = 18 if item.startswith("   ") else 0
        story.append(Paragraph(item.strip(), ParagraphStyle(
            'toc', parent=styles['Body'], leftIndent=indent, spaceAfter=2,
            fontSize=9.5, leading=12,
        )))
    story.append(PageBreak())

    # ====================================================================
    # SECTION 1: TWO-PHASE MODEL
    # ====================================================================
    story.append(Paragraph("1. Two-Phase Model (Paper Section 2)", styles['SectionHead']))

    story.append(Paragraph(
        "The model is assembled (not derived) from well-known components: "
        "Cahn-Hilliard for the phase field, a simplified Shliomis model for "
        "magnetization, magnetostatics for the magnetic field, and incompressible "
        "Navier-Stokes for fluid flow. The domain is a bounded convex "
        "polygon/polyhedron with phase field variable theta labeling ferrofluid "
        "(theta=1) vs non-magnetic fluid (theta=0).",
        styles['Body']
    ))

    # 1.1 CH
    story.append(Paragraph("1.1 Phase-Field: Cahn-Hilliard", styles['SubHead']))
    story.append(Paragraph(
        "The phase field theta evolves by the Cahn-Hilliard equation with "
        "constant mobility gamma > 0 and interface thickness 0 &lt; epsilon &lt;&lt; 1:",
        styles['Body']
    ))
    story.append(Paragraph(eqn("1",
        "theta_t = -gamma * Delta(psi)              in Omega"
    ), styles['Eq']))
    story.append(Paragraph(eqn("1",
        "psi    = epsilon * Delta(theta) - (1/epsilon) * f(theta)    in Omega"
    ), styles['Eq']))
    story.append(Paragraph(eqn("1",
        "dn(theta) = dn(psi) = 0                    on Gamma"
    ), styles['Eq']))

    story.append(Paragraph(
        "where psi is the chemical potential and f(theta) = F'(theta). "
        "The truncated double-well potential F is:",
        styles['Body']
    ))
    story.append(Paragraph(eqn("2",
        "F(theta) = { (theta+1)<super>2</super>          if theta in (-inf, -1]"
    ), styles['Eq']))
    story.append(Paragraph(eq(
        "         = { (1/4)(theta<super>2</super> - 1)<super>2</super>     if theta in [-1, 1]"
    ), styles['Eq']))
    story.append(Paragraph(eq(
        "         = { (theta-1)<super>2</super>          if theta in [1, +inf)"
    ), styles['Eq']))
    story.append(Paragraph(
        "This truncation ensures bounded derivatives:",
        styles['Body']
    ))
    story.append(Paragraph(eqn("3",
        "|f(theta)| = |F'(theta)| &lt;= 2|theta| + 1"
    ), styles['Eq']))
    story.append(Paragraph(eq(
        "|f'(theta)| = |F''(theta)| &lt;= 2        for all theta in R"
    ), styles['Eq']))
    story.append(Paragraph(
        "The CH equation is mass-conservative: d/dt integral(theta) dx = 0.",
        styles['Body']
    ))

    # 1.2 Shliomis
    story.append(Paragraph("1.2 Simplified Ferrohydrodynamics", styles['SubHead']))
    story.append(Paragraph(
        "The Shliomis model couples Navier-Stokes with a magnetization advection-reaction "
        "equation. The " + bold("full") + " monophase model is:",
        styles['Body']
    ))
    story.append(Paragraph(eqn("5a",
        "u_t + (u . nabla)u - nu*Delta(u) + nabla(p)"
    ), styles['Eq']))
    story.append(Paragraph(eq(
        "    = mu_0 * (m . nabla)h + (mu_0/2) * curl(m x h)"
    ), styles['Eq']))
    story.append(Paragraph(eqn("5b",
        "m_t + (u . nabla)m - (1/2)*curl(u) x m"
    ), styles['Eq']))
    story.append(Paragraph(eq(
        "    = -(1/T)(m - chi_0 * h) - beta * m x (m x h)"
    ), styles['Eq']))

    story.append(Paragraph(
        bold("Key simplification (Eq 8):") + " Since relaxation time T is very small "
        "(T ~ 1e-5 to 1e-9 for commercial ferrofluids), 1/T is huge. The dynamics is "
        "fast towards equilibrium m ~ chi_0 * h. Drop: (a) the (mu_0/2)*curl(m x h) "
        "from momentum, (b) the -(1/2)*curl(u) x m from magnetization, "
        "(c) the beta * m x (m x h) term. This gives the " + bold("simplified Shliomis") + ":",
        styles['Body']
    ))
    story.append(Paragraph(eqn("8a",
        "u_t + (u . nabla)u - nu*Delta(u) + nabla(p) = mu_0 * (m . nabla)h"
    ), styles['Eq']))
    story.append(Paragraph(eqn("8b",
        "m_t + (u . nabla)m + (1/T)*m = (chi_0/T)*h"
    ), styles['Eq']))
    story.append(Paragraph(
        "The convective term (u . nabla)m is kept for symmetry/energy reasons. "
        "The curl(u) x m term is dropped as it cannot be easily related to "
        "energy principles.",
        styles['Body']
    ))

    story.append(Paragraph(
        bold("CRITICAL:") + " Eq (8b) is the magnetization PDE that must be solved "
        "at every time step. It is NOT the algebraic equilibrium m = chi_0 * h. "
        "Even though T is small, the PDE form is essential for the energy-stable scheme.",
        styles['Remark']
    ))

    # 1.3 Capillary
    story.append(Paragraph("1.3 Capillary Force", styles['SubHead']))
    story.append(Paragraph(
        "The capillary force is derived from the capillary stress tensor "
        "sigma_c = lambda * nabla(theta) (x) nabla(theta). After manipulation:",
        styles['Body']
    ))
    story.append(Paragraph(eqn("10",
        "f_c := (lambda/epsilon) * theta * nabla(psi)"
    ), styles['Eq']))
    story.append(Paragraph(
        "This form allows the Ginzburg-Landau energy gradient terms to be absorbed "
        "into a redefined pressure (Remark 2.1), which is essential for the energy law.",
        styles['Body']
    ))

    # 1.4 Magnetostatics
    story.append(Paragraph("1.4 Magnetostatics and Poisson Equation", styles['SubHead']))
    story.append(Paragraph(
        "The magnetostatic equations curl(h) = 0, div(b) = 0 with b = mu_0(h + m) "
        "lead to h = nabla(phi) where phi satisfies:",
        styles['Body']
    ))
    story.append(Paragraph(eqn("13",
        "-Delta(phi) = div(m - h_a)    in Omega"
    ), styles['Eq']))
    story.append(Paragraph(eq(
        "dn(phi) = (h_a - m) . n       on Gamma"
    ), styles['Eq']))
    story.append(Paragraph(
        "where h_a is the (given) smooth applied magnetizing field (curl-free, div-free), "
        "and h_d is the demagnetizing field. The effective magnetizing field is:",
        styles['Body']
    ))
    story.append(Paragraph(eqn("11-12",
        "h := h_a + h_d,      h = nabla(phi)"
    ), styles['Eq']))

    # 1.5 Complete system
    story.append(Paragraph("1.5 Complete Two-Phase System (Eq 14)", styles['SubHead']))
    story.append(Paragraph(
        bold("This is the central system of the paper.") + " Collecting all components:",
        styles['Body']
    ))

    eq14 = [
        ("14a", "theta_t + div(u*theta) + gamma * Delta(psi) = 0"),
        ("14b", "psi - epsilon * Delta(theta) + (1/epsilon) * f(theta) = 0"),
        ("14c", "m_t + (u . nabla)m = -(1/T) * (m - chi_theta * h)"),
        ("14d", "-Delta(phi) = div(m - h_a)"),
        ("14e", "u_t + (u . nabla)u - div(nu_theta * T(u)) + nabla(p)"),
        ("",    "    = mu_0 * (m . nabla)h + (lambda/epsilon) * theta * nabla(psi)"),
        ("14f", "div(u) = 0"),
    ]
    for label, text in eq14:
        if label:
            story.append(Paragraph(eqn(label, text), styles['Eq']))
        else:
            story.append(Paragraph(eq(text), styles['Eq']))

    story.append(Paragraph(
        "where T(u) = (1/2)(nabla(u) + nabla(u)<super>T</super>) is the symmetric gradient.",
        styles['Body']
    ))

    story.append(Paragraph(
        bold("Important notes on Eq 14:") + "<br/>"
        "- Eq (14c): magnetization uses the " + ital("simplified") + " Shliomis (no curl term, no beta term)<br/>"
        "- Eq (14e): Kelvin force is mu_0*(m . nabla)h only (no curl(m x h) term)<br/>"
        "- Eq (14e): capillary force uses the specific form (lambda/epsilon)*theta*nabla(psi)<br/>"
        "- All nonlinear couplings preserved: u-theta (transport), m-h (Kelvin), theta-psi (capillary)",
        styles['Remark']
    ))

    # 1.6 BCs and material
    story.append(Paragraph("1.6 Boundary Conditions and Material Functions", styles['SubHead']))
    story.append(Paragraph(eqn("15",
        "dn(theta) = dn(psi) = 0,  u = 0,  dn(phi) = (h_a - m) . n    on Gamma"
    ), styles['Eq']))

    story.append(Paragraph(
        "The phase-dependent viscosity and susceptibility are Lipschitz functions of theta:",
        styles['Body']
    ))
    story.append(Paragraph(eqn("16",
        "0 &lt; min(nu_w, nu_f) &lt;= nu_theta &lt;= max(nu_w, nu_f)"
    ), styles['Eq']))
    story.append(Paragraph(eq(
        "0 &lt;= chi_theta &lt;= chi_0"
    ), styles['Eq']))

    story.append(Paragraph(
        "Practical choices use a regularized Heaviside (sigmoid):",
        styles['Body']
    ))
    story.append(Paragraph(eqn("17",
        "nu_theta = nu_w + (nu_f - nu_w) * H(theta/epsilon)"
    ), styles['Eq']))
    story.append(Paragraph(eq(
        "chi_theta = chi_0 * H(theta/epsilon)"
    ), styles['Eq']))
    story.append(Paragraph(eqn("18",
        "H(x) = 1 / (1 + exp(-x))"
    ), styles['Eq']))

    # 1.7 Gravity
    story.append(Paragraph("1.7 Gravity (Boussinesq Approximation)", styles['SubHead']))
    story.append(Paragraph(eqn("19",
        "f_g = (1 + r * H(theta/epsilon)) * g"
    ), styles['Eq']))
    story.append(Paragraph(eq(
        "r = |rho_f - rho_w| / min(rho_f, rho_w)"
    ), styles['Eq']))
    story.append(Paragraph(
        "This is added to the RHS of the momentum equation (14e). "
        "Valid when r &lt;&lt; 1 (nearly matching densities).",
        styles['Body']
    ))

    story.append(PageBreak())

    # ====================================================================
    # SECTION 2: ENERGY ESTIMATES
    # ====================================================================
    story.append(Paragraph("2. Energy Estimates (Paper Section 3)", styles['SectionHead']))

    story.append(Paragraph("2.1 Trilinear Forms and Key Identity", styles['SubHead']))
    story.append(Paragraph(
        "Define the Kelvin force trilinear form:",
        styles['Body']
    ))
    story.append(Paragraph(eqn("20",
        "B(m, h, u) = SUM_{i,j} integral( m<super>i</super> * h<super>j</super>_{x_i} * u<super>j</super> ) dx"
    ), styles['Eq']))

    story.append(Paragraph(
        bold("Lemma 3.1") + " (Key identity). For solenoidal h (curl h = 0) and u . n = 0 on Gamma:",
        styles['Body']
    ))
    story.append(Paragraph(eqn("21",
        "B(u, m, h) = -B(m, h, u)"
    ), styles['Eq']))
    story.append(Paragraph(
        "In other words: ((u . nabla)h, m) = -((m . nabla)h, u) - (m . h)*div(u) + boundary terms. "
        "This identity is " + bold("crucial") + " for the energy cancellation between "
        "Kelvin force and magnetization transport.",
        styles['Body']
    ))

    story.append(Paragraph("2.2 Energy Functional and Dissipation", styles['SubHead']))
    story.append(Paragraph(
        bold("Proposition 3.1") + " (Energy Estimate). If chi_0 &lt;= 4, solutions of system (14) satisfy:",
        styles['Body']
    ))
    story.append(Paragraph(eqn("23",
        "E(t_F) + integral_0^{t_F} D(s) ds &lt;= E(0) + integral_0^{t_F} F(h_a; s) ds"
    ), styles['Eq']))
    story.append(Paragraph("where the energy is:", styles['Body']))
    story.append(Paragraph(eqn("24-E",
        "E(u, m, h, theta; s) = (1/2)||u||<super>2</super>"
        " + (mu_0/2)||m||<super>2</super>"
        " + (mu_0/2)||h||<super>2</super>"
        " + (lambda/2)||nabla(theta)||<super>2</super>"
        " + (lambda/epsilon<super>2</super>)(F(theta), 1)"
    ), styles['Eq']))
    story.append(Paragraph("the dissipation is:", styles['Body']))
    story.append(Paragraph(eqn("24-D",
        "D(u, m, h, theta; s) = (mu_0/T)(1 - chi_0/4)||m||<super>2</super>"
        " + (mu_0/2T)||h||<super>2</super>"
    ), styles['Eq']))
    story.append(Paragraph(eq(
        "    + ||sqrt(nu_theta) T(u)||<super>2</super>"
        " + (lambda*gamma/epsilon)||nabla(psi)||<super>2</super>"
    ), styles['Eq']))
    story.append(Paragraph("and the source is:", styles['Body']))
    story.append(Paragraph(eqn("24-F",
        "F(h_a; s) = (mu_0/T)||h_a||<super>2</super>"
        " + (mu_0*T/tau)||dt(h_a)||<super>2</super>"
    ), styles['Eq']))

    story.append(Paragraph(
        bold("Proof sketch:") + " Multiply (14a) by psi, (14b) by theta_t, "
        "(14c) by mu_0*m, (14c) by mu_0*h, (14d) by phi_t, (14e) by u. "
        "The key cancellations are:<br/>"
        "- Capillary: -(lambda/epsilon)(theta*nabla(psi), u) from (14e) with "
        "+(lambda/epsilon)(u*theta, nabla(psi)) from modified (14a) = 0 (by parts)<br/>"
        "- Kelvin: mu_0*B(u,m,h) from (14e) dotted with u cancels with terms from "
        "(14c) dotted with mu_0*h via Lemma 3.1 (B(u,m,h) = -B(m,h,u))<br/>"
        "- Transport: (u . nabla)m . m = 0 by incompressibility and BCs",
        styles['Remark']
    ))

    story.append(Paragraph(
        bold("Remark 3.1") + " (chi_0 &lt;= 4 restriction). This comes from bounding "
        "(mu_0/T)(chi_theta*h, m) by Young's inequality. Covers commercial ferrofluids "
        "(chi_0 ranges 0.5 to 4.3).",
        styles['Body']
    ))

    story.append(PageBreak())

    # ====================================================================
    # SECTION 3: NUMERICAL SCHEME
    # ====================================================================
    story.append(Paragraph("3. Numerical Scheme (Paper Section 4)", styles['SectionHead']))

    story.append(Paragraph("3.1 Notation and Discrete Norms", styles['SubHead']))
    story.append(Paragraph(
        "K time steps, tau = T/K, t<super>k</super> = k*tau. "
        "Backward difference operator:",
        styles['Body']
    ))
    story.append(Paragraph(eqn("34",
        "delta(rho<super>k</super>) = rho<super>k</super> - rho<super>k-1</super>"
    ), styles['Eq']))
    story.append(Paragraph(
        "Key identity used repeatedly:",
        styles['Body']
    ))
    story.append(Paragraph(eqn("35",
        "2(a, a - b) = |a|<super>2</super> - |b|<super>2</super> + |a - b|<super>2</super>"
    ), styles['Eq']))

    story.append(Paragraph("3.2 Finite Element Spaces", styles['SubHead']))
    story.append(Paragraph(
        "Six finite-dimensional subspaces approximate the unknowns. "
        "All are CG " + bold("except magnetization which is DG") + ":",
        styles['Body']
    ))

    fe_data = [
        ['Field', 'Space', 'Regularity', 'FE Type', 'Degree'],
        ['Phase theta', 'G_h', 'H<super>1</super>(Omega)', 'CG', 'l >= 2'],
        ['Chem. pot. psi', 'Y_h', 'H<super>1</super>(Omega)', 'CG', 'l >= 2'],
        ['Magnetization m', 'M_h', 'L<super>2</super>(Omega)', bold('DG'), 'l - 1'],
        ['Mag. potential phi', 'X_h', 'H<super>1</super>(Omega)', 'CG', 'l >= 2'],
        ['Velocity u', 'U_h', 'H<super>1</super><sub>0</sub>(Omega)', 'CG', 'l >= 2'],
        ['Pressure P', 'P_h', 'L<super>2</super>(Omega)', bold('DG'), 'l - 1'],
    ]
    fe_table = Table(
        [[Paragraph(cell, styles['TableCell']) for cell in row] for row in fe_data],
        colWidths=[1.3*inch, 0.7*inch, 1.2*inch, 0.6*inch, 0.8*inch],
    )
    fe_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#d0d0e8')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1a1a6c')),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5ff')]),
    ]))
    story.append(fe_table)
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        bold("CRITICAL: M_h is DG") + " (discontinuous piecewise polynomials). "
        "This is equation (56): M_h = [P_{l-1}(T)]<super>d</super> for all T in T_h. "
        "Magnetization lives in the " + bold("same polynomial space as pressure") + " P_h. "
        "The requirement nabla(X_h) subset M_h is needed for energy stability.",
        styles['Remark']
    ))

    story.append(Paragraph(
        "Additional requirements:<br/>"
        "- (U_h, P_h) must satisfy LBB inf-sup condition (Eq 36)<br/>"
        "- (G_h, Y_h) must be inf-sup stable for bilaplacian discretization<br/>"
        "- Pressure P_h must contain continuous subspace of degree >= 1 (A1)<br/>"
        "- Each M_h component lives in same space as P_h (A2)<br/>"
        "- Stokes projector must converge in W<super>1,inf</super> norm (A3)",
        styles['Body']
    ))

    # 3.3 Discrete trilinear forms
    story.append(Paragraph("3.3 Discrete Trilinear Forms", styles['SubHead']))
    story.append(Paragraph(
        "For the NS convective term, a standard CG skew-symmetric form B_h is used. "
        "For the magnetization transport (14c) and Kelvin force (14e), a " +
        bold("DG upwind form") + " is required:",
        styles['Body']
    ))
    story.append(Paragraph(eqn("57",
        "B_h^m(U, V, W) = SUM_T integral_T [ (U . nabla)V . W"
        " + (1/2) div(U) V . W ] dx"
    ), styles['Eq']))
    story.append(Paragraph(eq(
        "              - SUM_F integral_F [ [[V]] . {{W}} ] (U . n_F) dS"
    ), styles['Eq']))

    story.append(Paragraph(
        "where F ranges over all " + bold("internal faces") + " (not boundary), "
        "[[V]] = V<super>+</super> - V<super>-</super> is the jump, "
        "{{W}} = (W<super>+</super> + W<super>-</super>)/2 is the average, "
        "and n_F is the face normal chosen arbitrarily but consistently.",
        styles['Body']
    ))

    story.append(Paragraph(
        bold("Skew-symmetry:") + " B_h^m(U, V, W) = -B_h^m(U, W, V) for all "
        "U in U_h (CG, div-free), V, W in M_h (DG). This ensures B_h^m(U, V, V) = 0, "
        "which is needed for energy stability.",
        styles['Remark']
    ))

    story.append(Paragraph(
        bold("Key simplification (Section 5.1, below Eq 71):") + " Since h = nabla(phi) "
        "is CG, the jumps [[h]] = 0 on all interior faces. Therefore in the Kelvin "
        "force term B_h^m(V, H, M), " + bold("all face integrals vanish") + ". "
        "Only the cell integrals survive. This greatly simplifies the NS assembler.",
        styles['Remark']
    ))

    # 3.4 Scheme
    story.append(Paragraph("3.4 Energy-Stable Scheme (Eq 42)", styles['SubHead']))
    story.append(Paragraph(
        bold("Initialization (Eq 41):"),
        styles['Body']
    ))
    story.append(Paragraph(eqn("41",
        "Theta<super>0</super> = I_{G_h}[theta(0)],   "
        "M<super>0</super> = I_{M_h}[m(0)],   "
        "U<super>0</super> = I_{U_h}[u(0)]"
    ), styles['Eq']))

    story.append(Paragraph(
        bold("For each k = 1, ..., K, find") + " {Theta<super>k</super>, Psi<super>k</super>, "
        "M<super>k</super>, Phi<super>k</super>, U<super>k</super>, P<super>k</super>} that solve:",
        styles['Body']
    ))

    story.append(Paragraph(bold("Cahn-Hilliard (Eq 42a-42b):"), styles['SubSubHead']))
    story.append(Paragraph(eqn("42a",
        "(delta(Theta<super>k</super>)/tau, Lambda)"
        " - (U<super>k</super>*Theta<super>k-1</super>, nabla(Lambda))"
        " - gamma*(nabla(Psi<super>k</super>), nabla(Lambda)) = 0"
    ), styles['Eq']))
    story.append(Paragraph(eqn("42b",
        "(Psi<super>k</super>, Y)"
        " + epsilon*(nabla(Theta<super>k</super>), nabla(Y))"
        " + (1/epsilon)*(f(Theta<super>k-1</super>), Y)"
        " + (1/eta)*(delta(Theta<super>k</super>), Y) = 0"
    ), styles['Eq']))
    story.append(Paragraph(
        "where eta > 0 is a stabilization parameter (eta &lt;= epsilon). "
        "The nonlinearity f(theta) is evaluated at Theta<super>k-1</super> "
        "(explicit/lagged), making the system " + bold("linear") + ".",
        styles['Body']
    ))

    story.append(Paragraph(bold("Magnetization + Poisson (Eq 42c-42d):"), styles['SubSubHead']))
    story.append(Paragraph(eqn("42c",
        "(delta(M<super>k</super>)/tau, Z)"
        " - B_h^m(U<super>k</super>, Z, M<super>k</super>)"
        " + (1/T)*(M<super>k</super>, Z)"
        " = (1/T)*(chi_theta * H<super>k</super>, Z)"
    ), styles['Eq']))
    story.append(Paragraph(eqn("42d",
        "(nabla(Phi<super>k</super>), nabla(X))"
        " = (h_a<super>k</super> - M<super>k</super>, nabla(X))"
    ), styles['Eq']))

    story.append(Paragraph(
        bold("Eq 42c details:") + "<br/>"
        "- B_h^m(U<super>k</super>, Z, M<super>k</super>) is the DG transport form (57): "
        "cell terms + face jumps<br/>"
        "- The sign convention: -B_h^m(U, Z, M) means B_h^m(U, M, Z) appears with + on LHS "
        "(skew-symmetry: -B_h^m(U,Z,M) = B_h^m(U,M,Z))<br/>"
        "- H<super>k</super> = nabla(Phi<super>k</super>) couples M and Phi<br/>"
        "- chi_theta = chi(Theta<super>k-1</super>) uses LAGGED phase field<br/>"
        "- The system (42c)-(42d) is COUPLED and solved by Picard iteration",
        styles['Remark']
    ))

    story.append(Paragraph(bold("Navier-Stokes (Eq 42e-42f):"), styles['SubSubHead']))
    story.append(Paragraph(eqn("42e",
        "(delta(U<super>k</super>)/tau, V)"
        " + B_h(U<super>k-1</super>, U<super>k</super>, V)"
        " + (nu_theta * T(U<super>k</super>), T(V))"
    ), styles['Eq']))
    story.append(Paragraph(eq(
        "    - (P<super>k</super>, div V)"
        " = mu_0 * B_h^m(V, H<super>k</super>, M<super>k</super>)"
    ), styles['Eq']))
    story.append(Paragraph(eq(
        "    + (lambda/epsilon) * (Theta<super>k-1</super> * nabla(Psi<super>k</super>), V)"
    ), styles['Eq']))
    story.append(Paragraph(eqn("42f",
        "(Q, div U<super>k</super>) = 0"
    ), styles['Eq']))

    story.append(Paragraph(
        bold("Eq 42e details:") + "<br/>"
        "- Kelvin force: mu_0 * B_h^m(V, H<super>k</super>, M<super>k</super>). "
        "Since H = nabla(Phi) is CG, face jumps [[H]] = 0, so only cell terms survive<br/>"
        "- Convection: B_h(U<super>k-1</super>, U<super>k</super>, V) is the standard CG skew form<br/>"
        "- Capillary: uses Theta<super>k-1</super> (lagged) and Psi<super>k</super> (current)<br/>"
        "- nu_theta = nu(Theta<super>k-1</super>) uses LAGGED phase field<br/>"
        "- The Kelvin force uses CURRENT M<super>k</super>, H<super>k</super> from the magnetization step",
        styles['Remark']
    ))

    # 3.5 Energy stability
    story.append(Paragraph("3.5 Discrete Energy Stability (Proposition 4.1)", styles['SubHead']))
    story.append(Paragraph(
        bold("Proposition 4.1.") + " If nabla(X_h) subset M_h, eta &lt;= epsilon, "
        "and chi_0 &lt;= 4, the scheme (42) satisfies:",
        styles['Body']
    ))
    story.append(Paragraph(eqn("43",
        "E<super>K</super> + SUM_{k=1}^{K} [ D_n(delta; k) + tau * D_p(k) ]"
        " &lt;= E<super>0</super> + SUM_{k=1}^{K} tau * F(h_a<super>k</super>)"
    ), styles['Eq']))
    story.append(Paragraph("where:", styles['Body']))
    story.append(Paragraph(eqn("43-E",
        "E<super>k</super> = (1/2)||U<super>k</super>||<super>2</super>"
        " + (mu_0/2)||M<super>k</super>||<super>2</super>"
        " + (mu_0/2)||nabla(Phi<super>k</super>)||<super>2</super>"
        " + (lambda/2)||nabla(Theta<super>k</super>)||<super>2</super>"
        " + (lambda/epsilon<super>2</super>)(F(Theta<super>k</super>), 1)"
    ), styles['Eq']))
    story.append(Paragraph(eqn("43-Dn",
        "D_n(delta; k) = (1/2)||delta(U)||<super>2</super>"
        " + (mu_0/2)||delta(M)||<super>2</super>"
        " + (mu_0/2)||delta(nabla Phi)||<super>2</super>"
        " + (lambda/2)||delta(nabla Theta)||<super>2</super>"
        " + (lambda/eta)||delta(Theta)||<super>2</super>"
    ), styles['Eq']))
    story.append(Paragraph(eqn("43-Dp",
        "D_p(k) = (mu_0/T)(1 - chi_0/4)||M<super>k</super>||<super>2</super>"
        " + (mu_0/T)||sqrt(chi_theta) H<super>k</super>||<super>2</super>"
        " + (mu_0/2T)||M<super>k</super>||<super>2</super>"
        " + 2*tau*||sqrt(nu) T(U)||<super>2</super>"
        " + (lambda*gamma*tau/epsilon)||nabla(Psi)||<super>2</super>"
    ), styles['Eq']))

    story.append(Paragraph(
        bold("Proof key steps:") + " Set test functions Lambda = (2*lambda*tau/epsilon)*Psi<super>k</super>, "
        "Y = (2*lambda/epsilon)*delta(Theta<super>k</super>), Z = 2*tau*mu_0*M<super>k</super>, "
        "X = (2*mu_0*tau/T)*Phi<super>k</super>, V = 2*tau*U<super>k</super>. "
        "Add all resulting equations. The convective terms vanish by skew-symmetry. "
        "The Kelvin force cancels with magnetization transport via identity (21/38). "
        "Capillary force cancels with CH coupling. Use Young's inequality for the "
        "chi_theta*H<super>k</super> coupling term (requires chi_0 &lt;= 4).",
        styles['Body']
    ))

    # 3.6 Picard
    story.append(Paragraph("3.6 Block-Gauss-Seidel (Picard) Iteration", styles['SubHead']))
    story.append(Paragraph(
        "The coupled system (42) is solved by a " + bold("3-block Gauss-Seidel") +
        " iteration (Section 6.1, p.520):",
        styles['Body']
    ))
    story.append(Paragraph(
        bold("Block 1:") + " Solve CH system (42a)-(42b) for (Theta<super>k</super>, Psi<super>k</super>) "
        "using U<super>k</super> from previous iteration (or U<super>k-1</super> for first iteration).",
        styles['Body']
    ))
    story.append(Paragraph(
        bold("Block 2:") + " Solve magnetization + Poisson (42c)-(42d) for "
        "(M<super>k</super>, Phi<super>k</super>) using U<super>k</super> from previous iteration. " +
        bold("This sub-system itself requires inner Picard iteration") +
        " since M and Phi are coupled through H<super>k</super> = nabla(Phi<super>k</super>).",
        styles['Body']
    ))
    story.append(Paragraph(
        bold("Block 3:") + " Solve NS (42e)-(42f) for (U<super>k</super>, P<super>k</super>) "
        "using the just-computed M<super>k</super>, H<super>k</super>, Theta<super>k</super>, Psi<super>k</super>.",
        styles['Body']
    ))
    story.append(Paragraph(
        bold("Implementation detail:") + " The paper states (p.520): "
        "'We make no attempt to prove, establish or show convergence of this iterative scheme.' "
        "In the Archived_Nochetto code, the M-Phi Picard uses up to 7 iterations with "
        "under-relaxation omega = 1/(1+chi_0) when chi_0 >= 1. "
        "Tolerance: ||M||_2 relative change &lt; 1e-4.",
        styles['Remark']
    ))

    story.append(PageBreak())

    # ====================================================================
    # SECTION 4: SIMPLIFIED MODEL
    # ====================================================================
    story.append(Paragraph("4. Simplified Model h = h_a (Paper Section 5)", styles['SectionHead']))

    story.append(Paragraph("4.1 The Simplification", styles['SubHead']))
    story.append(Paragraph(
        "When chi_0 &lt;&lt; 1 and the applied field h_a changes slowly, "
        "we can set h := h_a (drop the Poisson equation entirely). "
        "This is physically reasonable for small susceptibility ferrofluids. "
        "The magnetic field H<super>k</super> is simply the L<super>2</super> projection of h_a:",
        styles['Body']
    ))
    story.append(Paragraph(eqn("66",
        "H<super>k</super> := I_{M_h}[h_a<super>k</super>]"
    ), styles['Eq']))
    story.append(Paragraph(
        "Since H is now given data (no Poisson solve), the M equation decouples. "
        "The scheme becomes " + bold("directly solvable") + " (no Picard needed). "
        "Convergence can be proved for this case (Theorem 5.1).",
        styles['Body']
    ))

    story.append(Paragraph("4.2 Convergent Scheme (Eq 65)", styles['SubHead']))
    story.append(Paragraph(eqn("65a",
        "(delta(Theta<super>k</super>)/tau, Lambda)"
        " - (U<super>k</super>*Theta<super>k-1</super>, nabla(Lambda))"
        " - gamma*(nabla(Psi<super>k</super>), nabla(Lambda)) = 0"
    ), styles['Eq']))
    story.append(Paragraph(eqn("65b",
        "(Psi<super>k</super>, Y)"
        " + epsilon*(nabla(Theta<super>k</super>), nabla(Y))"
        " + (1/epsilon)*(f(Theta<super>k-1</super>), Y)"
        " + (1/eta)*(delta(Theta<super>k</super>), Y) = 0"
    ), styles['Eq']))
    story.append(Paragraph(eqn("65c",
        "(delta(M<super>k</super>)/tau, Z)"
        " - B_h^m(U<super>k</super>, Z, M<super>k</super>)"
        " + (1/T)*(M<super>k</super>, Z)"
        " = (1/T)*(chi_theta * H<super>k</super>, Z)"
    ), styles['Eq']))
    story.append(Paragraph(eqn("65d",
        "(delta(U<super>k</super>)/tau, V)"
        " + B_h(U<super>k-1</super>, U<super>k</super>, V)"
        " + (nu_theta * T(U<super>k</super>), T(V))"
        " - (P<super>k</super>, div V)"
    ), styles['Eq']))
    story.append(Paragraph(eq(
        "    = mu_0 * B_h^m(V, H<super>k</super>, M<super>k</super>)"
        " + (lambda/epsilon) * (Theta<super>k-1</super> * nabla(Psi<super>k</super>), V)"
    ), styles['Eq']))
    story.append(Paragraph(eqn("65e",
        "(Q, div U<super>k</super>) = 0"
    ), styles['Eq']))

    story.append(Paragraph(
        "Note: Eq (65) is identical to (42) except Poisson (42d) is removed and "
        "H<super>k</super> is given by (66).",
        styles['Body']
    ))

    story.append(Paragraph(
        bold("Remark 5.3:") + " For the simplified model, the chi_0 &lt;= 4 restriction "
        "is " + bold("NOT required") + ". The restriction only appears when bounding "
        "the M-H coupling, which is absent here since H is given data.",
        styles['Remark']
    ))

    story.append(PageBreak())

    # ====================================================================
    # SECTION 5: PRACTICAL FE SPACES
    # ====================================================================
    story.append(Paragraph("5. Practical Space Discretization", styles['SectionHead']))

    story.append(Paragraph("5.1 Finite Element Space Choices", styles['SubHead']))
    story.append(Paragraph(
        "The paper specifies (Eq 55-58, Remark 5.5):",
        styles['Body']
    ))
    fe_practical = [
        ['Space', 'Definition', 'deal.II equivalent'],
        ['G_h (theta)', 'CG P_l, l >= 2', 'FE_Q(2)'],
        ['Y_h (psi)', 'CG P_l, l >= 2', 'FE_Q(2)'],
        ['M_h (m)', '[P_{l-1}]<super>d</super> DG', 'FE_DGQ(1)<super>d</super>'],
        ['X_h (phi)', 'CG P_l, l >= 2', 'FE_Q(2)'],
        ['U_h (u)', 'P2+bubble enriched', 'FE_Q(2) + FE_Bernardi_Raugel'],
        ['P_h (p)', 'DG P_{l-1}', 'FE_DGP(1)'],
    ]
    fe_table2 = Table(
        [[Paragraph(cell, styles['TableCell']) for cell in row] for row in fe_practical],
        colWidths=[1.2*inch, 1.8*inch, 2.0*inch],
    )
    fe_table2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#d0d0e8')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5ff')]),
    ]))
    story.append(fe_table2)
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "The requirement nabla(X_h) subset M_h forces M_h to be at least P_{l-1} DG "
        "when X_h is P_l CG. The LBB condition for (U_h, P_h) is satisfied by the "
        "enriched Taylor-Hood or Bernardi-Raugel pairs.",
        styles['Body']
    ))

    story.append(Paragraph("5.2 Upwind Stabilization (Eq 94)", styles['SubHead']))
    story.append(Paragraph(
        "For convergence (not just stability), upwind stabilization is added to "
        "the DG transport form:",
        styles['Body']
    ))
    story.append(Paragraph(eqn("94",
        "S_h^up(U<super>k</super>, M<super>k</super>, Z)"
        " = (1/2) * SUM_F integral_F |U<super>k</super> . n_F| * [[M<super>k</super>]] . [[Z]] dS"
    ), styles['Eq']))
    story.append(Paragraph(
        "This is added to the LHS of (42c). The paper states (Remark 5.4): "
        "'Unlike Continuous Galerkin methods, DG schemes do not need any form of "
        "additional numerical stabilization in order to work. However without some "
        "form of linear stabilization ... they will deliver sub-optimal convergence.'",
        styles['Body']
    ))

    story.append(Paragraph("5.3 Key Implementation Constraint", styles['SubHead']))
    story.append(Paragraph(
        bold("nabla(X_h) subset M_h") + " is the central requirement. "
        "If X_h = CG Q2, then nabla(X_h) on each cell contains polynomials "
        "of degree 1 in each direction. So M_h = [DG Q1]<super>d</super> suffices "
        "(gradient of Q2 is in Q1). This means:<br/><br/>"
        "  phi: FE_Q(2)  ->  m: FE_DGQ(1)<super>d</super><br/><br/>"
        "Pressure: P_h = DG P1 (discontinuous, piecewise linear). "
        "The incompressibility constraint is enforced element-by-element.",
        styles['Remark']
    ))

    story.append(PageBreak())

    # ====================================================================
    # SECTION 6: NUMERICAL EXPERIMENTS
    # ====================================================================
    story.append(Paragraph("6. Numerical Experiments (Paper Section 6)", styles['SectionHead']))

    story.append(Paragraph("6.1 Rosensweig Instability (Uniform Field)", styles['SubHead']))
    story.append(Paragraph(
        "Domain: [0,1] x [0,0.6]. Ferrofluid pool of depth 0.2 at bottom.",
        styles['Body']
    ))

    params_rosen = [
        ['Parameter', 'Value', 'Description'],
        ['nu_w', '1.0', 'Water viscosity'],
        ['nu_f', '2.0', 'Ferrofluid viscosity'],
        ['mu_0', '1', 'Permeability of free space'],
        ['chi_0', '0.5', 'Magnetic susceptibility'],
        ['gamma', '0.0002', 'CH mobility'],
        ['lambda', '0.05', 'Surface tension coefficient'],
        ['epsilon', '0.01', 'Interface thickness'],
        ['r', '0.1', 'Density ratio (Boussinesq)'],
        ['g', '(0, -30000)<super>T</super>', 'Gravity (Eq 103: ~3e4)'],
        ['T (relax)', 'not explicit', 'Use 1/T >> 1 (fast relaxation)'],
        ['AMR levels', '4, 5, 6, 7', 'Max refinement levels tested'],
        ['Time steps', '1000, 2000, 4000, 8000', 'For t_F = 2'],
        ['t_F', '2.0', 'Final time'],
    ]
    params_table = Table(
        [[Paragraph(cell, styles['TableCell']) for cell in row] for row in params_rosen],
        colWidths=[1.2*inch, 1.5*inch, 2.3*inch],
    )
    params_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#d0d0e8')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5ff')]),
    ]))
    story.append(params_table)
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        bold("Applied field:") + " 5 dipoles at (x_s, y_s) = (-0.5,-15), (0,-15), "
        "(0.5,-15), (1,-15), (1.5,-15) with d = (0,1)<super>T</super>. "
        "Intensity alpha_s ramps linearly from 0 at t=0 to alpha_s=6000 at t=1.6, "
        "then constant. Using Eq (97):<br/>"
        "phi_s(x) = d . (x_s - x) / |x_s - x|<super>2</super>  (2D version)<br/>"
        "h_a = SUM_s alpha_s * nabla(phi_s)(x)",
        styles['Body']
    ))

    story.append(Paragraph(
        bold("Key result (Eq 103):") + " The gravity scaling g ~ 4*pi<super>2</super>*lambda "
        "/ (l_c<super>2</super> * Delta_rho * epsilon) ~ 3e4 gives approximately 4 peaks. "
        "The parametric study (Figs 3-4) shows spikes appear robustly across all "
        "refinement levels (4-7) and time discretizations (1000-8000 steps).",
        styles['Body']
    ))

    story.append(Paragraph(
        bold("Initial mesh:") + " 10 elements in x, 6 in y direction. "
        "AMR: Dorfler marking with refinement and coarsening every 5 steps. "
        "Error indicator: eta_T<super>2</super> = h_T * integral_{dT} [[d(theta)/dn]]<super>2</super> dS (Eq 99).",
        styles['Body']
    ))

    # 6.2 Hedgehog
    story.append(Paragraph("6.2 Ferrofluid Hedgehog (Non-Uniform Field)", styles['SubHead']))
    story.append(Paragraph(
        "Same domain [0,1] x [0,0.6] with modified parameters:",
        styles['Body']
    ))
    params_hedge = [
        ['Parameter', 'Value'],
        ['chi_0', '0.9'],
        ['epsilon', '0.005'],
        ['lambda', '0.025'],
        ['Pool depth', '0.11'],
        ['Initial mesh', '15 x 9'],
        ['AMR levels', '6'],
        ['Time steps', '24000'],
        ['t_F', '6.0'],
    ]
    params_table2 = Table(
        [[Paragraph(cell, styles['TableCell']) for cell in row] for row in params_hedge],
        colWidths=[1.5*inch, 2.0*inch],
    )
    params_table2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#d0d0e8')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5ff')]),
    ]))
    story.append(params_table2)
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        bold("Applied field:") + " 42 dipoles in 3 rows at y = -0.5, -0.75, -1.0 "
        "with 14 dipoles per row, equi-distributed in x. d = (0,1)<super>T</super>. "
        "alpha_s ramps from 0 to 4.3 over t in [0, 4.2], then constant until t=6. "
        "This creates a non-uniform field approximating a bar magnet.",
        styles['Body']
    ))

    story.append(Paragraph(
        bold("Fig 6 vs Fig 7:") + " The paper demonstrates that using the full "
        "magnetizing field h = h_a + h_d (with Poisson equation) gives spikes, "
        "while the simplified h = h_a " + bold("does NOT") + " give spikes for this "
        "non-uniform case. This proves the demagnetizing field h_d is " +
        bold("essential") + " for the hedgehog instability.",
        styles['Remark']
    ))

    story.append(PageBreak())

    # ====================================================================
    # SECTION 7: CODE CORRESPONDENCE
    # ====================================================================
    story.append(Paragraph("7. Code Correspondence Table", styles['SectionHead']))

    story.append(Paragraph(
        "Mapping between Nochetto paper equations and Semi_Coupled / Archived_Nochetto code.",
        styles['Body']
    ))

    code_data = [
        ['Paper Eq', 'Physics', 'Code File', 'Key Function/Detail'],
        ['(42a-42b)', 'Cahn-Hilliard', 'assembly/ch_assembler.cc', 'Coupled theta-psi system'],
        ['(42c)', 'Magnetization', 'assembly/magnetic_assembler.cc', 'DG transport + relaxation'],
        ['(42d)', 'Poisson', 'assembly/magnetic_assembler.cc', 'Monolithic with (42c)'],
        ['(42e-42f)', 'Navier-Stokes', 'assembly/ns_assembler.cc', 'Kelvin + capillary forces'],
        ['(57)', 'DG transport', 'physics/skew_forms.h', 'B_h^m: cell + face terms'],
        ['(94)', 'Upwind stab.', 'physics/skew_forms.h', 'S_h^up: |U.n|*[[M]]*[[Z]]'],
        ['(17-18)', 'Materials', 'physics/material_properties.h', 'nu(theta), chi(theta)'],
        ['(97-98)', 'Applied field', 'physics/applied_field.h', 'Dipole h_a computation'],
        ['(10)', 'Capillary', 'physics/kelvin_force.h', '(lambda/eps)*theta*nabla(psi)'],
        ['(19)', 'Gravity', 'physics/initial_conditions.h', 'Boussinesq f_g'],
        ['(41)', 'Init', 'core/phase_field_setup.cc', 'L2 projection of ICs'],
        ['(99)', 'AMR indicator', 'core/phase_field_amr.cc', 'Kelly error estimator'],
        ['Picard', 'Block GS', 'core/phase_field.cc', 'solve_poisson_magnetization_picard()'],
        ['All', 'Driver', 'drivers/*.cc', 'Time loop, block iteration'],
    ]
    code_table = Table(
        [[Paragraph(cell, styles['TableCell']) for cell in row] for row in code_data],
        colWidths=[0.8*inch, 1.0*inch, 2.0*inch, 1.7*inch],
    )
    code_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#d0d0e8')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1a1a6c')),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5ff')]),
    ]))
    story.append(code_table)
    story.append(Spacer(1, 12))

    # Key differences summary
    story.append(Paragraph("Key Differences: Nochetto vs Zhang", styles['SubHead']))
    diff_data = [
        ['Aspect', 'Nochetto (CMAME 2016)', 'Zhang (SIAM 2021)'],
        ['Magnetization FE', bold('DG') + ' (discontinuous)', bold('CG') + ' (continuous)'],
        ['Transport form', 'DG upwind B_h^m (Eq 57)', 'CG skew b(u,v,w) (Eq 3.3)'],
        ['Face integrals', 'YES (jumps, averages)', 'NO (all CG)'],
        ['M-Phi coupling', 'Picard iteration', 'Two-step: m_tilde then m'],
        ['Pressure', 'DG P1', 'CG Q_{l2-1}'],
        ['Extra NS terms', 'None (simplified)', '3 stabilization terms b_stab'],
        ['Momentum terms', 'mu_0*(m.nabla)h only', '+ (mu/2)*curl(mxh) + m x curl h'],
        ['Mag PDE terms', 'Simplified: no curl, no beta', 'Full: curl + beta (m x (m x h))'],
        ['Decoupling', 'Monolithic + Picard', 'Projection + splitting'],
        ['Energy proof', 'chi_0 <= 4 required', 'chi_0 <= chi_0 (no extra cond.)'],
    ]
    diff_table = Table(
        [[Paragraph(cell, styles['TableCell']) for cell in row] for row in diff_data],
        colWidths=[1.1*inch, 2.1*inch, 2.3*inch],
    )
    diff_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8d0d0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#6c1a1a')),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#fff5f5')]),
    ]))
    story.append(diff_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph(
        bold("Bottom line for Semi_Coupled debugging:") + " The Nochetto scheme "
        "requires (1) DG magnetization with face integrals (Eq 57), (2) Picard "
        "iteration for M-Phi coupling, (3) upwind stabilization (Eq 94), and "
        "(4) the full Poisson equation for h = nabla(phi) (not h = h_a). "
        "All four elements must be present for spikes to form.",
        styles['Remark']
    ))

    # Build
    doc = SimpleDocTemplate(
        OUTPUT_PATH,
        pagesize=letter,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch,
        leftMargin=0.85*inch,
        rightMargin=0.85*inch,
        title="Nochetto CMAME 2016 - Complete Formulation",
        author="Generated for PhD Project",
    )
    doc.build(story)
    print(f"PDF generated: {OUTPUT_PATH}")
    print(f"  Pages: ~{len(story)//20} (estimated)")


if __name__ == "__main__":
    build_document()
