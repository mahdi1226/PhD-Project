// ============================================================================
// utilities/questions.h - Open Questions and Assumptions Tracker
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
//
// This file tracks assumptions made during implementation that may need
// verification with paper authors or further analysis.
//
// Status: OPEN = needs investigation, RESOLVED = confirmed correct, BUG_FOUND = error fixed
// ============================================================================
#ifndef QUESTIONS_H
#define QUESTIONS_H

#include <iostream>
#include <string>
#include <vector>

struct QuestionEntry
{
    int id;
    std::string status;       // "OPEN", "RESOLVED", "BUG_FOUND"
    std::string file;
    int line;
    std::string assumption;
    std::string basis;
    std::string question;
    std::string resolution;   // Filled when resolved
};

inline std::vector<QuestionEntry> get_all_questions()
{
    return {
        // ====================================================================
        // Q1: η value for CH stabilization
        // ====================================================================
        {
            1,
            "OPEN",
            "utilities/parameters.h", 70,
            "η = 0.005 (stabilization parameter, η ≤ ε)",
            "Paper states η ≤ ε (Theorem 4.1, p.505; Proposition 5.1, p.508) "
            "but does not specify exact value used in Section 6.2 experiments. "
            "We use η = 0.5ε = 0.005.",
            "What value of η was used in the paper's numerical experiments?",
            ""
        },

        // ====================================================================
        // Q2: Dipole y-position
        // ====================================================================
        {
            2,
            "BUG_FOUND",
            "utilities/parameters.h", 96,
            "Dipole y-position is -15 (not -1.5)",
            "Section 6.2, p.520: 'five magnetic sources located at (-0.5, -15), "
            "(0, -15), (0.5, -15), (1, -15), (1.5, -15)'",
            "Initially implemented as y = -1.5, causing strong boundary effects.",
            "FIXED: Changed to y = -15 per paper. Dipoles are 15 units below domain."
        },

        // ====================================================================
        // Q3: Magnetization model - quasi-static vs dynamic
        // ====================================================================
        {
            3,
            "RESOLVED",
            "physics/kelvin_force.h", 1,
            "Magnetization is quasi-static: M = χ(θ)H, not a separate PDE",
            "Section 5, p.510: For small relaxation time T, m ≈ χ_θ h (equilibrium). "
            "Eq. 17: κ_θ = κ₀ H(θ/ε). We compute M inline during NS assembly.",
            "Is quasi-static approximation valid for Rosensweig experiments?",
            "YES - Paper Section 6.2 uses quasi-static. No separate magnetization PDE needed."
        },

        // ====================================================================
        // Q4: Kelvin force formula
        // ====================================================================
        {
            4,
            "OPEN",
            "assembly/ns_assembler.cc", 200,
            "Kelvin force: F_mag = μ₀ χ(θ) (H·∇)H where H = -∇φ",
            "Paper Eq. 14f uses μ₀(m·∇)h. With m = χ_θ h and h = -∇φ, "
            "we get F = μ₀ χ_θ (h·∇)h. Paper also mentions trilinear form B_h^m (Eq. 38).",
            "Is our simplified Kelvin force correct? Should we use skew-symmetric B_h^m?",
            ""
        },

        // ====================================================================
        // Q5: Capillary force formula
        // ====================================================================
        {
            5,
            "OPEN",
            "assembly/ns_assembler.cc", 180,
            "Capillary force: F_cap = (λ/ε) θ ∇ψ",
            "Paper Eq. 10, p.498: F_st = (λ/ε) θ ∇μ where μ is chemical potential. "
            "We use ψ as chemical potential. Verify sign and scaling.",
            "Is F_cap = (λ/ε)θ∇ψ exactly correct? Check sign convention.",
            ""
        },

        // ====================================================================
        // Q6: Gravity/Boussinesq term
        // ====================================================================
        {
            6,
            "OPEN",
            "assembly/ns_assembler.cc", 220,
            "Gravity: F_g = (1 + r·H(θ/ε)) g  where g = (0, -30000)",
            "Paper Eq. 19, p.502: f_g uses density ratio r = 0.1. "
            "H(θ/ε) transitions from 0 (water) to 1 (ferrofluid).",
            "Verify gravity formula and direction. Is g magnitude 30000 correct?",
            ""
        },

        // ====================================================================
        // Q7: Navier-Stokes boundary conditions
        // ====================================================================
        {
            7,
            "OPEN",
            "core/phase_field_setup.cc", 260,
            "No-slip BC (u = 0) on all four boundaries",
            "Paper Section 6.2 doesn't explicitly state velocity BCs for Rosensweig. "
            "No-slip is standard for enclosed containers.",
            "Are no-slip BCs on all boundaries correct for Rosensweig setup?",
            ""
        },

        // ====================================================================
        // Q8: Pressure uniqueness / mean-zero constraint
        // ====================================================================
        {
            8,
            "OPEN",
            "solvers/ns_solver.cc", 20,
            "No pressure pinning or mean-zero constraint applied",
            "Saddle-point system with pure Dirichlet velocity BCs has pressure "
            "determined up to a constant. UMFPACK handles singular systems.",
            "Should we pin one pressure DoF or add mean-zero constraint?",
            ""
        },

        // ====================================================================
        // Q9: Time stepping parameter θ
        // ====================================================================
        {
            9,
            "RESOLVED",
            "utilities/parameters.h", 51,
            "θ = 1.0 (Backward Euler, fully implicit viscosity)",
            "Paper Section 4.1, Eq. 42 uses θ-method for viscosity. "
            "θ = 1 is most stable (backward Euler).",
            "What value of θ was used in paper experiments? θ = 0.5 (Crank-Nicolson)?",
            "Using θ = 1 for stability. Can adjust if needed."
        },

        // ====================================================================
        // Q10: Permeability formula
        // ====================================================================
        {
            10,
            "RESOLVED",
            "physics/material_properties.h", 60,
            "μ(θ) = 1 + χ₀·H(θ/ε) where H(x) = 1/(1+exp(-x))",
            "Paper Eq. 17, p.501: μ_θ = 1 + κ_θ where κ_θ = κ₀ H(θ/ε). "
            "With χ₀ = κ₀ = 0.5, we get μ ∈ [1, 1.5].",
            "Verify permeability formula matches paper exactly.",
            "CONFIRMED: μ(θ=+1) ≈ 1.5, μ(θ=-1) ≈ 1.0. Matches expected range."
        },

        // ====================================================================
        // Q11: Viscosity formula
        // ====================================================================
        {
            11,
            "RESOLVED",
            "physics/material_properties.h", 50,
            "ν(θ) = ν_w + (ν_f - ν_w)·H(θ/ε)",
            "Paper Eq. 17, p.501: ν_θ = ν_w + (ν_f - ν_w) H(θ/ε). "
            "With ν_w = 1, ν_f = 2, we get ν ∈ [1, 2].",
            "Verify viscosity transition from water to ferrofluid.",
            "CONFIRMED: Matches paper formula exactly."
        },

        // ====================================================================
        // Q12: Dipole potential formula (2D)
        // ====================================================================
        {
            12,
            "RESOLVED",
            "assembly/poisson_assembler.cc", 50,
            "φ_dipole = α(t) × (d·r) / |r|^p where p = dim",
            "Paper Eq. 97, p.519: φ_s = d·(x_s - x) / |x_s - x|^p with p = dim. "
            "For 2D: p = 2, so φ = d·r / |r|². Also grad(φ_s) = 0 (harmonic).",
            "Is 2D dipole potential using correct power?",
            "CONFIRMED: p = dim = 2 for 2D. Formula φ = d·r / |r|² is correct."
        },

        // ====================================================================
        // Q13: Intensity ramp function
        // ====================================================================
        {
            13,
            "RESOLVED",
            "assembly/poisson_assembler.cc", 45,
            "α(t) = 6000 × min(t/1.6, 1)",
            "Paper Section 6.2, p.520: 'intensity increases linearly from 0 at t=0 "
            "to 6000 at t=1.6 and is kept constant from t=1.6 to t=2'",
            "Verify ramp function implementation.",
            "CONFIRMED: α(t) = α_max × min(t/t_ramp, 1) with α_max=6000, t_ramp=1.6"
        },

        // ====================================================================
        // Q14: Boundary effects from dipoles outside domain
        // ====================================================================
        {
            14,
            "RESOLVED",
            "utilities/parameters.h", 94,
            "Dipoles at x = -0.5 and x = 1.5 are outside domain [0, 1]",
            "Paper places 5 dipoles at x = -0.5, 0, 0.5, 1, 1.5 with domain [0,1]×[0,0.6]. "
            "This is intentional - creates field distribution across domain.",
            "Is corner spike formation expected physics or numerical artifact?",
            "Expected per paper setup. Corner effects reduced with y = -15 (far field)."
        },

        // ====================================================================
        // Q15: Pool depth / initial interface position
        // ====================================================================
        {
            15,
            "RESOLVED",
            "utilities/parameters.h", 145,
            "Initial ferrofluid pool depth = 0.2 (20% of domain height)",
            "Paper Section 6.2: 'pool depth 0.2'. Domain height is 0.6, "
            "so interface at y = 0.6 × 0.2 = 0.12.",
            "Verify initial condition: θ = +1 for y < 0.12, θ = -1 for y > 0.12",
            "CONFIRMED: Using tanh profile centered at y = pool_depth × (y_max - y_min)"
        },

        // ====================================================================
        // Q16: Stability of current time step
        // ====================================================================
        {
            16,
            "OPEN",
            "utilities/parameters.h", 49,
            "dt = 5e-4 (paper suggests dt ≈ t_F/4000 = 2/4000 = 5e-4)",
            "Observing potential instability in full solver. May need smaller dt "
            "or additional stabilization (grad-div, SUPG).",
            "What CFL condition should we enforce? Is dt = 5e-4 stable for all regimes?",
            ""
        },

        // ====================================================================
        // Q17: Convection term treatment in CH
        // ====================================================================
        {
            17,
            "OPEN",
            "assembly/ch_assembler.cc", 28,
            "Convection term sign: +(U·θ_old, ∇Λ) on RHS after IBP",
            "Eq. 42a has -(U^k Θ^{k-1}, ∇Λ) on LHS. Moving to RHS and using "
            "conservative form via IBP gives +(U·θ, ∇φ).",
            "Verify convection term sign. Should it be + or - on RHS?",
            ""
        },

        // ====================================================================
        // Q18: Need for grad-div stabilization in NS
        // ====================================================================
        {
            18,
            "OPEN",
            "assembly/ns_assembler.cc", 280,
            "Grad-div stabilization available but disabled (γ = 0)",
            "Paper doesn't mention grad-div stabilization. May be needed for "
            "high Reynolds number or dominant magnetic forces.",
            "Should we enable grad-div stabilization? What value of γ?",
            ""
        },

        // ====================================================================
        // Q19: CRITICAL - Poisson equation has wrong RHS
        // ====================================================================
        {
            19,
            "BUG_FOUND",
            "assembly/poisson_assembler.cc", 100,
            "We have: -∇·(μ∇φ) = 0. Paper has: ∇·(μ_θ ∇φ) = ∇·(μ₀ κ_θ h_a)",
            "Eq. 14d: The RHS is NOT zero! It's the divergence of the applied field "
            "weighted by susceptibility. h_a = -∇φ_s (dipole field).",
            "RHS of Poisson must include source term from applied field.",
            "MUST FIX: Add RHS = ∇·(μ₀ κ_θ h_a) to Poisson assembly."
        },

        // ====================================================================
        // Q20: CRITICAL - Poisson BC should be Neumann, not Dirichlet
        // ====================================================================
        {
            20,
            "BUG_FOUND",
            "assembly/poisson_assembler.cc", 60,
            "We use: φ = φ_dipole (Dirichlet). Paper uses: ∂_n(φ) = (h_a - m)·n (Neumann)",
            "The magnetic potential has Neumann BC based on interface condition "
            "for magnetic field continuity. With pure Neumann, need constraint for uniqueness.",
            "BC type is completely wrong - Dirichlet vs Neumann.",
            "MUST FIX: Change to Neumann BC with mean-zero constraint."
        },

        // ====================================================================
        // Q21: Capillary force - verify θ factor
        // ====================================================================
        {
            21,
            "OPEN",
            "assembly/ns_assembler.cc", 180,
            "We have: F_cap = (λ/ε) θ ∇ψ. Paper may have: f_c = (λ/ε) ∇ψ",
            "Paper Remark 2.1 says capillary force modifies pressure and simplifies. "
            "Need to verify if θ factor should be present or not.",
            "Is capillary force (λ/ε)θ∇ψ or just (λ/ε)∇ψ?",
            ""
        },

    };
}

inline void print_questions()
{
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║           OPEN QUESTIONS IN CODEBASE                           ║\n";
    std::cout << "║        Nochetto et al. CMAME 309 (2016)                        ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    auto questions = get_all_questions();

    int open_count = 0;
    int resolved_count = 0;
    int bug_count = 0;

    for (const auto& q : questions)
    {
        std::string status_marker;
        if (q.status == "OPEN") {
            status_marker = "[ ]";
            open_count++;
        } else if (q.status == "RESOLVED") {
            status_marker = "[✓]";
            resolved_count++;
        } else if (q.status == "BUG_FOUND") {
            status_marker = "[!]";
            bug_count++;
        }

        std::cout << "────────────────────────────────────────────────────────────────\n";
        std::cout << status_marker << " Q" << q.id << ": " << q.file << ":" << q.line << "\n";
        std::cout << "    Status: " << q.status << "\n";
        std::cout << "    ASSUMPTION: " << q.assumption << "\n";
        std::cout << "    BASIS:      " << q.basis << "\n";
        std::cout << "    QUESTION:   " << q.question << "\n";
        if (!q.resolution.empty())
            std::cout << "    RESOLUTION: " << q.resolution << "\n";
        std::cout << "\n";
    }

    std::cout << "════════════════════════════════════════════════════════════════\n";
    std::cout << "Summary: " << questions.size() << " total | "
              << open_count << " open | "
              << resolved_count << " resolved | "
              << bug_count << " bugs found\n";
    std::cout << "════════════════════════════════════════════════════════════════\n";
    std::cout << "\n";
}

#endif // QUESTIONS_H