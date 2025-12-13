// ============================================================================
// utilities/questions.h - Open Questions and Assumptions Tracker
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
//
// This file tracks assumptions made during implementation that may need
// verification with paper authors or further analysis.
//
// Status: OPEN = needs investigation, RESOLVED = confirmed correct
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
            "η = 0.5ε (stabilization parameter)",
            "Paper states η ≤ ε (Theorem 4.1, p.505; Proposition 5.1, p.508) "
            "but does not specify exact value used in Section 6.2 experiments.",
            "What value of η was used in the paper's numerical experiments?",
            ""
        },

        // ====================================================================
        // Q2: Convection term sign in CH equation
        // ====================================================================
        {
            2,
            "OPEN",
            "assembly/ch_assembler.cc", 28,
            "Convection term sign is +(U·θ_old, ∇Λ) on RHS",
            "Eq. 42a has -(U^k Θ^{k-1}, ∇Λ) on LHS. "
            "Moving to RHS with IBP gives +(U·θ, ∇φ) in conservative form.",
            "Verify convection term sign convention matches paper exactly.",
            ""
        },

        // ====================================================================
        // Q3: AMR constraint handling for coupled systems
        // ====================================================================
        {
            3,
            "OPEN",
            "assembly/ch_assembler.cc", 135,
            "Direct matrix addition without distribute_local_to_global for coupled system",
            "Works for uniform mesh; AMR with hanging nodes needs special handling "
            "when θ and ψ are in separate DoFHandlers.",
            "Need combined constraint matrix for coupled θ-ψ system with AMR?",
            ""
        },

        // ====================================================================
        // Q4: Poisson BC treatment for pure Neumann
        // ====================================================================
        {
            4,
            "OPEN",
            "assembly/poisson_assembler.cc", 140,
            "Pin first DoF to zero for pure Neumann problem",
            "Standard technique when φ determined only up to constant. "
            "Alternative: mean-zero constraint ∫φ dΩ = 0",
            "Is pinning DoF 0 the best approach? Mean-zero constraint alternative?",
            ""
        },

        // ====================================================================
        // Q5: Magnetization equilibrium assumption
        // ====================================================================
        {
            5,
            "OPEN",
            "assembly/magnetization_assembler.cc", 25,
            "Equilibrium m = χ_θ h_a uses applied field only, not total field h = h_a + ∇φ",
            "Avoids circular dependency (m needs h, Poisson needs m). "
            "Paper Section 5 (p.510) suggests h ≈ h_a when χ₀ << 1. "
            "With χ₀ = 0.5, this may not be accurate.",
            "Should we iterate Mag/Poisson to get self-consistent m and φ?",
            ""
        },

        // ====================================================================
        // Q6: Kelvin force trilinear form
        // ====================================================================
        {
            6,
            "OPEN",
            "assembly/ns_assembler.cc", 145,
            "Kelvin force simplified to μ₀(m·∇)h instead of full trilinear form B_h^m",
            "Paper uses skew-symmetric B_h^m (Eq. 38, p.504). "
            "We use explicit simpler form for clarity.",
            "Is simplified Kelvin force μ₀κ_θ(h·∇)h sufficient, or need full B_h^m?",
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