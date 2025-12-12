// ============================================================================
// utilities/questions.h - Open Questions Tracker
//
// Reference: Nochetto, Salgado & Tomas, CMAME 309 (2016) 497-531
// ============================================================================
#ifndef QUESTIONS_H
#define QUESTIONS_H

#include <iostream>
#include <string>
#include <vector>

struct QuestionEntry
{
    std::string file;
    int line;
    std::string assumption;
    std::string basis;
    std::string question;
};

inline std::vector<QuestionEntry> get_all_questions()
{
    return {
        {
            "utilities/parameters.h", 52,
            "η = 0.5ε (stabilization parameter)",
            "Paper states η ≤ ε (Theorem 4.1, p.505; Proposition 5.1)",
            "What value of η was used in the paper's numerical experiments?"
        },
        {
            "physics/applied_field.cc", 78,
            "We derived ∇_x φ_s = -d/|r|² - 2(d·r)r/|r|⁴ (2D) from φ_s = d·r/|r|² (Eq.97)",
            "Derivative w.r.t. x with r = x_s - x, so ∇_x r = -I. "
            "Chain rule: ∇_x[d·r/|r|²] = d·(-I)/|r|² + (d·r)·∇_x[|r|⁻²] = -d/|r|² - 2(d·r)r/|r|⁴",
            "Is our derived gradient ∇_x φ_s correct? Paper gives φ_s (Eq.97) but not ∇φ_s explicitly."
        },
        {
            "assembly/ch_assembler.cc", 28,
            "Convection term sign is +(U·θ_old, ∇Λ) on RHS",
            "Eq. 42a has -(U^k Θ^{k-1}, ∇Λ) on LHS, moved to RHS flips sign",
            "Verify convection term sign convention matches paper"
        },
        {
            "assembly/ch_assembler.cc", 135,
            "Direct matrix addition without distribute_local_to_global",
            "Works for uniform mesh; AMR needs proper constraint handling",
            "Need combined constraint matrix for coupled θ-ψ system with AMR?"
        },
        {
            "assembly/poisson_assembler.cc", 140,
            "Pin first DoF to zero for pure Neumann problem",
            "Standard technique; φ determined only up to constant",
            "Is pinning DoF 0 the best approach? Mean-zero constraint alternative?"
        },
        {
            "assembly/magnetization_assembler.cc", 25,
            "Equilibrium m = χ_θ h_a uses applied field only, not total field h = h_a + ∇φ",
            "Avoids circular dependency (m needs h, Poisson needs m). Paper Section 5 (p.510) suggests h ≈ h_a when χ₀ << 1",
            "Should we iterate Mag/Poisson to get self-consistent m and φ?"
        },
        {
            "assembly/ns_assembler.cc", 145,
            "Kelvin force simplified to μ₀(m·∇)h instead of full trilinear form B_h^m",
            "Paper uses skew-symmetric B_h^m (Eq. 38). We use explicit simpler form.",
            "Is simplified Kelvin force μ₀(m·h) sufficient, or need full B_h^m?"
        },
    };
}

inline void print_questions()
{
    std::cout << "\n========================================\n";
    std::cout << "  OPEN QUESTIONS IN CODEBASE\n";
    std::cout << "========================================\n\n";

    auto questions = get_all_questions();
    for (size_t i = 0; i < questions.size(); ++i)
    {
        const auto& q = questions[i];
        std::cout << "[Q" << (i + 1) << "] " << q.file << ":" << q.line << "\n";
        std::cout << "  ASSUMPTION: " << q.assumption << "\n";
        std::cout << "  BASIS:      " << q.basis << "\n";
        std::cout << "  QUESTION:   " << q.question << "\n\n";
    }
    std::cout << "Total: " << questions.size() << " open question(s)\n";
    std::cout << "========================================\n\n";
}

#endif // QUESTIONS_H
