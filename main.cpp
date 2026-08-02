#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "inference.h"
#include "potential.h"
#include "variable.h"

using bn::Inference;
using bn::Potential;
using bn::Variable;

namespace {

void printPotential(const std::string& label, const Potential& p) {
    const auto& vars = p.variables();
    std::vector<int> states;
    for (const auto& v : vars) states.push_back(v.num_states());
    std::vector<int> cur(states.size(), 0);

    std::cout << "\n" << label << " over {";
    for (std::size_t i = 0; i < vars.size(); ++i) {
        std::cout << (i == 0 ? "" : ", ") << vars[i].name();
    }
    std::cout << "}:\n";

    for (int k = 0; k < p.numEntries(); ++k) {
        std::cout << "  P(";
        for (std::size_t i = 0; i < vars.size(); ++i) {
            std::cout << (i == 0 ? "" : ", ") << vars[i].name() << "=" << cur[i];
        }
        std::cout << ") = " << std::fixed << std::setprecision(4)
                  << std::exp(p.logTable()[k]) << "\n";
        for (int r = static_cast<int>(states.size()) - 1; r >= 0; --r) {
            if (++cur[r] < states[r]) break;
            cur[r] = 0;
        }
    }
}

}  // namespace

int main() {
    //  A      two binary random variables form a diamond network
    // / \
    // B   C   with P(A) P(B|A) P(C|A) P(D|B,C)
    // \   /
    //   D
    const Variable a(0, "A", 2), b(1, "B", 2), c(2, "C", 2), d(3, "D", 2);

    const Potential pA({a}, {0.6f, 0.4f});
    const Potential pBgA({a, b}, {0.7f, 0.3f, 0.2f, 0.8f});
    const Potential pCgA({a, c}, {0.9f, 0.1f, 0.25f, 0.75f});
    const Potential pDgBC({b, c, d},
                          {0.8f, 0.2f, 0.3f, 0.7f, 0.6f, 0.4f, 0.1f, 0.9f});

    const Inference engine({pA, pBgA, pCgA, pDgBC});

    // 1. Full joint P(A, B, C, D)
    printPotential("Full joint P(A,B,C,D)", engine.fullJoint());

    // 2. Marginal / subset query P(D) and P(B, C)
    printPotential("Marginal P(D)", engine.marginal({d}));
    printPotential("Marginal P(B,C)", engine.marginal({b, c}));

    // 3. Conditional P(A | D) as a full table and with concrete evidence
    printPotential("Conditional P(A | D)", engine.conditional({a}, {d}));
    printPotential("Conditional P(A | D=1)", engine.conditionalGiven({a}, {{d, 1}}));
    printPotential("Conditional P(D | B=0, C=1)",
                   engine.conditionalGiven({d}, {{b, 0}, {c, 1}}));

    return 0;
}
