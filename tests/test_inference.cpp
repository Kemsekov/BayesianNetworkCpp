#include <gtest/gtest.h>

#include <map>
#include <stdexcept>
#include <vector>

#include "inference.h"
#include "potential.h"
#include "variable.h"

using bn::Inference;
using bn::Potential;
using bn::Variable;

namespace {

using Assignment = std::map<Variable, int>;

// ---------------------------------------------------------------------------
// Independent brute-force reference built on top of factor evaluation only.
// ---------------------------------------------------------------------------

std::vector<int> statesOf(const std::vector<Variable>& vars) {
    std::vector<int> states;
    for (const Variable& v : vars) states.push_back(v.num_states());
    return states;
}

double refJoint(const std::vector<Potential>& factors, const Assignment& full) {
    double p = 1.0;
    for (const Potential& f : factors) p *= f.probability(full);
    return p;
}

/// Full joint table over `allVars` keyed by assignment vector in `allVars` order.
std::map<std::vector<int>, double> buildFullTable(
    const std::vector<Potential>& factors, const std::vector<Variable>& allVars) {
    const std::vector<int> states = statesOf(allVars);
    std::vector<int> cur(states.size(), 0);
    int total = 1;
    for (int s : states) total *= s;

    std::map<std::vector<int>, double> table;
    for (int k = 0; k < total; ++k) {
        Assignment a;
        for (std::size_t i = 0; i < allVars.size(); ++i) a[allVars[i]] = cur[i];
        table[cur] = refJoint(factors, a);
        for (int r = static_cast<int>(states.size()) - 1; r >= 0; --r) {
            if (++cur[r] < states[r]) break;
            cur[r] = 0;
        }
    }
    return table;
}

int indexOf(const std::vector<Variable>& allVars, const Variable& v) {
    for (std::size_t i = 0; i < allVars.size(); ++i) {
        if (allVars[i] == v) return static_cast<int>(i);
    }
    return -1;
}

double refMarginal(const std::map<std::vector<int>, double>& table,
                   const std::vector<Variable>& allVars,
                   const std::vector<Variable>& query,
                   const std::vector<int>& qAssign) {
    double sum = 0.0;
    for (const auto& [full, p] : table) {
        bool match = true;
        for (std::size_t i = 0; i < query.size(); ++i) {
            if (full[indexOf(allVars, query[i])] != qAssign[i]) {
                match = false;
                break;
            }
        }
        if (match) sum += p;
    }
    return sum;
}

Assignment toAssignment(const std::vector<Variable>& vars,
                        const std::vector<int>& vals) {
    Assignment a;
    for (std::size_t i = 0; i < vars.size(); ++i) a[vars[i]] = vals[i];
    return a;
}

// ---------------------------------------------------------------------------
// Networks used by the tests.
// ---------------------------------------------------------------------------

std::vector<Potential> naiveBayesFactors() {
    // X -> Y and X -> Z, all binary.
    const Variable x(0, "X", 2), y(1, "Y", 2), z(2, "Z", 2);
    return {
        Potential({x}, {0.5f, 0.5f}),
        Potential({x, y}, {0.9f, 0.1f, 0.2f, 0.8f}),  // P(Y | X)
        Potential({x, z}, {0.8f, 0.2f, 0.1f, 0.9f}),  // P(Z | X)
    };
}

std::vector<Potential> diamondFactors() {
    // A -> B, A -> C, and D | (B, C); all binary.
    const Variable a(0, "A", 2), b(1, "B", 2), c(2, "C", 2), d(3, "D", 2);
    return {
        Potential({a}, {0.6f, 0.4f}),
        Potential({a, b}, {0.7f, 0.3f, 0.2f, 0.8f}),  // P(B | A)
        Potential({a, c}, {0.9f, 0.1f, 0.25f, 0.75f}),  // P(C | A)
        Potential({b, c, d},
                  {0.8f, 0.2f, 0.3f, 0.7f, 0.6f, 0.4f, 0.1f, 0.9f}),  // P(D | B, C)
    };
}

}  // namespace

// ---------------------------------------------------------------------------
// Naive Bayes network: X -> Y, X -> Z.
// ---------------------------------------------------------------------------

TEST(Inference, NaiveBayesFullJoint) {
    const auto factors = naiveBayesFactors();
    const Variable x(0, "X", 2), y(1, "Y", 2), z(2, "Z", 2);
    const std::vector<Variable> all = {x, y, z};
    const Inference engine(factors);
    const auto table = buildFullTable(factors, all);

    const Potential joint = engine.fullJoint();
    EXPECT_EQ(joint.numVariables(), 3);

    std::vector<int> cur(3, 0);
    const std::vector<int> states = {2, 2, 2};
    for (int k = 0; k < 8; ++k) {
        EXPECT_NEAR(joint.probability(toAssignment(all, cur)), table.at(cur), 1e-5)
            << "full joint cell " << cur[0] << cur[1] << cur[2];
        for (int r = 2; r >= 0; --r) {
            if (++cur[r] < states[r]) break;
            cur[r] = 0;
        }
    }

    double sum = 0.0;
    for (const auto& [assign, p] : table) sum += p;
    EXPECT_NEAR(sum, 1.0, 1e-6);
}

TEST(Inference, NaiveBayesMarginal) {
    const auto factors = naiveBayesFactors();
    const Variable x(0, "X", 2), y(1, "Y", 2), z(2, "Z", 2);
    const std::vector<Variable> all = {x, y, z};
    const Inference engine(factors);
    const auto table = buildFullTable(factors, all);

    const Potential pz = engine.marginal({z});
    EXPECT_EQ(pz.numVariables(), 1);
    EXPECT_NEAR(pz.probability({{z, 0}}), 0.45, 1e-5);
    EXPECT_NEAR(pz.probability({{z, 1}}), 0.55, 1e-5);

    const Potential py = engine.marginal({y});
    EXPECT_NEAR(py.probability({{y, 0}}), 0.55, 1e-5);
    EXPECT_NEAR(py.probability({{y, 1}}), 0.45, 1e-5);

    const Potential pxz = engine.marginal({x, z});
    EXPECT_EQ(pxz.numVariables(), 2);
    for (int xv = 0; xv < 2; ++xv) {
        for (int zv = 0; zv < 2; ++zv) {
            EXPECT_NEAR(pxz.probability({{x, xv}, {z, zv}}),
                        refMarginal(table, all, {x, z}, {xv, zv}), 1e-5);
        }
    }
}

TEST(Inference, NaiveBayesConditional) {
    const auto factors = naiveBayesFactors();
    const Variable x(0, "X", 2), y(1, "Y", 2), z(2, "Z", 2);
    const std::vector<Variable> all = {x, y, z};
    const Inference engine(factors);
    const auto table = buildFullTable(factors, all);

    const Potential cond = engine.conditional({x}, {y, z});
    EXPECT_EQ(cond.numVariables(), 3);
    for (int yv = 0; yv < 2; ++yv) {
        for (int zv = 0; zv < 2; ++zv) {
            const double pyz = refMarginal(table, all, {y, z}, {yv, zv});
            double sumx = 0.0;
            for (int xv = 0; xv < 2; ++xv) {
                const double expected =
                    refMarginal(table, all, {x, y, z}, {xv, yv, zv}) / pyz;
                EXPECT_NEAR(cond.probability({{x, xv}, {y, yv}, {z, zv}}),
                            expected, 1e-5);
                sumx += cond.probability({{x, xv}, {y, yv}, {z, zv}});
            }
            EXPECT_NEAR(sumx, 1.0, 1e-5);
        }
    }
}

TEST(Inference, NaiveBayesConditionalWithEvidence) {
    const auto factors = naiveBayesFactors();
    const Variable x(0, "X", 2), y(1, "Y", 2), z(2, "Z", 2);
    const Inference engine(factors);

    // P(X | Y=0, Z=1) = 0.5 by the symmetry of the chosen numbers.
    const Potential p = engine.conditionalGiven({x}, {{y, 0}, {z, 1}});
    EXPECT_EQ(p.numVariables(), 1);
    EXPECT_EQ(p.variables()[0], x);
    EXPECT_NEAR(p.probability({{x, 0}}), 0.5, 1e-5);
    EXPECT_NEAR(p.probability({{x, 1}}), 0.5, 1e-5);

    // P(X=0 | Y=1) = P(X=0)P(Y=1|X=0) / P(Y=1) = 0.5*0.1 / 0.45.
    const Potential q = engine.conditionalGiven({x}, {{y, 1}});
    EXPECT_NEAR(q.probability({{x, 0}}), 0.5 * 0.1 / 0.45, 1e-5);
    EXPECT_NEAR(q.probability({{x, 1}}), 0.5 * 0.8 / 0.45, 1e-5);
}

TEST(Inference, NaiveBayesConditionalWithoutEvidence) {
    const auto factors = naiveBayesFactors();
    const Variable x(0, "X", 2), y(1, "Y", 2), z(2, "Z", 2);
    const Inference engine(factors);

    const Potential p = engine.conditional({x}, {});
    EXPECT_EQ(p.numVariables(), 1);
    EXPECT_NEAR(p.probability({{x, 0}}), 0.5, 1e-5);
    EXPECT_NEAR(p.probability({{x, 1}}), 0.5, 1e-5);
}

// ---------------------------------------------------------------------------
// Diamond network: A -> B, A -> C, D | (B, C).
// ---------------------------------------------------------------------------

TEST(Inference, DiamondFullJoint) {
    const auto factors = diamondFactors();
    const Variable a(0, "A", 2), b(1, "B", 2), c(2, "C", 2), d(3, "D", 2);
    const std::vector<Variable> all = {a, b, c, d};
    const Inference engine(factors);
    const auto table = buildFullTable(factors, all);

    const Potential joint = engine.fullJoint();
    EXPECT_EQ(joint.numVariables(), 4);

    std::vector<int> cur(4, 0);
    const std::vector<int> states = {2, 2, 2, 2};
    for (int k = 0; k < 16; ++k) {
        EXPECT_NEAR(joint.probability(toAssignment(all, cur)), table.at(cur), 1e-5)
            << "cell " << cur[0] << cur[1] << cur[2] << cur[3];
        for (int r = 3; r >= 0; --r) {
            if (++cur[r] < states[r]) break;
            cur[r] = 0;
        }
    }

    double sum = 0.0;
    for (const auto& [assign, p] : table) sum += p;
    EXPECT_NEAR(sum, 1.0, 1e-6);
}

TEST(Inference, DiamondMarginal) {
    const auto factors = diamondFactors();
    const Variable a(0, "A", 2), b(1, "B", 2), c(2, "C", 2), d(3, "D", 2);
    const std::vector<Variable> all = {a, b, c, d};
    const Inference engine(factors);
    const auto table = buildFullTable(factors, all);

    const Potential pd = engine.marginal({d});
    EXPECT_EQ(pd.numVariables(), 1);
    EXPECT_NEAR(pd.probability({{d, 0}}), 0.52, 1e-5);
    EXPECT_NEAR(pd.probability({{d, 1}}), 0.48, 1e-5);

    const Potential pb = engine.marginal({b});
    EXPECT_NEAR(pb.probability({{b, 0}}), 0.50, 1e-5);
    EXPECT_NEAR(pb.probability({{b, 1}}), 0.50, 1e-5);

    const Potential pc = engine.marginal({c});
    EXPECT_NEAR(pc.probability({{c, 0}}), 0.64, 1e-5);
    EXPECT_NEAR(pc.probability({{c, 1}}), 0.36, 1e-5);

    const Potential pbc = engine.marginal({b, c});
    EXPECT_EQ(pbc.numVariables(), 2);
    EXPECT_NEAR(pbc.probability({{b, 0}, {c, 0}}),
                refMarginal(table, all, {b, c}, {0, 0}), 1e-5);
    EXPECT_NEAR(pbc.probability({{b, 1}, {c, 1}}),
                refMarginal(table, all, {b, c}, {1, 1}), 1e-5);
}

TEST(Inference, DiamondPosterior) {
    const auto factors = diamondFactors();
    const Variable a(0, "A", 2), b(1, "B", 2), c(2, "C", 2), d(3, "D", 2);
    const Inference engine(factors);

    // P(A | D=1): computed by hand as {0.3875, 0.6125}.
    const Potential pa = engine.conditionalGiven({a}, {{d, 1}});
    EXPECT_EQ(pa.numVariables(), 1);
    EXPECT_NEAR(pa.probability({{a, 0}}), 0.3875, 1e-5);
    EXPECT_NEAR(pa.probability({{a, 1}}), 0.6125, 1e-5);

    // P(D | B=0, C=1) = {0.3, 0.7}.
    const Potential pd = engine.conditionalGiven({d}, {{b, 0}, {c, 1}});
    EXPECT_NEAR(pd.probability({{d, 0}}), 0.3, 1e-5);
    EXPECT_NEAR(pd.probability({{d, 1}}), 0.7, 1e-5);
}

TEST(Inference, DiamondConditionalAgainstBruteForce) {
    const auto factors = diamondFactors();
    const Variable a(0, "A", 2), b(1, "B", 2), c(2, "C", 2), d(3, "D", 2);
    const std::vector<Variable> all = {a, b, c, d};
    const Inference engine(factors);
    const auto table = buildFullTable(factors, all);

    const Potential cond = engine.conditional({a}, {b, c});
    EXPECT_EQ(cond.numVariables(), 3);
    for (int bv = 0; bv < 2; ++bv) {
        for (int cv = 0; cv < 2; ++cv) {
            const double pbc =
                refMarginal(table, all, {b, c}, {bv, cv});
            double suma = 0.0;
            for (int av = 0; av < 2; ++av) {
                const double expected =
                    refMarginal(table, all, {a, b, c}, {av, bv, cv}) / pbc;
                EXPECT_NEAR(cond.probability({{a, av}, {b, bv}, {c, cv}}),
                            expected, 1e-5);
                suma += cond.probability({{a, av}, {b, bv}, {c, cv}});
            }
            EXPECT_NEAR(suma, 1.0, 1e-5);
        }
    }
}

TEST(Inference, MarginalSubsetAgainstBruteForce) {
    const auto factors = diamondFactors();
    const Variable a(0, "A", 2), b(1, "B", 2), c(2, "C", 2), d(3, "D", 2);
    const std::vector<Variable> all = {a, b, c, d};
    const Inference engine(factors);
    const auto table = buildFullTable(factors, all);

    const Potential pa = engine.marginal({a});
    EXPECT_NEAR(pa.probability({{a, 0}}), 0.6, 1e-5);
    EXPECT_NEAR(pa.probability({{a, 1}}), 0.4, 1e-5);

    const Potential pad = engine.marginal({a, d});
    EXPECT_EQ(pad.numVariables(), 2);
    for (int av = 0; av < 2; ++av) {
        for (int dv = 0; dv < 2; ++dv) {
            EXPECT_NEAR(pad.probability({{a, av}, {d, dv}}),
                        refMarginal(table, all, {a, d}, {av, dv}), 1e-5);
        }
    }
}

TEST(Inference, EmptyQueryAndEmptyMarginal) {
    const auto factors = naiveBayesFactors();
    const Inference engine(factors);

    const Potential identity = engine.marginal({});
    EXPECT_EQ(identity.numVariables(), 0);
    EXPECT_NEAR(identity.probabilities()[0], 1.0, 1e-6);
}

TEST(Inference, ThrowsForUnknownVariable) {
    const auto factors = naiveBayesFactors();
    const Variable ghost(99, "Ghost", 2);
    const Inference engine(factors);
    EXPECT_THROW(engine.marginal({ghost}), std::invalid_argument);
    EXPECT_THROW(engine.conditional({ghost}, {}), std::invalid_argument);
}

TEST(Inference, ThrowsWhenQueryAndEvidenceOverlap) {
    const auto factors = naiveBayesFactors();
    const Variable x(0, "X", 2), y(1, "Y", 2);
    const Inference engine(factors);
    EXPECT_THROW(engine.conditional({x}, {x}), std::invalid_argument);
    EXPECT_THROW(engine.conditionalGiven({x}, {{x, 0}}), std::invalid_argument);
}

TEST(Inference, ThrowsForOutOfRangeEvidenceState) {
    const auto factors = naiveBayesFactors();
    const Variable x(0, "X", 2), y(1, "Y", 2);
    const Inference engine(factors);
    EXPECT_THROW(engine.conditionalGiven({x}, {{y, 2}}), std::invalid_argument);
    EXPECT_THROW(engine.conditionalGiven({x}, {{y, -1}}), std::invalid_argument);
}

TEST(Inference, ThrowsForZeroProbabilityEvidence) {
    // P(Y=1) is impossible: Y is 1 with probability zero in both X rows.
    const Variable x(0, "X", 2), y(1, "Y", 2);
    std::vector<Potential> factors = {
        Potential({x}, {0.5f, 0.5f}),
        Potential({x, y}, {1.0f, 0.0f, 1.0f, 0.0f}),  // P(Y=1 | x) = 0 always
    };
    const Inference engine(factors);
    EXPECT_THROW(engine.conditionalGiven({x}, {{y, 1}}), std::runtime_error);

    // The consistent evidence still works.
    const Potential p = engine.conditionalGiven({x}, {{y, 0}});
    EXPECT_NEAR(p.probability({{x, 0}}), 0.5, 1e-5);
    EXPECT_NEAR(p.probability({{x, 1}}), 0.5, 1e-5);
}
