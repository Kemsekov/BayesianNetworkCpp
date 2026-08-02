#include <gtest/gtest.h>

#include <map>
#include <vector>

#include "inference.h"
#include "junction_tree.h"
#include "potential.h"
#include "variable.h"

using bn::Inference;
using bn::JunctionTree;
using bn::Potential;
using bn::Variable;

namespace {

const Variable x(0, "X", 2), y(1, "Y", 2), z(2, "Z", 2);
const Variable a(0, "A", 2), b(1, "B", 2), c(2, "C", 2), d(3, "D", 2);

std::vector<Potential> naiveBayesFactors() {
    return {
        Potential({x}, {0.5f, 0.5f}),
        Potential({x, y}, {0.9f, 0.1f, 0.2f, 0.8f}),
        Potential({x, z}, {0.8f, 0.2f, 0.1f, 0.9f}),
    };
}

std::vector<Potential> diamondFactors() {
    return {
        Potential({a}, {0.6f, 0.4f}),
        Potential({a, b}, {0.7f, 0.3f, 0.2f, 0.8f}),
        Potential({a, c}, {0.9f, 0.1f, 0.25f, 0.75f}),
        Potential({b, c, d},
                  {0.8f, 0.2f, 0.3f, 0.7f, 0.6f, 0.4f, 0.1f, 0.9f}),
    };
}

void expectSame(const Potential& p, const Potential& q, double tol = 1e-5) {
    ASSERT_EQ(p.variables().size(), q.variables().size());
    ASSERT_EQ(p.numEntries(), q.numEntries());
    const auto pv = p.probabilities();
    const auto qv = q.probabilities();
    for (std::size_t i = 0; i < pv.size(); ++i) {
        EXPECT_NEAR(pv[i], qv[i], tol) << "cell " << i;
    }
}

}  // namespace

TEST(JunctionTree, NaiveBayesMatchesInference) {
    const auto factors = naiveBayesFactors();
    const Inference inference(factors);
    JunctionTree jt(factors);
    EXPECT_GT(jt.numCliques(), 0);

    expectSame(jt.marginal({x}), inference.marginal({x}));
    expectSame(jt.marginal({y}), inference.marginal({y}));
    expectSame(jt.marginal({z}), inference.marginal({z}));
    expectSame(jt.marginal({x, y}), inference.marginal({x, y}));
    expectSame(jt.marginal({x, z}), inference.marginal({x, z}));
}

TEST(JunctionTree, DiamondMatchesInference) {
    const auto factors = diamondFactors();
    const Inference inference(factors);
    JunctionTree jt(factors);

    expectSame(jt.marginal({a}), inference.marginal({a}));
    expectSame(jt.marginal({b}), inference.marginal({b}));
    expectSame(jt.marginal({c}), inference.marginal({c}));
    expectSame(jt.marginal({d}), inference.marginal({d}));
    expectSame(jt.marginal({b, c}), inference.marginal({b, c}));

    // Sanity check a hand-computed value.
    const Potential pd = jt.marginal({d});
    EXPECT_NEAR(pd.probability({{d, 0}}), 0.52, 1e-5);
    EXPECT_NEAR(pd.probability({{d, 1}}), 0.48, 1e-5);
}

TEST(JunctionTree, EvidenceMatchesInference) {
    const auto factors = diamondFactors();
    const Inference inference(factors);
    JunctionTree jt(factors);

    jt.setEvidence({{d, 1}});
    const Potential p = jt.marginal({a});
    EXPECT_NEAR(p.probability({{a, 0}}), 0.3875, 1e-5);
    EXPECT_NEAR(p.probability({{a, 1}}), 0.6125, 1e-5);

    const Potential via_ve = inference.conditionalGiven({a}, {{d, 1}});
    expectSame(p, via_ve);

    // Two evidence values.
    jt.setEvidence({{b, 0}, {c, 1}});
    const Potential pd = jt.marginal({d});
    EXPECT_NEAR(pd.probability({{d, 0}}), 0.3, 1e-5);
    EXPECT_NEAR(pd.probability({{d, 1}}), 0.7, 1e-5);

    // Clearing the evidence restores the prior.
    jt.setEvidence({});
    expectSame(jt.marginal({d}), inference.marginal({d}));
}

TEST(JunctionTree, MapQuery) {
    const auto factors = diamondFactors();
    JunctionTree jt(factors);
    jt.setEvidence({{d, 1}});
    EXPECT_EQ(jt.mapQuery({a}).at(a), 1);

    jt.setEvidence({{b, 0}, {c, 1}});
    EXPECT_EQ(jt.mapQuery({d}).at(d), 1);
}

TEST(JunctionTree, MultiCliqueQueryFallsBack) {
    // {Y, Z} spans two cliques in the naive-Bayes tree -> falls back to VE.
    const auto factors = naiveBayesFactors();
    const Inference inference(factors);
    JunctionTree jt(factors);
    expectSame(jt.marginal({y, z}), inference.marginal({y, z}));
}

TEST(JunctionTree, ZeroProbabilityEvidenceThrows) {
    const Variable x0(0, "X", 2), y0(1, "Y", 2);
    std::vector<Potential> factors = {
        Potential({x0}, {0.5f, 0.5f}),
        Potential({x0, y0}, {1.0f, 0.0f, 1.0f, 0.0f}),
    };
    JunctionTree jt(factors);
    jt.setEvidence({{y0, 1}});
    EXPECT_THROW(jt.marginal({x0}), std::runtime_error);
}

TEST(JunctionTree, ThrowsForBadEvidence) {
    const auto factors = naiveBayesFactors();
    JunctionTree jt(factors);
    const Variable ghost(99, "Ghost", 2);
    EXPECT_THROW(jt.setEvidence({{ghost, 0}}), std::invalid_argument);
    EXPECT_THROW(jt.setEvidence({{y, 2}}), std::invalid_argument);
    EXPECT_THROW(jt.setEvidence({{y, -1}}), std::invalid_argument);
}
