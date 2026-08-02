#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "potential.h"

using bn::Potential;
using bn::Variable;

namespace {

const Variable x(0, "X", 2);
const Variable y(1, "Y", 2);
const Variable z(2, "Z", 3);

}  // namespace

TEST(Potential, ConstructFromProbabilities) {
    const Potential p({x, y}, {0.1f, 0.2f, 0.3f, 0.4f});
    EXPECT_EQ(p.numVariables(), 2);
    EXPECT_EQ(p.numEntries(), 4);
    const auto probs = p.probabilities();
    ASSERT_EQ(probs.size(), 4u);
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(probs[i], static_cast<float>(i + 1) / 10.0f, 1e-6);
    }
}

TEST(Potential, RejectsWrongNumberOfEntries) {
    EXPECT_THROW(Potential({x, y}, {0.5f, 0.5f}), std::invalid_argument);
    EXPECT_THROW(Potential({x, y}, {0.1f, 0.2f, 0.3f}), std::invalid_argument);
}

TEST(Potential, RejectsNegativeProbability) {
    EXPECT_THROW(Potential({x}, {-0.5f, 1.5f}), std::invalid_argument);
}

TEST(Potential, ValueAtAssignment) {
    // Row-major over (x, y): [x0y0, x0y1, x1y0, x1y1].
    const Potential p({x, y}, {0.1f, 0.2f, 0.3f, 0.4f});
    EXPECT_NEAR(p.probability({{x, 0}, {y, 0}}), 0.1f, 1e-6);
    EXPECT_NEAR(p.probability({{x, 0}, {y, 1}}), 0.2f, 1e-6);
    EXPECT_NEAR(p.probability({{x, 1}, {y, 0}}), 0.3f, 1e-6);
    EXPECT_NEAR(p.probability({{x, 1}, {y, 1}}), 0.4f, 1e-6);
    EXPECT_THROW(p.probability({{x, 0}}), std::invalid_argument);  // missing y
}

TEST(Potential, IdentityPotential) {
    const Potential identity;
    EXPECT_EQ(identity.numVariables(), 0);
    EXPECT_EQ(identity.numEntries(), 1);
    EXPECT_NEAR(identity.probabilities()[0], 1.0f, 1e-6);

    const Potential px({x}, {0.25f, 0.75f});
    const Potential joined = identity * px;
    EXPECT_EQ(joined.variables().size(), 1);
    EXPECT_NEAR(joined.probability({{x, 0}}), 0.25f, 1e-6);
    EXPECT_NEAR(joined.probability({{x, 1}}), 0.75f, 1e-6);
}

TEST(Potential, JoinOverSharedVariable) {
    const Potential p1({x}, {0.6f, 0.4f});
    const Potential p2({x, y}, {0.9f, 0.1f, 0.2f, 0.8f});
    const Potential joint = p1 * p2;

    ASSERT_EQ(joint.variables().size(), 2);
    EXPECT_EQ(joint.variables()[0], x);
    EXPECT_EQ(joint.variables()[1], y);

    EXPECT_NEAR(joint.probability({{x, 0}, {y, 0}}), 0.54f, 1e-5);
    EXPECT_NEAR(joint.probability({{x, 0}, {y, 1}}), 0.06f, 1e-5);
    EXPECT_NEAR(joint.probability({{x, 1}, {y, 0}}), 0.08f, 1e-5);
    EXPECT_NEAR(joint.probability({{x, 1}, {y, 1}}), 0.32f, 1e-5);
}

TEST(Potential, JoinOverDisjointVariables) {
    const Potential p1({x}, {0.6f, 0.4f});
    const Potential p2({z}, {0.2f, 0.3f, 0.5f});
    const Potential joint = p1 * p2;

    ASSERT_EQ(joint.variables().size(), 2);
    EXPECT_EQ(joint.variables()[0], x);
    EXPECT_EQ(joint.variables()[1], z);
    EXPECT_NEAR(joint.probability({{x, 0}, {z, 2}}), 0.6f * 0.5f, 1e-6);
    EXPECT_NEAR(joint.probability({{x, 1}, {z, 0}}), 0.4f * 0.2f, 1e-6);
}

TEST(Potential, JoinWithTernaryVariable) {
    // P(X) with 2 states and P(Z | X) where Z has 3 states.
    const Potential p1({x}, {0.5f, 0.5f});
    // rows of X: [0.2, 0.3, 0.5], [0.1, 0.4, 0.5]
    const Potential p2({x, z}, {0.2f, 0.3f, 0.5f, 0.1f, 0.4f, 0.5f});
    const Potential joint = p1 * p2;
    EXPECT_NEAR(joint.probability({{x, 1}, {z, 2}}), 0.5f * 0.5f, 1e-6);
}

TEST(Potential, Marginalize) {
    const Potential joint({x, y}, {0.54f, 0.06f, 0.08f, 0.32f});

    const Potential marg_y = joint.marginalize({y});
    ASSERT_EQ(marg_y.variables().size(), 1);
    EXPECT_EQ(marg_y.variables()[0], x);
    EXPECT_NEAR(marg_y.probability({{x, 0}}), 0.60f, 1e-6);
    EXPECT_NEAR(marg_y.probability({{x, 1}}), 0.40f, 1e-6);

    const Potential marg_x = joint.marginalize({x});
    ASSERT_EQ(marg_x.variables().size(), 1);
    EXPECT_EQ(marg_x.variables()[0], y);
    EXPECT_NEAR(marg_x.probability({{y, 0}}), 0.62f, 1e-6);
    EXPECT_NEAR(marg_x.probability({{y, 1}}), 0.38f, 1e-6);

    // Summing out everything yields the scalar partition function.
    const Potential scalar = joint.marginalize({x, y});
    EXPECT_EQ(scalar.numVariables(), 0);
    EXPECT_NEAR(scalar.probabilities()[0], 1.0f, 1e-5);
}

TEST(Potential, MarginalizeTernary) {
    const Variable a(0, "A", 2), b(1, "B", 2), c(2, "C", 3);
    // Row-major over (a, b, c): 2 * 2 * 3 = 12 entries.
    std::vector<float> probs;
    for (int i = 0; i < 12; ++i) probs.push_back(0.1f);
    const Potential joint({a, b, c}, probs);

    const Potential marg_ac = joint.marginalize({b});
    ASSERT_EQ(marg_ac.variables().size(), 2);
    // Each (a, c) cell sums over b -> 0.1 + 0.1 = 0.2.
    EXPECT_NEAR(marg_ac.probability({{a, 0}, {c, 2}}), 0.2f, 1e-6);
    EXPECT_NEAR(marg_ac.probability({{a, 1}, {c, 0}}), 0.2f, 1e-6);
}

TEST(Potential, Divide) {
    const Potential joint({x, y}, {0.54f, 0.06f, 0.08f, 0.32f});
    const Potential px({x}, {0.6f, 0.4f});
    const Potential conditional = joint / px;

    ASSERT_EQ(conditional.variables().size(), 2);
    EXPECT_EQ(conditional.variables()[0], x);
    EXPECT_EQ(conditional.variables()[1], y);
    EXPECT_NEAR(conditional.probability({{x, 0}, {y, 0}}), 0.9f, 1e-6);
    EXPECT_NEAR(conditional.probability({{x, 0}, {y, 1}}), 0.1f, 1e-6);
    EXPECT_NEAR(conditional.probability({{x, 1}, {y, 0}}), 0.2f, 1e-6);
    EXPECT_NEAR(conditional.probability({{x, 1}, {y, 1}}), 0.8f, 1e-6);
}

TEST(Potential, DivideThrowsForNonSubsetDivisor) {
    const Potential joint({x, y}, {0.54f, 0.06f, 0.08f, 0.32f});
    const Potential pz({z}, {0.2f, 0.3f, 0.5f});
    EXPECT_THROW(joint / pz, std::invalid_argument);
}

TEST(Potential, Normalize) {
    Potential p({x}, {2.0f, 8.0f});
    p.normalize();
    const auto probs = p.probabilities();
    EXPECT_NEAR(probs[0], 0.2f, 1e-6);
    EXPECT_NEAR(probs[1], 0.8f, 1e-6);
}

TEST(Potential, NormalizeOverQueryDims) {
    // Row-major over (x, y): columns of y sum to 1 before, rows of x do not.
    Potential p({x, y}, {0.50f, 0.50f, 0.10f, 0.90f});
    p.normalizeOver({x});
    // For y=0: {0.5, 0.1} -> {5/6, 1/6};  for y=1: {0.5, 0.9} -> {5/14, 9/14}.
    EXPECT_NEAR(p.probability({{x, 0}, {y, 0}}), 5.0f / 6.0f, 1e-6);
    EXPECT_NEAR(p.probability({{x, 1}, {y, 0}}), 1.0f / 6.0f, 1e-6);
    EXPECT_NEAR(p.probability({{x, 0}, {y, 1}}), 5.0f / 14.0f, 1e-6);
    EXPECT_NEAR(p.probability({{x, 1}, {y, 1}}), 9.0f / 14.0f, 1e-6);
    EXPECT_NEAR(p.probability({{x, 0}, {y, 0}}) + p.probability({{x, 1}, {y, 0}}),
                1.0f, 1e-6);
    EXPECT_NEAR(p.probability({{x, 0}, {y, 1}}) + p.probability({{x, 1}, {y, 1}}),
                1.0f, 1e-6);
}

TEST(Potential, Reorder) {
    const Potential p({x, y}, {0.54f, 0.06f, 0.08f, 0.32f});
    const Potential swapped = p.reorder({y, x});
    ASSERT_EQ(swapped.variables().size(), 2);
    EXPECT_EQ(swapped.variables()[0], y);
    EXPECT_EQ(swapped.variables()[1], x);
    // Row-major over (y, x): [y0x0, y0x1, y1x0, y1x1].
    EXPECT_NEAR(swapped.probability({{y, 0}, {x, 0}}), 0.54f, 1e-6);
    EXPECT_NEAR(swapped.probability({{y, 0}, {x, 1}}), 0.08f, 1e-6);
    EXPECT_NEAR(swapped.probability({{y, 1}, {x, 0}}), 0.06f, 1e-6);
    EXPECT_NEAR(swapped.probability({{y, 1}, {x, 1}}), 0.32f, 1e-6);
}

TEST(Potential, ReorderThrowsForNonPermutation) {
    const Potential p({x, y}, {0.54f, 0.06f, 0.08f, 0.32f});
    EXPECT_THROW(p.reorder({x}), std::invalid_argument);
    EXPECT_THROW(p.reorder({x, z}), std::invalid_argument);
}

TEST(Potential, Restrict) {
    const Potential p({x, y}, {0.54f, 0.06f, 0.08f, 0.32f});

    const Potential x0 = p.restrict({{x, 0}});
    ASSERT_EQ(x0.variables().size(), 1);
    EXPECT_EQ(x0.variables()[0], y);
    EXPECT_NEAR(x0.probability({{y, 0}}), 0.54f, 1e-6);
    EXPECT_NEAR(x0.probability({{y, 1}}), 0.06f, 1e-6);

    const Potential y1 = p.restrict({{y, 1}});
    ASSERT_EQ(y1.variables().size(), 1);
    EXPECT_NEAR(y1.probability({{x, 0}}), 0.06f, 1e-6);
    EXPECT_NEAR(y1.probability({{x, 1}}), 0.32f, 1e-6);
}
