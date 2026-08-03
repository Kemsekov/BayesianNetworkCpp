#include <gtest/gtest.h>

#include <random>
#include <string>
#include <vector>

#include "fit.h"
#include "inference.h"
#include "potential.h"
#include "variable.h"

using bn::Inference;
using bn::fitBayesian;
using bn::Potential;
using bn::Variable;

namespace {

std::vector<int> sampleChain(int nrows, unsigned seed) {
    // Chain X0 -> X1 -> X2, binary:
    //   P(X0=0)=0.6;  P(X1|X0): [0.9,0.1] / [0.2,0.8];  P(X2|X1): [0.7,0.3] / [0.4,0.6]
    std::vector<int> data(static_cast<std::size_t>(nrows) * 3);
    std::mt19937 rng(seed);
    for (int r = 0; r < nrows; ++r) {
        const int x0 = (rng() % 100 < 60) ? 0 : 1;
        const int x1 = (x0 == 0) ? (rng() % 100 < 90 ? 0 : 1)
                                 : (rng() % 100 < 20 ? 0 : 1);
        const int x2 = (x1 == 0) ? (rng() % 100 < 70 ? 0 : 1)
                                 : (rng() % 100 < 40 ? 0 : 1);
        data[static_cast<std::size_t>(r) * 3 + 0] = x0;
        data[static_cast<std::size_t>(r) * 3 + 1] = x1;
        data[static_cast<std::size_t>(r) * 3 + 2] = x2;
    }
    return data;
}

}  // namespace

TEST(FitBayesian, LearnsChainStructure) {
    const int nrows = 50000;
    const auto data = sampleChain(nrows, 42);
    const Inference inf = fitBayesian(data, nrows, 3, {"A", "B", "C"});

    const auto& factors = inf.factors();
    ASSERT_EQ(factors.size(), 3u);

    // Root (no parent) is A; B | A; C | B.
    bool has_A = false, has_AB = false, has_BC = false;
    for (const Potential& f : factors) {
        const auto& vs = f.variables();
        if (vs.size() == 1 && vs[0].id() == 0) has_A = true;
        if (vs.size() == 2 && vs[0].id() == 0 && vs[1].id() == 1) has_AB = true;
        if (vs.size() == 2 && vs[0].id() == 1 && vs[1].id() == 2) has_BC = true;
    }
    EXPECT_TRUE(has_A);
    EXPECT_TRUE(has_AB);
    EXPECT_TRUE(has_BC);
}

TEST(FitBayesian, MarginalsMatchEmpirical) {
    const int nrows = 50000;
    const auto data = sampleChain(nrows, 7);
    const Inference inf = fitBayesian(data, nrows, 3, {"A", "B", "C"});

    double emp_b0 = 0.0, emp_c0 = 0.0;
    for (int r = 0; r < nrows; ++r) {
        if (data[static_cast<std::size_t>(r) * 3 + 1] == 0) emp_b0 += 1.0;
        if (data[static_cast<std::size_t>(r) * 3 + 2] == 0) emp_c0 += 1.0;
    }
    emp_b0 /= nrows;
    emp_c0 /= nrows;

    const Variable b(1, "B", 2), c(2, "C", 2);
    const Potential pb = inf.marginal({b});
    const Potential pc = inf.marginal({c});
    EXPECT_NEAR(pb.probability({{b, 0}}), emp_b0, 0.01);
    EXPECT_NEAR(pb.probability({{b, 1}}), 1.0 - emp_b0, 0.01);
    EXPECT_NEAR(pc.probability({{c, 0}}), emp_c0, 0.01);

    // Conditional query is a valid distribution.
    const Potential p = inf.conditionalGiven({b}, {{c, 0}});
    const auto probs = p.probabilities();
    EXPECT_NEAR(probs[0] + probs[1], 1.0, 1e-5);
}

TEST(FitBayesian, DefaultNames) {
    const int nrows = 500;
    const auto data = sampleChain(nrows, 1);
    const Inference inf = fitBayesian(data, nrows, 3, {});
    const auto& vs = inf.factors()[0].variables();
    EXPECT_EQ(vs[0].name(), "x1");
    const auto& f1 = inf.factors()[1].variables();
    EXPECT_EQ(f1[0].name(), "x1");
    EXPECT_EQ(f1[1].name(), "x2");
}

TEST(FitBayesian, RejectsBadInput) {
    const int nrows = 100;
    std::vector<int> data(static_cast<std::size_t>(nrows) * 3, 0);
    data[0] = -1;  // negative state
    EXPECT_THROW(fitBayesian(data, nrows, 3, {}), std::invalid_argument);

    std::vector<int> data_ok(static_cast<std::size_t>(nrows) * 3, 0);
    EXPECT_THROW(fitBayesian(data_ok, nrows, 3, {"only-one"}), std::invalid_argument);
    EXPECT_THROW(fitBayesian(data_ok, nrows, 3, {"a", "b", "c", "d"}),
                 std::invalid_argument);
    EXPECT_THROW(fitBayesian({0, 1}, 3, 1, {}), std::invalid_argument);  // size mismatch
}
