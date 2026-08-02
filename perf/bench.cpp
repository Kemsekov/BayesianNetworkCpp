// Performance benchmark for the Bayesian network engine.
//
// Measures the wall-clock time of the main inference operations on several
// deterministic synthetic networks. The output is a plain table so that runs
// before/after an optimization can be diffed. Each measurement returns the
// best (minimum) time over several trials to be robust against timer noise.
//
// Networks:
//   chain<N>     X0 -> X1 -> ... -> X(N-1)          (treewidth 1)
//   ladder<C>    2 x C grid, acyclic orientation     (treewidth ~2)
//   naive<L>     one hub + L leaves (naive Bayes)
//
// Operations:
//   fullJoint()                P(all) - only for small networks (<= 18 vars)
//   single marginals           P(X_i) for every variable
//   pair marginals             P(X_i, X_{i+1})
//   conditionals               P(X_i | evidence on 2 other vars)
//   repeated query             same conditional repeated K times (Inference)
//   repeated query (JT)        same conditional answered by JunctionTree
//
// The JunctionTree section is compiled only when the engine provides it
// (src/junction_tree.h), so the same file builds on both branches.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "inference.h"
#include "potential.h"
#include "variable.h"

#if defined(__has_include)
#if __has_include("junction_tree.h")
#include "junction_tree.h"
#define BENCH_HAS_JT 1
#else
#define BENCH_HAS_JT 0
#endif
#else
#define BENCH_HAS_JT 0
#endif

using bn::Inference;
using bn::Potential;
using bn::Variable;

#if BENCH_HAS_JT
using bn::JunctionTree;
#endif

namespace {

struct Network {
    std::string name;
    std::vector<Variable> vars;
    std::vector<Potential> factors;
};

// Random normalized row over `n` states (fixed seed for reproducibility).
std::vector<float> randomRow(std::mt19937& rng, int n) {
    std::vector<float> w(n);
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        w[i] = static_cast<float>(1 + rng() % 100);
        sum += w[i];
    }
    for (auto& x : w) x = static_cast<float>(x / sum);
    return w;
}

std::vector<float> cptFor(const std::vector<Variable>& scope,
                          std::mt19937& rng) {
    const int parents = static_cast<int>(scope.size()) - 1;
    std::vector<float> probs;
    for (int combo = 0; combo < (1 << parents); ++combo) {
        const auto row = randomRow(rng, 2);
        probs.insert(probs.end(), row.begin(), row.end());
    }
    return probs;
}

Network makeChain(int n, unsigned seed) {
    Network net;
    net.name = "chain" + std::to_string(n);
    for (int i = 0; i < n; ++i) net.vars.emplace_back(i, "X" + std::to_string(i), 2);
    std::mt19937 rng(seed);
    net.factors.emplace_back(std::vector<Variable>{net.vars[0]}, randomRow(rng, 2));
    for (int i = 1; i < n; ++i) {
        const std::vector<Variable> scope{net.vars[i - 1], net.vars[i]};
        net.factors.emplace_back(scope, cptFor(scope, rng));
    }
    return net;
}

Network makeLadder(int cols, unsigned seed) {
    Network net;
    net.name = "ladder2x" + std::to_string(cols);
    const int n = 2 * cols;
    for (int i = 0; i < n; ++i) net.vars.emplace_back(i, "V" + std::to_string(i), 2);
    std::mt19937 rng(seed);
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < cols; ++c) {
            const int idx = r * cols + c;
            std::vector<Variable> scope;
            if (c > 0) scope.push_back(net.vars[r * cols + (c - 1)]);
            if (r == 1) scope.push_back(net.vars[0 * cols + c]);
            scope.push_back(net.vars[idx]);
            net.factors.emplace_back(scope, cptFor(scope, rng));
        }
    }
    return net;
}

Network makeMixed(int n, unsigned seed) {
    // Chain with alternating 2/3 state variables (exercises the weighted
    // elimination order for non-uniform cardinalities).
    Network net;
    net.name = "mixed" + std::to_string(n);
    for (int i = 0; i < n; ++i) {
        net.vars.emplace_back(i, "M" + std::to_string(i), i % 2 == 0 ? 2 : 3);
    }
    std::mt19937 rng(seed);
    net.factors.emplace_back(std::vector<Variable>{net.vars[0]},
                             randomRow(rng, net.vars[0].num_states()));
    for (int i = 1; i < n; ++i) {
        const std::vector<Variable> scope{net.vars[i - 1], net.vars[i]};
        std::vector<float> probs;
        for (int p = 0; p < net.vars[i - 1].num_states(); ++p) {
            const auto row = randomRow(rng, net.vars[i].num_states());
            probs.insert(probs.end(), row.begin(), row.end());
        }
        net.factors.emplace_back(scope, std::move(probs));
    }
    return net;
}

Network makeNaive(int leaves, unsigned seed) {
    Network net;
    net.name = "naive" + std::to_string(leaves);
    const int n = leaves + 1;
    for (int i = 0; i < n; ++i) net.vars.emplace_back(i, "N" + std::to_string(i), 2);
    std::mt19937 rng(seed);
    net.factors.emplace_back(std::vector<Variable>{net.vars[0]}, randomRow(rng, 2));
    for (int l = 1; l < n; ++l) {
        const std::vector<Variable> scope{net.vars[0], net.vars[l]};
        net.factors.emplace_back(scope, cptFor(scope, rng));
    }
    return net;
}

using Clock = std::chrono::steady_clock;

// Best-of-`trials` average wall-clock time of `fn` (each call runs `iters`
// invocations); a warm-up call precedes the measurements.
double measure(const std::function<void()>& fn, int trials, int iters) {
    fn();  // warm-up
    double best = 1e18;
    for (int t = 0; t < trials; ++t) {
        const auto t0 = Clock::now();
        for (int i = 0; i < iters; ++i) fn();
        const auto t1 = Clock::now();
        const double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
        best = std::min(best, ms);
    }
    return best;
}

bool checkNormalized(const Potential& p, const std::string& what) {
    double sum = 0.0;
    for (float v : p.probabilities()) sum += v;
    if (std::fabs(sum - 1.0) > 1e-3) {
        std::cerr << "  [WARN] " << what << " does not sum to 1 (" << sum << ")\n";
        return false;
    }
    return true;
}

void runSuite(const Network& net, int seed) {
    const std::size_t n = net.vars.size();
    std::cout << "\n## " << net.name << "  (" << n << " vars, "
              << net.factors.size() << " factors)\n";

    bool ok = true;
    if (n <= 18) {
        Inference inf(net.factors);
        const double full =
            measure([&] { const auto j = inf.fullJoint(); (void)j; }, 3, 30);
        std::cout << std::left << std::setw(58) << "fullJoint P(all)"
                  << std::right << std::setw(12) << std::setprecision(4) << full
                  << " ms\n";
        const Potential j = inf.fullJoint();
        ok &= checkNormalized(j, "fullJoint");
    }

    // Single marginals: one sweep over every variable.
    {
        Inference inf(net.factors);
        std::vector<Potential> results;
        const double ms = measure(
            [&] {
                results.clear();
                for (const Variable& v : net.vars) results.push_back(inf.marginal({v}));
            },
            3, 5);
        std::cout << std::left << std::setw(58)
                  << "marginal single (all " + std::to_string(n) + " vars / sweep)"
                  << std::right << std::setw(12) << std::setprecision(4) << ms
                  << " ms\n";
        for (const auto& p : results) ok &= checkNormalized(p, "marginal single");
    }

    // Pair marginals over consecutive pairs.
    {
        Inference inf(net.factors);
        std::vector<Potential> results;
        const double ms = measure(
            [&] {
                results.clear();
                for (std::size_t i = 0; i + 1 < n; ++i) {
                    results.push_back(inf.marginal({net.vars[i], net.vars[i + 1]}));
                }
            },
            3, 5);
        std::cout << std::left << std::setw(58)
                  << "marginal pair (all " + std::to_string(std::max<std::size_t>(1, n - 1)) +
                         " consecutive pairs / sweep)"
                  << std::right << std::setw(12) << std::setprecision(4) << ms
                  << " ms\n";
        for (const auto& p : results) ok &= checkNormalized(p, "marginal pair");
    }

    // Conditionals: P(X_i | X_{i+1}=0, X_{i+2}=1).
    {
        Inference inf(net.factors);
        const int q = static_cast<int>(std::min<std::size_t>(8, n));
        std::vector<Potential> results;
        const double ms = measure(
            [&] {
                results.clear();
                for (int i = 0; i < q; ++i) {
                    results.push_back(inf.conditionalGiven(
                        {net.vars[i]},
                        {{net.vars[(i + 1) % n], 0}, {net.vars[(i + 2) % n], 1}}));
                }
            },
            3, 5);
        std::cout << std::left << std::setw(58)
                  << "conditional P(X_i | X_..=0, X_..=1) x " + std::to_string(q)
                  << std::right << std::setw(12) << std::setprecision(4) << ms
                  << " ms\n";
        for (const auto& p : results) ok &= checkNormalized(p, "conditional");
    }

    // Repeated identical query (where caching pays off).
    {
        const int K = 200;
        Inference inf(net.factors);
        const double ms = measure(
            [&] {
                for (int i = 0; i < K; ++i) {
                    const auto p = inf.conditionalGiven(
                        {net.vars[0]}, {{net.vars[1], 0}});
                    (void)p;
                }
            },
            3, 1);
        std::cout << std::left << std::setw(58)
                  << "repeated same conditional x " + std::to_string(K) + " (Inference)"
                  << std::right << std::setw(12) << std::setprecision(4) << ms
                  << " ms\n";
    }

#if BENCH_HAS_JT
    {
        const int K = 200;
        JunctionTree jt(net.factors);
        jt.setEvidence({{net.vars[1], 0}});
        const double ms = measure(
            [&] {
                for (int i = 0; i < K; ++i) {
                    const auto p = jt.marginal({net.vars[0]});
                    (void)p;
                }
            },
            3, 1);
        std::cout << std::left << std::setw(58)
                  << "repeated same marginal x " + std::to_string(K) + " (JunctionTree)"
                  << std::right << std::setw(12) << std::setprecision(4) << ms
                  << " ms\n";
    }
#endif

    std::cout << "  correctness sanity: " << (ok ? "OK" : "FAILED") << "\n";
    (void)seed;
}

}  // namespace

int main() {
    std::cout << "Bayesian network engine performance benchmark\n"
              << "compiled with JunctionTree: " << (BENCH_HAS_JT ? "yes" : "no")
              << "\n";

    runSuite(makeChain(16, 1), 1);
    runSuite(makeChain(60, 2), 2);
    runSuite(makeLadder(14, 3), 3);
    runSuite(makeNaive(15, 4), 4);
    runSuite(makeMixed(30, 5), 5);

    return 0;
}
