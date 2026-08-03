#include "fit.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace bn {

namespace {

/// Empirical mutual information I(X_a; X_b) computed from joint counts.
double mutualInfo(const std::vector<int>& data, int nrows, int ncols, int a,
                 int b, const std::vector<int>& states) {
    const int sa = states[a];
    const int sb = states[b];
    std::vector<long long> joint(static_cast<std::size_t>(sa) * sb, 0);
    std::vector<long long> marg_a(sa, 0), marg_b(sb, 0);
    for (int r = 0; r < nrows; ++r) {
        const int va = data[r * ncols + a];
        const int vb = data[r * ncols + b];
        ++joint[static_cast<std::size_t>(va) * sb + vb];
        ++marg_a[va];
        ++marg_b[vb];
    }
    const double n = nrows;
    double mi = 0.0;
    for (int x = 0; x < sa; ++x) {
        for (int y = 0; y < sb; ++y) {
            const long long c = joint[static_cast<std::size_t>(x) * sb + y];
            if (c == 0) continue;
            const double pxy = c / n;
            const double px = static_cast<double>(marg_a[x]) / n;
            const double py = static_cast<double>(marg_b[y]) / n;
            if (px > 0.0 && py > 0.0) {
                mi += pxy * std::log(pxy / (px * py));
            }
        }
    }
    return mi;
}

}  // namespace

Inference fitBayesian(const std::vector<int>& data, int nrows, int ncols,
                      const std::vector<std::string>& names) {
    if (nrows <= 0 || ncols <= 0 ||
        data.size() != static_cast<std::size_t>(nrows) * ncols) {
        throw std::invalid_argument("fit_bayesian: bad data dimensions");
    }
    if (!names.empty() && names.size() != static_cast<std::size_t>(ncols)) {
        throw std::invalid_argument(
            "fit_bayesian: names must have ncols entries");
    }

    // Per-column state counts (max value + 1) and input validation.
    std::vector<int> states(ncols, 1);
    for (int c = 0; c < ncols; ++c) {
        for (int r = 0; r < nrows; ++r) {
            const int v = data[r * ncols + c];
            if (v < 0) {
                throw std::invalid_argument(
                    "fit_bayesian: negative state value");
            }
            states[c] = std::max(states[c], v + 1);
        }
    }

    // Pairwise mutual information for the complete graph.
    std::vector<std::vector<double>> mi(ncols, std::vector<double>(ncols, 0.0));
    for (int a = 0; a < ncols; ++a) {
        for (int b = a + 1; b < ncols; ++b) {
            mi[a][b] = mi[b][a] =
                mutualInfo(data, nrows, ncols, a, b, states);
        }
    }

    // Maximum spanning tree (Prim) rooted at variable 0.
    std::vector<int> parent(ncols, -1);
    std::vector<double> best(ncols, -1.0);
    std::vector<char> in_tree(ncols, 0);
    in_tree[0] = 1;
    for (int v = 1; v < ncols; ++v) {
        best[v] = mi[0][v];
        parent[v] = 0;
    }
    for (int step = 1; step < ncols; ++step) {
        int u = -1;
        for (int v = 1; v < ncols; ++v) {
            if (!in_tree[v] && (u == -1 || best[v] > best[u])) u = v;
        }
        in_tree[u] = 1;
        for (int v = 1; v < ncols; ++v) {
            if (!in_tree[v] && mi[u][v] > best[v]) {
                best[v] = mi[u][v];
                parent[v] = u;
            }
        }
    }

    // Variables and maximum-likelihood CPTs.
    std::vector<Variable> vars;
    vars.reserve(ncols);
    for (int c = 0; c < ncols; ++c) {
        const std::string nm =
            names.empty() ? "x" + std::to_string(c + 1) : names[c];
        vars.emplace_back(c, nm, states[c]);
    }

    std::vector<Potential> factors;
    factors.reserve(ncols);
    for (int c = 0; c < ncols; ++c) {
        const int p = parent[c];
        if (p < 0) {
            std::vector<float> probs(states[c], 0.0f);
            for (int r = 0; r < nrows; ++r) {
                ++probs[data[r * ncols + c]];
            }
            for (float& v : probs) v /= nrows;
            factors.emplace_back(std::vector<Variable>{vars[c]},
                                 std::move(probs));
        } else {
            std::vector<long long> counts(
                static_cast<std::size_t>(states[p]) * states[c], 0);
            for (int r = 0; r < nrows; ++r) {
                ++counts[static_cast<std::size_t>(data[r * ncols + p]) *
                             states[c] +
                         data[r * ncols + c]];
            }
            std::vector<float> probs(counts.size());
            for (int x = 0; x < states[p]; ++x) {
                long long row_sum = 0;
                for (int y = 0; y < states[c]; ++y) {
                    row_sum += counts[static_cast<std::size_t>(x) * states[c] + y];
                }
                for (int y = 0; y < states[c]; ++y) {
                    const long long cnt =
                        counts[static_cast<std::size_t>(x) * states[c] + y];
                    probs[static_cast<std::size_t>(x) * states[c] + y] =
                        row_sum ? static_cast<float>(cnt) / row_sum : 0.0f;
                }
            }
            factors.emplace_back(std::vector<Variable>{vars[p], vars[c]},
                                 std::move(probs));
        }
    }
    return Inference(std::move(factors));
}

}  // namespace bn
