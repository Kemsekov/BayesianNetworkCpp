#include "inference.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace bn {

namespace {

bool disjointSets(const std::vector<Variable>& a, const std::vector<Variable>& b) {
    for (const Variable& x : a) {
        for (const Variable& y : b) {
            if (x.id() == y.id()) return false;
        }
    }
    return true;
}

/// Greedy min-fill elimination order for the variables not in `query_ids`.
/// The optimal order is NP-hard; min-fill (fewest new edges in the moral
/// graph) is the standard near-optimal heuristic, with min-degree and then
/// lowest id as tie-breakers. The simulated graph exactly tracks the scopes
/// of the current factors, so the order is valid for `eliminateInOrder`.
std::vector<int> computeEliminationOrder(const std::vector<Potential>& factors,
                                         const std::set<int>& query_ids) {
    std::map<int, std::set<int>> adj;
    const auto addEdge = [&](int a, int b) {
        if (a == b) return;
        adj[a].insert(b);
        adj[b].insert(a);
    };
    for (const Potential& f : factors) {
        const auto& vs = f.variables();
        for (std::size_t i = 0; i < vs.size(); ++i) {
            for (std::size_t j = i + 1; j < vs.size(); ++j) {
                addEdge(vs[i].id(), vs[j].id());
            }
        }
    }

    std::set<int> remaining;
    for (const Potential& f : factors) {
        for (const Variable& v : f.variables()) {
            if (!query_ids.count(v.id())) remaining.insert(v.id());
        }
    }

    std::vector<int> order;
    while (!remaining.empty()) {
        int best = -1;
        int best_fill = std::numeric_limits<int>::max();
        int best_deg = std::numeric_limits<int>::max();
        for (int v : remaining) {
            const std::set<int>& nbr = adj[v];
            int fill = 0;
            for (int a : nbr) {
                const std::set<int>& set_a = adj[a];
                for (int b : nbr) {
                    if (b > a && !set_a.count(b)) ++fill;
                }
            }
            const int deg = static_cast<int>(nbr.size());
            const bool better =
                fill < best_fill ||
                (fill == best_fill &&
                 (deg < best_deg ||
                  (deg == best_deg && (best == -1 || v < best))));
            if (better) {
                best = v;
                best_fill = fill;
                best_deg = deg;
            }
        }
        remaining.erase(best);
        order.push_back(best);

        // Simulate the elimination: connect the neighbors of `best` (fill-in).
        const std::set<int> nbr = adj[best];
        adj.erase(best);
        for (int a : nbr) adj[a].erase(best);
        for (int a : nbr) {
            for (int b : nbr) {
                if (a < b) addEdge(a, b);
            }
        }
    }
    return order;
}

/// Eliminate the variables listed in `order` from `factors` and return the
/// product of the remaining factors. `order` must contain every non-query
/// variable exactly once. Factors are stored in a pool and referenced by
/// index, so no factor is copied or re-allocated during elimination.
Potential eliminateInOrder(const std::vector<Potential>& factors,
                           const std::vector<int>& order) {
    std::vector<Potential> pool = factors;  // single copy of the input list
    std::vector<char> active(pool.size(), 1);
    for (int v : order) {
        std::vector<int> idxs;
        for (int i = 0; i < static_cast<int>(pool.size()); ++i) {
            if (active[i] && pool[i].contains(v)) idxs.push_back(i);
        }
        Potential bundle;
        for (int i : idxs) {
            bundle = bundle * pool[i];
            active[i] = 0;
        }
        const auto it = std::find_if(
            bundle.variables().begin(), bundle.variables().end(),
            [&](const Variable& x) { return x.id() == v; });
        pool.push_back(bundle.marginalize({*it}));
        active.push_back(1);
    }
    Potential result;
    for (int i = 0; i < static_cast<int>(pool.size()); ++i) {
        if (active[i]) result = result * pool[i];
    }
    return result;
}

}  // namespace

Inference::Inference(std::vector<Potential> factors)
    : factors_(std::move(factors)) {}

std::vector<Variable> Inference::globalOrder() const {
    std::vector<Variable> order;
    std::set<int> seen;
    for (const Potential& f : factors_) {
        for (const Variable& v : f.variables()) {
            if (!seen.count(v.id())) {
                seen.insert(v.id());
                order.push_back(v);
            }
        }
    }
    return order;
}

void Inference::checkVariables(const std::vector<Variable>& vars) const {
    const std::vector<Variable> order = globalOrder();
    for (const Variable& v : vars) {
        const bool present = std::any_of(
            order.begin(), order.end(),
            [&](const Variable& x) { return x.id() == v.id(); });
        if (!present) {
            throw std::invalid_argument(
                "Inference: variable '" + v.name() + "' does not occur in the "
                "network factors");
        }
    }
}

Potential Inference::eliminate(const std::set<int>& query_ids) const {
    auto it = order_cache_.find(query_ids);
    if (it == order_cache_.end()) {
        it = order_cache_
                 .emplace(query_ids, computeEliminationOrder(factors_, query_ids))
                 .first;
    }
    return eliminateInOrder(factors_, it->second);
}

Potential Inference::fullJoint() const {
    Potential result;
    for (const Potential& f : factors_) result = result * f;
    return result;
}

Potential Inference::marginal(const std::vector<Variable>& query) const {
    checkVariables(query);
    std::set<int> query_ids;
    for (const Variable& v : query) query_ids.insert(v.id());

    Potential raw = eliminate(query_ids);

    // Present the result in the network's global order restricted to the query.
    std::vector<Variable> wanted;
    for (const Variable& v : globalOrder()) {
        if (query_ids.count(v.id())) wanted.push_back(v);
    }
    return raw.reorder(wanted);
}

Potential Inference::conditional(const std::vector<Variable>& query,
                                 const std::vector<Variable>& evidence) const {
    checkVariables(query);
    checkVariables(evidence);
    if (!disjointSets(query, evidence)) {
        throw std::invalid_argument(
            "Inference: query and evidence must be disjoint");
    }

    std::set<int> query_ids;
    for (const Variable& v : query) query_ids.insert(v.id());

    // Joint scope: query followed by evidence (in global order) not in query.
    std::vector<Variable> num_vars = query;
    for (const Variable& v : globalOrder()) {
        if (!query_ids.count(v.id())) {
            const bool is_evidence = std::any_of(
                evidence.begin(), evidence.end(),
                [&](const Variable& e) { return e.id() == v.id(); });
            if (is_evidence) num_vars.push_back(v);
        }
    }
    std::set<int> keep;
    for (const Variable& v : num_vars) keep.insert(v.id());

    // A single elimination over query ∪ evidence is enough: the denominator
    // P(evidence) is the joint with the query variables summed out.
    Potential joint = eliminate(keep);
    std::vector<Variable> wanted;
    for (const Variable& v : globalOrder()) {
        if (keep.count(v.id())) wanted.push_back(v);
    }
    joint = joint.reorder(wanted);

    const Potential denominator = joint.marginalize(query);
    Potential ratio = joint / denominator;  // over query ∪ evidence
    return ratio.normalizeOver(query);
}

Potential Inference::conditionalGiven(const std::vector<Variable>& query,
                                      const std::map<Variable, int>& evidence) const {
    checkVariables(query);
    std::vector<Variable> evidence_vars;
    for (const auto& [v, state] : evidence) evidence_vars.push_back(v);
    checkVariables(evidence_vars);
    if (!disjointSets(query, evidence_vars)) {
        throw std::invalid_argument(
            "Inference: query and evidence must be disjoint");
    }
    for (const auto& [v, state] : evidence) {
        if (state < 0 || state >= v.num_states()) {
            throw std::invalid_argument(
                "Inference: evidence state out of range");
        }
    }

    if (evidence.empty()) return marginal(query);

    // Apply the evidence into the factors first: every factor mentioning an
    // evidence variable is restricted, which shrinks its scope. A single
    // elimination pass then sums out everything outside the query, giving
    // P(query, evidence) directly.
    std::vector<Potential> restricted;
    restricted.reserve(factors_.size());
    for (const Potential& f : factors_) restricted.push_back(f.restrict(evidence));

    std::set<int> query_ids;
    for (const Variable& v : query) query_ids.insert(v.id());

    Potential p = eliminateInOrder(restricted,
                                   computeEliminationOrder(restricted, query_ids));

    // Zero-probability evidence guard: if every entry is log(0).
    const std::vector<float>& logt = p.logTable();
    const bool all_zero = !logt.empty() &&
        std::all_of(logt.begin(), logt.end(),
                    [](float x) { return !std::isfinite(x); });
    if (all_zero) {
        throw std::runtime_error(
            "Inference::conditionalGiven: evidence has zero probability");
    }

    p.normalize();

    std::vector<Variable> wanted;
    for (const Variable& v : globalOrder()) {
        if (query_ids.count(v.id())) wanted.push_back(v);
    }
    return p.reorder(wanted);
}

std::map<Variable, int> Inference::mapQuery(
    const std::vector<Variable>& query,
    const std::map<Variable, int>& evidence) const {
    const Potential p = conditionalGiven(query, evidence);
    const std::vector<float>& logt = p.logTable();
    int best = 0;
    for (int i = 1; i < static_cast<int>(logt.size()); ++i) {
        if (logt[i] > logt[best]) best = i;
    }
    std::map<Variable, int> result;
    const auto& vars = p.variables();
    std::vector<int> strides;
    strides.reserve(vars.size());
    int acc = 1;
    for (int i = static_cast<int>(vars.size()) - 1; i >= 0; --i) {
        strides.insert(strides.begin(), acc);
        acc *= vars[i].num_states();
    }
    for (std::size_t i = 0; i < vars.size(); ++i) {
        result[vars[i]] = (best / strides[i]) % vars[i].num_states();
    }
    return result;
}

}  // namespace bn
