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

/// Greedy weighted min-fill elimination order for the variables not in
/// `query_ids`. The optimal order is NP-hard; min-fill (fewest new edges in
/// the moral graph) is the standard near-optimal heuristic, with the size of
/// the resulting intermediate factor (product of neighbour state counts),
/// min-degree and then lowest id as tie-breakers. Variable ids are remapped
/// to dense indices first so the greedy loop uses cheap vector indexing
/// instead of std::map lookups.
std::vector<int> computeEliminationOrder(const std::vector<Potential>& factors,
                                         const std::set<int>& query_ids) {
    // One-time remap of variable ids to 0..n-1.
    std::map<int, int> dense_of;
    std::vector<int> dense_to_id;
    std::vector<int> states;             // dense id -> number of states
    std::vector<std::vector<int>> scopes;  // dense ids per factor scope
    for (const Potential& f : factors) {
        std::vector<int> dv;
        for (const Variable& v : f.variables()) {
            auto it = dense_of.find(v.id());
            if (it == dense_of.end()) {
                it = dense_of.emplace(v.id(), static_cast<int>(dense_to_id.size())).first;
                dense_to_id.push_back(v.id());
                states.push_back(v.num_states());
            }
            dv.push_back(it->second);
        }
        scopes.push_back(std::move(dv));
    }
    const int n = static_cast<int>(dense_to_id.size());

    std::vector<std::set<int>> adj(n);
    const auto addEdge = [&](int a, int b) {
        if (a == b) return;
        adj[a].insert(b);
        adj[b].insert(a);
    };
    for (const auto& dv : scopes) {
        for (std::size_t i = 0; i < dv.size(); ++i) {
            for (std::size_t j = i + 1; j < dv.size(); ++j) {
                addEdge(dv[i], dv[j]);
            }
        }
    }

    std::set<int> remaining;
    for (const auto& dv : scopes) {
        for (int d : dv) {
            if (!query_ids.count(dense_to_id[d])) remaining.insert(d);
        }
    }

    std::vector<int> order;
    while (!remaining.empty()) {
        int best = -1;
        int best_fill = std::numeric_limits<int>::max();
        long long best_weight = std::numeric_limits<long long>::max();
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
            long long weight = 1;  // size of the intermediate factor
            for (int a : nbr) weight *= states[a];
            const int deg = static_cast<int>(nbr.size());
            const bool better =
                fill < best_fill ||
                (fill == best_fill &&
                 (weight < best_weight ||
                  (weight == best_weight &&
                   (deg < best_deg ||
                    (deg == best_deg &&
                     (best == -1 || dense_to_id[v] < dense_to_id[best]))))));
            if (better) {
                best = v;
                best_fill = fill;
                best_weight = weight;
                best_deg = deg;
            }
        }
        remaining.erase(best);
        order.push_back(dense_to_id[best]);

        // Simulate the elimination: connect the neighbors of `best` (fill-in).
        const std::set<int> nbr = adj[best];
        adj[best].clear();
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
/// variable exactly once. Factors live in a pool referenced by index, and a
/// per-variable index of active factors avoids scanning the whole pool on
/// every elimination step.
Potential eliminateInOrder(const std::vector<Potential>& factors,
                           const std::vector<int>& order) {
    std::vector<Potential> pool = factors;  // single copy of the input list
    std::vector<char> active(pool.size(), 1);
    std::map<int, std::vector<int>> by_var;  // variable id -> active pool indices
    for (int i = 0; i < static_cast<int>(pool.size()); ++i) {
        for (const Variable& v : pool[i].variables()) {
            by_var[v.id()].push_back(i);
        }
    }
    for (int v : order) {
        Potential bundle;
        std::vector<int>& list = by_var[v];
        for (int i : list) {
            if (!active[i]) continue;
            bundle = bundle * pool[i];
            active[i] = 0;
        }
        // marginalize matches by id, so a placeholder Variable suffices.
        const int new_idx = static_cast<int>(pool.size());
        pool.push_back(bundle.marginalize({Variable(v, "", 0)}));
        active.push_back(1);
        for (const Variable& x : pool[new_idx].variables()) {
            by_var[x.id()].push_back(new_idx);
        }
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
