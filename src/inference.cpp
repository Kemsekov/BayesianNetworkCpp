#include "inference.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace bn {

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
    std::vector<Potential> factors = factors_;
    std::set<int> present;
    for (const Potential& f : factors) {
        for (const Variable& v : f.variables()) present.insert(v.id());
    }

    // Variables to sum out: every variable that is not part of the query.
    std::set<int> to_eliminate;
    for (int id : present) {
        if (!query_ids.count(id)) to_eliminate.insert(id);
    }

    const auto factorCount = [&](int id) {
        int count = 0;
        for (const Potential& f : factors) {
            if (f.contains(id)) ++count;
        }
        return count;
    };

    // Greedy min-degree elimination order (ties broken by lowest id) so that
    // intermediate tables stay as small as possible.
    while (!to_eliminate.empty()) {
        int best = -1;
        int best_count = std::numeric_limits<int>::max();
        for (int id : to_eliminate) {
            const int count = factorCount(id);
            if (count < best_count ||
                (count == best_count && (best == -1 || id < best))) {
                best = id;
                best_count = count;
            }
        }
        to_eliminate.erase(best);

        // Multiply every factor that mentions `best`, then sum it out.
        Potential bundle;
        std::vector<Potential> rest;
        rest.reserve(factors.size());
        for (Potential& f : factors) {
            if (f.contains(best)) {
                bundle = bundle * f;
            } else {
                rest.push_back(std::move(f));
            }
        }
        const auto it = std::find_if(
            bundle.variables().begin(), bundle.variables().end(),
            [&](const Variable& v) { return v.id() == best; });
        rest.push_back(bundle.marginalize({*it}));
        factors = std::move(rest);
    }

    Potential result;
    for (const Potential& f : factors) result = result * f;
    return result;
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

    const Potential numerator = marginal(num_vars);
    const Potential denominator = marginal(evidence);
    Potential ratio = numerator / denominator;  // over query ∪ evidence
    return ratio.normalizeOver(query);
}

Potential Inference::conditionalGiven(const std::vector<Variable>& query,
                                      const std::map<Variable, int>& evidence) const {
    std::vector<Variable> evidence_vars;
    for (const auto& [v, state] : evidence) evidence_vars.push_back(v);
    const Potential table = conditional(query, evidence_vars);
    return table.restrict(evidence);
}

}  // namespace bn
