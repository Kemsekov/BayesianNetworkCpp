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

/// Multiplies all factors and sums out every variable not in `query_ids`.
/// `factors` are moved through but never modified in place.
Potential eliminateOn(const std::vector<Potential>& factors,
                      const std::set<int>& query_ids) {
    std::vector<Potential> work = factors;
    std::set<int> present;
    for (const Potential& f : work) {
        for (const Variable& v : f.variables()) present.insert(v.id());
    }

    // Variables to sum out: every variable that is not part of the query.
    std::set<int> to_eliminate;
    for (int id : present) {
        if (!query_ids.count(id)) to_eliminate.insert(id);
    }

    const auto factorCount = [&](int id) {
        int count = 0;
        for (const Potential& f : work) {
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
        rest.reserve(work.size());
        for (Potential& f : work) {
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
        work = std::move(rest);
    }

    Potential result;
    for (const Potential& f : work) result = result * f;
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
    return eliminateOn(factors_, query_ids);
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

    Potential p = eliminateOn(restricted, query_ids);

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

}  // namespace bn
