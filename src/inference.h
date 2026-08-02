#pragma once

#include <map>
#include <set>
#include <vector>

#include "potential.h"

namespace bn {

/// Exact inference over a set of factors via variable elimination.
///
/// The factors (typically the conditional distributions P(X | Parents(X)) of a
/// Bayesian network) are such that their product equals the full joint
/// distribution P(X, Y, Z, ...). Instead of materializing the full joint and
/// summing out variables one by one, `marginal` eliminates the variables that
/// are not part of the query greedily (min-degree order), which drastically
/// reduces the size of the intermediate tables. It supports:
///   * full joint:                     P(X, Y, Z, ...)
///   * marginals / subset queries:     P(X, Z), others summed out
///   * conditionals:                   P(X | Y, Z) and P(X | Y=y, Z=z)
class Inference {
public:
    /// @param factors network potentials; their product must be the joint.
    explicit Inference(std::vector<Potential> factors);

    /// P(X, Y, Z, ...) over every variable appearing in the factors.
    Potential fullJoint() const;

    /// P(query): sums out every other variable via variable elimination.
    /// The resulting scope is the query re-ordered by global network order.
    Potential marginal(const std::vector<Variable>& query) const;

    /// P(query | evidence) as a table over query ∪ evidence, normalized over
    /// the query variables for every joint assignment of the evidence.
    Potential conditional(const std::vector<Variable>& query,
                          const std::vector<Variable>& evidence) const;

    /// P(query | evidence = values): a normalized distribution over the query
    /// variables for the specific observed evidence assignment.
    Potential conditionalGiven(const std::vector<Variable>& query,
                               const std::map<Variable, int>& evidence) const;

private:
    /// Variables ordered by first appearance across the factors.
    std::vector<Variable> globalOrder() const;
    /// Throws if any variable does not occur in the network.
    void checkVariables(const std::vector<Variable>& vars) const;
    /// Multiplies all factors and sums out every variable not in `query_ids`.
    Potential eliminate(const std::set<int>& query_ids) const;

    std::vector<Potential> factors_;
};

}  // namespace bn
