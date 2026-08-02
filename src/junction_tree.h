#pragma once

#include <map>
#include <set>
#include <utility>
#include <vector>

#include "inference.h"
#include "potential.h"
#include "variable.h"

namespace bn {

/// Exact inference via a junction tree (clique tree).
///
/// The clique tree is built once from the factor scopes: the moral graph is
/// triangulated with a min-fill elimination order, the maximal cliques form
/// the nodes, and a maximum-weight spanning tree over shared variables yields
/// the tree. Evidence is applied by restricting the factors and re-propagating
/// messages (Hugin-style collect + distribute), so many marginal queries can
/// be answered from the calibrated cliques without re-running elimination.
///
/// A query whose scope is contained in a single clique is answered directly
/// from the calibrated clique potential; any other query falls back to
/// variable elimination for correctness.
class JunctionTree {
public:
    explicit JunctionTree(std::vector<Potential> factors);

    /// Set the evidence and re-propagate. An empty map clears the evidence.
    /// @throws std::invalid_argument for an unknown variable or bad state.
    void setEvidence(const std::map<Variable, int>& evidence);

    /// P(query) under the current evidence, ordered by global network order.
    /// @throws std::runtime_error if the evidence has zero probability.
    Potential marginal(const std::vector<Variable>& query) const;

    /// Most likely joint assignment over `query` under the current evidence.
    std::map<Variable, int> mapQuery(const std::vector<Variable>& query) const;

    int numCliques() const { return static_cast<int>(cliques_.size()); }
    const std::vector<Variable>& cliqueScope(int i) const { return cliques_[i].scope; }

private:
    void buildStructure();
    void initializePotentials(const std::vector<Potential>& factors);
    void propagate();

    struct Clique {
        std::vector<Variable> scope;      // ordered by global variable order
        std::set<int> ids;                // sorted variable ids
        std::vector<int> factor_indices;  // indices into `factors_`
        Potential potential;              // calibrated, log-space over `scope`
    };

    std::vector<Potential> factors_;
    std::vector<Clique> cliques_;
    std::vector<std::pair<int, int>> edges_;  // clique tree edges (u < v)
    std::vector<Variable> global_order_;
    std::map<int, Variable> id_to_var_;
    std::map<Variable, int> evidence_;
    Inference fallback_;  // for queries not contained in a single clique
};

}  // namespace bn
