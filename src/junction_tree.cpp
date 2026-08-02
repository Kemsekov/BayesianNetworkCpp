#include "junction_tree.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace bn {

namespace {

int intersectionSize(const std::set<int>& a, const std::set<int>& b) {
    int n = 0;
    auto ia = a.begin();
    auto ib = b.begin();
    while (ia != a.end() && ib != b.end()) {
        if (*ia < *ib) {
            ++ia;
        } else if (*ib < *ia) {
            ++ib;
        } else {
            ++n;
            ++ia;
            ++ib;
        }
    }
    return n;
}

int productStates(const std::vector<Variable>& scope) {
    int p = 1;
    for (const Variable& v : scope) p *= v.num_states();
    return p;
}

struct Dsu {
    std::vector<int> parent;
    explicit Dsu(int n) : parent(n) {
        for (int i = 0; i < n; ++i) parent[i] = i;
    }
    int find(int x) { return parent[x] == x ? x : parent[x] = find(parent[x]); }
    void unite(int a, int b) {
        a = find(a);
        b = find(b);
        if (a != b) parent[a] = b;
    }
};

}  // namespace

JunctionTree::JunctionTree(std::vector<Potential> factors)
    : factors_(std::move(factors)), fallback_(factors_) {
    buildStructure();
    setEvidence({});
}

void JunctionTree::buildStructure() {
    // Global variable order (first appearance) and id -> Variable map.
    std::set<int> seen;
    for (const Potential& f : factors_) {
        for (const Variable& v : f.variables()) {
            if (seen.insert(v.id()).second) {
                global_order_.push_back(v);
                id_to_var_[v.id()] = v;
            }
        }
    }

    // Moral graph over factor scopes.
    std::map<int, std::set<int>> adj;
    const auto addEdge = [&](int a, int b) {
        if (a == b) return;
        adj[a].insert(b);
        adj[b].insert(a);
    };
    for (const Potential& f : factors_) {
        const auto& vs = f.variables();
        for (std::size_t i = 0; i < vs.size(); ++i) {
            for (std::size_t j = i + 1; j < vs.size(); ++j) {
                addEdge(vs[i].id(), vs[j].id());
            }
        }
    }

    // Triangulate with a min-fill elimination order, recording clusters.
    std::set<int> remaining;
    for (const auto& [id, v] : id_to_var_) remaining.insert(id);
    std::vector<std::set<int>> clusters;
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
        std::set<int> cluster = adj[best];
        cluster.insert(best);
        clusters.push_back(cluster);

        const std::set<int> nbr = adj[best];
        adj.erase(best);
        for (int a : nbr) adj[a].erase(best);
        for (int a : nbr) {
            for (int b : nbr) {
                if (a < b) addEdge(a, b);
            }
        }
    }

    // Keep maximal cliques (dedupe, drop cliques contained in another).
    std::vector<std::set<int>> maximal;
    for (const std::set<int>& c : clusters) {
        bool duplicate = false;
        for (const std::set<int>& d : maximal) {
            if (c == d) {
                duplicate = true;
                break;
            }
        }
        if (duplicate) continue;
        bool dominated = false;
        for (const std::set<int>& d : clusters) {
            if (c == d) continue;
            if (c.size() < d.size() &&
                std::includes(d.begin(), d.end(), c.begin(), c.end())) {
                dominated = true;
                break;
            }
        }
        if (!dominated) maximal.push_back(c);
    }

    cliques_.clear();
    for (const std::set<int>& c : maximal) {
        Clique cl;
        cl.ids = c;
        for (const Variable& v : global_order_) {
            if (c.count(v.id())) cl.scope.push_back(v);
        }
        cliques_.push_back(std::move(cl));
    }

    // Assign each factor to the first clique whose scope contains it.
    for (std::size_t fi = 0; fi < factors_.size(); ++fi) {
        std::set<int> fids;
        for (const Variable& v : factors_[fi].variables()) fids.insert(v.id());
        bool assigned = false;
        for (std::size_t ci = 0; ci < cliques_.size(); ++ci) {
            if (std::includes(cliques_[ci].ids.begin(), cliques_[ci].ids.end(),
                              fids.begin(), fids.end())) {
                cliques_[ci].factor_indices.push_back(static_cast<int>(fi));
                assigned = true;
                break;
            }
        }
        if (!assigned) {
            throw std::runtime_error(
                "JunctionTree: factor scope not contained in any clique");
        }
    }

    // Clique tree: maximum-weight spanning tree (Kruskal) over shared vars.
    struct Cand {
        int u, v, w;
    };
    std::vector<Cand> cands;
    for (std::size_t i = 0; i < cliques_.size(); ++i) {
        for (std::size_t j = i + 1; j < cliques_.size(); ++j) {
            const int w = intersectionSize(cliques_[i].ids, cliques_[j].ids);
            if (w > 0) cands.push_back({static_cast<int>(i), static_cast<int>(j), w});
        }
    }
    std::sort(cands.begin(), cands.end(),
              [](const Cand& a, const Cand& b) { return a.w > b.w; });
    Dsu dsu(static_cast<int>(cliques_.size()));
    edges_.clear();
    for (const Cand& c : cands) {
        if (dsu.find(c.u) != dsu.find(c.v)) {
            dsu.unite(c.u, c.v);
            edges_.push_back({c.u, c.v});
        }
    }
}

void JunctionTree::initializePotentials(const std::vector<Potential>& factors) {
    for (Clique& cl : cliques_) {
        cl.potential = Potential(cl.scope, std::vector<float>(productStates(cl.scope), 1.0f));
        for (int fi : cl.factor_indices) {
            cl.potential = cl.potential * factors[fi];
        }
    }
}

void JunctionTree::propagate() {
    std::map<int, std::vector<int>> adj;
    for (const auto& [u, v] : edges_) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    std::vector<char> visited(cliques_.size(), 0);
    for (int s = 0; s < static_cast<int>(cliques_.size()); ++s) {
        if (visited[s]) continue;

        // BFS from s; `queue` keeps parents before children.
        std::vector<int> parent(cliques_.size(), -2);
        std::vector<std::vector<int>> children(cliques_.size());
        std::vector<int> queue{s};
        parent[s] = -1;
        visited[s] = 1;
        for (std::size_t qi = 0; qi < queue.size(); ++qi) {
            const int u = queue[qi];
            for (int w : adj[u]) {
                if (!visited[w]) {
                    visited[w] = 1;
                    parent[w] = u;
                    children[u].push_back(w);
                    queue.push_back(w);
                }
            }
        }

        // Collect: messages travel from the leaves towards the root.
        std::map<int, Potential> msg;
        for (auto it = queue.rbegin(); it != queue.rend(); ++it) {
            const int u = *it;
            Potential combined = cliques_[u].potential;
            for (int c : children[u]) combined = combined * msg[c];
            cliques_[u].potential = combined;
            if (parent[u] != -1) {
                std::vector<Variable> to_sum;
                for (const Variable& v : cliques_[u].scope) {
                    if (!cliques_[parent[u]].ids.count(v.id())) to_sum.push_back(v);
                }
                msg[u] = combined.marginalize(to_sum);
            }
        }

        // Distribute: messages travel from the root back to the leaves.
        for (int u : queue) {
            for (int c : children[u]) {
                std::vector<Variable> to_sum;
                for (const Variable& v : cliques_[u].scope) {
                    if (!cliques_[c].ids.count(v.id())) to_sum.push_back(v);
                }
                const Potential m = cliques_[u].potential.marginalize(to_sum);
                cliques_[c].potential = cliques_[c].potential * m;
            }
        }
    }
}

void JunctionTree::setEvidence(const std::map<Variable, int>& evidence) {
    for (const auto& [v, state] : evidence) {
        const auto it = id_to_var_.find(v.id());
        if (it == id_to_var_.end()) {
            throw std::invalid_argument(
                "JunctionTree: evidence variable '" + v.name() +
                "' does not occur in the network");
        }
        if (state < 0 || state >= it->second.num_states()) {
            throw std::invalid_argument("JunctionTree: evidence state out of range");
        }
    }
    evidence_ = evidence;

    std::vector<Potential> restricted;
    restricted.reserve(factors_.size());
    for (const Potential& f : factors_) restricted.push_back(f.restrict(evidence));

    initializePotentials(restricted);
    propagate();
}

Potential JunctionTree::marginal(const std::vector<Variable>& query) const {
    if (query.empty()) return Potential::fromLog({}, {0.0f});

    std::set<int> q;
    for (const Variable& v : query) q.insert(v.id());

    for (const Clique& cl : cliques_) {
        if (std::includes(cl.ids.begin(), cl.ids.end(), q.begin(), q.end())) {
            std::vector<Variable> to_sum;
            for (const Variable& v : cl.scope) {
                if (!q.count(v.id())) to_sum.push_back(v);
            }
            Potential p = cl.potential.marginalize(to_sum);
            const std::vector<float>& lt = p.logTable();
            const bool all_zero = !lt.empty() &&
                std::all_of(lt.begin(), lt.end(),
                            [](float x) { return !std::isfinite(x); });
            if (all_zero) {
                throw std::runtime_error(
                    "JunctionTree::marginal: evidence has zero probability");
            }
            p.normalize();
            std::vector<Variable> wanted;
            for (const Variable& v : global_order_) {
                if (q.count(v.id())) wanted.push_back(v);
            }
            return p.reorder(wanted);
        }
    }
    return fallback_.conditionalGiven(query, evidence_);
}

std::map<Variable, int> JunctionTree::mapQuery(const std::vector<Variable>& query) const {
    const Potential p = marginal(query);
    const std::vector<float>& logt = p.logTable();
    int best = 0;
    for (int i = 1; i < static_cast<int>(logt.size()); ++i) {
        if (logt[i] > logt[best]) best = i;
    }
    std::map<Variable, int> result;
    const auto& vars = p.variables();
    std::vector<int> strides;
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
