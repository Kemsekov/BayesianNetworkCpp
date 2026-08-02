// Head-to-head benchmark driver (C++ side).
//
// Reads the same inputs as bench_py.py and computes the same queries with the
// local Bayesian network engine:
//   voting.csv       cleaned integer dataset (header row = variable order)
//   structure.txt    node<TAB>parent1,parent2 per node (same order as CSV)
//   queries.txt      id<TAB>q1,q2<TAB>e1=0;e2=1
//   cpts_py.txt      node<TAB>parents<TAB>row-major MLE CPTs from pgmpy
//
// Writes:
//   cpts_cpp.txt              MLE CPTs fitted from voting.csv
//   results_cpp.txt           query results using C++-fitted CPTs
//   results_cpp_from_py.txt   query results using pgmpy's exact CPTs
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "inference.h"
#include "potential.h"
#include "variable.h"

using bn::Inference;
using bn::Potential;
using bn::Variable;

namespace {

std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == delim) {
            out.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    out.push_back(cur);
    return out;
}

std::string trim(const std::string& s) {
    const std::size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    const std::size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

struct Csv {
    std::vector<std::string> names;
    std::vector<std::vector<int>> rows;
};

Csv loadCsv(const std::string& path) {
    Csv csv;
    std::ifstream in(path);
    if (!in) {
        std::cerr << "cannot open " << path << "\n";
        std::exit(1);
    }
    std::string line;
    bool first = true;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty()) continue;
        const auto parts = split(line, ',');
        if (first) {
            for (const auto& p : parts) csv.names.push_back(trim(p));
            first = false;
            continue;
        }
        std::vector<int> row;
        for (const auto& p : parts) row.push_back(std::stoi(trim(p)));
        csv.rows.push_back(std::move(row));
    }
    return csv;
}

struct Query {
    std::string id;
    std::vector<std::string> vars;
    std::map<std::string, int> evidence;
};

std::vector<Query> loadQueries(const std::string& path) {
    std::vector<Query> out;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        const auto parts = split(line, '\t');
        if (parts.size() < 2) continue;
        Query q;
        q.id = trim(parts[0]);
        for (const auto& v : split(parts[1], ',')) q.vars.push_back(trim(v));
        if (parts.size() >= 3) {
            for (const auto& ev : split(parts[2], ';')) {
                const auto kv = split(ev, '=');
                if (kv.size() == 2) q.evidence[trim(kv[0])] = std::stoi(trim(kv[1]));
            }
        }
        out.push_back(std::move(q));
    }
    return out;
}

/// Row-major strides over n binary dimensions (first dimension slowest).
std::vector<int> binStrides(int n) {
    std::vector<int> s(n);
    int acc = 1;
    for (int i = n - 1; i >= 0; --i) {
        s[i] = acc;
        acc *= 2;
    }
    return s;
}

/// Fit an MLE CPT for `node` given `parents` (indices), row-major over
/// (parents..., node) with the node varying fastest.
Potential fitCpt(const std::vector<Variable>& vars,
                 const std::vector<std::vector<int>>& rows,
                 int node_index, const std::vector<int>& parents) {
    const int ncols = static_cast<int>(parents.size()) + 1;
    const std::vector<int> strides = binStrides(ncols);
    const int total = 1 << ncols;
    std::vector<long long> counts(total, 0);
    for (const auto& row : rows) {
        int idx = 0;
        for (int c = 0; c < ncols; ++c) {
            const int col = (c < static_cast<int>(parents.size())) ? parents[c] : node_index;
            idx += row[col] * strides[c];
        }
        ++counts[idx];
    }
    // Normalize each parent combination (last dimension = node).
    const int parent_combos = 1 << static_cast<int>(parents.size());
    std::vector<float> probs(total);
    for (int combo = 0; combo < parent_combos; ++combo) {
        const long long sum = counts[combo * 2] + counts[combo * 2 + 1];
        if (sum > 0) {
            probs[combo * 2] = static_cast<float>(static_cast<double>(counts[combo * 2]) / sum);
            probs[combo * 2 + 1] = static_cast<float>(static_cast<double>(counts[combo * 2 + 1]) / sum);
        }
    }
    std::vector<Variable> scope;
    for (int p : parents) scope.push_back(vars[p]);
    scope.push_back(vars[node_index]);
    return Potential(std::move(scope), std::move(probs));
}

void writeResults(const std::string& path, const std::vector<Query>& queries,
                  const std::function<Potential(const Query&)>& runner) {
    std::ofstream out(path);
    out << "# id\tvars\tvalues\n";
    for (const Query& q : queries) {
        const Potential p = runner(q);
        out << q.id << "\t";
        for (std::size_t i = 0; i < p.variables().size(); ++i) {
            out << (i ? "," : "") << p.variables()[i].name();
        }
        out << "\t";
        const auto probs = p.probabilities();
        for (std::size_t i = 0; i < probs.size(); ++i) {
            out << (i ? "," : "") << std::setprecision(12) << probs[i];
        }
        out << "\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::string dir = (argc > 1) ? argv[1] : ".";
    const auto join = [&](const std::string& f) { return dir + "/" + f; };

    const Csv csv = loadCsv(join("voting.csv"));
    std::cout << "csv: " << csv.names.size() << " vars, " << csv.rows.size() << " rows\n";

    // Build variables in CSV order (binary states 0/1).
    std::vector<Variable> vars;
    std::map<std::string, int> name_to_index;
    for (std::size_t i = 0; i < csv.names.size(); ++i) {
        name_to_index[csv.names[i]] = static_cast<int>(i);
        vars.emplace_back(static_cast<int>(i), csv.names[i], 2);
    }

    // Read structure (parents per node).
    std::vector<std::vector<int>> parents_of(csv.names.size());
    {
        std::ifstream in(join("structure.txt"));
        std::string line;
        while (std::getline(in, line)) {
            line = trim(line);
            if (line.empty() || line[0] == '#') continue;
            const auto parts = split(line, '\t');
            const auto it = name_to_index.find(trim(parts[0]));
            if (it == name_to_index.end()) {
                std::cerr << "structure.txt references unknown node: " << parts[0] << "\n";
                return 1;
            }
            if (parts.size() >= 2 && !trim(parts[1]).empty()) {
                for (const auto& p : split(parts[1], ',')) {
                    const auto pit = name_to_index.find(trim(p));
                    if (pit == name_to_index.end()) {
                        std::cerr << "unknown parent: " << p << "\n";
                        return 1;
                    }
                    parents_of[it->second].push_back(pit->second);
                }
            }
        }
    }

    // ---- Fit MLE CPTs from the dataset (identical formula to Python) -------
    std::vector<Potential> fitted;
    {
        std::ofstream out(join("cpts_cpp.txt"));
        out << "# node\tparents\tvalues\n";
        for (std::size_t i = 0; i < vars.size(); ++i) {
            Potential cpt = fitCpt(vars, csv.rows, static_cast<int>(i), parents_of[i]);
            fitted.push_back(cpt);
            out << vars[i].name() << "\t";
            for (std::size_t j = 0; j < parents_of[i].size(); ++j) {
                out << (j ? "," : "") << vars[parents_of[i][j]].name();
            }
            out << "\t";
            const auto probs = cpt.probabilities();
            for (std::size_t j = 0; j < probs.size(); ++j) {
                out << (j ? "," : "") << std::setprecision(12) << probs[j];
            }
            out << "\n";
        }
    }

    // ---- Load pgmpy's exact CPTs (for engine isolation) --------------------
    std::vector<Potential> from_py;
    {
        std::ifstream in(join("cpts_py.txt"));
        std::string line;
        while (std::getline(in, line)) {
            line = trim(line);
            if (line.empty() || line[0] == '#') continue;
            const auto parts = split(line, '\t');
            const auto it = name_to_index.find(trim(parts[0]));
            if (it == name_to_index.end()) continue;
            std::vector<Variable> scope;
            const std::vector<std::string> pnames = split(parts[1], ',');
            for (const auto& p : pnames) {
                if (!trim(p).empty()) scope.push_back(vars[name_to_index[trim(p)]]);
            }
            scope.push_back(vars[it->second]);
            std::vector<float> probs;
            for (const auto& v : split(parts[2], ',')) probs.push_back(std::stof(trim(v)));
            from_py.push_back(Potential(std::move(scope), std::move(probs)));
        }
    }

    // ---- Queries ------------------------------------------------------------
    const std::vector<Query> queries = loadQueries(join("queries.txt"));

    const Inference infer_fitted(fitted);
    const Inference infer_py(from_py);

    // Resolve query variables against a variable table.
    auto resolve = [&](const std::vector<std::string>& names) {
        std::vector<Variable> qv;
        for (const auto& n : names) {
            const auto it = name_to_index.find(n);
            if (it == name_to_index.end()) {
                std::cerr << "unknown query variable: " << n << "\n";
                std::exit(1);
            }
            qv.push_back(vars[it->second]);
        }
        return qv;
    };
    auto resolveEvidence = [&](const std::map<std::string, int>& ev) {
        std::map<Variable, int> out;
        for (const auto& [n, v] : ev) out[vars[name_to_index.at(n)]] = v;
        return out;
    };

    auto runner = [&](const Inference& inf) {
        return [&](const Query& q) {
            return inf.conditionalGiven(resolve(q.vars), resolveEvidence(q.evidence));
        };
    };

    writeResults(join("results_cpp.txt"), queries, runner(infer_fitted));
    writeResults(join("results_cpp_from_py.txt"), queries, runner(infer_py));

    std::cout << "queries run: " << queries.size() << "\n";
    std::cout << "wrote cpts_cpp.txt, results_cpp.txt, results_cpp_from_py.txt\n";
    return 0;
}
