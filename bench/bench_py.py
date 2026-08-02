#!/usr/bin/env python3
"""Head-to-head benchmark driver (Python / pgmpy side).

1. Loads the cleaned voting.csv dataset.
2. Learns a Chow-Liu tree structure (deterministic, community standard).
3. Fits CPTs by exact maximum-likelihood relative frequencies with a canonical
   [0,1] state ordering (identical formula to the C++ side).
4. Runs exact variable-elimination queries with pgmpy.
5. Writes: structure.txt, cpts_py.txt, queries.txt, results_py.txt
"""
import itertools
import sys

import numpy as np
import pandas as pd
from pgmpy.estimators import TreeSearch
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

sys.path.insert(0, ".")

df = pd.read_csv("voting.csv")
nodes = list(df.columns)
print(f"nodes ({len(nodes)}): {nodes}")

# ---- 1. Learn a Chow-Liu tree (deterministic) --------------------------------
learner = TreeSearch(df)
dag = learner.estimate(estimator_type="chow-liu", show_progress=False)
edges = sorted((str(u), str(v)) for u, v in dag.edges())
print(f"learned tree edges ({len(edges)}): {edges}")

# ---- 2. Fit exact MLE CPTs with canonical [0,1] states -----------------------
def fit_counts(df, node, parents, card=2):
    """Counts flat in row-major order over (parents..., node), node fastest."""
    ncols = len(parents) + 1
    data = df[parents + [node]].to_numpy(dtype=np.int64)
    shape = [card] * ncols
    strides = []
    acc = 1
    for i in range(ncols - 1, -1, -1):
        strides.insert(0, acc)
        acc *= card
    flat = data[:, -1].copy()
    for i, p in enumerate(parents):
        flat += data[:, i] * strides[i]
    counts = np.bincount(flat, minlength=acc).astype(np.float64)
    counts = counts.reshape(shape)
    # normalize over node (last axis)
    denom = counts.sum(axis=-1, keepdims=True)
    probs = np.divide(counts, denom, out=np.zeros_like(counts), where=denom != 0)
    # row-major over (parents..., node)
    return probs.ravel()

parent_of = {v: [] for v in nodes}
for u, v in edges:
    parent_of[v].append(u)

cpds = []
flat_cpts = {}  # node -> (parents, row-major probs)
for node in nodes:
    parents = parent_of[node]
    probs = fit_counts(df, node, parents)
    flat_cpts[node] = (parents, probs)
    if not parents:
        cpd = TabularCPD(
            node, 2, [[probs[0]], [probs[1]]], state_names={node: [0, 1]}
        )
    else:
        values = probs.reshape(2, 2 ** len(parents)).T.tolist()  # rows=node states
        cpd = TabularCPD(
            node,
            2,
            values,
            evidence=parents,
            evidence_card=[2] * len(parents),
            state_names={node: [0, 1], **{p: [0, 1] for p in parents}},
        )
    cpds.append(cpd)

model = DiscreteBayesianNetwork(edges)
model.add_cpds(*cpds)

# ---- 2b. Brute-force sanity check of the pgmpy setup ------------------------
def brute_marginal(vars_, evidence=None):
    evidence = evidence or {}
    query_vars = [v for v in vars_]
    acc = {}
    for combo in itertools.product([0, 1], repeat=len(nodes)):
        a = dict(zip(nodes, combo))
        if any(a[k] != v for k, v in evidence.items()):
            continue
        p = 1.0
        for n in nodes:
            parents = parent_of[n]
            row = flat_cpts[n][1]
            idx = 0
            for par in parents:
                idx = idx * 2 + a[par]
            idx = idx * 2 + a[n]
            p *= row[idx]
        key = tuple(a[v] for v in query_vars)
        acc[key] = acc.get(key, 0.0) + p
    return acc

infer = VariableElimination(model)
max_err = 0.0
for vars_ in [["physician-fee-freeze"], ["immigration", "crime"],
              ["anti-satellite-test-ban"]]:
    f = infer.query(variables=vars_)
    fvals = np.asarray(f.values).ravel()
    bf = brute_marginal(vars_)
    bfvals = np.array([bf[k] for k in sorted(bf)])
    max_err = max(max_err, np.abs(fvals - bfvals).max())
print(f"pgmpy-vs-bruteforce max abs err: {max_err:.3e}")

# ---- 3. Build the query set --------------------------------------------------
queries = []  # (id, query_vars, evidence dict)
for n in nodes:
    queries.append((f"marg_{n}", [n], {}))
pair_pool = edges[:6]
for i, (u, v) in enumerate(pair_pool):
    queries.append((f"joint_{i}_{u}_{v}", [u, v], {}))
for i, n in enumerate(nodes[:8]):
    queries.append((f"cond_{n}_0", [n], {parent_of[n][0]: 0} if parent_of[n] else {}))
    queries.append((f"cond_{n}_1", [n], {parent_of[n][0]: 1} if parent_of[n] else {}))
queries.append(("cond_multi_0", ["physician-fee-freeze"],
                {"immigration": 0, "crime": 1}))
queries.append(("cond_multi_1", ["el-salvador-aid", "mx-missile"],
                {"religious-groups-in-schools": 0}))
queries.append(("cond_multi_2", ["education-spending"],
                {"class": 1, "crime": 0}))
queries.append(("cond_multi_3", ["adoption-of-the-budget-resolution"],
                {"physician-fee-freeze": 0, "el-salvador-aid": 1}))

# ---- 4. Run the queries with pgmpy ------------------------------------------
out_lines = ["# id\tvars\tvalues"]
for qid, vars_, evidence in queries:
    f = infer.query(variables=vars_, evidence=evidence)
    varnames = [str(v) for v in f.variables]
    vals = ",".join(f"{v:.12g}" for v in np.asarray(f.values).ravel())
    out_lines.append(f"{qid}\t{','.join(varnames)}\t{vals}")
with open("results_py.txt", "w") as fh:
    fh.write("\n".join(out_lines) + "\n")

# ---- 5. Write sidecar files --------------------------------------------------
with open("structure.txt", "w") as fh:
    for n in nodes:
        fh.write(f"{n}\t{','.join(parent_of[n])}\n")

with open("cpts_py.txt", "w") as fh:
    for n in nodes:
        parents = parent_of[n]
        vals = ",".join(f"{v:.12g}" for v in flat_cpts[n][1])
        fh.write(f"{n}\t{','.join(parents)}\t{vals}\n")

with open("queries.txt", "w") as fh:
    for qid, vars_, evidence in queries:
        ev = ";".join(f"{k}={v}" for k, v in sorted(evidence.items()))
        fh.write(f"{qid}\t{','.join(vars_)}\t{ev}\n")

print(f"queries: {len(queries)}")
print("wrote structure.txt, cpts_py.txt, queries.txt, results_py.txt")
