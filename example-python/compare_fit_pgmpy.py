#!/usr/bin/env python3
"""Showcase: fit_bayesian (Chow-Liu + MLE) vs pgmpy.

Learns a Bayesian network structure + parameters from the same integer-coded
data using:
  - pgmpy   : TreeSearch (chow-liu) for structure + maximum-likelihood CPTs
  - bayesian: fit_bayesian() (C++ Chow-Liu tree + MLE)
and compares the learned structure quality, the query outputs and the
wall-clock fitting time.

Structure comparison is done two ways:
  * optimality: the total mutual-information weight of our learned tree must
    equal that of a maximum spanning tree (both are optimal; on near-tie
    weights Prim and Kruskal may pick different but equally-optimal trees),
  * identity: on data sampled from a strongly-determined chain the true tree
    is unique, so both implementations must recover the exact same edges.

Usage:
    pip install bayesian-cpp pgmpy numpy pandas networkx scikit-learn
    python compare_fit_pgmpy.py
"""
import os
import sys
import time

import networkx as nx
import numpy as np
import pandas as pd
from pgmpy.estimators import TreeSearch
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork
from sklearn.metrics import mutual_info_score

import bayesian
from bayesian import Variable

TOL = 1e-6


# ---------------------------------------------------------------------------
# pgmpy side: Chow-Liu structure + canonical MLE CPTs.
# ---------------------------------------------------------------------------
def build_cpd(df, node, parents, card=2):
    cols = df[[*(parents), node]].to_numpy(dtype=np.int64)
    flat = cols[:, -1].copy()
    for k, p in enumerate(parents):
        flat += cols[:, k] << (len(parents) - k)
    counts = np.bincount(flat, minlength=2 ** (len(parents) + 1)).astype(float)
    counts = counts.reshape([2] * (len(parents) + 1))
    denom = counts.sum(axis=-1, keepdims=True)
    probs = np.divide(counts, denom, out=np.zeros_like(counts), where=denom != 0)
    if not parents:
        return TabularCPD(node, 2, [[probs[0]], [probs[1]]],
                          state_names={node: [0, 1]})
    values = probs.reshape(2, 2 ** len(parents)).T.tolist()
    return TabularCPD(node, 2, values, evidence=list(parents),
                      evidence_card=[2] * len(parents),
                      state_names={node: [0, 1], **{p: [0, 1] for p in parents}})


def fit_pgmpy(df):
    # n_jobs=1: pgmpy's joblib workers segfault on scipy import when our C++
    # extension is already loaded in the parent process.
    learner = TreeSearch(df, n_jobs=1)
    t0 = time.perf_counter()
    dag = learner.estimate(estimator_type="chow-liu", show_progress=False)
    edges = sorted((str(u), str(v)) for u, v in dag.edges())
    parents = {c: [] for c in df.columns}
    for u, v in edges:
        parents[v].append(u)
    cpds = [build_cpd(df, c, parents[c]) for c in df.columns]
    model = DiscreteBayesianNetwork(edges)
    model.add_cpds(*cpds)
    infer = VariableElimination(model)
    elapsed = (time.perf_counter() - t0) * 1e3
    return infer, edges, elapsed


def fit_ours(data_int32, names):
    t0 = time.perf_counter()
    inf = bayesian.fit_bayesian(data_int32, names=names)
    elapsed = (time.perf_counter() - t0) * 1e3
    return inf, elapsed


def factor_to_dict(varnames, values):
    """Map a flat row-major probability vector to {sorted (var, state) key: p}."""
    out = {}
    n = len(varnames)
    for idx, v in enumerate(values):
        states = []
        rem = idx
        for name in reversed(varnames):
            states.insert(0, rem % 2)
            rem //= 2
        out[tuple(sorted(zip(varnames, states)))] = float(v)
    return out


def compare(names, data_int32, label, require_exact_structure):
    df = pd.DataFrame(data_int32, columns=names)
    pg_infer, pg_edges, t_pg = fit_pgmpy(df)
    inf, t_bs = fit_ours(data_int32, names)

    # ---- structure ---------------------------------------------------------
    name2id = {n: i for i, n in enumerate(names)}
    bs_edges = set()
    for f in inf.factors:
        vs = f.variables
        if len(vs) == 2:
            bs_edges.add(tuple(sorted((vs[0].id, vs[1].id))))
    pg_edges_ids = set(tuple(sorted((name2id[u], name2id[v]))) for u, v in pg_edges)

    # total mutual-information weight of each tree (our MI formula)
    def our_mi(a, b):
        sa, sb = int(a.max()) + 1, int(b.max()) + 1
        joint = np.zeros((sa, sb))
        np.add.at(joint, (a, b), 1)
        joint /= len(a)
        px, py = joint.sum(1), joint.sum(0)
        out = 0.0
        for x in range(sa):
            for y in range(sb):
                if joint[x, y] > 0 and px[x] > 0 and py[y] > 0:
                    out += joint[x, y] * np.log(joint[x, y] / (px[x] * py[y]))
        return out

    mi = np.zeros((len(names), len(names)))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            m = our_mi(data_int32[:, i], data_int32[:, j])
            mi[i, j] = mi[j, i] = m
    w_ours = sum(mi[i, j] for i, j in bs_edges)
    w_pg = sum(mi[i, j] for i, j in pg_edges_ids)
    optimal = abs(w_ours - w_pg) < 1e-9

    # ---- queries -----------------------------------------------------------
    qvars = [Variable(i, names[i], 2) for i in range(len(names))]
    name2var = {n: v for n, v in zip(names, qvars)}
    queries = []
    for i in range(len(names)):
        queries.append(("marg", [names[i]], {}))
    for i in range(len(names) - 1):
        queries.append(("pair", [names[i], names[i + 1]], {}))
    for i in range(min(6, len(names))):
        j = (i + 1) % len(names)
        queries.append(("cond", [names[i]], {names[j]: 0}))

    max_err = 0.0
    for kind, q, ev in queries:
        qvars_py = [name2var[name] for name in q]
        ev_py = {name2var[k]: v for k, v in ev.items()}
        if kind == "cond":
            pgf = pg_infer.query(variables=q, evidence=ev)
            bsf = inf.conditional_given(qvars_py, ev_py)
        else:
            pgf = pg_infer.query(variables=q)
            bsf = inf.marginal(qvars_py)
        pgd = factor_to_dict([str(v) for v in pgf.variables],
                             np.asarray(pgf.values).ravel())
        bsd = factor_to_dict([v.name for v in bsf.variables],
                             bsf.probabilities())
        for key in set(pgd) | set(bsd):
            max_err = max(max_err, abs(pgd.get(key, 0.0) - bsd.get(key, 0.0)))

    exact = bs_edges == pg_edges_ids
    ok = optimal and (not require_exact_structure or (exact and max_err <= TOL))
    status = "PASS" if ok else "FAIL"
    print(f"\n[{status}] {label}")
    print(f"  tree optimality : our total MI = {w_ours:.6f} vs pgmpy = {w_pg:.6f} "
          f"(equal -> {optimal})")
    print(f"  structure       : edges identical = {exact} "
          f"(ours {len(bs_edges)}, pgmpy {len(pg_edges_ids)})")
    print(f"  queries ({len(queries)}): max |pgmpy - bayesian| = {max_err:.3e}")
    print(f"  fit time        : pgmpy {t_pg:8.3f} ms | bayesian {t_bs:8.4f} ms "
          f"({t_pg / max(t_bs, 1e-9):6.1f}x faster)")
    return ok


def make_chain_data(n_vars=16, n_samples=50000, seed=7):
    """Strong chain X0 -> X1 -> ... -> X(n-1); child flips parent with p=0.12."""
    rng = np.random.default_rng(seed)
    names = [f"v{i}" for i in range(n_vars)]
    data = np.zeros((n_samples, n_vars), dtype=np.int32)
    data[:, 0] = (rng.random(n_samples) < 0.5).astype(np.int32)
    for i in range(1, n_vars):
        flip = rng.random(n_samples) < 0.12
        data[:, i] = (data[:, i - 1] ^ flip.astype(np.int32))
    return names, data


def main():
    ok = True

    # 1) synthetic strong chain: the true tree is unique -> exact recovery.
    names, data = make_chain_data()
    ok &= compare(names, data, "synthetic chain (16 vars, 50000 samples)",
                  require_exact_structure=True)

    # 2) real dataset (UCI congressional voting): verify optimality + report
    #    query agreement (near-tie weights allow alternative optimal trees).
    path = os.path.join(os.path.dirname(__file__), "..", "bench", "voting.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        cols = list(df.columns)
        ok &= compare(cols, df[cols].to_numpy(dtype=np.int32),
                      "real voting dataset (232 samples, 16 vars)",
                      require_exact_structure=False)
    else:
        print("\n[skip] bench/voting.csv not found")

    print("\nOVERALL:", "ALL PASS" if ok else "FAILURES PRESENT")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
