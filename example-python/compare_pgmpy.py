#!/usr/bin/env python3
"""Showcase: pgmpy vs bayesian-cpp on the same model.

Builds a random tree Bayesian network, samples a dataset from it with pgmpy,
fits exact MLE CPTs in numpy (canonical [0,1] state order), then runs the same
marginal and conditional queries through pgmpy's variable elimination and
through the exported C++ engine (bayesian), comparing the outputs and the
wall-clock performance.

Usage:
    pip install bayesian-cpp pgmpy numpy pandas
    python compare_pgmpy.py [num_variables] [num_samples]
"""
import sys
import time

import numpy as np
import pandas as pd
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

import bayesian
from bayesian import Inference, JunctionTree, Potential, Variable

N_VARS = int(sys.argv[1]) if len(sys.argv) > 1 else 16
N_SAMPLES = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
SEED = 7
TOL = 1e-6


def random_cpt(rng, n):
    w = np.array([1 + rng.integers(100) for _ in range(n)], dtype=float)
    return w / w.sum()


def make_tree(n):
    """Random binary tree: node i > 0 has parent (i - 1) // 2."""
    rng = np.random.default_rng(SEED)
    names = [f"V{i}" for i in range(n)]
    parents = {i: [] if i == 0 else [(i - 1) // 2] for i in range(n)}
    cpts = {}
    for i in range(n):
        ps = parents[i]
        cpts[i] = random_cpt(rng, 2) if not ps else np.concatenate(
            [random_cpt(rng, 2) for _ in range(2 ** len(ps))]
        )
    return names, parents, cpts


def fit_cpt(data, node, parents, card=2):
    """Exact MLE CPT, row-major over (parents..., node) with node fastest."""
    cols = data[[*(names[p] for p in parents), names[node]]].to_numpy(
        dtype=np.int64)
    flat = cols[:, -1].copy()
    for k, p in enumerate(parents):
        flat += cols[:, k] << (len(parents) - k)
    counts = np.bincount(flat, minlength=2 ** (len(parents) + 1)).astype(float)
    counts = counts.reshape([2] * (len(parents) + 1))
    denom = counts.sum(axis=-1, keepdims=True)
    probs = np.divide(counts, denom, out=np.zeros_like(counts), where=denom != 0)
    return probs.ravel()


def build_pg_cpds(fit_cpts):
    cpds = []
    for i in range(N_VARS):
        ps = parents[i]
        state_names = {names[i]: [0, 1], **{names[p]: [0, 1] for p in ps}}
        if not ps:
            cpds.append(TabularCPD(names[i], 2, [[fit_cpts[i][0]], [fit_cpts[i][1]]],
                                   state_names=state_names))
        else:
            values = fit_cpts[i].reshape(2, 2 ** len(ps)).T.tolist()
            cpds.append(TabularCPD(
                names[i], 2, values, evidence=[names[p] for p in ps],
                evidence_card=[2] * len(ps), state_names=state_names))
    return cpds


names, parents, cpts = make_tree(N_VARS)
print(f"network: {N_VARS} binary vars, tree structure, seed={SEED}")

# ---- sample a dataset from the true model with pgmpy -----------------------
true_model = DiscreteBayesianNetwork(
    [(names[(i - 1) // 2], names[i]) for i in range(1, N_VARS)])
true_model.add_cpds(*build_pg_cpds(cpts))
data = true_model.simulate(n_samples=N_SAMPLES, seed=1)
print(f"sampled dataset: {len(data)} rows x {N_VARS} columns")

# ---- fit MLE CPTs from the dataset (shared by both engines) ----------------
fit_cpts = {i: fit_cpt(data, i, parents[i]) for i in range(N_VARS)}

# ---- build both models ------------------------------------------------------
pg_model = DiscreteBayesianNetwork(
    [(names[(i - 1) // 2], names[i]) for i in range(1, N_VARS)])
pg_model.add_cpds(*build_pg_cpds(fit_cpts))
pg_infer = VariableElimination(pg_model)

vars_ = [Variable(i, names[i], 2) for i in range(N_VARS)]
factors = [
    Potential([*(vars_[p] for p in parents[i]), vars_[i]], list(fit_cpts[i]))
    for i in range(N_VARS)
]
engine = Inference(factors)

# ---- queries ---------------------------------------------------------------
marginal_queries = list(range(N_VARS))
cond_queries = [(i, [(p, 0) for p in parents[i]]) for i in range(N_VARS)
                if parents[i]]


def run_queries(impl):
    """Run the query set and return the flat probability arrays."""
    out = []
    if impl == "pg":
        for i in marginal_queries:
            out.append(np.asarray(
                pg_infer.query(variables=[names[i]]).values).ravel())
        for i, ev in cond_queries:
            evd = {names[e]: val for e, val in ev}
            out.append(np.asarray(
                pg_infer.query(variables=[names[i]], evidence=evd).values).ravel())
    else:
        for i in marginal_queries:
            out.append(np.asarray(engine.marginal([vars_[i]]).probabilities()))
        for i, ev in cond_queries:
            out.append(np.asarray(engine.conditional_given(
                [vars_[i]], {vars_[e]: val for e, val in ev}).probabilities()))
    return out


# ---- correctness comparison -------------------------------------------------
res_pg = run_queries("pg")
res_bs = run_queries("bs")
max_err = 0.0
for a, b in zip(res_pg, res_bs):
    max_err = max(max_err, float(np.abs(a - b).max()))
nq = len(marginal_queries) + len(cond_queries)
print(f"\ncorrectness: max |pgmpy - bayesian| over {nq} queries = {max_err:.3e}")


def bench(fn, repeat, iters):
    best = float("inf")
    fn()  # warmup
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        best = min(best, (time.perf_counter() - t0) / iters)
    return best * 1e3  # ms


t_pg = bench(lambda: run_queries("pg"), 3, 5)
t_bs = bench(lambda: run_queries("bs"), 3, 50)
print(f"\nperformance (one full query batch of {nq} queries, best of 3 runs):")
print(f"  pgmpy    : {t_pg:9.3f} ms")
print(f"  bayesian : {t_bs:9.4f} ms")
print(f"  speedup  : {t_pg / max(t_bs, 1e-9):7.1f}x")

# ---- repeated single-query performance (caching story) ----------------------
K = 500
qvar = 0
ev = {parents[qvar][0]: 0} if parents[qvar] else {}
evd = {names[e]: val for e, val in ev.items()}
jt = JunctionTree(factors)
jt.set_evidence({vars_[e]: val for e, val in ev.items()})

t_pg1 = bench(lambda: pg_infer.query(variables=[names[qvar]], evidence=evd), 3, K)
t_bs1 = bench(
    lambda: engine.conditional_given([vars_[qvar]], {vars_[e]: v for e, v in ev.items()}),
    3, K)
t_jt1 = bench(lambda: jt.marginal([vars_[qvar]]), 3, K)
print(f"\nperformance (single query P({names[qvar]} | "
      f"{', '.join(f'{names[e]}={v}' for e, v in ev.items())}), per query, best of 3):")
print(f"  pgmpy     : {t_pg1:8.4f} ms")
print(f"  bayesian  : {t_bs1:8.4f} ms")
print(f"  + JunctionTree (cached): {t_jt1:8.4f} ms "
      f"({t_pg1 / max(t_jt1, 1e-9):6.1f}x faster than pgmpy)")

print("\n" + ("PASS: outputs match within tolerance"
              if max_err <= TOL else f"FAIL: max err {max_err} > {TOL}"))
sys.exit(0 if max_err <= TOL else 1)
