#!/usr/bin/env python3
"""Quick tour of the bayesian-cpp Python API.

Run after `pip install bayesian-cpp`:
    python api_smoke.py
"""
import bayesian
from bayesian import Inference, JunctionTree, Potential, Variable

print(f"bayesian {bayesian.__version__}\n")

# --- variables and potentials ------------------------------------------------
x = Variable(0, "X", 2)
y = Variable(1, "Y", 2)
z = Variable(2, "Z", 2)
print("Variable:", x, "| id:", x.id, "| states:", x.num_states)

# P(X); P(Y | X); P(Z | X)
factors = [
    Potential([x], [0.5, 0.5]),
    Potential([x, y], [0.9, 0.1, 0.2, 0.8]),
    Potential([x, z], [0.8, 0.2, 0.1, 0.9]),
]
joint = factors[0] * factors[1] * factors[2]
print("\nfull joint entries:", joint.num_entries)
print("P(X=0, Y=1)         :", round(factors[0].value({x: 0}) * factors[1].value({x: 0, y: 1}), 4))

# --- inference ----------------------------------------------------------------
engine = Inference(factors)
print("\nP(Y)                 :", [round(v, 4) for v in engine.marginal([y]).probabilities()])
print("P(X, Z)              :", [round(v, 4) for v in engine.marginal([x, z]).probabilities()])
print("P(X | Y=0, Z=1)      :", [round(v, 4) for v in engine.conditional_given([x], {y: 0, z: 1}).probabilities()])
print("MAP(X | Y=1)         :", engine.map_query([x], {y: 1}))

# --- junction tree (repeated queries are cached) -------------------------------
jt = JunctionTree(factors)
jt.set_evidence({z: 0})
print("\nJT cliques:", jt.num_cliques)
print("JT P(X | Z=0)        :", [round(v, 4) for v in jt.marginal([x]).probabilities()])

# --- operators / normalization --------------------------------------------------
p = engine.marginal([x, y]) / engine.marginal([y])      # P(X,Y)/P(Y)
probs = p.reorder([y, x]).probabilities()               # row-major over (y, x)
rows = [[round(v, 4) for v in probs[i * 2:(i + 1) * 2]] for i in range(2)]
print("\nP(X|Y) rows (each sums to 1):", rows)

# --- edge cases raise Python exceptions -----------------------------------------
for label, fn in [
    ("negative probability   ", lambda: Potential([x], [-0.1, 1.1])),
    ("unknown variable       ", lambda: engine.marginal([Variable(99, "ghost", 2)])),
    ("overlapping query/ev   ", lambda: engine.conditional_given([x], {x: 0})),
]:
    try:
        fn()
        print(f"\n[FAIL] {label}: no exception raised")
    except (ValueError, RuntimeError) as e:
        print(f"\n[{type(e).__name__:>10}] {label}: {e}")

print("\nAPI smoke test OK")
