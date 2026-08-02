# bayesian-cpp

**A fast Bayesian network inference engine written in C++20, exposed to Python
via pybind11.**

`bayesian-cpp` performs **exact probabilistic inference** on discrete Bayesian
networks: marginals, full joint distributions, conditionals with evidence, and
MAP queries. All probability tables are kept in **log-space** for numerical
stability, and inference runs in native C++ — orders of magnitude faster than
pure-Python implementations such as pgmpy.

## Features

- **Variable elimination** — exact marginal / conditional inference with a
  min-fill elimination order and per-query order caching.
- **Junction tree propagation** — build the clique tree once and answer many
  repeated queries (or new evidence) cheaply from calibrated cliques.
- **Queries**: full joint `P(X, Y, Z, …)`, marginals `P(X, Z)` (all other
  variables summed out), conditionals `P(X | Y, Z)` and `P(X | Y=0, Z=1)`, and
  maximum a posteriori (`MAP`) assignments.
- **Log-space tables** with log-sum-exp marginalization — robust against
  underflow on large networks.
- **Self-contained** — pure C++20, no third-party C++ dependencies; the Python
  extension builds from source with only pybind11 (fetched automatically by
  pip).

## Install

```bash
pip install bayesian-cpp
```

Requires Python ≥ 3.8 and a C++20 compiler. On Linux a prebuilt `manylinux`
wheel is installed; on other platforms pip builds from the source distribution.

## Quick start

```python
import bayesian
from bayesian import Variable, Potential, Inference, JunctionTree

x = Variable(0, "X", 2)
y = Variable(1, "Y", 2)
z = Variable(2, "Z", 2)

factors = [
    Potential([x], [0.5, 0.5]),                     # P(X)
    Potential([x, y], [0.9, 0.1, 0.2, 0.8]),       # P(Y | X)
    Potential([x, z], [0.8, 0.2, 0.1, 0.9]),       # P(Z | X)
]

engine = Inference(factors)

p_y          = engine.marginal([y])                          # P(Y)
p_xz         = engine.marginal([x, z])                       # P(X, Z)
p_x_given    = engine.conditional_given([x], {y: 0, z: 1})   # P(X | Y=0, Z=1)
posterior    = engine.full_joint()                           # P(X, Y, Z)
mode         = engine.map_query([x], {y: 1})                 # MAP assignment

# Repeated queries are cheap after the junction tree is calibrated once.
jt = JunctionTree(factors)
jt.set_evidence({z: 0})
p_x = jt.marginal([x])
```

Every query returns a `Potential`; use `.probabilities()` for a flat list in
row-major order over `.variables`, or `.value({var: state, ...})` for a single
cell.

## Performance

Because inference runs in native C++ with log-space arithmetic, it is far
faster than pure-Python engines. On a 16-variable tree network (dataset fitted
with MLE, same queries):

| | pgmpy | bayesian-cpp |
|---|---|---|
| 31 marginals + conditionals | ~4.3 ms | ~0.6 ms |
| repeated single query | ~68 µs | ~0.8 µs (JunctionTree) |

Outputs agree with pgmpy to within ~1e-7 (float32 vs float64 rounding).

## Exceptions

- `ValueError` — invalid arguments (negative probabilities, unknown variables,
  overlapping query/evidence, out-of-range evidence states).
- `RuntimeError` — evidence with zero probability.

## Repository

Source: https://github.com/Kemsekov/BayesianNetworkCpp

The repository contains the full C++ engine (`src/`), a pytest/gtest test suite,
a benchmark suite, and a `example-python/` folder comparing `bayesian-cpp`
head-to-head against pgmpy (outputs and wall-clock performance).
