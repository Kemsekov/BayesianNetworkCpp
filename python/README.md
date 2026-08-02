# bayesian-cpp

Python bindings (via pybind11) for the C++ Bayesian network engine in this
repository: exact inference by **variable elimination** and by **junction tree
propagation**, with log-space tables for numerical stability.

## Install

The package is self-contained: pip builds the C++ extension from source and
fetches its only build-time dependency (pybind11) automatically.

```bash
pip install .
```

or, from any directory:

```bash
pip install /path/to/this/repository
```

Requires a C++20 compiler (gcc >= 10 / clang >= 12 / MSVC >= 2019).

## Usage

```python
import bayesian
from bayesian import Variable, Potential, Inference, JunctionTree

x = Variable(0, "X", 2)
y = Variable(1, "Y", 2)
z = Variable(2, "Z", 2)

factors = [
    Potential([x], [0.5, 0.5]),
    Potential([x, y], [0.9, 0.1, 0.2, 0.8]),   # P(Y | X)
    Potential([x, z], [0.8, 0.2, 0.1, 0.9]),   # P(Z | X)
]

engine = Inference(factors)
joint = engine.full_joint()                 # P(X, Y, Z)
p_y   = engine.marginal([y])                # P(Y)
p_x_given_yz = engine.conditional_given([x], {y: 0, z: 1})  # P(X | Y=0, Z=1)
mode  = engine.map_query([x], {y: 1})       # MAP assignment over X

jt = JunctionTree(factors)
jt.set_evidence({z: 0})
p_x = jt.marginal([x])                      # repeated queries are cached
```

Every probability query returns a `Potential`; use `.probabilities()` to get a
flat list (row-major over `.variables`) or `.value({var: state, ...})` to read
a single cell.

## Exceptions

- `ValueError` is raised for invalid arguments (e.g. negative probabilities,
  unknown variables, overlapping query/evidence).
- `RuntimeError` is raised when the evidence has zero probability.

## Development

```bash
python -m pip install .  # or: pip install -e .
```
