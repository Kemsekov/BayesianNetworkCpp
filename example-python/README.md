# example-python

Showcase for the exported `bayesian-cpp` Python package: it runs the same model
and the same queries through **pgmpy** (pure Python, battle-tested) and through
**bayesian** (the C++ engine exposed via pybind11) and compares outputs and
performance.

## Install

```bash
pip install bayesian-cpp        # builds the C++ extension from source
pip install pgmpy numpy pandas  # for the reference implementation
```

## Files

- `compare_pgmpy.py` — the main showcase:
  1. builds a random tree Bayesian network (16 binary variables, fixed seed),
  2. samples a dataset from it with pgmpy,
  3. fits exact MLE CPTs in numpy (canonical `[0,1]` state order),
  4. constructs the model in both pgmpy and `bayesian`,
  5. runs the same marginals + conditionals and compares the probabilities,
  6. benchmarks a full query batch and a repeated single query.

  ```bash
  python compare_pgmpy.py [num_variables] [num_samples]
  ```

  Expected output (last digits may vary):

  ```
  correctness: max |pgmpy - bayesian| over 31 queries = 1.2e-07
  performance (one full query batch of 31 queries, best of 3 runs):
    pgmpy    :    4.28 ms
    bayesian :   0.634 ms
    speedup  :     6.7x
  ...
  PASS: outputs match within tolerance
  ```

- `api_smoke.py` — a quick tour of the API surface (variables, potentials,
  inference, junction tree, MAP, exception translation):

  ```bash
  python api_smoke.py
  ```

## Notes

- Probabilities are compared with `|pgmpy - bayesian| < 1e-6`; the observed
  error is ~1e-7, i.e. float32-vs-float64 rounding only.
- `bayesian` runs inference in C++ (log-space tables); the speedup grows with
  network size and with repeated queries (JunctionTree caches propagation).
