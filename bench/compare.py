#!/usr/bin/env python3
"""Compare the C++ engine results against the pgmpy (Python) results.

Reads:
  results_py.txt           pgmpy results
  results_cpp.txt          C++ results (C++-fitted CPTs)
  results_cpp_from_py.txt  C++ results (pgmpy's exact CPTs)
  cpts_py.txt / cpts_cpp.txt  fitted CPTs from both sides

Reports per-file max absolute error between matching assignment keys.
"""
import sys


def parse_results(path):
    """-> {id: {assignment_key: prob}} where assignment key uses each side's
    own declared variable order (e.g. 'A=0,B=1')."""
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            qid, vars_str, vals_str = line.split("\t")
            varnames = [v for v in vars_str.split(",") if v]
            values = [float(v) for v in vals_str.split(",")]
            table = {}
            n = len(varnames)
            for idx, p in enumerate(values):
                key_parts = []
                for i in range(n):
                    key_parts.append(f"{varnames[i]}={idx % (2 ** (n - i)) // (2 ** (n - i - 1))}")
                table[",".join(key_parts)] = p
            out[qid] = table
    return out


def compare(a, b, label, tol=1e-5):
    common = sorted(set(a) & set(b))
    missing_a = sorted(set(b) - set(a))
    missing_b = sorted(set(a) - set(b))
    max_err = 0.0
    worst = None
    for qid in common:
        ka, kb = a[qid], b[qid]
        for key in set(ka) & set(kb):
            err = abs(ka[key] - kb[key])
            if err > max_err:
                max_err = err
                worst = (qid, key, ka[key], kb[key])
    n_keys = sum(len(a[q]) for q in common)
    status = "PASS" if max_err <= tol and not missing_a and not missing_b else "FAIL"
    print(f"[{status}] {label}: max abs err = {max_err:.3e} over {n_keys} keys "
          f"({len(common)} queries)")
    if max_err > tol:
        print(f"      worst: {worst}")
    if missing_a or missing_b:
        print(f"      query ids only in A: {missing_a[:5]}...; only in B: {missing_b[:5]}...")
    return status, max_err


def parse_cpts(path):
    """-> {node: (parents, values_flat)}"""
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            node, parents_str, vals = line.split("\t")
            parents = [p for p in parents_str.split(",") if p]
            values = [float(v) for v in vals.split(",")]
            out[node] = (parents, values)
    return out


def main():
    ok = True
    py = parse_results("results_py.txt")
    cpp = parse_results("results_cpp.txt")
    cpp_py = parse_results("results_cpp_from_py.txt")

    s, e = compare(cpp, py, "inference: C++ (own CPTs)  vs pgmpy          ", tol=1e-5)
    ok &= s == "PASS"
    s, e = compare(cpp_py, py, "inference: C++ (py CPTs)  vs pgmpy          ", tol=1e-6)
    ok &= s == "PASS"
    s, e = compare(cpp, cpp_py, "inference: C++ (own)      vs C++ (py CPTs)  ", tol=1e-5)
    ok &= s == "PASS"

    pyc = parse_cpts("cpts_py.txt")
    cppc = parse_cpts("cpts_cpp.txt")
    max_err = 0.0
    n_vals = 0
    for node, (parents, values) in pyc.items():
        p2 = cppc[node]
        n_vals += len(values)
        for a, b in zip(values, p2[1]):
            max_err = max(max_err, abs(a - b))
    s = "PASS" if max_err <= 1e-7 else "FAIL"
    ok &= s == "PASS"
    print(f"[{s}] fitted CPTs: C++ (own) vs pgmpy max abs err = {max_err:.3e} "
          f"over {n_vals} values")
    print("\nOVERALL:", "ALL PASS" if ok else "FAILURES PRESENT")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
