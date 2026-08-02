"""Python bindings for the C++ Bayesian network inference engine."""

from ._native import Inference, JunctionTree, Potential, Variable

__all__ = ["Variable", "Potential", "Inference", "JunctionTree"]
__version__ = "0.1.2"
