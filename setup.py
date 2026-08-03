import sys

# pybind11.setup_helpers is the stable build API across pybind11 2.x and 3.x.
from pybind11.setup_helpers import Pybind11Extension
from setuptools import find_packages, setup

# The engine is pure C++20 with no third-party C++ dependencies; pybind11 is
# fetched automatically by pip's build isolation. Metadata lives in
# pyproject.toml ([project]).
ext_modules = [
    Pybind11Extension(
        "bayesian._native",
        sources=[
            "python/bayesian/_native.cpp",
            "src/potential.cpp",
            "src/inference.cpp",
            "src/junction_tree.cpp",
            "src/fit.cpp",
        ],
        include_dirs=["src"],
        cxx_std=20,
        extra_compile_args=["-O3", "-fvisibility=hidden"] if sys.platform != "win32"
        else ["/O2"],
    )
]

setup(
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=ext_modules,
    zip_safe=False,
)
