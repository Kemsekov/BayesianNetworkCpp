#!/bin/bash
set -e

# Build and run the gtest test suite
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --target bayesian_tests
ctest --output-on-failure
