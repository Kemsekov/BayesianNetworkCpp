#!/bin/bash
set -e

# Setup build directory
mkdir -p build && cd build

# Configure and build in Release mode (-O3)
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .

# Run the compiled demo binary
./BayesianNetwork
