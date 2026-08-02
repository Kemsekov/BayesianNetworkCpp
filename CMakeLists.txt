cmake_minimum_required(VERSION 3.15)
project(BayesianNetwork LANGUAGES CXX)

# Set C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Apply O3 optimizations for Release builds
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Bayesian network engine library
add_library(bayesian STATIC
    src/potential.cpp
    src/inference.cpp
    src/junction_tree.cpp
)
target_include_directories(bayesian PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Demo executable
add_executable(BayesianNetwork main.cpp)
target_link_libraries(BayesianNetwork PRIVATE bayesian)

# Tests
find_package(GTest REQUIRED)
include(GoogleTest)
enable_testing()

add_executable(bayesian_tests
    tests/test_variable.cpp
    tests/test_potential.cpp
    tests/test_inference.cpp
    tests/test_junction_tree.cpp
)
target_link_libraries(bayesian_tests PRIVATE bayesian GTest::gtest_main)
gtest_discover_tests(bayesian_tests)

# Head-to-head benchmark vs pgmpy (run from build dir with bench/ as arg)
add_executable(bench_cpp bench/bench_cpp.cpp)
target_link_libraries(bench_cpp PRIVATE bayesian)

# Performance benchmark
add_executable(perf_bench perf/bench.cpp)
target_link_libraries(perf_bench PRIVATE bayesian)
