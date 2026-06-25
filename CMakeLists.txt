cmake_minimum_required(VERSION 3.15)
project(MyEigenProject LANGUAGES CXX)

# Set C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Apply O3 optimizations for Release builds
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Find the Eigen package
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Define your executable using the project name
add_executable(${PROJECT_NAME} main.cpp)

# Link Eigen to your executable
target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen)
