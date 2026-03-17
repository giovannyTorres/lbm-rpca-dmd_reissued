cmake_minimum_required(VERSION 3.16)
project(lbm_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(lbm_sim
    src/config.cpp
    src/io.cpp
    src/solver.cpp
    src/main.cpp
)

target_include_directories(lbm_sim PRIVATE include)
