cmake_minimum_required(VERSION 3.25)
project(nano_vllm_cpp26 VERSION 0.1.0 LANGUAGES CXX)

# Enable C++26 (-std=c++26 / -std=c++2c)
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -ffast-math)
endif()

find_package(Threads REQUIRED)

add_library(nano_vllm_core STATIC
    src/kv_cache.cpp
    src/scheduler.cpp
    src/model.cpp
    src/engine.cpp
    src/server.cpp
)
target_include_directories(nano_vllm_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(nano_vllm_core PUBLIC Threads::Threads)

# 1. HTTP/SSE Inference Server Executable
add_executable(nano_vllm_server src/main.cpp)
target_link_libraries(nano_vllm_server PRIVATE nano_vllm_core)

# 2. Continuous vs. Static Batching Benchmark Executable
add_executable(nano_vllm_benchmark src/benchmark.cpp)
target_link_libraries(nano_vllm_benchmark PRIVATE nano_vllm_core)

# 3. Unit & Concurrent Integration Test Suite
enable_testing()
add_executable(nano_vllm_tests tests/test_main.cpp)
target_link_libraries(nano_vllm_tests PRIVATE nano_vllm_core)
add_test(NAME nano_vllm_tests COMMAND nano_vllm_tests)
