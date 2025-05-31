#
# This example can be built as part of ggml or
# as a separate project with an installed ggml.

cmake_minimum_required(VERSION 3.14)
project(ggml-simple)

if (NOT TARGET ggml::ggml)
    # Building as a separate project
    find_package(ggml CONFIG REQUIRED)

    set(CMAKE_CXX_STANDARD 17)
endif()

#
# simple-ctx

set(TEST_TARGET simple-ctx)
add_executable(${TEST_TARGET} simple-ctx.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml::ggml)

#
# simple-backend

set(TEST_TARGET simple-backend)
add_executable(${TEST_TARGET} simple-backend.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml::ggml)

if (GGML_CUDA)
    add_compile_definitions(GGML_USE_CUDA)
endif()

if (GGML_METAL)
    add_compile_definitions(GGML_USE_METAL)
endif()
