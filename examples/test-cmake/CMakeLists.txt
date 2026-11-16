cmake_minimum_required(VERSION 3.14)
project(ggml-simple)

set(CMAKE_CXX_STANDARD 17)

find_package(ggml CONFIG REQUIRED PATHS ${CMAKE_PREFIX_PATH} NO_DEFAULT_PATH)
message(STATUS "Found ggml ${ggml_VERSION}: ${ggml_CONFIG}")

set(TEST_TARGET test-cmake)
add_executable(test-cmake test-cmake.cpp)
target_link_libraries(test-cmake PRIVATE ggml::ggml)
