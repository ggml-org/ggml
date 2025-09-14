include(ExternalProject)
find_package(Git REQUIRED)

include(FetchContent)

FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.1)
set(BENCHMARK_ENABLE_TESTING
        OFF
        CACHE BOOL "Turn off BENCHMARK_ENABLE_TESTING" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

add_executable(benchmark-mul-mat ${PROJECT_SOURCE_DIR}/benchmarks/benchmark-mul-mat.cpp)
target_link_libraries(benchmark-mul-mat PRIVATE benchmark::benchmark ggml)
