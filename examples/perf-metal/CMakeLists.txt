#
# perf-metal

set(TEST_TARGET perf-metal)
add_executable(${TEST_TARGET} perf-metal.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)

#
# bench-metal

set(BENCH_TARGET bench-metal)
add_executable(${BENCH_TARGET} bench-metal.cpp)
target_link_libraries(${BENCH_TARGET} PRIVATE ggml)
