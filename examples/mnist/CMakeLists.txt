#
# mnist-common

set(TEST_TARGET mnist-common)
add_library(${TEST_TARGET} STATIC mnist-common.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common)

#
# mnist-eval

set(TEST_TARGET mnist-eval)
add_executable(${TEST_TARGET} mnist-eval.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common mnist-common)

#
# mnist-train

set(TEST_TARGET mnist-train)
add_executable(${TEST_TARGET} mnist-train.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common mnist-common)


#
# mnist-wasm
if (EMSCRIPTEN)
    set(TARGET mnist)

    add_executable(${TARGET} mnist-common.cpp)
    target_link_libraries(${TARGET} PRIVATE ggml ggml-cpu)

    set_target_properties(${TARGET} PROPERTIES LINK_FLAGS " \
        --bind \
        -s FORCE_FILESYSTEM=1 \
        -s USE_PTHREADS=1 \
        -s PTHREAD_POOL_SIZE=10 \
        -s ASSERTIONS=1 \
        -s WASM=1 \
        -s EXPORTED_RUNTIME_METHODS=\"['ccall', 'cwrap', 'setValue', 'getValue']\" \
        -s EXPORTED_FUNCTIONS=\"['_wasm_eval','_wasm_random_digit','_malloc','_free']\" \
        -s ALLOW_MEMORY_GROWTH=1 \
        --preload-file ${CMAKE_CURRENT_SOURCE_DIR}/mnist-f32.gguf@/ \
        --preload-file ${CMAKE_CURRENT_SOURCE_DIR}/t10k-images-idx3-ubyte@/ \
        ")

    # Copy output to web directory
    add_custom_command(
        TARGET ${TARGET} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_BINARY_DIR}/bin/mnist.js
            ${CMAKE_CURRENT_SOURCE_DIR}/web/mnist.js
        COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_BINARY_DIR}/bin/mnist.wasm
            ${CMAKE_CURRENT_SOURCE_DIR}/web/mnist.wasm
        COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_BINARY_DIR}/bin/mnist.worker.js
            ${CMAKE_CURRENT_SOURCE_DIR}/web/mnist.worker.js
        )
endif()
