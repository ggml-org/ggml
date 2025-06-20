#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <benchmark/benchmark.h>

#include <cstring>
#include <random>
#include <stdexcept>
#include <thread>

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

template <typename ValueT>
void randomize(std::vector<ValueT>& data) {
    static_assert(std::is_floating_point_v<ValueT>);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<ValueT> dist(-1.0, 1.0);
    for (auto& element : data) {
        element = dist(gen);
    }
}

enum class BackendDevice { kCPU, kCUDA, kMETAL };

// initialize the backend
template <BackendDevice Device>
ggml_backend_t init_backend() {
    ggml_backend_t backend = NULL;

    if constexpr (Device == BackendDevice::kCPU) {
        backend = ggml_backend_cpu_init();
        if (!backend) {
            throw std::runtime_error("Failed to initialize CPU backend.");
        }
        return backend;
    } else if constexpr (Device == BackendDevice::kCUDA) {
#ifdef GGML_USE_CUDA
        backend = ggml_backend_cuda_init(0);
        if (!backend) {
            throw std::runtime_error("Failed to initialize CUDA backend.");
        }
        return backend;
#endif
    } else if constexpr (Device == BackendDevice::kMETAL) {
#ifdef GGML_USE_METAL
        backend = ggml_backend_metal_init();
        if (!backend) {
            throw std::runtime_error("Failed to initialize METAL backend.");
        }
        return backend;
#endif
    } else {
        throw std::runtime_error("Unknown backend type.");
    }
    if (!backend) {
        throw std::runtime_error("Failed to initialize backend.");
    }
    return NULL;
}


template <BackendDevice Device, typename InputT, ggml_type GGML_TYPE>
static void bmMulMat(benchmark::State& state) {
    // define the matrix size, use power-of-two square-matrices of equal size for simplicity
    const uint8_t log_2_size = state.range(0);
    assert(log_2_size < sizeof(size_t) * 8);
    const size_t M = 1 << log_2_size;
    const size_t N = 1 << log_2_size;
    const size_t K = 1 << log_2_size;

    const uint8_t log_2_threads = state.range(1);
    assert(log_2_threads < sizeof(uint8_t) * 8);
    const uint8_t num_cpu_threads = 1 << log_2_threads;
    if (num_cpu_threads > std::thread::hardware_concurrency()) {
        state.SkipWithError("Not enough CPU threads available.");
    }

    // define input and output matrices
    std::vector<InputT> matrixA(M * K);
    std::vector<InputT> matrixB(K * N);
    std::vector<InputT> matrixC(M * N);

    // randomize input matrices
    randomize<InputT>(matrixA);
    randomize<InputT>(matrixB);

    // calculate GGML buffer sizes
    const size_t buffer_size_A = (M * N) * ggml_type_size(GGML_TYPE);
    const size_t buffer_size_B = (N * K) * ggml_type_size(GGML_TYPE);
    const size_t overhead = 1024;
    const size_t buffer_size = buffer_size_A + buffer_size_B + overhead;

    constexpr uint8_t num_tensors = 2;
    struct ggml_init_params params{
        /*.mem_size   =*/ggml_tensor_overhead() * num_tensors,
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/true,
    };

    // initialize the backend
    ggml_backend_t backend = init_backend<Device>();

    // allocate buffers
    ggml_backend_buffer_t buffer = ggml_backend_alloc_buffer(backend, buffer_size);

    // create context
    struct ggml_context* ctx = ggml_init(params);

    // create tensors
    struct ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE, K, M);
    struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE, K, N);

    // create an allocator
    struct ggml_tallocr alloc = ggml_tallocr_new(buffer);

    // alloc memory
    if (ggml_tallocr_alloc(&alloc, a) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to allocate tensor a.");
    }
    if (ggml_tallocr_alloc(&alloc, b) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to allocate tensor b.");
    }

    // load data to buffer
    if (ggml_backend_is_cpu(backend)
#ifdef GGML_USE_METAL
        || ggml_backend_is_metal(backend)
#endif
    ) {
        memcpy(a->data, matrixA.data(), ggml_nbytes(a));
        memcpy(b->data, matrixB.data(), ggml_nbytes(b));
    } else {
        ggml_backend_tensor_set(a, matrixA.data(), 0, ggml_nbytes(a));
        ggml_backend_tensor_set(b, matrixB.data(), 0, ggml_nbytes(b));
    }

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

    size_t buf_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params params0 = {
        /*.mem_size   =*/buf_size,
        /*.mem_buffer =*/buf.data(),
        /*.no_alloc   =*/true, // the tensors will be allocated later by ggml_gallocr_alloc_graph()
    };

    // create a temporary context to build the graph
    struct ggml_context* ctx0 = ggml_init(params0);

    struct ggml_cgraph* gf = ggml_new_graph(ctx0);

    // zT = x @ yT
    struct ggml_tensor* result = ggml_mul_mat(ctx0, a, ggml_cont(ctx0, b));

    // z = (zT)T
    ggml_build_forward_expand(gf, ggml_cont(ctx0, ggml_transpose(ctx0, result)));

    // compute the required memory
    ggml_gallocr_reserve(allocr, gf);

    // allocate tensors
    ggml_gallocr_alloc_graph(allocr, gf);

    if (ggml_backend_is_cpu(backend)) {
        ggml_backend_cpu_set_n_threads(backend, num_cpu_threads);
    }

    for (auto _ : state) {
        ggml_backend_graph_compute(backend, gf);
    }

    const size_t flops_per_iteration = 2 * M * N * K;
    state.counters["Flop/s"] = benchmark::Counter(flops_per_iteration, benchmark::Counter::kIsIterationInvariantRate,
                                                  benchmark::Counter::kIs1000);

    // free memory
    ggml_free(ctx0);
    ggml_free(ctx);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_gallocr_free(allocr);
}

const std::vector<int64_t> log_2_sizes{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
const std::vector<int64_t> log_2_threads{0, 1, 2, 3, 4, 5, 6, 7, 8};

// CPU single-threaded, MxNxK: 1 x 1 x 1 -> 2048 x 2048 x 2048
BENCHMARK_TEMPLATE(bmMulMat, BackendDevice::kCPU, float, GGML_TYPE_F16)->ArgsProduct({log_2_sizes, {0}});
BENCHMARK_TEMPLATE(bmMulMat, BackendDevice::kCPU, float, GGML_TYPE_F32)->ArgsProduct({log_2_sizes, {0}});
// CPU multithreaded, MxNxK: 2048 x 2048 x 2048
BENCHMARK_TEMPLATE(bmMulMat, BackendDevice::kCPU, float, GGML_TYPE_F16)->ArgsProduct({{11}, log_2_threads});
BENCHMARK_TEMPLATE(bmMulMat, BackendDevice::kCPU, float, GGML_TYPE_F32)->ArgsProduct({{11}, log_2_threads});

#ifdef GGML_USE_CUDA
// CUDA single-device, MxNxK: 1 x 1 x 1 -> 2048 x 2048 x 2048
BENCHMARK_TEMPLATE(bmMulMat, BackendDevice::kCUDA, float, GGML_TYPE_F16)->ArgsProduct({log_2_sizes, {0}});
BENCHMARK_TEMPLATE(bmMulMat, BackendDevice::kCUDA, float, GGML_TYPE_F32)->ArgsProduct({log_2_sizes, {0}});
#endif

#ifdef GGML_USE_METAL
// METAL single-device, MxNxK: 1 x 1 x 1 -> 2048 x 2048 x 2048
BENCHMARK_TEMPLATE(bmMulMat, BackendDevice::kMETAL, float, GGML_TYPE_F16)->ArgsProduct({log_2_sizes, {0}});
BENCHMARK_TEMPLATE(bmMulMat, BackendDevice::kMETAL, float, GGML_TYPE_F32)->ArgsProduct({log_2_sizes, {0}});
#endif

BENCHMARK_MAIN();
