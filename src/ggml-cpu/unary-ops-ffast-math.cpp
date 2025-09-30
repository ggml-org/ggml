#include "unary-ops.inc"

// This file is compiled with -ffast-math only ifdef GGML_CPU_FFAST_MATH.
// libmvec allows sine/cos vectorization but not bit-identically to libm.
// Backends (e.g. CUDA) aren't bit-identical either, but more people expect the CPU backend to be.

static inline float op_sin(float x) {
    return sinf(x);
}

static inline float op_cos(float x) {
    return cosf(x);
}

void ggml_compute_forward_sin(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sin>(params, dst);
}

void ggml_compute_forward_cos(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_cos>(params, dst);
}
