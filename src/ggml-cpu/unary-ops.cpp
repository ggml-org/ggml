#include "unary-ops.inc"

static inline float op_abs(float x) {
    return fabsf(x);
}

static inline float op_sgn(float x) {
    return (x > 0.f) ? 1.f : ((x < 0.f) ? -1.f : 0.f);
}

static inline float op_neg(float x) {
    return -x;
}

static inline float op_step(float x) {
    return (x > 0.f) ? 1.f : 0.f;
}

static inline float op_tanh(float x) {
    return tanhf(x);
}

static inline float op_elu(float x) {
    return (x > 0.f) ? x : expm1f(x);
}

static inline float op_relu(float x) {
    return (x > 0.f) ? x : 0.f;
}

static inline float op_sigmoid(float x) {
    return 1.f / (1.f + expf(-x));
}

static inline float op_hardsigmoid(float x) {
    return fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

static inline float op_exp(float x) {
    return expf(x);
}

static inline float op_hardswish(float x) {
    return x * fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

static inline float op_sqr(float x) {
    return x * x;
}

static inline float op_sqrt(float x) {
    return sqrtf(x);
}

static inline float op_log(float x) {
    return logf(x);
}

void ggml_compute_forward_abs(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_abs>(params, dst);
}

void ggml_compute_forward_sgn(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sgn>(params, dst);
}

void ggml_compute_forward_neg(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_neg>(params, dst);
}

void ggml_compute_forward_step(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_step>(params, dst);
}

void ggml_compute_forward_tanh(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_tanh>(params, dst);
}

void ggml_compute_forward_elu(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_elu>(params, dst);
}

void ggml_compute_forward_relu(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_relu>(params, dst);
}

void ggml_compute_forward_sigmoid(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sigmoid>(params, dst);
}

void ggml_compute_forward_hardsigmoid(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_hardsigmoid>(params, dst);
}

void ggml_compute_forward_exp(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_exp>(params, dst);
}

void ggml_compute_forward_hardswish(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_hardswish>(params, dst);
}

void ggml_compute_forward_sqr(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sqr>(params, dst);
}

void ggml_compute_forward_sqrt(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_sqrt>(params, dst);
}

void ggml_compute_forward_log(const ggml_compute_params * params, ggml_tensor * dst) {
    unary_op<op_log>(params, dst);
}
