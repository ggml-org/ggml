#include "unary_ops.h"

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
    return expm1f(x);
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

static inline float op_sin(float x) {
    return sinf(x);
}

static inline float op_cos(float x) {
    return cosf(x);
}

static inline float op_log(float x) {
    return logf(x);
}

// TODO: Use the 'traits' lookup table (for type conversion fns), instead of squeezing them into templates
template <float (*op)(float), typename src0_t, float (*src0_to_f32)(src0_t), typename dst_t, dst_t (*f32_to_dst)(float)>
static inline void vec_unary_op(int64_t n, dst_t * y, const src0_t * x) {
    for (int i = 0; i < n; i++) {
        y[i] = f32_to_dst(op(src0_to_f32(x[i])));
    }
}

template <float (*op)(float), typename src0_t, float (*src0_to_f32)(src0_t), typename dst_t, dst_t (*f32_to_dst)(float)>
static void apply_unary_op(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous_1(src0) && ggml_is_contiguous_1(dst) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == sizeof(dst_t));
    GGML_ASSERT(nb00 == sizeof(src0_t));

    const auto [ir0, ir1] = get_thread_range(params, src0);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i03 = ir/(ne02*ne01);
        const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        dst_t  * dst_ptr  = (dst_t  *) ((char *) dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
        src0_t * src0_ptr = (src0_t *) ((char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);

        vec_unary_op<op, src0_t, src0_to_f32, dst_t, f32_to_dst>(ne0, dst_ptr, src0_ptr);
    }
}

// TODO: Use the 'traits' lookup table (for type conversion fns), instead of a mass of 'if' conditions with long templates
template <float (*op)(float)>
static void unary_op(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];

    /*  */ if (src0->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_unary_op<op, F32_SRC, F32_DST>(params, dst);
    } else if (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_unary_op<op, F16_SRC, F16_DST>(params, dst);
    } else if (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_unary_op<op, BF16_SRC, BF16_DST>(params, dst);
    } else if (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, BF16_SRC, F32_DST>(params, dst);
    } else if (src0->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F32) {
        apply_unary_op<op, F16_SRC, F32_DST>(params, dst);
    } else {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type));
        GGML_ABORT("fatal error");
    }
}

void ggml_compute_forward_abs(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    unary_op<op_abs>(params, dst);
}

void ggml_compute_forward_sgn(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    unary_op<op_sgn>(params, dst);
}

void ggml_compute_forward_neg(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    unary_op<op_neg>(params, dst);
}

void ggml_compute_forward_step(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    unary_op<op_step>(params, dst);
}

void ggml_compute_forward_tanh(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    unary_op<op_tanh>(params, dst);
}

void ggml_compute_forward_elu(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    unary_op<op_elu>(params, dst);
}

void ggml_compute_forward_relu(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    unary_op<op_relu>(params, dst);
}

void ggml_compute_forward_sigmoid(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    unary_op<op_sigmoid>(params, dst);
}

void ggml_compute_forward_hardsigmoid(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    unary_op<op_hardsigmoid>(params, dst);
}

void ggml_compute_forward_exp(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    unary_op<op_exp>(params, dst);
}

void ggml_compute_forward_hardswish(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    unary_op<op_hardswish>(params, dst);
}

void ggml_compute_forward_sqr(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    unary_op<op_sqr>(params, dst);
}

void ggml_compute_forward_sqrt(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    unary_op<op_sqrt>(params, dst);
}

void ggml_compute_forward_sin(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    unary_op<op_sin>(params, dst);
}

void ggml_compute_forward_cos(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    unary_op<op_cos>(params, dst);
}

void ggml_compute_forward_log(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    unary_op<op_log>(params, dst);
}
