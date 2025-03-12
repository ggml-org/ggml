#include "binary_ops.h"

#if defined(GGML_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>

#define vDSP_fn_t(f) void (*f)(const float *, vDSP_Stride, const float *, vDSP_Stride, float *, vDSP_Stride, vDSP_Length)
#endif

static inline float op_add(float a, float b) {
    return a + b;
}

static inline float op_sub(float a, float b) {
    return a - b;
}

static inline float op_mul(float a, float b) {
    return a * b;
}

static inline float op_div(float a, float b) {
    return a / b;
}

// TODO: Use the 'traits' lookup table (for type conversion fns), instead of squeezing them into templates
template <float (*op)(float, float), typename src0_t, float (*src0_to_f32)(src0_t), typename src1_t, float (*src1_to_f32)(src1_t), typename dst_t, dst_t (*f32_to_dst)(float)>
static inline void vec_binary_op_contiguous(const int64_t n, dst_t * z, const src0_t * x, const src1_t * y) {
    for (int i = 0; i < n; i++) {
        z[i] = f32_to_dst(op(src0_to_f32(x[i]), src1_to_f32(y[i])));
    }
}

template <float (*op)(float, float), typename src0_t, float (*src0_to_f32)(src0_t), typename src1_t, float (*src1_to_f32)(src1_t), typename dst_t, dst_t (*f32_to_dst)(float)>
static inline void vec_binary_op_non_contiguous(const int64_t n, const int64_t ne10, const int64_t nb10, dst_t * z, const src0_t * x, const src1_t * y) {
    for (int i = 0; i < n; i++) {
        int i10 = i % ne10;
        src1_t * y_ptr = (src1_t *)((char *)y + i10*nb10);
        z[i] = f32_to_dst(op(src0_to_f32(x[i]), src1_to_f32(*y_ptr)));
    }
}

template <float (*op)(float, float), typename src0_t, float (*src0_to_f32)(src0_t), typename src1_t, float (*src1_to_f32)(src1_t), typename dst_t, dst_t (*f32_to_dst)(float)>
static void apply_binary_op(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT( nb0 == sizeof(dst_t));
    GGML_ASSERT(nb00 == sizeof(src0_t));

    const auto [ir0, ir1] = get_thread_range(params, src0);
    const bool is_src1_contiguous = (nb10 == sizeof(src1_t));

    if (!is_src1_contiguous) { // broadcast not implemented yet for non-contiguous
        GGML_ASSERT(ggml_are_same_shape(src0, src1));
    }

#ifdef GGML_USE_ACCELERATE
    vDSP_fn_t(vDSP_op) = nullptr;
    // TODO - avoid the f32-only check using type 'trait' lookup tables and row-based src-to-float conversion functions
    if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        if (op == op_add) {
            vDSP_op = vDSP_vadd;
        } else if (op == op_sub) {
            vDSP_op = vDSP_vsub;
        } else if (op == op_mul) {
            vDSP_op = vDSP_vmul;
        } else if (op == op_div) {
            vDSP_op = vDSP_vdiv;
        }
    }
#endif

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i03 = ir/(ne02*ne01);
        const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        const int64_t i13 = i03 % ne13;
        const int64_t i12 = i02 % ne12;
        const int64_t i11 = i01 % ne11;

        dst_t  * dst_ptr  = (dst_t  *) ((char *) dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
        src0_t * src0_ptr = (src0_t *) ((char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);
        src1_t * src1_ptr = (src1_t *) ((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11);

        if (is_src1_contiguous) {
            // src1 is broadcastable across src0 and dst in i1, i2, i3
            const int64_t nr0 = ne00 / ne10;

            for (int64_t r = 0; r < nr0; ++r) {
#ifdef GGML_USE_ACCELERATE
                if (vDSP_op != nullptr) {
                    vDSP_op((float *)src1_ptr, 1, (float *)src0_ptr + r*ne10, 1, (float *)dst_ptr + r*ne10, 1, ne10);
                    continue;
                }
#endif
                vec_binary_op_contiguous<op, src0_t, src0_to_f32, src1_t, src1_to_f32, dst_t, f32_to_dst>(ne10, dst_ptr + r*ne10, src0_ptr + r*ne10, src1_ptr);
            }
        } else {
            vec_binary_op_non_contiguous<op, src0_t, src0_to_f32, src1_t, src1_to_f32, dst_t, f32_to_dst>(ne0, ne10, nb10, dst_ptr, src0_ptr, src1_ptr);
        }
    }
}

// TODO: Use the 'traits' lookup table (for type conversion fns), instead of a mass of 'if' conditions with long templates
template <float (*op)(float, float)>
static void binary_op(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    /*  */ if (src0->type == GGML_TYPE_F32  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_binary_op<op, F32_SRC, F32_SRC, F32_DST>(params, dst);
    } else if (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_binary_op<op, F16_SRC, F16_SRC, F16_DST>(params, dst);
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_binary_op<op, BF16_SRC, BF16_SRC, BF16_DST>(params, dst);
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_BF16) {
        apply_binary_op<op, BF16_SRC, F32_SRC, BF16_DST>(params, dst);
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) {
        apply_binary_op<op, BF16_SRC, F32_SRC, F32_DST>(params, dst);
    } else if (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F16) {
        apply_binary_op<op, F16_SRC, F32_SRC, F16_DST>(params, dst);
    } else if (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) {
        apply_binary_op<op, F16_SRC, F32_SRC, F32_DST>(params, dst);
    } else {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s, src1: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type), ggml_type_name(src1->type));
        GGML_ABORT("fatal error");
    }
}

void ggml_compute_forward_add_non_quantized(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    binary_op<op_add>(params, dst);
}

void ggml_compute_forward_sub(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    binary_op<op_sub>(params, dst);
}

void ggml_compute_forward_mul(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    binary_op<op_mul>(params, dst);
}

void ggml_compute_forward_div(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    binary_op<op_div>(params, dst);
}
