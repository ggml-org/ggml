#pragma once

#include "ggml.h"
#include "ggml-cpu-traits.h"
#include "ggml-cpu-impl.h"
#include "ggml-impl.h"

#ifdef __cplusplus

#include <utility>

extern "C" {
#endif

// convenience functions/macros for use in template calls
// note: these won't be required after the 'traits' lookup table is used.
static inline ggml_fp16_t f32_to_f16(float x) {
    return GGML_FP32_TO_FP16(x);
}

static inline float f16_to_f32(ggml_fp16_t x) {
    return GGML_FP16_TO_FP32(x);
}

static inline ggml_bf16_t f32_to_bf16(float x) {
    return GGML_FP32_TO_BF16(x);
}

static inline float bf16_to_f32(ggml_bf16_t x) {
    return GGML_BF16_TO_FP32(x);
}

static inline float f32_to_f32(float x) {
    return x;
}

// data type, conversion function.
#define F16_SRC     ggml_fp16_t,    f16_to_f32
#define BF16_SRC    ggml_bf16_t,    bf16_to_f32
#define F32_SRC     float,          f32_to_f32

#define F16_DST     ggml_fp16_t,    f32_to_f16
#define BF16_DST    ggml_bf16_t,    f32_to_bf16
#define F32_DST     float,          f32_to_f32

#ifdef __cplusplus
}

static std::pair<int64_t, int64_t> get_thread_range(const struct ggml_compute_params * params, const struct ggml_tensor * src0) {
    const int64_t ith = params->ith;
    const int64_t nth = params->nth;

    const int64_t nr  = ggml_nrows(src0);

    // rows per thread
    const int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int64_t ir0 = dr*ith;
    const int64_t ir1 = MIN(ir0 + dr, nr);

    return {ir0, ir1};
}

#endif
