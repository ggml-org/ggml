#include "unary.cuh"

template <class T>
static __global__ void neg(const T * x, T * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = -x[i];
}

template <class T>
static __global__ void step(const T * x, T * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = x[i] > (T)0.0f;
}

template <class T>
static __global__ void gelu(const T * x, T * dst, const int k) {
    const T GELU_COEF_A    = 0.044715f;
    const T SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    T xi = x[i];
    dst[i] = (T)0.5f*xi*((T)1.0f + (T)tanhf(SQRT_2_OVER_PI*xi*((T)1.0f + GELU_COEF_A*xi*xi)));
}

template <class T>
static __global__ void gelu_quick(const T * x, T * dst, int k) {
    const T GELU_QUICK_COEF = -1.702f;
    const int i  = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    dst[i] = x[i] * ((T)1.0f / ((T)1.0f + (T)expf(GELU_QUICK_COEF * x[i])));
}

template <class T>
static __global__ void silu(const T * x, T * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = x[i] / ((T)1.0f + (T)expf(-x[i]));
}

template <class T>
static __global__ void silu_back(
        const T * grad, const T * xf, T * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    const T xfi = xf[i];
    const T s = (T)1.0f / ((T)1.0f + (T)expf(-xfi));
    dst[i] = grad[i] * s * ((T)1.0f + xfi * ((T)1.0f - s));
}

template <class T>
static __global__ void tanh(const T * x, T * dst, int k) {
    const int i  = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    dst[i] = tanhf(x[i]);
}

template <class T>
static __global__ void relu(const T * x, T * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = fmaxf(x[i], 0);
}

template <class T>
static __global__ void sigmoid(const T * x, T * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = (T)1.0f / ((T)1.0f + (T)expf(-x[i]));
}

template <class T>
static __global__ void hardsigmoid(const T * x, T * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = fminf(1.0f, fmaxf(0.0f, (x[i] + (T)3.0f) / (T)6.0f));
}

template <class T>
static __global__ void hardswish(const T * x, T * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = x[i] * (T)fminf(1.0f, fmaxf(0.0f, (x[i] + (T)3.0f) / (T)6.0f));
}

template <class T>
static __global__ void exp(const T * x, T * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = expf(x[i]);
}

template <class T>
static __global__ void leaky_relu(const T * x, T * dst, const int k, const float negative_slope) {
    const int i  = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    dst[i] = (T)fmaxf(x[i], 0) + (T)fminf(x[i], 0.0f) * (T)negative_slope;
}

template <class T>
static __global__ void sqr(const T * x, T * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = x[i] * x[i];
}

template <class T>
static __global__ void sqrt(const T * x, T * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = sqrtf(x[i]);
}

template <class T>
static __global__ void sin(const T * x, T * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = sinf(x[i]);
}

template <class T>
static __global__ void cos(const T * x, T * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = cosf(x[i]);
}

template <class T>
static void neg_cuda(const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_NEG_BLOCK_SIZE - 1) / CUDA_NEG_BLOCK_SIZE;
    neg<<<num_blocks, CUDA_NEG_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

template <class T>
static void step_cuda(const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_STEP_BLOCK_SIZE - 1) / CUDA_STEP_BLOCK_SIZE;
    step<<<num_blocks, CUDA_STEP_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

template <class T>
static void gelu_cuda(const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_GELU_BLOCK_SIZE - 1) / CUDA_GELU_BLOCK_SIZE;
    gelu<<<num_blocks, CUDA_GELU_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

template <class T>
static void gelu_quick_cuda(const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_GELU_BLOCK_SIZE - 1) / CUDA_GELU_BLOCK_SIZE;
    gelu_quick<<<num_blocks, CUDA_GELU_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

template <class T>
static void silu_cuda(const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SILU_BLOCK_SIZE - 1) / CUDA_SILU_BLOCK_SIZE;
    silu<<<num_blocks, CUDA_SILU_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

template <class T>
static void silu_back_cuda(const T * grad, const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SILU_BACK_BLOCK_SIZE - 1) / CUDA_SILU_BLOCK_SIZE;
    silu_back<<<num_blocks, CUDA_SILU_BACK_BLOCK_SIZE, 0, stream>>>(grad, x, dst, k);
}

template <class T>
static void tanh_cuda(const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_TANH_BLOCK_SIZE - 1) / CUDA_TANH_BLOCK_SIZE;
    tanh<<<num_blocks, CUDA_TANH_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

template <class T>
static void relu_cuda(const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_RELU_BLOCK_SIZE - 1) / CUDA_RELU_BLOCK_SIZE;
    relu<<<num_blocks, CUDA_RELU_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

template <class T>
static void sigmoid_cuda(const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SIGMOID_BLOCK_SIZE - 1) / CUDA_SIGMOID_BLOCK_SIZE;
    sigmoid<<<num_blocks, CUDA_SIGMOID_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

template <class T>
static void hardsigmoid_cuda(const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_HARDSIGMOID_BLOCK_SIZE - 1) / CUDA_HARDSIGMOID_BLOCK_SIZE;
    hardsigmoid<<<num_blocks, CUDA_HARDSIGMOID_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

template <class T>
static void hardswish_cuda(const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_HARDSWISH_BLOCK_SIZE - 1) / CUDA_HARDSWISH_BLOCK_SIZE;
    hardswish<<<num_blocks, CUDA_HARDSWISH_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

template <class T>
static void exp_cuda(const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_EXP_BLOCK_SIZE - 1) / CUDA_EXP_BLOCK_SIZE;
    exp<<<num_blocks, CUDA_EXP_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

template <class T>
static void leaky_relu_cuda(const T * x, T * dst, const int k, const float negative_slope, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_RELU_BLOCK_SIZE - 1) / CUDA_RELU_BLOCK_SIZE;
    leaky_relu<<<num_blocks, CUDA_RELU_BLOCK_SIZE, 0, stream>>>(x, dst, k, negative_slope);
}

template <class T>
static void sqr_cuda(const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SQR_BLOCK_SIZE - 1) / CUDA_SQR_BLOCK_SIZE;
    sqr<<<num_blocks, CUDA_SQR_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

template <class T>
static void sqrt_cuda(const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SQRT_BLOCK_SIZE - 1) / CUDA_SQRT_BLOCK_SIZE;
    sqrt<<<num_blocks, CUDA_SQRT_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

template <class T>
static void sin_cuda(const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SIN_BLOCK_SIZE - 1) / CUDA_SIN_BLOCK_SIZE;
    sin<<<num_blocks, CUDA_SIN_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

template <class T>
static void cos_cuda(const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_COS_BLOCK_SIZE - 1) / CUDA_COS_BLOCK_SIZE;
    cos<<<num_blocks, CUDA_COS_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

void ggml_cuda_op_neg(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        neg_cuda((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), stream);
    } else {
        neg_cuda((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
}

void ggml_cuda_op_step(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        step_cuda((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), stream);
    } else {
        step_cuda((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
}

void ggml_cuda_op_gelu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        gelu_cuda((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), stream);
    } else {
        gelu_cuda((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
}

void ggml_cuda_op_silu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        silu_cuda((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), stream);
    } else {
        silu_cuda((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
}

void ggml_cuda_op_silu_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // input from forward pass
    const ggml_tensor * src1 = dst->src[1]; // grads of forward pass output

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       * dst_d  = (float       *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        silu_back_cuda((const half *)src0_d, (const half *)src1_d, (half *)dst_d, ggml_nelements(src0), stream);
    } else {
        silu_back_cuda((const float*)src0_d, (const float*)src1_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
}

void ggml_cuda_op_gelu_quick(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        gelu_quick_cuda((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), stream);
    } else {
        gelu_quick_cuda((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
}

void ggml_cuda_op_tanh(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        tanh_cuda((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), stream);
    } else {
        tanh_cuda((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
}

void ggml_cuda_op_relu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        relu_cuda((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), stream);
    } else {
        relu_cuda((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
}

void ggml_cuda_op_sigmoid(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        sigmoid_cuda((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), stream);
    } else {
        sigmoid_cuda((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
}

void ggml_cuda_op_hardsigmoid(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        hardsigmoid_cuda((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), stream);
    } else {
        hardsigmoid_cuda((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
}

void ggml_cuda_op_hardswish(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        hardswish_cuda((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), stream);
    } else {
        hardswish_cuda((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
}

void ggml_cuda_op_exp(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        exp_cuda((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), stream);
    } else {
        exp_cuda((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
}

void ggml_cuda_op_leaky_relu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    float negative_slope;
    memcpy(&negative_slope, dst->op_params, sizeof(float));

    if (src0->type == GGML_TYPE_F16) {
        leaky_relu_cuda((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), negative_slope, stream);
    } else {
        leaky_relu_cuda((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), negative_slope, stream);
    }
}

void ggml_cuda_op_sqr(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        sqr_cuda((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), stream);
    } else {
        sqr_cuda((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
}

void ggml_cuda_op_sqrt(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        sqrt_cuda((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), stream);
    } else {
        sqrt_cuda((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
}

void ggml_cuda_op_sin(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        sin_cuda((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), stream);
    } else {
        sin_cuda((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
}

void ggml_cuda_op_cos(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        cos_cuda((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), stream);
    } else {
        cos_cuda((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
}
