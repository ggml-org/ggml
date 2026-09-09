#include "conv2d-transpose.cuh"
#include "convert.cuh"

// Each thread computes CONV2D_TRANSPOSE_CO_PER_THREAD output channels for one output
// pixel. The c_in reduction is unrolled to hide global memory latency.
#define CONV2D_TRANSPOSE_CO_PER_THREAD 16
#define CONV2D_TRANSPOSE_CI_UNROLL     8

// Fast path for kernel 2x2 / stride 2 / pad 0 (all SAM3 neck/memory-encoder deconvs):
// each output pixel maps to a single input pixel using all four taps, so a thread reads
// one input value per c_in step, reuses it for COPT*4 FMAs and loads the 4 taps as one
// vector load. Output layout: kernel ne = [kw, kh, c_out, c_in].
#define CONV2D_TRANSPOSE_FAST_COPT 8

template <typename kernel_t>
struct conv2d_transpose_fast_wload;

template <>
struct conv2d_transpose_fast_wload<half> {
    __device__ __forceinline__ static float4 load(const half * k, const int kidx) {
        const float2 w = __ldg((const float2 *) (k + kidx));
        const half2 w0 = *(const half2 *) &w.x;
        const half2 w1 = *(const half2 *) &w.y;
        const float2 f0 = __half22float2(w0);
        const float2 f1 = __half22float2(w1);
        return make_float4(f0.x, f0.y, f1.x, f1.y);
    }
};

template <>
struct conv2d_transpose_fast_wload<float> {
    __device__ __forceinline__ static float4 load(const float * k, const int kidx) {
        return __ldg((const float4 *) (k + kidx));
    }
};

template <typename kernel_t>
static __global__ void conv2d_transpose_kernel_fast(const float * __restrict__ input,
                                                    const kernel_t * __restrict__ kernel,
                                                    float * __restrict__ output,
                                                    const int in_w,
                                                    const int in_h,
                                                    const int out_w,
                                                    const int out_h,
                                                    const int c_in,
                                                    const int c_out,
                                                    const int batches) {
    const int spatial_in = in_w * in_h;
    const int n_co_groups = c_out / CONV2D_TRANSPOSE_FAST_COPT;
    const int total_threads = spatial_in * batches * n_co_groups;

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_threads) {
        return;
    }

    const int pixel_id = tid % (spatial_in * batches);
    const int co_group = tid / (spatial_in * batches);
    const int co_base  = co_group * CONV2D_TRANSPOSE_FAST_COPT;

    const int n_idx = pixel_id / spatial_in;
    const int pix   = pixel_id % spatial_in;
    const int in_x  = pix % in_w;
    const int in_y  = pix / in_w;

    const int input_stride_ci  = spatial_in;
    const int kernel_stride_ci = 4 * c_out;

    const int input_base  = n_idx * (spatial_in * c_in) + in_y * in_w + in_x;
    const int output_base = n_idx * (out_w * out_h * c_out) + co_base * out_w * out_h + in_y * 2 * out_w + in_x * 2;

    float acc[CONV2D_TRANSPOSE_FAST_COPT][4] = {};

    for (int ci = 0; ci < c_in; ++ci) {
        const float vin = __ldg(input + input_base + ci * input_stride_ci);
        #pragma unroll
        for (int cl = 0; cl < CONV2D_TRANSPOSE_FAST_COPT; ++cl) {
            const float4 w = conv2d_transpose_fast_wload<kernel_t>::load(kernel, ci * kernel_stride_ci + (co_base + cl) * 4);
            acc[cl][0] = fmaf(vin, w.x, acc[cl][0]);
            acc[cl][1] = fmaf(vin, w.y, acc[cl][1]);
            acc[cl][2] = fmaf(vin, w.z, acc[cl][2]);
            acc[cl][3] = fmaf(vin, w.w, acc[cl][3]);
        }
    }

    const int output_stride_ci = out_w * out_h;
    #pragma unroll
    for (int cl = 0; cl < CONV2D_TRANSPOSE_FAST_COPT; ++cl) {
        const int ooff = output_base + cl * output_stride_ci;
        output[ooff]         = acc[cl][0];
        output[ooff + 1]     = acc[cl][1];
        output[ooff + out_w] = acc[cl][2];
        output[ooff + out_w + 1] = acc[cl][3];
    }
}

template <typename kernel_t>
static __global__ void conv2d_transpose_kernel(const float * __restrict__ input,
                                               const kernel_t * __restrict__ kernel,
                                               float * __restrict__ output,
                                               const int in_w,
                                               const int in_h,
                                               const int out_w,
                                               const int out_h,
                                               const int kernel_w,
                                               const int kernel_h,
                                               const int stride,
                                               const int c_in,
                                               const int c_out,
                                               const int batches) {
    const int spatial = out_w * out_h;
    const int n_co_groups = (c_out + CONV2D_TRANSPOSE_CO_PER_THREAD - 1) / CONV2D_TRANSPOSE_CO_PER_THREAD;
    const int total_pixels = spatial * batches;
    const int total_threads = total_pixels * n_co_groups;

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_threads) {
        return;
    }

    const int pixel_id = tid % total_pixels;
    const int co_group = tid / total_pixels;
    const int co_base  = co_group * CONV2D_TRANSPOSE_CO_PER_THREAD;

    const int n_idx = pixel_id / spatial;
    const int pix   = pixel_id % spatial;
    const int out_x = pix % out_w;
    const int out_y = pix / out_w;

    const int input_stride_ci  = in_w * in_h;
    const int kernel_stride_ci = kernel_h * kernel_w * c_out;
    const int input_base       = n_idx * (input_stride_ci * c_in);

    float acc[CONV2D_TRANSPOSE_CO_PER_THREAD] = {};

    for (int kh = 0; kh < kernel_h; ++kh) {
        const int in_y_nom = out_y - kh;
        if (in_y_nom < 0 || in_y_nom % stride != 0) {
            continue;
        }
        const int in_y = in_y_nom / stride;
        if (in_y >= in_h) {
            continue;
        }

        for (int kw = 0; kw < kernel_w; ++kw) {
            const int in_x_nom = out_x - kw;
            if (in_x_nom < 0 || in_x_nom % stride != 0) {
                continue;
            }
            const int in_x = in_x_nom / stride;
            if (in_x >= in_w) {
                continue;
            }

            const int tap         = kh * kernel_w + kw;
            const int input_off   = input_base + in_y * in_w + in_x;
            const int kernel_taps = kernel_h * kernel_w;

            int ci = 0;
            for (; ci + CONV2D_TRANSPOSE_CI_UNROLL <= c_in; ci += CONV2D_TRANSPOSE_CI_UNROLL) {
                float v[CONV2D_TRANSPOSE_CI_UNROLL];
                #pragma unroll
                for (int u = 0; u < CONV2D_TRANSPOSE_CI_UNROLL; ++u) {
                    v[u] = __ldg(input + input_off + (ci + u) * input_stride_ci);
                }
                #pragma unroll
                for (int u = 0; u < CONV2D_TRANSPOSE_CI_UNROLL; ++u) {
                    const float vin = v[u];
                    #pragma unroll
                    for (int cl = 0; cl < CONV2D_TRANSPOSE_CO_PER_THREAD; ++cl) {
                        const int co = co_base + cl;
                        if (co < c_out) {
                            const int kidx = (ci + u) * kernel_stride_ci + co * kernel_taps + tap;
                            acc[cl] = fmaf(vin, ggml_cuda_cast<float>(__ldg(kernel + kidx)), acc[cl]);
                        }
                    }
                }
            }
            for (; ci < c_in; ++ci) {
                const float vin = __ldg(input + input_off + ci * input_stride_ci);
                #pragma unroll
                for (int cl = 0; cl < CONV2D_TRANSPOSE_CO_PER_THREAD; ++cl) {
                    const int co = co_base + cl;
                    if (co < c_out) {
                        const int kidx = ci * kernel_stride_ci + co * kernel_taps + tap;
                        acc[cl] = fmaf(vin, ggml_cuda_cast<float>(__ldg(kernel + kidx)), acc[cl]);
                    }
                }
            }
        }
    }

    const int output_stride_ci = out_w * out_h;
    const int output_off       = n_idx * (output_stride_ci * c_out) + out_y * out_w + out_x;
    #pragma unroll
    for (int cl = 0; cl < CONV2D_TRANSPOSE_CO_PER_THREAD; ++cl) {
        const int co = co_base + cl;
        if (co < c_out) {
            output[output_off + co * output_stride_ci] = acc[cl];
        }
    }
}

//input is (W, H, C_in, N), Kernel is (W, H, C_out, C_in)
void ggml_cuda_conv_2d_transpose_p0(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * kernel = dst->src[0];
    const ggml_tensor * input  = dst->src[1];

    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(input->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    const float * input_data  = (const float *) input->data;
    float *       output_data = (float *) dst->data;
    const void *  kernel_data = kernel->data;

    const int input_w      = input->ne[0];
    const int input_h      = input->ne[1];
    const int output_w     = dst->ne[0];
    const int output_h     = dst->ne[1];
    const int channels_in  = input->ne[2];
    const int channels_out = kernel->ne[2];
    const int kernel_w     = kernel->ne[0];
    const int kernel_h     = kernel->ne[1];
    const int stride       = dst->op_params[0];
    const int batches      = input->ne[3];

    GGML_ASSERT(channels_in == kernel->ne[3]);
    GGML_ASSERT(stride > 0);

    cudaStream_t st = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(input));
    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int n_co_groups = (channels_out + CONV2D_TRANSPOSE_CO_PER_THREAD - 1) / CONV2D_TRANSPOSE_CO_PER_THREAD;
    const int total       = output_w * output_h * batches * n_co_groups;
    const int blocks      = (total + CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE - 1) / CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE;

    const bool use_fast = kernel_w == 2 && kernel_h == 2 && stride == 2 &&
                          channels_out % CONV2D_TRANSPOSE_FAST_COPT == 0 &&
                          channels_in % 8 == 0;

    if (use_fast) {
        const int spatial_in  = input_w * input_h;
        const int n_co_groups = channels_out / CONV2D_TRANSPOSE_FAST_COPT;
        const int total       = spatial_in * batches * n_co_groups;
        const int blocks      = (total + CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE - 1) / CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE;

        if (kernel->type == GGML_TYPE_F16) {
            conv2d_transpose_kernel_fast<half><<<blocks, CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE, 0, st>>>(
                input_data, (const half *) kernel_data, output_data, input_w, input_h, output_w, output_h,
                channels_in, channels_out, batches);
        } else {
            conv2d_transpose_kernel_fast<float><<<blocks, CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE, 0, st>>>(
                input_data, (const float *) kernel_data, output_data, input_w, input_h, output_w, output_h,
                channels_in, channels_out, batches);
        }
        return;
    }

    if (kernel->type == GGML_TYPE_F16) {
        conv2d_transpose_kernel<half><<<blocks, CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE, 0, st>>>(
            input_data, (const half *) kernel_data, output_data, input_w, input_h, output_w, output_h, kernel_w,
            kernel_h, stride, channels_in, channels_out, batches);

    } else {
        conv2d_transpose_kernel<float><<<blocks, CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE, 0, st>>>(
            input_data, (const float *) kernel_data, output_data, input_w, input_h, output_w, output_h, kernel_w,
            kernel_h, stride, channels_in, channels_out, batches);
    }
}
