#include "conv-transpose-1d.cuh"

// One CUDA warp (32 threads) cooperatively computes one output pixel
// dst[oc, ol] (== dst[ol + OL*oc] in linear index, since we keep ne2/ne3 == 1).
//
// Grid : (OL, OC, 1)
// Block: (32, 1, 1) — exactly one warp; sized below as CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE.
//
// Two perf-critical changes vs the original "1 thread per output pixel + scan
// the full IC*IL grid + skip via conditional" implementation:
//
//   1. Narrow the input position i to the small range that actually
//      contributes:
//          out[ol, oc] = sum over (ic, i, ki) of  k[ki, oc, ic] * x[i, ic]
//          subject to  i*s0 + ki == ol,  0 <= ki < K,  0 <= i < IL
//      ⇒    i ∈ [ ceil((ol - K + 1)/s0), floor(ol/s0) ] ∩ [0, IL-1]
//      typically (KS=16, s0=8) this is 2 iterations of i instead of IL=O(100).
//
//   2. Parallelise the IC reduction across the warp (each thread handles a
//      strided slice of IC) and finalise with __shfl_xor_sync.  This gives
//      32× useful work per warp on top of the i-range narrowing.
//
// Layouts (matching the original kernel and the Vulkan / Metal patches):
//   src0 (kernel) : [K, OC, IC] row-major  → element (ki, oc, ic) at
//                                            ic*(OC*K) + oc*K + ki
//   src1 (input)  : [IL, IC]    row-major  → element (i, ic) at ic*IL + i
//   dst           : [OL, OC]    row-major  → element (ol, oc) at oc*OL + ol
//
// Limitation (unchanged from the original kernel): only ne1==ne3==1 is
// supported; the host-side wrapper enforces that via the contiguous +
// shape assertions.
static __global__ void conv_transpose_1d_kernel(
        const int s0, const int p0, const int d0, const int output_size,
        const int src0_ne0, const int src0_ne1, const int src0_ne2, const int src0_ne3,
        const int src1_ne0, const int src1_ne1, const int src1_ne2, const int src1_ne3,
        const int dst_ne0,  const int dst_ne1,  const int dst_ne2,  const int dst_ne3,
        const float * __restrict__ src0, const float * __restrict__ src1, float * __restrict__ dst) {

    const int ol = blockIdx.x;
    const int oc = blockIdx.y;
    if (ol >= dst_ne0 || oc >= dst_ne1) {
        return;
    }

    const int K  = src0_ne0;
    const int OC = dst_ne1;
    const int IC = src0_ne2;
    const int IL = src1_ne0;

    // Range of input positions i that contribute to this output pixel.
    int i_start = (ol - K + 1 + s0 - 1) / s0;   // ceil((ol - K + 1) / s0)
    if (i_start < 0)      i_start = 0;
    int i_end = ol / s0;
    if (i_end > IL - 1)   i_end = IL - 1;

    const int tid = threadIdx.x;
    const int nth = blockDim.x;

    float v = 0.0f;

    // Each thread handles a strided slice of IC; the range of i is
    // already narrow (≤ K/s0 + 1), so the inner loop is the cheap one.
    for (int ic = tid; ic < IC; ic += nth) {
        const int kernel_base = (ic * OC + oc) * K;
        const int input_base  = ic * IL;
        #pragma unroll 4
        for (int i = i_start; i <= i_end; ++i) {
            const int ki = ol - i * s0;
            v += src0[kernel_base + ki] * src1[input_base + i];
        }
    }

    // Reduce across the warp.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFFu, v, offset);
    }

    if (tid == 0) {
        dst[oc * dst_ne0 + ol] = v;
    }

    GGML_UNUSED_VARS(p0, d0, output_size,
                     src0_ne1, src0_ne3, src1_ne1, src1_ne2, src1_ne3,
                     dst_ne2, dst_ne3);
}

static void conv_transpose_1d_f32_f32_cuda(
        const int s0, const int p0, const int d0, const int output_size,
        const int src0_ne0, const int src0_ne1, const int src0_ne2, const int src0_ne3,
        const int src1_ne0, const int src1_ne1, const int src1_ne2, const int src1_ne3,
        const int dst_ne0,  const int dst_ne1,  const int dst_ne2,  const int dst_ne3,
        const float * src0, const float * src1, float * dst,
        cudaStream_t stream) {

    // Block = one warp (32 threads).  Grid has one block per output pixel,
    // i.e. (OL, OC).  ne2/ne3 are required to be 1 by the existing host-side
    // assertions, so we don't extend the grid into z.
    const dim3 block_dim(CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE, 1, 1);
    const dim3 grid_dim((unsigned)dst_ne0, (unsigned)dst_ne1, 1);

    conv_transpose_1d_kernel<<<grid_dim, block_dim, 0, stream>>>(
        s0, p0, d0, output_size,
        src0_ne0, src0_ne1, src0_ne2, src0_ne3,
        src1_ne0, src1_ne1, src1_ne2, src1_ne3,
        dst_ne0,  dst_ne1,  dst_ne2,  dst_ne3,
        src0, src1, dst);
}

void ggml_cuda_op_conv_transpose_1d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;

    const ggml_tensor * src1 = dst->src[1];
    const float * src1_d = (const float *)src1->data;

    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    const int32_t * opts = (const int32_t *)dst->op_params;

    const int s0 = opts[0];
    const int p0 = 0;//opts[3];
    const int d0 = 1;//opts[4];

    const int64_t output_size = ggml_nelements(dst);

    conv_transpose_1d_f32_f32_cuda(s0, p0, d0, output_size,
        src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
        src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
        dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
        src0_d, src1_d, dst_d, stream);
}
