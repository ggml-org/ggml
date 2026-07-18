#include "win-part.cuh"

// ============================================================================
// WIN_PART - Window partition kernel
// ============================================================================

template <typename T>
static __global__ void win_part_kernel(
    const T * __restrict__ src,
    T * __restrict__ dst,
    int64_t C,        // channels
    int64_t H_src,    // source height
    int64_t W_src,    // source width
    int64_t w,        // window size
    int64_t nep_x,    // number of windows in X
    int64_t nep_y) {  // number of windows in Y

    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = C * w * w * nep_x * nep_y;

    if (idx >= total) return;

    // Decompose output index: [C, w, w, num_windows]
    const int64_t c = idx % C;
    int64_t tmp = idx / C;
    const int64_t x_in_window = tmp % w;
    tmp /= w;
    const int64_t y_in_window = tmp % w;
    const int64_t window_idx = tmp / w;

    // Which window (px, py)?
    const int64_t px = window_idx % nep_x;
    const int64_t py = window_idx / nep_x;

    // Position in source image
    const int64_t x_src = px * w + x_in_window;
    const int64_t y_src = py * w + y_in_window;

    // Check if we're in the image or in padding
    if (y_src >= H_src || x_src >= W_src) {
        dst[idx] = (T)0.0f;  // padding
    } else {
        // Calculate source index: [C, H, W]
        const int64_t src_idx = c + y_src * W_src * C + x_src * C;
        dst[idx] = src[src_idx];
    }
}

template <typename T>
static void win_part_cuda(
    const T * src, T * dst,
    int64_t C, int64_t H, int64_t W, int64_t w,
    int64_t nep_x, int64_t nep_y,
    cudaStream_t stream) {

    const int64_t total = C * w * w * nep_x * nep_y;
    const int num_blocks = (total + CUDA_WIN_PART_BLOCK_SIZE - 1) / CUDA_WIN_PART_BLOCK_SIZE;

    win_part_kernel<<<num_blocks, CUDA_WIN_PART_BLOCK_SIZE, 0, stream>>>(
        src, dst, C, H, W, w, nep_x, nep_y);
}

// ============================================================================
// WIN_UNPART - Window unpartition kernel
// ============================================================================

template <typename T>
static __global__ void win_unpart_kernel(
    const T * __restrict__ src,
    T * __restrict__ dst,
    int64_t C,        // channels
    int64_t H_dst,    // destination height
    int64_t W_dst,    // destination width
    int64_t w,        // window size
    int64_t nep_x) {  // number of windows in X

    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = C * H_dst * W_dst;

    if (idx >= total) return;

    // Decompose destination index: [C, H, W]
    const int64_t c = idx % C;
    int64_t tmp = idx / C;
    const int64_t x_dst = tmp % W_dst;
    const int64_t y_dst = tmp / W_dst;

    // Which window contains this pixel?
    const int64_t px = x_dst / w;
    const int64_t py = y_dst / w;
    const int64_t window_idx = py * nep_x + px;

    // Position in window
    const int64_t x_in_window = x_dst % w;
    const int64_t y_in_window = y_dst % w;

    // Calculate source index: [C, w, w, num_windows]
    const int64_t src_idx = c + x_in_window * C + y_in_window * w * C + window_idx * w * w * C;

    dst[idx] = src[src_idx];
}

template <typename T>
static void win_unpart_cuda(
    const T * src, T * dst,
    int64_t C, int64_t H, int64_t W, int64_t w,
    int64_t nep_x,
    cudaStream_t stream) {

    const int64_t total = C * H * W;
    const int num_blocks = (total + CUDA_WIN_PART_BLOCK_SIZE - 1) / CUDA_WIN_PART_BLOCK_SIZE;

    win_unpart_kernel<<<num_blocks, CUDA_WIN_PART_BLOCK_SIZE, 0, stream>>>(
        src, dst, C, H, W, w, nep_x);
}

// ============================================================================
// Public ggml API functions
// ============================================================================

void ggml_cuda_op_win_part(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    // Get parameters from op_params
    const int32_t nep0 = ((const int32_t *)(dst->op_params))[0];  // number of windows X
    const int32_t nep1 = ((const int32_t *)(dst->op_params))[1];  // number of windows Y
    const int32_t w    = ((const int32_t *)(dst->op_params))[2];  // window size

    // Source dimensions [ne0, ne1, ne2, ne3] = [C, H, W, B]
    const int64_t C = src0->ne[0];
    const int64_t H = src0->ne[1];
    const int64_t W = src0->ne[2];

    if (src0->type == GGML_TYPE_F16) {
        win_part_cuda(
            (const half *)src0_d, (half *)dst_d,
            C, H, W, w, nep0, nep1, stream);
    } else {
        win_part_cuda(
            (const float *)src0_d, (float *)dst_d,
            C, H, W, w, nep0, nep1, stream);
    }
}

void ggml_cuda_op_win_unpart(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    // Get window size from op_params
    const int32_t w = ((const int32_t *)(dst->op_params))[0];

    // Destination dimensions [ne0, ne1, ne2, ne3] = [C, H, W, B]
    const int64_t C = dst->ne[0];
    const int64_t H = dst->ne[1];
    const int64_t W = dst->ne[2];

    // Calculate number of windows in X
    const int64_t nep_x = (W + w - 1) / w;

    if (src0->type == GGML_TYPE_F16) {
        win_unpart_cuda(
            (const half *)src0_d, (half *)dst_d,
            C, H, W, w, nep_x, stream);
    } else {
        win_unpart_cuda(
            (const float *)src0_d, (float *)dst_d,
            C, H, W, w, nep_x, stream);
    }
}

