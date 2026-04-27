#include "common.cuh"

// One warp per output pixel; see conv-transpose-1d.cu for why.
#define CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE 32

void ggml_cuda_op_conv_transpose_1d(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
