#include "common.cuh"

#define CUDA_WIN_PART_BLOCK_SIZE 256

void ggml_cuda_op_win_part(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_win_unpart(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

