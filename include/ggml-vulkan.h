#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_VK_NAME "Vulkan"
#define GGML_VK_MAX_DEVICES 16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_vk_init(size_t dev_num);

GGML_BACKEND_API bool ggml_backend_is_vk(ggml_backend_t backend);
GGML_BACKEND_API int  ggml_backend_vk_get_device_count(void);
GGML_BACKEND_API void ggml_backend_vk_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_vk_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_vk_buffer_type(size_t dev_num);
// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_vk_host_buffer_type(void);

// Scan cgraph for MATMUL scratch sizes and record them in a thread-local hint.
// The next ggml_backend_vk_buffer_type_alloc_buffer() for this backend inflates
// the compute allocation to cover scratch, sub-allocating all four prealloc
// buffers from the same VkDeviceMemory as the compute region.
// Call BEFORE ggml_gallocr_alloc_graph() / alloc_compute_buffer().
GGML_BACKEND_API void ggml_backend_vk_set_scratch_workspace_hint(ggml_backend_t backend,
                                                                   const ggml_cgraph * cgraph);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_vk_reg(void);

#ifdef  __cplusplus
}
#endif
