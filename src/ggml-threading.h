#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

GGML_API void ggml_critical_section_start(void);
GGML_API void ggml_critical_section_end(void);

void ggml_print_backtrace(void);
void ggml_uncaught_exception_init(void);

#ifdef __cplusplus
}
#endif
