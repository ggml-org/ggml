#include "ggml-backend.h"
#include "ggml.h"
#include <cstdio>

int main(void) {
    printf("[test-cmake] Using ggml version %s\n", ggml_version());
    printf("[test-cmake] Loading all backends...\n");
    ggml_backend_load_all();
    printf("[test-cmake] Succesfully loaded all backend.\n");
    return 0;
}
