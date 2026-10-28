#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static void write_u32(FILE * f, uint32_t v) { fwrite(&v, sizeof(v), 1, f); }
static void write_u64(FILE * f, uint64_t v) { fwrite(&v, sizeof(v), 1, f); }

// Writes a minimal synthetic GGUF v3 file with 1 tensor of given dimensions
static std::string write_minimal_gguf(const char * path, const std::vector<uint64_t> & ne) {
    FILE * f = fopen(path, "wb");
    GGML_ASSERT(f != nullptr);

    // Header: magic ("GGUF"), version (3), n_tensors (1), n_kv (0)
    fwrite("GGUF", 1, 4, f);
    write_u32(f, 3);
    write_u64(f, 1);
    write_u64(f, 0);

    // Tensor info: name ("test_tensor"), n_dims, ne[...], type (GGML_TYPE_F32 = 0), offset (0)
    const char * name = "test_tensor";
    write_u64(f, strlen(name));
    fwrite(name, 1, strlen(name), f);

    write_u32(f, (uint32_t)ne.size());
    for (uint64_t d : ne) {
        write_u64(f, d);
    }

    write_u32(f, (uint32_t)GGML_TYPE_F32);
    write_u64(f, 0); // offset

    // Pad to 32-byte alignment
    long pos = ftell(f);
    long pad = (32 - (pos % 32)) % 32;
    for (long i = 0; i < pad; ++i) {
        fputc(0, f);
    }

    fclose(f);
    return std::string(path);
}

static bool try_load_gguf_shape(const std::vector<uint64_t> & ne) {
    const char * tmp_path = "test_gguf_shape_tmp.gguf";
    write_minimal_gguf(tmp_path, ne);

    struct ggml_context * ctx = nullptr;
    struct gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &ctx,
    };

    struct gguf_context * gctx = gguf_init_from_file(tmp_path, params);
    remove(tmp_path);

    bool loaded = (gctx != nullptr);
    if (gctx) {
        gguf_free(gctx);
    }
    if (ctx) {
        ggml_free(ctx);
    }
    return loaded;
}

int main() {
    // 1. Valid zero-dimension shape should load cleanly
    GGML_ASSERT(try_load_gguf_shape({1, 0, 3, 6}) == true);

    // 2. Direct overflow shape should fail cleanly (return NULL, not abort)
    GGML_ASSERT(try_load_gguf_shape({1000, (uint64_t)1 << 62, 1, 1}) == false);

    // 3. Zero-dimension followed by huge dimension (Issue #1668 case 1) should fail cleanly, not abort
    GGML_ASSERT(try_load_gguf_shape({1000, 0, (uint64_t)1 << 62, 1}) == false);

    // 4. Boundary equality INT64_MAX / ne[i] == current_nelements (Issue #1668 case 2) should fail cleanly, not abort
    GGML_ASSERT(try_load_gguf_shape({2, (uint64_t)(INT64_MAX / 2), 1, 1}) == false);

    printf("test-gguf: all tensor shape overflow checks passed!\n");
    return 0;
}
