// Regression test for in-place aliasing bug in
// ggml_compute_forward_soft_max_ext_back_f32. GGML_OP_SOFT_MAX_BACK is listed
// in ggml_op_can_inplace, so the scheduler may reuse src0 (dy) or src1 (y) as
// the destination. The CPU implementation must remain correct under either
// aliasing.
//
// Compare two paths:
//   (1) ggml_graph_compute_with_ctx     — buffers allocated by the legacy
//                                          context (no aliasing)
//   (2) ggml_backend_sched_graph_compute — buffers allocated by the scheduler
//                                          (may alias)
// In one of the cases below, src0 is marked OUTPUT, which forces the allocator
// to skip src0 and fall through to src1 — exercising the previously-broken
// dst-aliases-src1 path.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct test_case {
    const char *       name;
    int64_t            ne0;
    int64_t            ne1;
    std::vector<float> dy;
    std::vector<float> y;     // softmax output (normalized per row)
    float              scale;
    bool               mark_dy_output;  // force allocator to skip src0
    bool               mark_y_output;
};

static bool run_legacy(const test_case & tc, std::vector<float> & out) {
    const size_t n_elem = (size_t) tc.ne0 * (size_t) tc.ne1;
    const size_t mem_size = 4 * (sizeof(float) * n_elem) + 256 * 1024;

    ggml_init_params params = {};
    params.mem_size   = mem_size;
    params.mem_buffer = nullptr;
    params.no_alloc   = false;

    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    ggml_tensor * t_dy = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, tc.ne0, tc.ne1);
    ggml_tensor * t_y  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, tc.ne0, tc.ne1);
    std::memcpy(t_dy->data, tc.dy.data(), n_elem * sizeof(float));
    std::memcpy(t_y->data,  tc.y.data(),  n_elem * sizeof(float));

    ggml_tensor * t_out = ggml_soft_max_ext_back(ctx, t_dy, t_y, tc.scale, 0.0f);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, t_out);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    out.assign((const float *) t_out->data, (const float *) t_out->data + n_elem);
    ggml_free(ctx);
    return true;
}

static bool run_sched(const test_case & tc, std::vector<float> & out) {
    const size_t n_elem = (size_t) tc.ne0 * (size_t) tc.ne1;

    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    if (!backend) return false;

    ggml_backend_sched_t sched = ggml_backend_sched_new(
        &backend, NULL, 1, GGML_DEFAULT_GRAPH_SIZE, false, true);

    const size_t ctx_buf_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE
                              + ggml_graph_overhead();
    std::vector<uint8_t> ctx_buf(ctx_buf_size, 0);

    ggml_init_params params = {};
    params.mem_size   = ctx_buf_size;
    params.mem_buffer = ctx_buf.data();
    params.no_alloc   = true;

    ggml_context * ctx = ggml_init(params);
    ggml_tensor * t_dy = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, tc.ne0, tc.ne1);
    ggml_tensor * t_y  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, tc.ne0, tc.ne1);
    if (tc.mark_dy_output) ggml_set_output(t_dy);
    if (tc.mark_y_output)  ggml_set_output(t_y);

    ggml_tensor * t_out = ggml_soft_max_ext_back(ctx, t_dy, t_y, tc.scale, 0.0f);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, t_out);

    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_alloc_graph(sched, gf)) return false;

    ggml_backend_tensor_set(t_dy, tc.dy.data(), 0, n_elem * sizeof(float));
    ggml_backend_tensor_set(t_y,  tc.y.data(),  0, n_elem * sizeof(float));
    ggml_backend_sched_graph_compute(sched, gf);

    out.assign(n_elem, 0.0f);
    ggml_backend_tensor_get(t_out, out.data(), 0, n_elem * sizeof(float));

    ggml_free(ctx);
    ggml_backend_sched_free(sched);
    ggml_backend_free(backend);
    return true;
}

static int run_case(const test_case & tc) {
    std::vector<float> a, b;
    if (!run_legacy(tc, a)) return 1;
    if (!run_sched (tc, b)) return 1;

    const float atol = 1e-5f, rtol = 1e-5f;
    bool ok = true;
    for (size_t i = 0; i < a.size(); i++) {
        const float d = std::fabs(a[i] - b[i]);
        if (d > atol + rtol * std::fabs(a[i])) {
            ok = false;
            std::fprintf(stderr, "[%s] idx %zu: legacy=%g sched=%g diff=%g\n",
                tc.name, i, a[i], b[i], d);
        }
    }
    std::printf("[%s] %s\n", tc.name, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

static test_case make_case(const char * name, bool mark_dy, bool mark_y,
                           int64_t cols, int64_t rows) {
    test_case tc;
    tc.name = name;
    tc.ne0 = cols;
    tc.ne1 = rows;
    tc.scale = 1.0f;
    tc.mark_dy_output = mark_dy;
    tc.mark_y_output  = mark_y;
    tc.dy.resize((size_t) cols * rows);
    tc.y.resize((size_t) cols * rows);
    for (int64_t r = 0; r < rows; r++) {
        float row_sum = 0;
        for (int64_t c = 0; c < cols; c++) {
            size_t i = (size_t) r * cols + c;
            tc.y[i]  = std::exp(0.1f * (c + 1));
            tc.dy[i] = std::sin(0.31f * (i + 1)) + 0.2f;
            row_sum += tc.y[i];
        }
        for (int64_t c = 0; c < cols; c++) {
            tc.y[(size_t) r * cols + c] /= row_sum;
        }
    }
    return tc;
}

int main(void) {
    int rc = 0;

    // Default allocator: typically picks src0 for in-place reuse — historically safe.
    rc |= run_case(make_case("default-8x4",  false, false, 8,  4));
    rc |= run_case(make_case("default-64x16", false, false, 64, 16));

    // dy marked OUTPUT: allocator skips src0 and falls through to src1.
    // This is the path that previously corrupted y mid-computation.
    rc |= run_case(make_case("dy-output-8x4",  true, false, 8,  4));
    rc |= run_case(make_case("dy-output-64x16", true, false, 64, 16));

    // y marked OUTPUT: allocator picks src0 (safe).
    rc |= run_case(make_case("y-output-8x4",  false, true,  8,  4));

    // Both outputs: allocator can't reuse either, fresh buffer (safe).
    rc |= run_case(make_case("both-output-8x4", true,  true,  8,  4));

    return rc;
}
