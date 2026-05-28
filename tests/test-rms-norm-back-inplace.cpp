// Regression test for ggml-org/ggml#1491: ggml_rms_norm_back produces wrong
// output via the backend scheduler on the CPU backend when GGML_OP_RMS_NORM_BACK
// is allocated in-place over its src0 (gradient) input.
//
// The op is listed in ggml_op_can_inplace, so the allocator may reuse a parent
// buffer (src0 = dz or src1 = x) as the destination buffer. The CPU
// implementation must remain correct under that aliasing.
//
// This test runs the same graph two ways:
//   (1) ggml_graph_compute_with_ctx     — buffers allocated by the legacy
//                                          context (no aliasing)
//   (2) ggml_backend_sched_graph_compute — buffers allocated by the scheduler
//                                          (may alias src0 with dst)
// and asserts the two results agree element-wise.

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
    const char *           name;
    int64_t                ne0;
    int64_t                ne1;
    std::vector<float>     a;   // gradient input (src0, dz)
    std::vector<float>     b;   // forward x  (src1)
    float                  eps;
};

static bool run_legacy(const test_case & tc, std::vector<float> & out) {
    const size_t n_elem = (size_t) tc.ne0 * (size_t) tc.ne1;
    const size_t mem_size = 4 * (sizeof(float) * n_elem) + 256 * 1024;

    ggml_init_params params = {};
    params.mem_size   = mem_size;
    params.mem_buffer = nullptr;
    params.no_alloc   = false;

    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::fprintf(stderr, "ggml_init failed\n");
        return false;
    }

    ggml_tensor * t_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, tc.ne0, tc.ne1);
    ggml_tensor * t_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, tc.ne0, tc.ne1);
    std::memcpy(t_a->data, tc.a.data(), n_elem * sizeof(float));
    std::memcpy(t_b->data, tc.b.data(), n_elem * sizeof(float));

    ggml_tensor * t_out = ggml_rms_norm_back(ctx, t_a, t_b, tc.eps);
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
    if (!backend) {
        std::fprintf(stderr, "backend init failed\n");
        return false;
    }

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
    ggml_tensor * t_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, tc.ne0, tc.ne1);
    ggml_tensor * t_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, tc.ne0, tc.ne1);
    ggml_tensor * t_out = ggml_rms_norm_back(ctx, t_a, t_b, tc.eps);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, t_out);

    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        std::fprintf(stderr, "ggml_backend_sched_alloc_graph failed\n");
        return false;
    }

    ggml_backend_tensor_set(t_a, tc.a.data(), 0, n_elem * sizeof(float));
    ggml_backend_tensor_set(t_b, tc.b.data(), 0, n_elem * sizeof(float));
    ggml_backend_sched_graph_compute(sched, gf);

    out.assign(n_elem, 0.0f);
    ggml_backend_tensor_get(t_out, out.data(), 0, n_elem * sizeof(float));

    ggml_free(ctx);
    ggml_backend_sched_free(sched);
    ggml_backend_free(backend);
    return true;
}

static int run_case(const test_case & tc) {
    std::vector<float> out_legacy;
    std::vector<float> out_sched;

    if (!run_legacy(tc, out_legacy)) return 1;
    if (!run_sched (tc, out_sched))  return 1;

    const float atol = 1e-5f;
    const float rtol = 1e-5f;

    bool ok = true;
    for (size_t i = 0; i < out_legacy.size(); i++) {
        const float a = out_legacy[i];
        const float b = out_sched[i];
        const float d = std::fabs(a - b);
        if (d > atol + rtol * std::fabs(a)) {
            ok = false;
            std::fprintf(stderr,
                "[%s] idx %zu: legacy=%g sched=%g diff=%g\n",
                tc.name, i, a, b, d);
        }
    }

    std::printf("[%s] %s\n", tc.name, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

int main(void) {
    int rc = 0;

    {
        // Exact reproducer from ggml-org/ggml#1491
        test_case tc;
        tc.name = "issue-1491-reporter-4x1";
        tc.ne0 = 4;
        tc.ne1 = 1;
        tc.a   = {1.0f, 0.0f, 0.0f, 0.0f};
        tc.b   = {1.0f, 0.0f, 0.0f, 0.0f};
        tc.eps = 1e-4f;
        rc |= run_case(tc);
    }

    {
        // Same shape with a different non-trivial input
        test_case tc;
        tc.name = "non-trivial-4x1";
        tc.ne0 = 4;
        tc.ne1 = 1;
        tc.a   = { 0.5f, -1.0f, 2.0f,  0.25f};
        tc.b   = {-0.3f,  0.7f, 1.5f, -0.8f };
        tc.eps = 1e-6f;
        rc |= run_case(tc);
    }

    {
        // Multi-row case to ensure per-row independence
        test_case tc;
        tc.name = "multi-row-16x4";
        tc.ne0 = 16;
        tc.ne1 = 4;
        tc.a.resize((size_t) tc.ne0 * tc.ne1);
        tc.b.resize((size_t) tc.ne0 * tc.ne1);
        for (size_t i = 0; i < tc.a.size(); i++) {
            tc.a[i] = std::sin((float) i * 0.31f);
            tc.b[i] = std::cos((float) i * 0.17f) + 0.1f;
        }
        tc.eps = 1e-5f;
        rc |= run_case(tc);
    }

    return rc;
}
