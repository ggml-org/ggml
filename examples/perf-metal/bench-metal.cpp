// bench-metal.cpp — comprehensive Metal backend benchmark
//
// Tests the key operations that dominate LLM inference on Apple Silicon:
//   1. MUL_MAT (f16, q4_0, q8_0 weights × f32 activations) — various sizes
//   2. Flash Attention (f16 KV)
//   3. RMS Norm
//   4. Softmax
//   5. RoPE
//   6. Element-wise ADD
//   7. Mixed graph simulating a transformer decode step
//
// Usage:
//   cmake --build build --config Release -j && ./build/bin/bench-metal [n_iter]
//
// Default: 100 iterations per test, ~2-3 minutes total on M-series.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-metal.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

struct bench_result {
    const char * name;
    double       time_ms;     // mean per iteration
    double       gflops;      // if applicable, else 0
    double       gb_s;        // memory bandwidth if applicable, else 0
};

static std::vector<bench_result> results;

static void report(const char * name, int n_iter, int64_t elapsed_us,
                   double flops_per_iter = 0.0, double bytes_per_iter = 0.0) {
    double ms = (double)elapsed_us / 1000.0 / n_iter;
    double gflops = (flops_per_iter > 0) ? (flops_per_iter / (ms * 1e6)) : 0.0;
    double gb_s   = (bytes_per_iter > 0) ? (bytes_per_iter / (ms * 1e6)) : 0.0;

    printf("  %-45s %8.3f ms", name, ms);
    if (gflops > 0) printf("  %8.2f GFLOPS", gflops);
    if (gb_s   > 0) printf("  %8.2f GB/s", gb_s);
    printf("\n");

    results.push_back({name, ms, gflops, gb_s});
}

static void fill_random(void * data, size_t nbytes) {
    // fast PRNG fill
    uint32_t * p = (uint32_t *)data;
    uint32_t seed = 42;
    for (size_t i = 0; i < nbytes / 4; i++) {
        seed = seed * 1664525u + 1013904223u;
        p[i] = seed;
    }
}

static void fill_random_f32(float * data, size_t n) {
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < n; i++) {
        data[i] = dist(rng);
    }
}

// ---------------------------------------------------------------------------
// Single-op benchmark runner
// ---------------------------------------------------------------------------

typedef void (*build_graph_fn)(struct ggml_context * ctx, struct ggml_cgraph * gf, void * ud);

struct bench_config {
    const char    * name;
    build_graph_fn  build;
    void          * ud;
    double          flops_per_iter;
    double          bytes_per_iter;
};

// Runs a single benchmark: allocates, warms up, times, frees.
static void run_bench(ggml_backend_t backend, const bench_config & cfg,
                      int n_iter, struct ggml_context * ctx_input,
                      ggml_backend_buffer_t buf_input) {
    // Build graph
    const size_t ctx_size = 64 * ggml_tensor_overhead() + ggml_graph_overhead_custom(64, false);
    struct ggml_init_params params = { ctx_size, NULL, true };
    struct ggml_context * ctx_gf = ggml_init(params);

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx_gf, 64, false);
    cfg.build(ctx_gf, gf, cfg.ud);

    // Allocate compute buffers
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(allocr, gf);

    // Warm-up (3 iterations)
    for (int i = 0; i < 3; i++) {
        ggml_backend_graph_compute(backend, gf);
    }
    ggml_backend_synchronize(backend);

    // Timed run
    int64_t t0 = ggml_time_us();
    for (int i = 0; i < n_iter; i++) {
        ggml_backend_graph_compute(backend, gf);
    }
    ggml_backend_synchronize(backend);
    int64_t t1 = ggml_time_us();

    report(cfg.name, n_iter, t1 - t0, cfg.flops_per_iter, cfg.bytes_per_iter);

    ggml_gallocr_free(allocr);
    ggml_free(ctx_gf);
}

// ============================================================================
// Benchmark definitions
// ============================================================================

// --- MUL_MAT ---------------------------------------------------------------

struct mm_params {
    struct ggml_tensor * weight;  // pre-allocated on device
    int M, N, K;
};

static void build_mul_mat(struct ggml_context * ctx, struct ggml_cgraph * gf, void * ud) {
    auto * p = (mm_params *)ud;
    // activation: K x M (f32)
    struct ggml_tensor * act = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, p->K, p->M);
    struct ggml_tensor * res = ggml_mul_mat(ctx, p->weight, act);
    ggml_build_forward_expand(gf, res);
}

// --- Flash Attention -------------------------------------------------------

struct fa_params {
    int n_head;
    int n_kv;
    int d_head;
    int n_batch;
};

static void build_flash_attn(struct ggml_context * ctx, struct ggml_cgraph * gf, void * ud) {
    auto * p = (fa_params *)ud;
    // Q: [d_head, n_batch, n_head, 1]  — (ne0=d_head, ne1=seq_len, ne2=n_head, ne3=1)
    struct ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, p->d_head, p->n_batch, p->n_head, 1);
    // K: [d_head, n_kv, n_head, 1]
    struct ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, p->d_head, p->n_kv, p->n_head, 1);
    // V: [d_head, n_kv, n_head, 1]
    struct ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, p->d_head, p->n_kv, p->n_head, 1);

    float scale = 1.0f / sqrtf((float)p->d_head);
    struct ggml_tensor * res = ggml_flash_attn_ext(ctx, q, k, v, NULL, scale, 0.0f, 0.0f);
    ggml_build_forward_expand(gf, res);
}

// --- RMS Norm --------------------------------------------------------------

struct norm_params {
    int n_embd;
    int n_batch;
};

static void build_rms_norm(struct ggml_context * ctx, struct ggml_cgraph * gf, void * ud) {
    auto * p = (norm_params *)ud;
    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, p->n_embd, p->n_batch);
    struct ggml_tensor * res = ggml_rms_norm(ctx, x, 1e-5f);
    ggml_build_forward_expand(gf, res);
}

// --- Softmax ---------------------------------------------------------------

struct softmax_params {
    int n_cols;
    int n_rows;
};

static void build_softmax(struct ggml_context * ctx, struct ggml_cgraph * gf, void * ud) {
    auto * p = (softmax_params *)ud;
    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, p->n_cols, p->n_rows);
    struct ggml_tensor * res = ggml_soft_max(ctx, x);
    ggml_build_forward_expand(gf, res);
}

// --- RoPE ------------------------------------------------------------------

struct rope_params {
    int n_embd;
    int n_head;
    int n_batch;
};

static void build_rope(struct ggml_context * ctx, struct ggml_cgraph * gf, void * ud) {
    auto * p = (rope_params *)ud;
    int d_head = p->n_embd / p->n_head;
    // x: [n_embd, n_head, n_batch]  — but RoPE sees [d_head, n_head, n_batch]
    struct ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_head, p->n_head, p->n_batch);
    // positions: int32 [n_batch]
    struct ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, p->n_batch);
    struct ggml_tensor * res = ggml_rope(ctx, x, pos, d_head, 0);
    ggml_build_forward_expand(gf, res);
}

// --- Element-wise ADD ------------------------------------------------------

struct add_params {
    int ne0, ne1;
};

static void build_add(struct ggml_context * ctx, struct ggml_cgraph * gf, void * ud) {
    auto * p = (add_params *)ud;
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, p->ne0, p->ne1);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, p->ne0, p->ne1);
    struct ggml_tensor * res = ggml_add(ctx, a, b);
    ggml_build_forward_expand(gf, res);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char ** argv) {
    int n_iter = 100;
    if (argc > 1) {
        n_iter = atoi(argv[1]);
        if (n_iter < 1) n_iter = 100;
    }

    printf("=== ggml Metal Benchmark ===\n");
    printf("Iterations per test: %d\n\n", n_iter);

    // Init Metal backend
    ggml_backend_t backend = ggml_backend_metal_init();
    if (!backend) {
        fprintf(stderr, "ERROR: ggml_backend_metal_init() failed\n");
        return 1;
    }
    printf("Backend: %s\n\n", ggml_backend_name(backend));

    // ------------------------------------------------------------------
    // 1. MUL_MAT benchmarks (the dominant op in LLM inference)
    // ------------------------------------------------------------------
    printf("--- MUL_MAT (weight × activation) ---\n");
    printf("  Format: [N,K] × [K,M] -> [N,M]  (weight × act)\n\n");

    // Typical LLM shapes: K=4096, N=4096, M=1 (decode), M=512 (prefill)
    struct {
        const char * label;
        enum ggml_type wtype;
        int K, N, M;
    } mm_tests[] = {
        // Decode (M=1): bandwidth-bound
        { "f16  4096x4096 × 4096x1   (decode)",  GGML_TYPE_F16,  4096, 4096,    1 },
        { "q4_0 4096x4096 × 4096x1   (decode)",  GGML_TYPE_Q4_0, 4096, 4096,    1 },
        { "q8_0 4096x4096 × 4096x1   (decode)",  GGML_TYPE_Q8_0, 4096, 4096,    1 },
        // Small batch (M=8)
        { "f16  4096x4096 × 4096x8",             GGML_TYPE_F16,  4096, 4096,    8 },
        { "q4_0 4096x4096 × 4096x8",             GGML_TYPE_Q4_0, 4096, 4096,    8 },
        { "q8_0 4096x4096 × 4096x8",             GGML_TYPE_Q8_0, 4096, 4096,    8 },
        // Prefill (M=512): compute-bound
        { "f16  4096x4096 × 4096x512 (prefill)", GGML_TYPE_F16,  4096, 4096,  512 },
        { "q4_0 4096x4096 × 4096x512 (prefill)", GGML_TYPE_Q4_0, 4096, 4096,  512 },
        { "q8_0 4096x4096 × 4096x512 (prefill)", GGML_TYPE_Q8_0, 4096, 4096,  512 },
        // Large (7B-like hidden → FFN)
        { "q4_0 4096x11008 × 4096x1  (FFN dec)", GGML_TYPE_Q4_0, 4096, 11008,   1 },
        { "q4_0 4096x11008 × 4096x512 (FFN pre)",GGML_TYPE_Q4_0, 4096, 11008, 512 },
    };

    for (auto & t : mm_tests) {
        // Allocate weight on device
        size_t weight_ctx_size = ggml_tensor_overhead() + 16;
        struct ggml_init_params wp = { weight_ctx_size, NULL, true };
        struct ggml_context * wctx = ggml_init(wp);
        struct ggml_tensor * w = ggml_new_tensor_2d(wctx, t.wtype, t.K, t.N);
        ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);

        // Fill weight with random data
        std::vector<uint8_t> wdata(ggml_nbytes(w));
        fill_random(wdata.data(), wdata.size());
        ggml_backend_tensor_set(w, wdata.data(), 0, wdata.size());

        mm_params mp = { w, t.M, t.N, t.K };

        // FLOPS: 2*M*N*K per matmul
        double flops = 2.0 * t.M * t.N * t.K;
        // Bytes: weight read + activation read + result write
        double bytes = (double)ggml_nbytes(w) + (double)t.K * t.M * sizeof(float) + (double)t.N * t.M * sizeof(float);

        bench_config cfg = { t.label, build_mul_mat, &mp, flops, bytes };
        run_bench(backend, cfg, n_iter, NULL, NULL);

        ggml_backend_buffer_free(wbuf);
        ggml_free(wctx);
    }

    // ------------------------------------------------------------------
    // 2. Flash Attention
    // ------------------------------------------------------------------
    printf("\n--- Flash Attention (f16 KV) ---\n\n");

    struct {
        const char * label;
        int n_head, n_kv, d_head, n_batch;
    } fa_tests[] = {
        // d=128 (LLaMA-style)
        { "FA  heads=32 d=128 kv=512   batch=1",   32,  512,  128,   1 },
        { "FA  heads=32 d=128 kv=2048  batch=1",   32, 2048,  128,   1 },
        { "FA  heads=32 d=128 kv=8192  batch=1",   32, 8192,  128,   1 },
        { "FA  heads=32 d=128 kv=2048  batch=32",  32, 2048,  128,  32 },
        { "FA  heads=32 d=128 kv=512   batch=512", 32,  512,  128, 512 },
        // d=64
        { "FA  heads=32 d=64  kv=2048  batch=1",   32, 2048,   64,   1 },
        { "FA  heads=32 d=64  kv=2048  batch=32",  32, 2048,   64,  32 },
        // d=256 (large-head models)
        { "FA  heads=16 d=256 kv=2048  batch=1",   16, 2048,  256,   1 },
        { "FA  heads=16 d=256 kv=2048  batch=32",  16, 2048,  256,  32 },
    };

    for (auto & t : fa_tests) {
        fa_params fp = { t.n_head, t.n_kv, t.d_head, t.n_batch };
        // FLOPS approximation: 2 * n_batch * n_head * n_kv * d_head (QK) + same for V
        double flops = 4.0 * t.n_batch * t.n_head * t.n_kv * t.d_head;
        bench_config cfg = { t.label, build_flash_attn, &fp, flops, 0 };
        run_bench(backend, cfg, n_iter, NULL, NULL);
    }

    // ------------------------------------------------------------------
    // 3. RMS Norm
    // ------------------------------------------------------------------
    printf("\n--- RMS Norm ---\n\n");

    struct {
        const char * label;
        int n_embd, n_batch;
    } norm_tests[] = {
        { "RMS Norm  4096  batch=1",     4096,    1 },
        { "RMS Norm  4096  batch=512",   4096,  512 },
        { "RMS Norm  8192  batch=1",     8192,    1 },
    };

    for (auto & t : norm_tests) {
        norm_params np = { t.n_embd, t.n_batch };
        double bytes = 2.0 * t.n_embd * t.n_batch * sizeof(float); // read + write
        bench_config cfg = { t.label, build_rms_norm, &np, 0, bytes };
        run_bench(backend, cfg, n_iter, NULL, NULL);
    }

    // ------------------------------------------------------------------
    // 4. Softmax
    // ------------------------------------------------------------------
    printf("\n--- Softmax ---\n\n");

    struct {
        const char * label;
        int n_cols, n_rows;
    } sm_tests[] = {
        { "Softmax  2048  rows=32",     2048,   32 },
        { "Softmax  8192  rows=32",     8192,   32 },
        { "Softmax  2048  rows=16384",  2048, 16384 },
    };

    for (auto & t : sm_tests) {
        softmax_params sp = { t.n_cols, t.n_rows };
        double bytes = 2.0 * t.n_cols * t.n_rows * sizeof(float);
        bench_config cfg = { t.label, build_softmax, &sp, 0, bytes };
        run_bench(backend, cfg, n_iter, NULL, NULL);
    }

    // ------------------------------------------------------------------
    // 5. RoPE
    // ------------------------------------------------------------------
    printf("\n--- RoPE ---\n\n");

    struct {
        const char * label;
        int n_embd, n_head, n_batch;
    } rope_tests[] = {
        { "RoPE  d=128 heads=32 batch=1",   4096, 32,   1 },
        { "RoPE  d=128 heads=32 batch=512", 4096, 32, 512 },
    };

    for (auto & t : rope_tests) {
        rope_params rp = { t.n_embd, t.n_head, t.n_batch };
        int d_head = t.n_embd / t.n_head;
        double bytes = 2.0 * d_head * t.n_head * t.n_batch * sizeof(float);
        bench_config cfg = { t.label, build_rope, &rp, 0, bytes };
        run_bench(backend, cfg, n_iter, NULL, NULL);
    }

    // ------------------------------------------------------------------
    // 6. Element-wise ADD
    // ------------------------------------------------------------------
    printf("\n--- Element-wise ADD ---\n\n");

    struct {
        const char * label;
        int ne0, ne1;
    } add_tests[] = {
        { "ADD  4096x4096",   4096, 4096 },
        { "ADD  4096x1",      4096,    1 },
        { "ADD  8192x2048",   8192, 2048 },
    };

    for (auto & t : add_tests) {
        add_params ap = { t.ne0, t.ne1 };
        double bytes = 3.0 * t.ne0 * t.ne1 * sizeof(float); // 2 reads + 1 write
        bench_config cfg = { t.label, build_add, &ap, 0, bytes };
        run_bench(backend, cfg, n_iter, NULL, NULL);
    }

    // ------------------------------------------------------------------
    // Summary
    // ------------------------------------------------------------------
    printf("\n=== Summary ===\n");
    printf("  %-45s %8s  %8s  %8s\n", "Test", "ms", "GFLOPS", "GB/s");
    printf("  %-45s %8s  %8s  %8s\n", "----", "--", "------", "----");
    for (auto & r : results) {
        printf("  %-45s %8.3f", r.name, r.time_ms);
        if (r.gflops > 0) printf("  %8.2f", r.gflops);
        else               printf("  %8s", "-");
        if (r.gb_s > 0)   printf("  %8.2f", r.gb_s);
        else               printf("  %8s", "-");
        printf("\n");
    }

    ggml_backend_free(backend);
    printf("\nDone.\n");
    return 0;
}
