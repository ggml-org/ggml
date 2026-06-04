// Unit test for ggml_adreno_version_from_description() — the pure GPU-string
// parser behind the Android Adreno backend-selection policy in
// ggml-backend-reg.cpp (see src/ggml-adreno.h).
//
// Pure / header-only: does not link the ggml runtime or need a GPU, so it runs
// on every host/CI. The hardware-dependent half of the policy
// (ggml_backend_min_adreno_version + the OpenCL load decision) is exercised
// end-to-end on the device farm via transcription-whispercpp's GPU test.

#include "ggml-adreno.h"

#include <cstdio>
#include <string>

static int g_failures = 0;

static void expect_version(const std::string & description, int expected) {
    const int got = ggml_adreno_version_from_description(description);
    if (got != expected) {
        std::printf("FAIL: \"%s\" -> %d (expected %d)\n", description.c_str(), got, expected);
        g_failures++;
    } else {
        std::printf("ok:   \"%s\" -> %d\n", description.c_str(), got);
    }
}

static void expect_policy(int min_adreno_version, bool load_opencl, bool unload_vulkan) {
    const ggml_adreno_backend_policy got = ggml_adreno_resolve_backend_policy(min_adreno_version);
    if (got.load_opencl != load_opencl || got.unload_vulkan != unload_vulkan) {
        std::printf("FAIL: policy(%d) -> {load_opencl=%d, unload_vulkan=%d} (expected {%d, %d})\n",
                    min_adreno_version, got.load_opencl, got.unload_vulkan, load_opencl, unload_vulkan);
        g_failures++;
    } else {
        std::printf("ok:   policy(%d) -> {load_opencl=%d, unload_vulkan=%d}\n",
                    min_adreno_version, got.load_opencl, got.unload_vulkan);
    }
}

int main() {
    // Real Adreno descriptions (as reported via the Vulkan device name).
    expect_version("Adreno (TM) 830", 830);   // Samsung S25 (Snapdragon 8 Elite)
    expect_version("Adreno (TM) 750", 750);   // Snapdragon 8 Gen 3
    expect_version("Adreno (TM) 740", 740);
    expect_version("Adreno (TM) 660", 660);
    expect_version("Adreno 730", 730);        // no "(TM)" variant
    expect_version("Adreno(TM)619", 619);     // no spaces

    // Case-insensitive (hardening over the raw substring check in fabric-llm).
    expect_version("ADRENO 830", 830);
    expect_version("adreno 612", 612);

    // Combined OpenCL device description: ggml-opencl reports
    // "<CL_DEVICE_NAME> (<CL_DEVICE_VERSION>)", where CL_DEVICE_VERSION embeds the
    // gen after an "OpenCL <api>" prefix. The "3.0" API version must be skipped so
    // the real generation (not 3) is returned.
    expect_version("QUALCOMM Adreno(TM) (OpenCL 3.0 Adreno(TM) 740)", 740);
    expect_version("QUALCOMM Adreno(TM) (OpenCL 3.0 Adreno(TM) 830)", 830);
    expect_version("QUALCOMM Adreno(TM) (OpenCL 3.0 Adreno(TM) 750)", 750);
    // Bare OpenCL CL_DEVICE_NAME (no version, so no 3-digit gen) -> -3, not 3.
    expect_version("QUALCOMM Adreno(TM)", -3);

    // Non-Adreno GPUs -> -1 (not Adreno). These are the devices that must keep
    // using Vulkan / Metal, never OpenCL.
    expect_version("Mali-G715", -1);          // Pixel 9 (proven to work on Vulkan)
    expect_version("Mali-G78 MP14", -1);
    expect_version("NVIDIA GeForce RTX 5090", -1);
    expect_version("AMD Radeon (RADV RAPHAEL_MENDOCINO)", -1);
    expect_version("Apple M2", -1);
    expect_version("llvmpipe (LLVM 20.1.2, 256 bits)", -1);
    expect_version("Intel(R) Arc(TM) A770 Graphics", -1);
    expect_version("", -1);

    // "dreno" present but no parseable number -> -3 (treated as "no usable
    // Adreno version" by the caller; distinct from "not Adreno").
    expect_version("Adreno (TM)", -3);
    expect_version("Adreno", -3);

    // Backend policy {load_opencl, unload_vulkan} per Adreno generation.
    // Non-Adreno / no GPU -> no OpenCL, keep Vulkan/CPU.
    expect_policy(-2, false, false);   // null Vulkan backend
    expect_policy(-1, false, false);   // no Adreno GPU (e.g. Mali)
    // Adreno 7xx / 8xx -> load OpenCL (Vulkan kept; consumer picks OpenCL).
    expect_policy(830, true, false);
    expect_policy(750, true, false);
    expect_policy(730, true, false);
    expect_policy(701, true, false);
    // Boundary: exactly 700 is NOT > 700 -> CPU-only tier.
    expect_policy(700, false, true);
    // Adreno 1..700 -> CPU only (unload Vulkan, no OpenCL). This now includes
    // Adreno <= 600, which is treated the same as 601..700 (stricter than
    // qvac-fabric-llm.cpp, which loaded OpenCL on <= 600).
    expect_policy(660, false, true);
    expect_policy(601, false, true);
    expect_policy(600, false, true);   // <= 600 now CPU-only (was OpenCL)
    expect_policy(500, false, true);   // <= 600 now CPU-only (was OpenCL)
    expect_policy(1, false, true);     // smallest positive Adreno -> CPU only

    if (g_failures == 0) {
        std::printf("All Adreno-version parsing cases passed.\n");
        return 0;
    }
    std::printf("%d Adreno-version parsing case(s) failed.\n", g_failures);
    return 1;
}
