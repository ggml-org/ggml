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

    if (g_failures == 0) {
        std::printf("All Adreno-version parsing cases passed.\n");
        return 0;
    }
    std::printf("%d Adreno-version parsing case(s) failed.\n", g_failures);
    return 1;
}
