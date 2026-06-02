#pragma once

// Adreno GPU detection helper, shared between the Android backend-selection
// logic in ggml-backend-reg.cpp and its unit test (tests/test-adreno-version.cpp).
//
// Header-only (inline) and dependency-free so the unit test can exercise the
// pure parsing logic without linking the ggml runtime or needing a GPU.
//
// Mirrors the Adreno-version policy used by qvac-fabric-llm.cpp's ggml fork so
// the speech stack (whisper / parakeet / tts) selects GPU backends the same
// way the LLM stack does: on Android, detect the GPU through Vulkan (present
// on virtually every Android GPU) and only fall back to OpenCL for Adreno,
// whose Vulkan compute path is unstable.

#include <algorithm>
#include <cctype>
#include <regex>
#include <string>

// Parse the Adreno GPU generation from a ggml device description.
//   e.g. "Adreno (TM) 830" -> 830, "Adreno 730" -> 730
// Returns:
//   > 0 : the parsed Adreno generation number
//   -1  : the description is not an Adreno GPU
//   -3  : the description is an Adreno GPU but the version failed to parse
//
// The description is lowercased before matching so "Adreno"/"ADRENO"/"adreno"
// are all recognised (a small hardening over the raw substring check in
// qvac-fabric-llm.cpp; it never produces a false negative for Adreno).
//
// Limitation (inherited from qvac-fabric-llm.cpp): the first digit run wins, so
// a non-numeric model name like "Adreno X1-85" would parse as 1. That naming is
// Snapdragon-X (Windows-on-ARM) only; Android phone Adrenos are 5xx/6xx/7xx/8xx,
// which parse correctly.
inline int ggml_adreno_version_from_description(const std::string & gpu_description) {
    std::string lowered = gpu_description;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    if (lowered.find("dreno") == std::string::npos) {
        return -1;
    }

    static const std::regex digits_regex(R"((\d+))");
    std::smatch matches;
    if (std::regex_search(lowered, matches, digits_regex) && matches.size() > 1) {
        try {
            return std::stoi(matches[1].str());
        } catch (const std::exception &) {
            return -3;
        }
    }
    return -3;
}

// Android OpenCL/Vulkan backend policy decision, factored out of
// ggml_backend_load_all_from_path() so the version thresholds are unit-testable
// without a GPU. Input is the smallest Adreno generation among the GPU devices
// (the value ggml_backend_min_adreno_version() returns: a positive generation,
// or <= 0 when no Adreno GPU is present). Mirrors qvac-fabric-llm.cpp's ggml
// fork:
//   not Adreno (<= 0) -> no OpenCL; keep Vulkan/CPU
//   Adreno > 700      -> load OpenCL (kept alongside Vulkan; the consumer picks
//                        OpenCL over Vulkan -- see transcription-whispercpp)
//   Adreno 601..700   -> CPU only: unload Vulkan and don't load OpenCL (both GPU
//                        paths are unstable on this tier)
//   Adreno 1..600     -> load OpenCL (kept alongside Vulkan), matching fabric
struct ggml_adreno_backend_policy {
    bool load_opencl;
    bool unload_vulkan;
};

inline ggml_adreno_backend_policy ggml_adreno_resolve_backend_policy(int min_adreno_version) {
    if (min_adreno_version <= 0) {
        return ggml_adreno_backend_policy{ /*load_opencl=*/ false, /*unload_vulkan=*/ false };
    }
    if (min_adreno_version > 700) {
        return ggml_adreno_backend_policy{ /*load_opencl=*/ true, /*unload_vulkan=*/ false };
    }
    if (min_adreno_version > 600) {
        return ggml_adreno_backend_policy{ /*load_opencl=*/ false, /*unload_vulkan=*/ true };
    }
    return ggml_adreno_backend_policy{ /*load_opencl=*/ true, /*unload_vulkan=*/ false };
}
