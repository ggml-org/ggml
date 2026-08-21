# ggml

<div align="center">

<img src="https://raw.githubusercontent.com/ggml-org/media/refs/heads/master/logo/ggml-logo.jpg" width="256" height="256" alt="ggml logo" />

<b>Tensor library for machine learning</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/ggml?filter=v*)](https://github.com/ggml-org/ggml/releases)
[![CI](https://github.com/ggml-org/ggml/actions/workflows/build-cpu.yml/badge.svg)](https://github.com/ggml-org/ggml/actions/workflows/build-cpu.yml)

</div>

## Quick start

Build from source:

```bash
git clone https://github.com/ggml-org/ggml
cd ggml

mkdir build && cd build
cmake ..
cmake --build . --config Release -j 8
```

For a minimal, fully commented example (matrix multiplication), see [examples/simple](examples/simple).

## Description

The main goal of `ggml` is to be a simple, portable, and efficient tensor library for machine learning with minimal setup.

- Plain C/C++ implementation without any dependencies
- Cross-platform - x86, ARM, RISC-V, LoongArch, PowerPC, s390x, and WebAssembly
- SIMD-optimized kernels for x86, ARM, and RISC-V
- Broad backend support - CPU, GPU, NPU, and browser
- 2- to 8-bit integer quantization, plus MXFP4 and NVFP4 microscaling formats
- Zero memory allocations during runtime

## Documentation

- [The GGUF file format](docs/gguf.md)
- [Introduction to ggml](https://huggingface.co/blog/introduction-to-ggml)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)

## Contributing

- For changes to the core `ggml` library (including to the CMake build system), please open a PR in [llama.cpp](https://github.com/ggml-org/llama.cpp) - doing so will make your PR more visible, better tested, and more likely to be reviewed
