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
- Low-level, cross-platform support (x86, ARM, RISC-V, LoongArch, PowerPC, s390x, WebAssembly)
- SIMD support - AVX, AVX2, AVX512 and AMX for x86; NEON, i8mm, dotprod and MLA for ARM (including KleidiAI kernels); RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE for RISC-V
- Broad backend support - run the same graph on CPU, GPU, NPU, or in the browser
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization, plus MXFP4 and NVFP4 microscaling formats, for faster inference and reduced memory use
- Automatic differentiation
- ADAM and L-BFGS optimizers
- Zero memory allocations during runtime

## Documentation

- [The GGUF file format](docs/gguf.md)
- [Introduction to ggml](https://huggingface.co/blog/introduction-to-ggml)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)

## Contributing

- For changes to the core `ggml` library (including to the CMake build system), please open a PR in [llama.cpp](https://github.com/ggml-org/llama.cpp) - doing so will make your PR more visible, better tested, and more likely to be reviewed
