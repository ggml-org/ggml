# ggml

<div align="center">

<b>Tensor library for machine learning</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/ggml?filter=v*)](https://github.com/ggml-org/ggml/releases)
[![CI](https://github.com/ggml-org/ggml/actions/workflows/build-cpu.yml/badge.svg)](https://github.com/ggml-org/ggml/actions/workflows/build-cpu.yml)

[GGUF file format](docs/gguf.md) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/ggml/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Abartowski1182%20OR%20author%3Ahipudding%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3A0cc4m%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)

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
