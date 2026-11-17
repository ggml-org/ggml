# ggml

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

<div align="center">

<img src="https://raw.githubusercontent.com/ggml-org/media/refs/heads/master/logo/ggml-logo.jpg" width="256" height="256" alt="ggml logo" />

<b>专为机器学习打造的高性能张量库</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/ggml?filter=v*)](https://github.com/ggml-org/ggml/releases)
[![CI](https://github.com/ggml-org/ggml/actions/workflows/build-cpu.yml/badge.svg)](https://github.com/ggml-org/ggml/actions/workflows/build-cpu.yml)

</div>

## 快速上手

从源码编译构建：

```bash
git clone https://github.com/ggml-org/ggml
cd ggml

mkdir build && cd build
cmake ..
cmake --build . --config Release -j 8
```

关于极简且带有完整注释的代码范例（矩阵乘法），请参阅 [examples/simple](examples/simple)。

## 项目简介

`ggml` 的核心目标是以极简的配置，成为一个轻量、便携且高效的机器学习张量库。

- 无任何外部依赖的原生纯 C/C++ 实现
- 跨平台支持 - 支持 x86、ARM、RISC-V、LoongArch（龙芯）、PowerPC、s390x 与 WebAssembly
- 针对 x86、ARM 与 RISC-V 架构深度优化的 SIMD 算子内核
- 广泛的硬件后端支持 - CPU、GPU、NPU 及浏览器端
- 支持 2 至 8 位整型量化，以及 MXFP4 和 NVFP4 显微缩放（microscaling）格式
- 运行时零内存分配（Zero memory allocations during runtime）

## 文档导航

- [GGUF 文件格式规范 (The GGUF file format)](docs/gguf.md)
- [ggml 入门指南 (Introduction to ggml)](https://huggingface.co/blog/introduction-to-ggml)
- [GGML 技巧与实用指南 (GGML tips & tricks)](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)

## 参与贡献

- 针对核心 `ggml` 库的变更（包括 CMake 构建系统），请前往 [llama.cpp](https://github.com/ggml-org/llama.cpp) 提交 PR - 这样能让改动获得更高的曝光度、更全面的自动化测试，以及更及时的维护者审查。

---

> **翻译说明（Translation Disclaimer）**
> 本文档由社区开发者维护，旨在降低中文开发者的阅读门槛。最后同步日期：2026年11月17日。若译文与英文原版存在任何出入或滞后，请以 [英文原版 (README.md)](README.md) 为准。欢迎通过 PR 协助改进译文！
