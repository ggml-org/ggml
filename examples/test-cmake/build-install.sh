#!/bin/bash

set -e

# Remove installation target directory
rm -rf install-dl install ggml-build-install-dl ggml-build-install

### Build standard installation
build_dir=ggml-build-install
install_dir=install

cmake --fresh -S ../../. -B $build_dir -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_INSTALL_PREFIX="${PWD}/${install_dir}"

cmake --build $build_dir --parallel 12
cmake --install $build_dir

### Build dynamic backend
build_dir=ggml-build-install-dl
install_dir=install-dl

# Build dynamic backend modules and install them in libexec, mirroring a package
# manager's private plugin directory. GGML_BACKEND_DIR is compiled into ggml as
# its plugin search path, and cmake --install copies the backend modules there.
cmake --fresh -S ../../. -B $build_dir -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_BACKEND_DL=ON \
  -DGGML_CPU_ALL_VARIANTS=ON \
  -DCMAKE_INSTALL_PREFIX="${PWD}/${install_dir}" \
  -DGGML_BACKEND_DIR="${PWD}/${install_dir}/libexec"

cmake --build $build_dir --parallel 12
cmake --install $build_dir
