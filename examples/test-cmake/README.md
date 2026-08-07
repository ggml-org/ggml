## cmake-test

This directory can be built as a separate project to test/verify a ggml
installation.

### Usage
The following will configure, build, and install ggml using two different
configurations. One will use dynamically linked backends (GGML_BACKEND_DL=ON)
and one without.

Configuring, build, and install ggml:
```console
./build-install.sh
```

Build this project twice using the installations created above:
```console
./build.sh
```

_wip_
