# CUDA 11.4 build environment for llama.cpp / whisper.cpp on L4T r35.
# Built on the workstation under QEMU (no CUDA compilation happens here), then the
# CUDA compile itself runs on the Jetson with `docker build --network none`.
ARG BUILDER_BASE=nvcr.io/nvidia/l4t-jetpack:r35.4.1
FROM ${BUILDER_BASE}
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc-10 g++-10 git ninja-build python3-pip ccache pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir "cmake>=3.27,<3.31"
# CUDA 11.4 supports gcc <= 10; Ubuntu 20.04's cmake (3.16) is too old for current ggml.
ENV CC=gcc-10 CXX=g++-10 CUDAHOSTCXX=g++-10 PATH=/usr/local/cuda/bin:${PATH} \
    LIBRARY_PATH=/usr/local/cuda/lib64/stubs
RUN nvcc --version && cmake --version
