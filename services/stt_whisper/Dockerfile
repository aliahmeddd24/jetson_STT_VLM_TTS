# whisper.cpp with CUDA for sm_87. Build ON THE JETSON, offline:
#   docker build --network none -f services/stt_whisper/Dockerfile -t g1voice/whisper:r35 .
ARG BUILDER_IMAGE=g1voice/builder:r35
FROM ${BUILDER_IMAGE}
COPY third_party/whisper.cpp /src/whisper.cpp
WORKDIR /src/whisper.cpp
# libcuda is linked from the toolkit stubs at build time; the real one is mounted by
# the NVIDIA runtime at run time. VMM is not available on L4T 35 / CUDA 11.4.
RUN cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DGGML_CUDA_NO_VMM=ON \
        -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_EXAMPLES=ON -DWHISPER_SDL2=OFF \
        -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
    && cmake --build build -j4 \
    && (cp build/bin/whisper-server /usr/local/bin/ || cp build/bin/server /usr/local/bin/whisper-server) \
    && cp build/bin/whisper-cli /usr/local/bin/ 2>/dev/null || true \
    && find build -name "*.so*" -exec cp -a {} /usr/local/lib/ \; ; ldconfig
ENTRYPOINT ["whisper-server"]
CMD ["-m", "/models/whisper/ggml-base.en.bin", "--host", "127.0.0.1", "--port", "8091", "-t", "4"]
