# llama.cpp llama-server with CUDA for sm_87 (VLM stretch fallback). Build ON THE JETSON:
#   docker build --network none -f services/llamacpp/Dockerfile -t g1voice/llamacpp:r35 .
ARG BUILDER_IMAGE=g1voice/builder:r35
FROM ${BUILDER_IMAGE}
COPY third_party/llama.cpp /src/llama.cpp
WORKDIR /src/llama.cpp
RUN cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DGGML_CUDA_NO_VMM=ON \
        -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=ON \
        -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
    && cmake --build build -j4 --target llama-server \
    && cp build/bin/llama-server /usr/local/bin/ \
    && find build -name "*.so*" -exec cp -a {} /usr/local/lib/ \; ; ldconfig
ENTRYPOINT ["llama-server"]
CMD ["-m", "/models/llamacpp/model.gguf", "--mmproj", "/models/llamacpp/mmproj.gguf", \
     "--host", "127.0.0.1", "--port", "8080", "-ngl", "99", "-c", "4096", "--parallel", "1"]
