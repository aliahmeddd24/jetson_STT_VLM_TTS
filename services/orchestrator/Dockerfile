# App image (orchestrator + action server + optional Piper). CPU only, linux/arm64.
# Built on the workstation:  docker buildx build --platform linux/arm64 --target app ...
# Needs third_party/cyclonedds and third_party/unitree_sdk2_python (scripts/offline/fetch_sources.sh).
FROM python:3.10-slim-bookworm AS app
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git libglib2.0-0 iproute2 iputils-ping curl \
    && rm -rf /var/lib/apt/lists/*

COPY third_party/cyclonedds /tmp/cyclonedds
RUN cmake -S /tmp/cyclonedds -B /tmp/cyclonedds/build -DCMAKE_INSTALL_PREFIX=/opt/cyclonedds \
        -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF -DBUILD_IDLC_TESTING=OFF \
    && cmake --build /tmp/cyclonedds/build --target install -j"$(nproc)" \
    && rm -rf /tmp/cyclonedds
ENV CYCLONEDDS_HOME=/opt/cyclonedds LD_LIBRARY_PATH=/opt/cyclonedds/lib

COPY services/orchestrator/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY third_party/unitree_sdk2_python /opt/unitree_sdk2_python
# --no-deps: its opencv-python/numpy pins are already satisfied by requirements.txt
RUN pip install --no-cache-dir --no-deps -e /opt/unitree_sdk2_python \
    && python -c "import unitree_sdk2py, cyclonedds, onnxruntime, riva.client; print('imports ok')"

COPY services/orchestrator /app/services/orchestrator
COPY services/action_server /app/services/action_server
COPY services/tts_piper /app/services/tts_piper
COPY services/vlm /app/services/vlm
COPY scripts /app/scripts
COPY tests /app/tests
ENV PYTHONPATH=/app/services/orchestrator:/app/services/action_server PYTHONUNBUFFERED=1
WORKDIR /work

# Optional Piper layer (separate target so a piper-phonemize wheel problem cannot break the app image)
FROM app AS piper
RUN pip install --no-cache-dir piper-tts==1.2.0 \
    && python -c "from piper.voice import PiperVoice; print('piper ok')"
