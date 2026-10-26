#!/usr/bin/env bash
# ============================================================================
# Run on the WORKSTATION (internet + Docker). Produces everything the Jetson needs:
#   third_party/  pinned sources (CycloneDDS, unitree_sdk2(_python), whisper.cpp, llama.cpp, G1 URDF)
#   models/       Silero VAD, VILA1.5-3B, Piper voice, whisper ggml, (optional) Qwen2.5-VL GGUF, Riva RMIRs
#   offline-bundle/images/*.tar.gz   arm64 images: vlm, app, piper, builder (+ Riva if NGC key)
# Then copy the WHOLE repo directory to the Jetson (see the command printed at the end).
#
# Options (env): SKIP_IMAGES=1 SKIP_MODELS=1 WITH_LLAMACPP=1 (default 1) WITH_RIVA=1 (default 1)
#   NGC_API_KEY=...  needed for Riva (never stored in the repo)
#   *_REF variables pin source revisions (defaults below; resolved SHAs are recorded).
# Host changes on the WORKSTATION only: QEMU binfmt (arm64 emulation) if missing, a .venv-ws/.
# ============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
[ -f .env ] || cp .env.example .env
set -a; source .env; set +a
B=offline-bundle; mkdir -p "$B/images" models third_party logs
MAN="$B/BUNDLE_MANIFEST.md"
WITH_LLAMACPP="${WITH_LLAMACPP:-1}"; WITH_RIVA="${WITH_RIVA:-1}"
CYCLONEDDS_REF="${CYCLONEDDS_REF:-0.10.2}"
UNITREE_PY_REF="${UNITREE_PY_REF:-master}"
UNITREE_CPP_REF="${UNITREE_CPP_REF:-main}"
UNITREE_ROS_REF="${UNITREE_ROS_REF:-master}"
WHISPER_REF="${WHISPER_REF:-master}"
LLAMA_REF="${LLAMA_REF:-master}"
SILERO_REF="${SILERO_REF:-v5.1.2}"
NANO_LLM_CANDIDATES="${NANO_LLM_IMAGE} dustynv/nano_llm:24.5-r35.4.1 dustynv/nano_llm:24.4.1-r35.4.1 dustynv/nano_llm:r35.4.1"
FAILED=()

step() { printf '\n\033[1m=== %s\033[0m\n' "$*"; }
note() { echo "- $*" | tee -a "$MAN"; }
fail() { FAILED+=("$*"); echo "!! FAILED: $*" | tee -a "$MAN"; }
need() { command -v "$1" >/dev/null || { echo "missing tool: $1"; exit 1; }; }

echo "# Offline bundle manifest ($(date -Is), built on $(hostname))" > "$MAN"
need docker; need git; need python3; need curl; need gzip

step "0. arm64 emulation + private python tools (workstation only)"
if ! docker run --rm --platform linux/arm64 alpine:3.20 uname -m 2>/dev/null | grep -q aarch64; then
  echo "Installing QEMU binfmt handlers on this workstation (docker run --privileged tonistiigi/binfmt)."
  docker run --privileged --rm tonistiigi/binfmt --install arm64
fi
python3 -m venv .venv-ws
.venv-ws/bin/pip install -q --upgrade pip "huggingface_hub[cli]>=0.24,<1.0"
HF=".venv-ws/bin/huggingface-cli"

clone() {  # name url ref [depth-ok]
  local d="third_party/$1"
  if [ ! -d "$d/.git" ]; then git clone --filter=blob:none "$2" "$d"; fi
  git -C "$d" fetch --tags -q origin || true
  git -C "$d" checkout -q "$3" && git -C "$d" pull -q --ff-only 2>/dev/null || true
  git -C "$d" submodule update --init --recursive -q || true
  note "source $1 @ $3 = $(git -C "$d" rev-parse HEAD)"
}

step "1. sources"
clone cyclonedds           https://github.com/eclipse-cyclonedds/cyclonedds.git        "$CYCLONEDDS_REF"
clone unitree_sdk2_python  https://github.com/unitreerobotics/unitree_sdk2_python.git  "$UNITREE_PY_REF"
clone unitree_sdk2         https://github.com/unitreerobotics/unitree_sdk2.git         "$UNITREE_CPP_REF"
clone whisper.cpp          https://github.com/ggml-org/whisper.cpp.git                 "$WHISPER_REF"
clone llama.cpp            https://github.com/ggml-org/llama.cpp.git                   "$LLAMA_REF"
clone reachy-mini-jetson-assistant https://github.com/NVIDIA-AI-IOT/reachy-mini-jetson-assistant.git main || fail "reachy reference clone"
if [ ! -d third_party/unitree_ros/.git ]; then
  git clone --filter=blob:none --sparse https://github.com/unitreerobotics/unitree_ros.git third_party/unitree_ros
fi
git -C third_party/unitree_ros sparse-checkout set robots/g1_description
git -C third_party/unitree_ros checkout -q "$UNITREE_ROS_REF" || true
rm -rf third_party/g1_description && cp -a third_party/unitree_ros/robots/g1_description third_party/g1_description
note "unitree_ros g1_description @ $(git -C third_party/unitree_ros rev-parse HEAD); URDFs: $(cd third_party/g1_description && ls *.urdf 2>/dev/null | tr '\n' ' ')"
note "FSM-id reference: grep -rn 'FSM\|fsm' third_party/unitree_sdk2/include/unitree/robot/g1/loco/ (confirm on robot before use)"

if [ "${SKIP_MODELS:-0}" != 1 ]; then
  step "2. models"
  mkdir -p models/piper models/whisper models/llamacpp
  curl -fL "https://github.com/snakers4/silero-vad/raw/${SILERO_REF}/src/silero_vad/data/silero_vad.onnx" \
       -o models/silero_vad.onnx && note "silero_vad.onnx @ ${SILERO_REF}" || fail "silero vad download"
  $HF download Efficient-Large-Model/VILA1.5-3b --local-dir models/VILA1.5-3b \
      && note "VILA1.5-3b HF snapshot ($(du -sh models/VILA1.5-3b | cut -f1))" || fail "VILA1.5-3b download"
  $HF download rhasspy/piper-voices en/en_US/lessac/medium/en_US-lessac-medium.onnx \
      en/en_US/lessac/medium/en_US-lessac-medium.onnx.json --local-dir models/.piper_raw \
      && cp models/.piper_raw/en/en_US/lessac/medium/* models/piper/ && note "piper en_US-lessac-medium" || fail "piper voice"
  $HF download ggerganov/whisper.cpp ggml-base.en.bin --local-dir models/whisper \
      && note "whisper ggml-base.en.bin" || fail "whisper model"
  if [ "$WITH_LLAMACPP" = 1 ]; then
    $HF download ggml-org/Qwen2.5-VL-3B-Instruct-GGUF --include "*Q4_K_M*.gguf" "mmproj*" --local-dir models/.llamacpp_raw \
      || fail "Qwen2.5-VL GGUF"
    M=$(ls models/.llamacpp_raw/*Q4_K_M*.gguf 2>/dev/null | grep -v mmproj | head -1 || true)
    P=$(ls models/.llamacpp_raw/mmproj*Q8_0*.gguf models/.llamacpp_raw/mmproj*f16*.gguf 2>/dev/null | head -1 || true)
    if [ -n "$M" ] && [ -n "$P" ]; then cp "$M" models/llamacpp/model.gguf; cp "$P" models/llamacpp/mmproj.gguf
      note "llama.cpp VLM: $(basename "$M") + $(basename "$P")"; else fail "no GGUF/mmproj pair found"; fi
  fi
fi

if [ "$WITH_RIVA" = 1 ]; then
  step "3. Riva embedded 2.14.0 (needs NGC_API_KEY and the ngc CLI)"
  if [ -n "${NGC_API_KEY:-}" ] && command -v ngc >/dev/null; then
    [ -d third_party/riva_quickstart_arm64_v2.14.0 ] || \
      ngc registry resource download-version "nvidia/riva/riva_quickstart_arm64:2.14.0" --dest third_party \
      || fail "riva quickstart"
    echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin >/dev/null
    docker pull --platform linux/arm64 "$RIVA_IMAGE" && note "riva image $RIVA_IMAGE" || fail "riva image pull"
    bash scripts/offline/riva_models.sh || fail "riva RMIR download (fallback: temporary connection sharing, docs/RUNBOOK.md)"
  else
    fail "Riva skipped: set NGC_API_KEY and install the ngc CLI, or plan for the Piper/whisper fallbacks"
  fi
fi

if [ "${SKIP_IMAGES:-0}" != 1 ]; then
  step "4. images (linux/arm64)"
  CHOSEN=""
  for img in $NANO_LLM_CANDIDATES; do
    if docker pull --platform linux/arm64 "$img"; then CHOSEN="$img"; break; fi
  done
  if [ -n "$CHOSEN" ]; then
    note "nano_llm base: $CHOSEN"; sed -i "s#^NANO_LLM_IMAGE=.*#NANO_LLM_IMAGE=$CHOSEN#" .env
    docker build --platform linux/arm64 --build-arg NANO_LLM_IMAGE="$CHOSEN" -t g1voice/vlm:r35 services/vlm || fail "vlm image"
  else fail "no dustynv/nano_llm r35 tag could be pulled (tried: $NANO_LLM_CANDIDATES)"; fi
  docker build --platform linux/arm64 --target app -t g1voice/app:latest -f services/orchestrator/Dockerfile . || fail "app image"
  docker build --platform linux/arm64 --target piper -t g1voice/piper:latest -f services/orchestrator/Dockerfile . || fail "piper image"
  docker pull --platform linux/arm64 "$BUILDER_BASE" || fail "builder base pull"
  docker build --platform linux/arm64 --build-arg BUILDER_BASE="$BUILDER_BASE" -t g1voice/builder:r35 services/builder || fail "builder image"

  step "5. self-test inside the arm64 app image (QEMU)"
  docker run --rm --platform linux/arm64 g1voice/app:latest python -m unittest discover -s /app/tests \
    && note "unit tests passed inside arm64 app image" || fail "tests inside app image"

  step "6. save images"
  for img in g1voice/vlm:r35 g1voice/app:latest g1voice/piper:latest g1voice/builder:r35 "$RIVA_IMAGE"; do
    f="$B/images/$(echo "$img" | tr '/:' '__').tar.gz"
    if docker image inspect "$img" >/dev/null 2>&1; then
      [ -f "$f" ] || docker save "$img" | gzip -1 > "$f"
      note "image $img -> $f ($(du -h "$f" | cut -f1))"
    fi
  done
fi

step "7. checksums"
( cd "$B" && sha256sum images/*.tar.gz > SHA256SUMS 2>/dev/null || true )
find models -type f -not -path "*/.*" -size +1M -print0 | xargs -0 -r sha256sum > "$B/MODELS.sha256" || true
du -sh "$B" models third_party 2>/dev/null | tee -a "$MAN"

echo
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "Completed WITH FAILURES (see $MAN):"; printf '  - %s\n' "${FAILED[@]}"
else echo "Bundle complete."; fi
cat <<MSG

Next: copy the repo to the Jetson over Ethernet (example: PC2 at 192.168.123.164):
  rsync -a --info=progress2 --exclude .venv-ws "$ROOT/" unitree@192.168.123.164:~/g1-voice/
Then on the Jetson:  cd ~/g1-voice && make install && make discover
MSG
