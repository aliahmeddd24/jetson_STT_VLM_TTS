#!/usr/bin/env bash
# Riva embedded (<= 2.14.0, the last line for JetPack 5) helper.
#
#   riva.sh prepare        copy the quickstart into third_party/riva_qs and patch it:
#                          ASR+TTS en-US only, Orin platform, and `--runtime nvidia` on every
#                          `docker run`, so Docker's default runtime does NOT have to change.
#   riva.sh deploy-offline build TensorRT engines from pre-downloaded RMIRs (no internet)
#   riva.sh init           NVIDIA's riva_init.sh (needs internet: use only with connection sharing)
#   riva.sh start|stop|status
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="${RIVA_QS_SRC:-$ROOT/third_party/riva_quickstart_arm64_v2.14.0}"
QS="$ROOT/third_party/riva_qs"
IMAGE="${RIVA_IMAGE:-nvcr.io/nvidia/riva/riva-speech:2.14.0-l4t-aarch64}"

patch_cfg() {
  local c="$QS/config.sh"
  sed -i -E \
    -e 's/^service_enabled_asr=.*/service_enabled_asr=true/' \
    -e 's/^service_enabled_nlp=.*/service_enabled_nlp=false/' \
    -e 's/^service_enabled_tts=.*/service_enabled_tts=true/' \
    -e 's/^service_enabled_nmt=.*/service_enabled_nmt=false/' \
    -e 's/^riva_tegra_platform=.*/riva_tegra_platform="orin"/' \
    -e 's/^asr_language_code=.*/asr_language_code=("en-US")/' \
    -e 's/^tts_language_code=.*/tts_language_code=("en-US")/' "$c"
  for s in "$QS"/riva_*.sh; do
    sed -i -E '/docker run/{/--runtime/!s/docker run /docker run --runtime nvidia /}' "$s"
  done
}

model_loc() { (cd "$QS" && source ./config.sh && echo "${riva_model_loc}"); }

case "${1:-}" in
  prepare)
    [ -d "$SRC" ] || { echo "quickstart not found at $SRC (see scripts/offline/fetch_all.sh)"; exit 1; }
    rm -rf "$QS" && cp -a "$SRC" "$QS"
    patch_cfg
    echo "patched copy in $QS; diff vs original:"
    diff -ru "$SRC" "$QS" || true
    ;;
  deploy-offline)
    # VERIFY on first use: mirrors what riva_init.sh does for tegra after its download step.
    loc="$(model_loc)"
    case "$loc" in /*) ;; *) loc="$QS/$loc";; esac
    mkdir -p "$loc/rmir" "$loc/models"
    if [ -d "$ROOT/models/riva_rmir" ]; then
      find "$ROOT/models/riva_rmir" -name "*.rmir" -exec cp -n {} "$loc/rmir/" \;
    fi
    n=$(find "$loc/rmir" -name "*.rmir" | wc -l)
    [ "$n" -gt 0 ] || { echo "no .rmir files in $loc/rmir"; exit 1; }
    echo "deploying $n RMIR(s) into $loc/models (takes a while; TensorRT engines are built for this GPU)"
    docker run --runtime nvidia --rm --network none -v "$loc:/data" --entrypoint /bin/bash "$IMAGE" -c \
      'set -e; for r in /data/rmir/*.rmir; do echo "== $r"; riva-deploy -f "$r" /data/models; done'
    ;;
  init)   (cd "$QS" && bash ./riva_init.sh) ;;
  start)  (cd "$QS" && bash ./riva_start.sh) ;;
  stop)   (cd "$QS" && bash ./riva_stop.sh) ;;
  status)
    docker ps --filter name=riva --format '{{.Names}} {{.Status}}'
    (exec 3<>/dev/tcp/127.0.0.1/50051) 2>/dev/null && echo "port 50051 open" || echo "port 50051 closed"
    ;;
  *) sed -n '2,12p' "$0"; exit 1 ;;
esac
