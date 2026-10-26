#!/usr/bin/env bash
# JETSON (offline): load images, check models, prepare Riva. No host-level changes:
# no apt, no pip into system Python, no daemon.json edits.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
B=offline-bundle
[ "$(uname -m)" = aarch64 ] || { echo "run this on the Jetson"; exit 1; }
grep -q "R35" /etc/nv_tegra_release || echo "WARNING: not L4T R35; images here are built for r35"
[ -f .env ] || cp .env.example .env
mkdir -p logs state models/nano_llm_data tests/recordings; chmod 700 state

echo "== verifying checksums"
( cd "$B" && sha256sum -c SHA256SUMS )
echo "== loading images"
for f in "$B"/images/*.tar.gz; do echo "  $f"; gunzip -c "$f" | docker load; done

echo "== models"
chk() { if [ -e "$1" ]; then printf '  %-8s %s\n' ok "$1"; else printf '  %-8s %s (%s)\n' MISSING "$1" "$2"; fi; }
chk models/silero_vad.onnx "VAD; energy fallback is much worse"
chk models/VILA1.5-3b/config.json "default VLM"
chk models/piper/en_US-lessac-medium.onnx "TTS fallback"
chk models/whisper/ggml-base.en.bin "STT fallback"
chk models/llamacpp/model.gguf "VLM stretch fallback"
chk third_party/g1_description "URDF for clip validation"
ls models/riva_rmir/*/*.rmir >/dev/null 2>&1 && echo "  ok       Riva RMIRs" || echo "  MISSING  Riva RMIRs"

if [ -d third_party/riva_quickstart_arm64_v2.14.0 ]; then bash services/riva/riva.sh prepare >/dev/null && echo "== Riva quickstart prepared (third_party/riva_qs)"; fi
cat <<MSG

Done. Next (docs/RUNBOOK.md):
  make discover        # Phase 0, read-only
  make gpu-sanity      # Phase 1
  make mic-capture     # verify the G1 mic format
  make riva-deploy     # build Riva engines from the bundled RMIRs
  make build-native    # optional: whisper.cpp / llama.cpp fallbacks (offline, time-boxed)
  make up              # everything, action server in dry-run
MSG
