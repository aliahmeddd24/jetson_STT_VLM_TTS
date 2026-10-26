#!/usr/bin/env bash
# JETSON (offline): build the CUDA fallbacks from bundled sources, each time-boxed (default 2 h).
# Failures are recorded and do not stop anything else.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
BOX="${TIMEBOX:-2h}"; TS=$(date +%Y%m%d-%H%M%S); mkdir -p logs
for t in whisper:stt_whisper llamacpp:llamacpp; do
  name="${t%%:*}"; dir="${t##*:}"; log="logs/build-$name-$TS.log"
  echo "== building g1voice/$name:r35 (time box $BOX, log $log)"
  start=$(date +%s)
  if timeout "$BOX" docker build --network=none --build-arg BUILDER_IMAGE=g1voice/builder:r35 \
       -f "services/$dir/Dockerfile" -t "g1voice/$name:r35" . > "$log" 2>&1; then
    echo "   OK in $(( ($(date +%s) - start) / 60 )) min"
  else
    echo "   FAILED after $(( ($(date +%s) - start) / 60 )) min: tail of log:"; tail -15 "$log"
    echo "   -> record in docs/DECISIONS.md; the default stack does not need this image."
  fi
done
