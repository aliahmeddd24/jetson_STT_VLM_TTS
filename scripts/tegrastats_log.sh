#!/usr/bin/env bash
# Timestamped tegrastats log (read-only). Usage: scripts/tegrastats_log.sh logs/teg.log [seconds]
set -euo pipefail
OUT="${1:-logs/tegrastats-$(date +%Y%m%d-%H%M%S).log}"; SECS="${2:-0}"
mkdir -p "$(dirname "$OUT")"
if [ "$SECS" -gt 0 ]; then T="timeout $SECS"; else T=""; fi
echo "logging to $OUT (Ctrl-C to stop)"
$T tegrastats --interval 500 | while IFS= read -r l; do printf '%s %s\n' "$(date +%s.%N)" "$l"; done > "$OUT"
