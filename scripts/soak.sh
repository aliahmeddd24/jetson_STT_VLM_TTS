#!/usr/bin/env bash
# Phase 6 soak: replay recorded utterances through the running stack for N seconds while
# logging tegrastats. Requires `make up` (services running) and tests/recordings/*.wav.
set -euo pipefail
SECS="${1:-1800}"; TS=$(date +%Y%m%d-%H%M%S)
cd "$(dirname "$0")/.."
bash scripts/tegrastats_log.sh "logs/soak-tegrastats-$TS.log" "$((SECS + 30))" &
TPID=$!
docker compose run --rm --no-deps orchestrator python /app/scripts/bench/bench_e2e.py \
  --dir tests/recordings --metrics "logs/soak-$TS.jsonl" --loop-seconds "$SECS" || true
wait $TPID || true
python3 scripts/bench/mem_report.py "logs/soak-tegrastats-$TS.log" | tee "logs/soak-mem-$TS.md"
python3 scripts/bench/report.py logs --only "soak-$TS" | tee "logs/soak-report-$TS.md"
