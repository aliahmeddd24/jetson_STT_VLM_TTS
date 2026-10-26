#!/usr/bin/env bash
# WORKSTATION: find out which RMIRs riva_init.sh would download for our patched config, then
# download them with the ngc CLI into models/riva_rmir/. riva_init.sh is run with stub
# `docker`, `ngc`, `nvidia-smi` and `uname` on PATH, so nothing is executed for real; we
# only capture the model names it asks for. VERIFY: if the list comes back empty, the
# quickstart's layout differs; fall back to temporary connection sharing (RUNBOOK).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
bash services/riva/riva.sh prepare >/dev/null
SHIM="$(mktemp -d)"; CALLS="$SHIM/calls.log"; : > "$CALLS"
for c in docker ngc nvidia-smi; do
  printf '#!/bin/sh\necho "%s $*" >> "%s"\nexit 0\n' "$c" "$CALLS" > "$SHIM/$c"; chmod +x "$SHIM/$c"
done
printf '#!/bin/sh\nif [ "$1" = "-m" ]; then echo aarch64; else /bin/uname "$@"; fi\n' > "$SHIM/uname"; chmod +x "$SHIM/uname"
( cd third_party/riva_qs && PATH="$SHIM:$PATH" bash ./riva_init.sh </dev/null >/dev/null 2>&1 || true )
grep -oE '[A-Za-z0-9_-]+/[A-Za-z0-9_-]+/(rmir|models)_[A-Za-z0-9_.-]+:[A-Za-z0-9_.-]+' "$CALLS" | sort -u > "$SHIM/models.txt" || true
echo "models requested by riva_init.sh:"; cat "$SHIM/models.txt"
[ -s "$SHIM/models.txt" ] || { echo "no models captured"; cp "$CALLS" logs/riva_init_calls.log; exit 1; }
mkdir -p models/riva_rmir
while read -r m; do ngc registry model download-version "$m" --dest models/riva_rmir; done < "$SHIM/models.txt"
cp "$SHIM/models.txt" models/riva_rmir/MODELS.txt
find models/riva_rmir -name "*.rmir" | sed 's/^/  /'
