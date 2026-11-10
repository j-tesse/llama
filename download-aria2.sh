#!/usr/bin/env bash
# Fallback for ./download-model.sh when xet stalls (it restarts big files from zero).
# aria2 in a throw-away container: parallel ranges, resumable. INCLUDE must be a literal file path, no glob.
set -euo pipefail
cd "$(dirname "$0")"
source ./model.conf

# --- configuration ---------------------------------------------------------
MODELS_DIR="/home/joran/models/gguf"
CONNECTIONS=16
# ---------------------------------------------------------------------------

DESTINATION="$MODELS_DIR/$NAME"
mkdir -p "$DESTINATION"
# root for apk; chown back to us at the end
list="https://huggingface.co/$REPOSITORY/resolve/main/$INCLUDE\n  out=$INCLUDE\n"
for f in "${DRAFT_FILES[@]}"; do list+="https://huggingface.co/$DRAFT_REPOSITORY/resolve/main/$f\n  out=$f\n"; done
printf "$list" >"$DESTINATION/.aria2-input"
docker run --rm -v "$DESTINATION:/out" alpine:3 sh -c "
  apk add -q --no-cache aria2 && \
  aria2c -c -x $CONNECTIONS -s $CONNECTIONS -k 64M --max-tries=0 --retry-wait=5 \
    --lowest-speed-limit=100K --summary-interval=60 --console-log-level=warn \
    -d /out -i /out/.aria2-input; \
  rc=\$?; rm -f /out/.aria2-input; chown -R $(id -u):$(id -g) /out; exit \$rc"
rm -rf "${DESTINATION:?}/.hfcache" "${DESTINATION:?}/.cache"
exec ./verify-model.sh
