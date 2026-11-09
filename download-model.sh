#!/usr/bin/env bash
# Download the GGUF from model.conf into ~/models/gguf/<NAME> using a throw-away
# container (nothing is installed on the host). Resumable — safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"
source ./model.conf

# --- configuration ---------------------------------------------------------
MODELS_DIR="/home/joran/models/gguf"    # NOT the HF cache — see AGENTS.md
MAX_WORKERS=8
# ---------------------------------------------------------------------------

DESTINATION="$MODELS_DIR/$NAME"
mkdir -p "$DESTINATION"

need_gb=$(( SIZE_GB + 16 ))
avail_gb=$(df -BG --output=avail "$DESTINATION" | tail -1 | tr -dc '0-9')
if [ "$avail_gb" -lt "$need_gb" ]; then
  echo "need ${need_gb}G free at $DESTINATION, have ${avail_gb}G" >&2
  exit 1
fi
echo "downloading $REPOSITORY [$INCLUDE] -> $DESTINATION (${avail_gb}G free)"

# --user keeps the downloaded files owned by us, not root.
# HF_HOME is parked on the same big filesystem so the xet chunk cache can't fill /.
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e HF_HOME=/out/.hfcache \
  -v "$DESTINATION:/out" \
  python:3.12-slim \
  sh -c "pip install --quiet --user huggingface_hub && \
         /tmp/.local/bin/hf download '$REPOSITORY' \
           --include '$INCLUDE' \
           --local-dir /out \
           --max-workers $MAX_WORKERS"

rm -rf "${DESTINATION:?}/.hfcache"
exec ./verify-model.sh
