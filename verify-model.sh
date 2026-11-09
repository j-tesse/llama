#!/usr/bin/env bash
# Check the GGUF shards from model.conf are all present and self-consistent.
set -euo pipefail
cd "$(dirname "$0")"
source ./model.conf

# --- configuration ---------------------------------------------------------
MODELS_DIR="/home/joran/models/gguf"
# ---------------------------------------------------------------------------

first="$MODELS_DIR/$NAME/$MODEL_FILE"
[ -f "$first" ] || { echo "missing $first — run ./download-model.sh" >&2; exit 1; }

# Shard names end in -0000N-of-0000M.gguf; M must match what we actually have.
base="${first%.gguf}"; base="${base%-00001-of-*}"
shopt -s nullglob
shards=("$base"*.gguf)
expected=$(basename "$first" | sed -nE 's/.*-of-0*([0-9]+)\.gguf/\1/p')
if [ -n "$expected" ] && [ "$expected" != "${#shards[@]}" ]; then
  echo "incomplete: found ${#shards[@]} shard(s), manifest says $expected — re-run ./download-model.sh" >&2
  exit 1
fi

# A truncated download leaves a short final shard; llama.cpp only fails minutes in.
for f in "${shards[@]}"; do
  [ -s "$f" ] || { echo "empty shard: $f" >&2; exit 1; }
  head -c4 "$f" | grep -qa GGUF || { echo "bad magic (not a GGUF): $f" >&2; exit 1; }
done

echo "OK  ${#shards[@]} shard(s), $(du -shc "${shards[@]}" | tail -1 | cut -f1) total"
echo "$first"
