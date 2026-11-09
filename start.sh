#!/usr/bin/env bash
# Start the model from model.conf on port 8000 as a drop-in replacement for vLLM.
# The GPU holds one LLM at a time, so this stops any running vLLM first.
set -euo pipefail
cd "$(dirname "$0")"
source ./model.conf

# --- configuration ---------------------------------------------------------
IMAGE="llama-cpp:latest"
CONTAINER="llama_cpp"
MODELS_DIR="/home/joran/models/gguf"
PORT=8000
THREADS=12          # physical cores only; SMT siblings thrash cache
LOAD_MODE=mmap      # DO NOT set to `none`/--no-mmap on this box: pins the CPU weights unreclaimable
                    # and the OOM killer took out the agent. See AGENTS.md.
# ---------------------------------------------------------------------------

C='\033[96m'; G='\033[92m'; Y='\033[93m'; R='\033[91m'; RS='\033[0m'
say() { echo -e "${C}==>${RS} $*"; }

[ -f "$MODELS_DIR/$NAME/$MODEL_FILE" ] || {
  echo -e "${R}model missing:${RS} $MODELS_DIR/$NAME/$MODEL_FILE"
  echo "run: ./download-model.sh"
  exit 1
} >&2

docker image inspect "$IMAGE" >/dev/null 2>&1 || {
  echo -e "${R}image missing:${RS} $IMAGE — run ./build-async.sh start && ./build-async.sh wait" >&2
  exit 1
}

# 1. Free the GPU. Our own container first — the flags below are baked in at create
#    time, so we always recreate and must release its VRAM before the drain check.
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

# `docker stop` (not kill) so `unless-stopped` won't resurrect it.
vllm_running=$(docker ps -q --filter "name=vllm_")
if [ -n "$vllm_running" ]; then
  say "stopping vLLM to free the GPU"
  echo "$vllm_running" | xargs -r docker stop
fi

say "waiting for VRAM to drain"
for _ in $(seq 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$used" -lt 1000 ] && break
  sleep 2
done
[ "$used" -lt 1000 ] || { echo -e "${R}VRAM still ${used} MiB in use${RS} — check nvidia-smi" >&2; exit 1; }
say "VRAM free (${used} MiB used)"

# 2. Create.
say "creating container for $ALIAS (model load can take minutes)"
# --restart no: vLLM's container is `unless-stopped` and also binds 8000;
# two auto-restarting containers would race for the port after a reboot.
docker run -d --name "$CONTAINER" \
  --runtime nvidia --gpus all \
  --restart no \
  -v "${MODELS_DIR}/${NAME}:/model:ro" \
  -p "${PORT}:${PORT}" \
  --ipc=host \
  --health-cmd "curl -f http://localhost:${PORT}/health || exit 1" \
  --health-interval 30s \
  --health-start-interval 5s \
  --health-timeout 5s \
  --health-retries 60 \
  --health-start-period 30s \
  "$IMAGE" \
  --model "/model/${MODEL_FILE}" \
  --alias "$ALIAS" \
  --n-gpu-layers 999 \
  --threads "$THREADS" \
  --ctx-size "$CONTEXT_SIZE" \
  --load-mode "$LOAD_MODE" \
  --cont-batching \
  --parallel 1 \
  --jinja \
  --flash-attn on \
  --host 0.0.0.0 --port "$PORT" \
  "${SERVER_ARGS[@]}" >/dev/null

# 3. Wait for healthy, logging startup duration like vllm.sh/llama.sh do.
START_TIME=$(date +%s)
say "waiting for health"
while true; do
  status=$(docker inspect --format='{{.State.Health.Status}}' "$CONTAINER" 2>/dev/null || echo gone)
  elapsed=$(( $(date +%s) - START_TIME ))
  case "$status" in
    healthy) echo -e "\r${G}healthy${RS} after ${elapsed}s        "; break ;;
    gone|unhealthy)
      echo -e "\r${R}container ${status}${RS} after ${elapsed}s"
      docker logs --tail 40 "$CONTAINER" 2>&1 || true
      exit 1 ;;
  esac
  if ! docker ps -q -f name="^${CONTAINER}$" | grep -q .; then
    echo -e "\r${R}container exited${RS} after ${elapsed}s"
    docker logs --tail 40 "$CONTAINER" 2>&1 || true
    exit 1
  fi
  printf "\r  %ss elapsed (%s)" "$elapsed" "$status"
  sleep 2
done

CSV=/home/joran/scripts/llama-startups.csv
[ -f "$CSV" ] || echo "timestamp,container,duration_seconds" > "$CSV"
echo "$(date -Is),${CONTAINER}/${NAME},$(( $(date +%s) - START_TIME ))" >> "$CSV"

# 4. Re-point pi + OpenClaw at whatever :8000 now serves.
if [ -x /home/joran/scripts/update-agent-models.sh ]; then
  say "re-pointing agents at :${PORT}"
  # That script reads the context from vLLM's `max_model_len`, which llama.cpp
  # does not report; without this it silently falls back to its 32768 default.
  CONTEXT_WINDOW="$CONTEXT_SIZE" /home/joran/scripts/update-agent-models.sh \
    || echo -e "${Y}agent sync failed (non-fatal)${RS}"
fi

say "serving $ALIAS on http://localhost:${PORT}/v1  — ./stop.sh returns the GPU to vLLM"
