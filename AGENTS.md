# llama
Test-bench for running quantized models with `llama.cpp` in docker, **one model at a time**.
The model under test is defined in `model.conf`; per-model findings live in `models/<NAME>.md`.
Current model: **DeepSeek V4 Flash 0731** → `models/deepseek-v4-flash.md`.

Serves an OpenAI-compatible API on **port 8000** as a drop-in replacement for the vLLM service.

## Machine
- headless machine running Ubuntu Server
- 96gb system memory - not all of it is available (~91gb total, ~81gb available)
- RTX 5090 with 32gb VRAM - **shared, not exclusive**: a `vllm_*` container (`~/scripts/vllm.sh`) normally owns
  all of it (`--gpu-memory-utilization 0.982`, `restart: unless-stopped`). One GPU LLM at a time.
  `start.sh` stops vLLM, `stop.sh` restarts the most recently stopped `vllm_*` (no-op if one runs).
- docker and nvidia docker tools already installed
- Ryzen 9 9900X (12C/24T), driver 610.43.02 / CUDA 13.3, GPU is Blackwell **sm_120**
- RAM: 2x48gb DDR5-6000 (already at rated speed), ~60-70 GB/s practical
- Crucial T705 Gen5 NVMe, ~670gb free

## Architecture
```
llama/
├── model.conf          THE model under test: repo, files, alias, ctx, model-specific server args
├── models/<NAME>.md    findings per model (fit, tuning tables, dead ends)
├── llama.cpp/          upstream submodule, pinned to a release tag (b12140). Never edited.
├── build.sh            builds upstream's .devops/cuda.Dockerfile (target: server) via build args
├── build-async.sh      start|status [--json]|wait|log|stop — build takes ~10-20min
├── update.sh           move submodule to a newer tag (ff-only, refuses dirty tree) + rebuild
├── download-model.sh   throw-away container pulls the GGUF into ~/models/gguf/<NAME>
├── verify-model.sh     shard count + GGUF magic check
├── start.sh            free GPU → run container → wait healthy → re-point agents
├── stop.sh             stop container → restart vLLM  (`--no-restore` to skip)
├── bench.sh            single-request prefill/decode + PCIe counters against :8000
├── bench-spec.sh       echo-a-file decode bench — the case speculative decoding should win
├── tune.sh             sweep server flags, restarting the container per variant → tune.results
├── tune-summary.sh     tune.results → markdown table, repeated variants merged as min-max
├── needle-test.sh      3 needles in a ~22k-token prompt — run after changing -ub
└── build.number/history
```

Every script sources `model.conf`; nothing else is model-specific. Machine-level constants (image,
container, port, threads, models dir) stay at the top of each script.

No custom Dockerfile: upstream's `.devops/cuda.Dockerfile` already does what we need, driven by
`--build-arg CUDA_VERSION` / `CUDA_DOCKER_ARCH`. If a patch ever becomes necessary, add
`<file>.patched.<ext>` here plus a thin Dockerfile that `COPY`s it over the submodule — never edit
`llama.cpp/` in place.

Image tags per build: `llama-cpp:b<N>`, `:<upstream-describe>`, `:latest`. `build.number` bumps only on
success. Container: `llama_cpp` (fixed name, so `stop.sh` finds it regardless of model).

## Switching to the next model
1. `./stop.sh --no-restore`; optionally `rm -rf ~/models/gguf/<old NAME>` (disk)
2. new arch? `./update.sh` to a llama.cpp tag that supports it
3. edit `model.conf` (old values stay in git history + `models/<old>.md`), create `models/<NAME>.md`
4. `./download-model.sh && ./start.sh`, then `./bench.sh`, `./needle-test.sh`, `./tune.sh`
5. record fit / measurements / dead ends in `models/<NAME>.md`; fold the tune winner into `SERVER_ARGS`

## Carry-over lessons (measured on DeepSeek V4 Flash, apply to any big MoE)
Details and numbers in `models/deepseek-v4-flash.md`.
- **MoE that doesn't fit VRAM**: dense layers + KV on GPU (`--n-gpu-layers 999`), experts split GPU/RAM
  with `--n-cpu-moe`. Mind RAM headroom — ~81gb available.
- **Prefill is PCIe-bound**: op-offload streams RAM-resident experts to the GPU once per ubatch, so big
  `-b/-ub` is the biggest prefill lever; free VRAM goes to ubatch buffers before extra expert layers.
- **Never `--no-mmap` / `--load-mode none` / `--mlock` at 96gb**: pinned host memory is 1.7x faster at
  prefill but unreclaimable; the OOM killer took the `claude` agent. mmap degrades to NVMe paging instead.
  Expect a slow first request after start (weights page in lazily).
- **Decode is RAM-bandwidth-bound** with experts in RAM; speculative decoding / MTP can't win when
  routing is sparse (adjacent tokens hit disjoint experts). `--moe-cache-mib` thrashed for the same reason.
- `--threads 12`: physical cores only; SMT siblings thrash cache. 10/12/16 measured identical.

### Models live in `~/models/gguf/<NAME>`, NOT the HF cache
vLLM bind-mounts `~/.cache/huggingface` and pulls by repo id, but `~/scripts/hf-cache-cleanup.sh`
builds its KEEP set by grepping `MODEL_ID=` out of `~/scripts/vllm.sh` and `rm -rf`s everything else
under `hub/`. A GGUF there would be treated as an orphan and deleted. Keeping it at `~/models/gguf`
(the path `~/scripts/llama.sh` already declares) avoids that, and llama.cpp wants a plain file path
anyway rather than the hub's blobs/snapshots symlink layout.

## Generic flag choices (`start.sh`)
- `--restart no`: vLLM's container is `unless-stopped` and also binds 8000 — two auto-restarting
  containers would race for the port after a reboot.
- `--jinja`, `--flash-attn on`, `--parallel 1`, f16 KV (no `--cache-type-k/-v`) by default; a model can
  add/override flags via `SERVER_ARGS` in `model.conf`.
- Health retries are generous — loading 100gb+ takes minutes.
- `start.sh` always recreates the container: the flags are baked in at create time, so reusing a
  stale one would silently ignore edits to the constants.

## Agent config sync
`start.sh` calls `~/scripts/update-agent-models.sh` (same as vllm.sh / colibri) to point pi + OpenClaw
at whatever :8000 serves. Two llama.cpp-specific details:
- `--alias` (`ALIAS` in `model.conf`) is what `/v1/models` reports; without it the id is the raw
  container path of the GGUF, which is what would land in the agent configs.
- That script reads the context window from vLLM's `max_model_len`, **which llama.cpp does not report**.
  It would silently fall back to its own 32768 default, so `start.sh` passes `CONTEXT_WINDOW` explicitly.
  Change `CONTEXT_SIZE` in `model.conf` and both agents follow; verified end-to-end at 40960.

## Ports
8000 = this service / vLLM (mutually exclusive). 8080 = open-webui, do not use. owui already has
`http://host.docker.internal:8000/v1` registered, so it picks this up with no config change.

## Usage
```
./build-async.sh start && ./build-async.sh wait
./download-model.sh
./start.sh
./stop.sh
```

## AGENTS.md Guidance
- broad architecture of this repo must be recorded in `AGENTS.md`; model-specific findings in `models/<NAME>.md`
- file must be kept in sync with the repo, apply edits to `AGENTS.md` alongside changes to the repo
- use terse language

## Work Guidance
- scripts are argument-free and configured by the `# --- configuration ---` constants at the top,
  plus `model.conf` for everything model-specific
  - do NOT use env variables to tune our scripts — edit the constant, so there is nothing to memorize
- skip code comments if the code is self-explanatory
- code comments must be extremely terse
- less code is better code
- create and persist small scripts instead of running medium to long bash commands ad-hoc
  - e.g. create `start.sh`, `stop.sh`, `rebuild.sh`, ...
- do not install anything globally
  - example: if a 1-time dependency is required to download a model from hugging-face, just write a script to create and start a throw-away container to get it done
- prefer additive patches over editing
  - example: instead of cloning a repo and applying changes in place, prefer cloning a repo in a subfolder, and "override" files from the subfolder via Dockerfile instructions at build time
