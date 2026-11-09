# DeepSeek V4 Flash 0731
Tested 2026-08 → 2026-11 on llama.cpp b10244 / b12140. Config: `model.conf` (or its git history once
another model is current).

## Model
`unsloth/DeepSeek-V4-Flash-0731-GGUF`, quant **UD-IQ3_XXS** (104gb, 4 shards).
284B total / 13B active MoE, 43 layers, 256 routed experts (6 active), 1M ctx max.
`DeepSeek-V4.1-Flash-1017` exists but is 763B — smallest quant (UD-IQ1_S) is 233gb. Does not fit.

**Routed experts are ~97.5% of the weights** (~277B of 284B); the dense part is only ~6.5B. So the
dense layers + KV cache go on the GPU and the experts are split GPU/RAM with `--n-cpu-moe`.

Fit: 81gb RAM + 32gb VRAM = 113gb usable > 104gb.
- GPU: ~8gb dense + ~4gb KV + ~0.6gb overhead → ~19gb of experts
- experts ≈ 96gb / 43 layers ≈ 2.23gb per layer → ~8 expert layers fit on GPU
- `--n-cpu-moe 34` leaves ~76gb of experts in RAM vs ~85gb free. **Headroom is ~9gb — tight.**
- VRAM at `-ncmoe 34 -ub 6144` via `start.sh`: 30.4gb of 32.6gb. `-ncmoe 33` fits only at `-ub` ≤4096.

Measured 2026-11-05 at `-ncmoe 34 -ub 6144` (b12140): **21-23 tok/s decode**, **~225-250 / ~830 /
~1160-1220 tok/s prefill** at 0.6k / 4.6k / 18k-token prompts. For reference a DGX Spark managed
15.5-17.4 tok/s decode / 148 tok/s prompt on a comparable quant. Expect a slower first request after each
start — mmap pages the weights in from NVMe lazily (the server reports healthy in ~16s, long before the
weights are resident).

## Prefill is PCIe-bound, not CPU-bound
llama.cpp's op-offload streams the RAM-resident expert weights to the GPU **once per ubatch** (measured
7-16 GB/s sustained PCIe rx during prefill, ~26gb per 512-token ubatch). The cost is per-ubatch and
fixed, so it amortises over ubatch size — this is the single biggest lever in the whole setup.
Free VRAM is split between expert layers (`-ncmoe`) and ubatch compute buffers (`-ub`); the buffers win.

`./tune.sh` sweeps, `./tune-summary.sh` tabulates. Full re-tune 2026-11-05 on b12140: stage 1 = 21-cell
grid (`-ncmoe` 33-37 x `-ub` 1024-4096, 1 run), stage 2 = finalists x3 runs, interleaved. Means:
| `-ncmoe` / `-ub` | 0.6k | 4.6k | 18k | decode | VRAM |
|---|---|---|---|---|---|
| 36 / 512 | 169 | 311 | 328 | ~21 | 23.7gb |
| 36 / 2048 (Aug pick) | ~215 | ~575 | ~785 | ~21 | 24.1gb |
| 36 / 4096 | 216 | 673 | 976 | 20.9 | 24.5gb |
| 33 / 4096 | 249 | 722 | 1035 | 21.7 | 31.0gb |
| 34 / 4096 | 221 | 697 | 993 | 21.2 | 28.7gb |
| **34 / 6144 (current)** | **224** | **830** | **1157** | **21.3** | **29.7gb** |
| 36 / 6144 | 218 | 823 | 1107 | 20.7 | 25.4gb |
| 36 / 8192 | 214 | 772 | 1188 | 20.1 | 26.4gb |

**+45% prefill at 4.6k and 18k over 36/2048, decode unchanged.** Prefill scales with the number of
ubatches a prompt needs, so bigger `-ub` keeps winning; 8192 starts losing at 4.6k. The August
"regression above 2048 at ≤4k" did not reproduce. 0.6k prefill is flat ~210-250 everywhere (one ubatch).
Decode spread across `-ncmoe` 33-36 is ~1 tok/s, about one stdev of run noise (theory: ~0.6 tok/s per
layer). `--threads` 10 / 12 / 16: no difference, keep 12.

Verified at `-ub 6144`: `./needle-test.sh` (22.6k tokens = 4 ubatches) returns all three needles, so the
upstream `ubatch >= 32` KV-corruption report does not reproduce here.

## `--no-mmap` is 1.7x faster and still must stay off — it OOM-killed the agent
With mmap the CPU experts sit in pageable page cache, so every per-ubatch copy to the GPU goes through
a bounce buffer (measured 7-16 GB/s on a link good for ~50). `--load-mode none` (= `--no-mmap`) puts
them in **pinned host memory** instead — `make_cpu_buft_list()` prefers the CUDA host buffer type, and
the loader warns about this at startup. Measured 2026-08-04 at `-ncmoe 36 -ub 2048` (b10244; `mmap` row is the outlier run, see above):

| load mode | 0.6k | 4.6k | 18k | decode | load time |
|---|---|---|---|---|---|
| `mmap` (current) | 310 | 715 | 808 | 23-24 | 16s |
| `none` | 620 | 1292 | 1344 | 23 | 36s |

**4x prefill over the original baseline — and unusable.** It pins ~78gb that the kernel cannot reclaim,
dropping MemAvailable from ~88gb to ~10gb. Four minutes after load the OOM killer fired and took the
`claude` agent process (`oom_score_adj:200` makes it the preferred victim); llama.cpp and the other
containers survived. The mmap safety net is exactly what stops this. Do not re-enable it at 96gb.

This is what a RAM upgrade would buy: at 128gb, `--load-mode none` fits with ~50gb spare and the 4x
prefill is free. It is the single highest-value hardware change for this box — worth far more than
anything left in the flags.

## `--moe-cache-mib` (b12140+) is a loss — leave it off
VRAM SLRU cache for hot experts of the RAM layers. Measured 2026-11-05 at `-ncmoe 36 -ub 2048`:
| cache MiB | 0 | 2048 | 4096 | 6144 | 7168 |
|---|---|---|---|---|---|
| decode tok/s | 21-23 | 9-10 | 14-15 | 17-21 | 17-24 |
| VRAM | 24.7gb | 26.6gb | 28.7gb | 30.8gb | 31.8gb |

PCIe rx stays at 12-18 GB/s during decode: 6-of-256 routing has too little reuse, so the cache
thrashes and every miss pays an upload on top of the CPU read. Prefill moved ±30% with no trend
(2048 looked faster, 4096+ slower) — not worth halving decode for.

## Decode is near its ceiling — don't expect much
Analysis from 2026-08-04 at `-ncmoe 36`.
Decode reads ~1.9gb/token of expert weights from RAM (36 layers x 6 of 256 experts). At 23.5 tok/s
that is ~44 GB/s effective against a 96 GB/s theoretical / ~60-70 GB/s practical dual-channel ceiling.
RAM is already at its rated 6000 MT/s (2x48gb CMK96GX5M2B6000Z30), so there is no BIOS win. The only
real lever left is reading fewer bytes: fall back to `UD-IQ2_M` (90.9gb, ~12% fewer bytes) via
`INCLUDE`/`MODEL_FILE`/`SIZE_GB` in `model.conf`, at a quality cost.

**Speculative decoding — including MTP — cannot help here. Do not spend time on it.** Decode is ~100%
expert-weight streaming: 36 layers x 6/256 x 2.23gb ≈ 1.9gb/token, which at 44 GB/s is 42.7ms of the
42.5ms/token budget (the GPU sits at sm=0% during decode). With 6-of-256 routing, adjacent tokens pick
near-disjoint expert sets, so verifying K drafts touches `256*(1-(1-6/256)^K)` experts per layer —
11.9 for K=2 vs 12 sequential, 23.1 for K=4 vs 24. **Batching saves 1-4% even at 100% acceptance**,
so any draft miss is a net loss. Measured on an echo-the-file prompt: off 23 tok/s, `ngram-mod` 21,
`ngram-map-k` 18.

MTP is obtainable — `convert_hf_to_gguf.py --mtp` emits the MTP head as a small standalone draft GGUF
(and `--dspark` a V4 DSpark draft), so it would not require reconverting 284B — but it changes nothing
about the arithmetic above. This quant ships no `nextn` tensors (no `deepseek4.nextn_layer_count`)
regardless. Speculation only pays once the experts are in VRAM, which 32gb cannot do.

mmap stays enabled (the default) so RAM overflow degrades to NVMe paging instead of OOM. Do not add
`--mlock` or `--no-mmap`.

## Flag choices
- **f16 KV cache** (no `--cache-type-k/-v`): upstream reports garbage output with quantized KV on this
  architecture. Deliberate deviation from `~/scripts/llama.sh`'s `q8_0` default.
- `--jinja`: required for V4's chat template / tool calling.
- `-b/-ub 6144`, `-ncmoe 34`: see the prefill tuning table above. The default `-ub 512` costs 3.5x
  prefill at long context.
- Upstream (PR #24162, merged 2026-06-29) had post-merge reports of KV corruption at `ubatch >= 32`
  *with expert offloading*. Did not reproduce at `-ub 6144` (`./needle-test.sh`).
