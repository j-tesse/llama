# Ripple 65B Preview NVFP4
Tested 2026-11 on llama.cpp b12238. Config: `model.conf` (or its git history once another model is current).

## Model
`prithivMLmods/Ripple-65B-Preview-NVFP4-GB202-GGUF`, single file, 57.2gb. Ricursive Intelligence, Apache-2.0.
**Dense** 65B, 80 layers (4 SWA-1024 + 1 global NoPE per 5), hidden 5120, ffn 40960, 80 q / 8 kv heads.
Arch `ricursive` needs llama.cpp ≥ b12167 (PR #31692). No patch: b12238's generic pass loads the
NVFP4 `.scale` / `.input_scale` tensors and the ricursive graph uses `ffn_*_s`.

Tensor types: FFN gate/up/down **NVFP4** (~0.35gb/layer); attention q/k/v/o/gate, embd, output **BF16**
(~0.34gb/layer, ~27gb total — modelopt left `self_attn` unquantized). ~0.69gb per layer.

KV is small: only the 16 global layers keep full context (64kb/token f16 → 2gb at 32k); SWA layers 1024 tokens.

## Fit
57gb > 32gb VRAM, and dense means every token reads every layer. Partial offload by whole layers
(`--n-gpu-layers`, overrides start.sh's 999). Expected decode ≈ RAM bytes / RAM bandwidth.
Smaller alternative that fits VRAM fully: `AtomicChat/Ripple-65B-Preview-GGUF` `AD-IQ3_XXS` (28gb).

## Download
xet stalled twice at ~13-16gb and restarted from zero; `HF_HUB_DISABLE_XET` is refused for files this
size. single-stream curl fell from ~30 to ~2MB/s; `./download-aria2.sh` (16 parallel ranges, resumable).
