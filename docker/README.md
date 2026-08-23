# keyframe-server on RunPod

Qwen-Image-Edit-2511 behind an authenticated HTTP endpoint, mirroring the fal
request/response shape so `client/client.py` can target it unchanged.

**Docker Hub:** `davidjbarnes/keyframe-server`

## How it works

1. Container starts and runs `provision.sh`, which downloads the models to
   `$HF_HOME` (default `/workspace/hf`) — **the network volume, not the image**.
   Idempotent, so a restart costs seconds rather than 57.7 GB.
2. `auth-proxy.py` starts on port **8888**, enforcing the API key.
3. `server.py` starts on **127.0.0.1:8189** — loopback only, so the proxy is the
   sole public door.

The model is not baked into the image on purpose: at 57.7 GB it would be
impractical to push, pull and store, and it would be re-pulled on every image
update rather than persisting on the volume.

## RunPod template settings

| Setting | Value |
|---|---|
| Container Image | `davidjbarnes/keyframe-server:latest` |
| Exposed HTTP Port | `8888` |
| Container Disk | 20 GB |
| **Network Volume** | **Required — 80 GB+, mounted at `/workspace`** |
| GPU | RTX 3090 / 4090 (24 GB) — see *Other GPUs* below |

**The network volume is not optional.** Without it the models land on ephemeral
container disk and the full 57.7 GB is re-downloaded on every pod start.

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `API_KEY` | *(empty)* | **Set this.** Empty means the endpoint is open to anyone who finds the pod URL. |
| `HF_TOKEN` | — | Only needed if a repo becomes gated |
| `HF_HOME` | `/workspace/hf` | Model cache. Must be on the network volume. |
| `QUANT` | `fp8` | `fp8`, `none`, `nunchaku` — see below |
| `LIGHTNING` | `1` | `0` for full 40-step sampling |
| `SKIP_DOWNLOAD` | `0` | `1` to skip provisioning when the volume is already populated |
| `SERVER_PORT` | `8189` | Internal, loopback only |
| `PROXY_PORT` | `8888` | Public, the one RunPod exposes |

## Usage

```bash
export QWEN_EDIT_URL_RUNPOD="https://<podid>-8888.proxy.runpod.net/edit"
export KEYFRAME_API_KEY="<the API_KEY you set>"

./client.py --model runpod -i in.png -p "Change her shirt to a green sweater." -o out.png
```

Health, unauthenticated so RunPod can probe it:

```bash
curl https://<podid>-8888.proxy.runpod.net/health
# {"status":"ok","vram_free_gb":24.7,"vram_total_gb":25.3}
# 503 with "still loading weights" while the model loads — this is honest, not broken
```

## Quant modes

Measured on a 3090 (24 GB), 512x768 input:

| mode | works on 24 GB | notes |
|---|---|---|
| `fp8` | **yes — the default** | ~20 GB transformer, fits as one resident component. Lightning applies → 4 steps. **~44–72 s/edit** |
| `none` | **no — OOMs** | `enable_model_cpu_offload()` swaps whole models and the bf16 transformer is ~40 GB. Viable on 48 GB+ cards, where it is better quality. |
| `nunchaku` | **no — broken upstream** | Incompatible with diffusers 0.40 (`QwenEmbedRope.forward` signature change), and PEFT cannot patch its INT4 layers so Lightning is forfeited anyway. |

### Other GPUs

This template assumes a 24 GB card. On 48 GB+ (A6000, L40S) set `QUANT=none`
for better quality at the same step count — the OOM that forces fp8 here is
purely a 24 GB constraint.

## Notes

- **No 2511 Lightning LoRA exists.** The 2509 4-step LoRA applies cleanly to the
  2511 transformer and is what gives 4-step sampling. If it ever fails to attach,
  the server logs it and falls back to 40 steps rather than silently producing
  4-step output with no LoRA, which looks like garbage rather than an error.
- **First boot downloads 57.7 GB.** Subsequent starts reuse the volume.
- Output resolution is chosen by the model (~1 MP), not by the input.
