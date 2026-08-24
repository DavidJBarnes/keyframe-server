# keyframe-server

Local **Qwen-Image-Edit-2511** inference behind a FastAPI endpoint, plus a CLI client
for building LTX multi-frame conditioning keyframes.

The server mirrors the fal.ai request/response shape, so the same client can target
either a hosted fal endpoint or your own GPU.

- `server.py` — FastAPI wrapper around the Qwen edit pipeline (`POST /edit`, `GET /health`)
- `client/client.py` — edit-based keyframe factory ([client README](client/README.md))
- `docker/` — RunPod image and template ([docker README](docker/README.md))
- `docs/pipeline-notes.md` — measured findings from the proof-of-concept shots

---

## How to run

### On RunPod (recommended)

The packaged path. `davidjbarnes/keyframe-server` on Docker Hub, RunPod template
`keyframe-server`. Handles the model download, an auth proxy, and correct flags for
you — see [docker/README.md](docker/README.md).

### Locally

```bash
python server.py --quant fp8 --port 8189
```

Port **8189**, not 8188: ComfyUI commonly occupies 8188 on GPU boxes.

**Setup:**

```bash
python3.11 -m venv venv          # 3.11 or 3.12 — see note below
source venv/bin/activate
pip install torch torchvision    # install FIRST, on its own
pip install -r requirements.txt
```

Install `torch` **before** the rest, and let pip pick the default PyPI wheels — they
bundle their own CUDA runtime. Pinning an old `--index-url` (e.g. `cu124`) makes pip
backtrack for a long time hunting for a compatible build and may find none.

**Python 3.11 or 3.12, not 3.13** — only because the nunchaku wheel requires `<3.13`.
Since nunchaku is currently broken anyway (below), 3.13 is fine if you never intend to
use it.

**Pre-download the weights** (optional; moves the multi-GB pull out of server startup so
a slow download doesn't look like a hung server):

```bash
pip install -U "huggingface_hub[cli]"

# base model, 57.7 GB
hf download Qwen/Qwen-Image-Edit-2511

# Lightning 4-step LoRA, 0.85 GB. Note this is the 2509 file: no 2511 LoRA has ever
# been published, and the 2509 one applies cleanly to the 2511 transformer.
hf download lightx2v/Qwen-Image-Lightning \
    Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors
```

Everything lands in `$HF_HOME` (default `~/.cache/huggingface`), which is where the
server looks at load time. Downloads resume if interrupted. Set `HF_HOME` first if you
want the cache elsewhere — on RunPod it must point at the network volume.

**Options:**

| Flag | Default | Notes |
|---|---|---|
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8188` | Listen port — pass `8189` if ComfyUI has 8188 |
| `--quant` | `fp8` | `fp8` \| `none` \| `nunchaku` — see below |
| `--no-lightning` | off | Full 40-step sampling at CFG 4.0 instead of 4-step Lightning |
| `--nunchaku-variant` | `balance` | Only relevant if nunchaku is ever fixed |

### Quant modes — measured, not theoretical

On a 24 GB 3090, 512x768 input (2026-08-22):

| mode | 24 GB | notes |
|---|---|---|
| **`fp8`** | **works — the default** | torchao float8 weight-only, ~20 GB transformer, fits as one resident component under model-level offload. Lightning applies → 4 steps. **~44–72 s/edit** (200 s at 40 steps) |
| `none` | **OOMs** | `enable_model_cpu_offload()` swaps *whole models*, and the bf16 transformer is ~40 GB — it cannot fit in 23.5 GB regardless of scheduling. Fine on 48 GB+ cards, where it is the better-quality option. |
| `nunchaku` | **broken** | Two independent blockers — see below |

**nunchaku is currently unusable.** It calls `self.pos_embed(img_shapes, txt_seq_lens,
device=...)`, but diffusers 0.40 changed the signature to
`QwenEmbedRope.forward(video_fhw, device=None, max_txt_seq_len=None)`, so it dies with
`TypeError: got multiple values for argument 'device'`. nunchaku declares
`diffusers>=0.36` while its CI pins `==0.36`. Separately, PEFT cannot patch its INT4
`SVDQW4A4Linear` layers, so the Lightning LoRA never attaches and you are stuck at 40
steps — which removes most of the reason to want INT4 in the first place. Its wheels
also top out at torch 2.12.

**Quality vs speed:** the Lightning LoRA is loaded by default (4 steps, cfg 1.0). For
hard edits, `--no-lightning` gives full 40-step sampling at cfg 4.0. Per-request
overrides beat both — `num_inference_steps` and `true_cfg_scale` in the POST body, or
`--steps` / `--cfg` on the client.

The server prints what it actually resolved at startup, which is worth reading rather
than assuming:

```
[lightning] active -> 4 steps, cfg 1.0
[pipeline] ready: quant=fp8 steps=4 cfg=1.0
```

**Verify it's up:**

```bash
curl -s localhost:8189/health
# {"status":"ok","vram_free_gb":21.4,"vram_total_gb":25.4}
```

**Hitting the endpoint directly:**

```bash
curl -s localhost:8189/edit \
  -H 'content-type: application/json' \
  -d '{
    "prompt": "Change her shirt to a dark green sweater. Keep everything else identical.",
    "image_urls": ["data:image/png;base64,iVBORw0KG..."],
    "num_images": 1,
    "seed": 42
  }' | jq -r '.images[0].url' | sed 's/^data:image\/png;base64,//' | base64 -d > out.png
```

`image_urls` takes 1–3 entries, each a **data URI** or an **http(s) URL** — file paths
are rejected. Responses come back as base64 data URIs; the server never writes to disk.
Output resolution is chosen by the model (~1 MP), not by your input.

Optional body fields: `seed`, `num_inference_steps`, `true_cfg_scale`.

**Prompt phrasing matters.** Imperative edits (`"Change X to Y. Keep everything else
identical."`) behave; descriptive restatement (`"the same woman, now wearing X"`) can
make the model emit a side-by-side before/after pair. See `docs/pipeline-notes.md`.

**Running it as a background service:**

```bash
nohup python server.py --quant fp8 --port 8189 > server.log 2>&1 &
tail -f server.log
```

Model load happens *before* uvicorn binds the port, so a refused connection during the
first minutes means it's still loading, not that it crashed.

---

## Repo hygiene

Generated media never gets committed. Two layers enforce this:

1. `.gitignore` covers `*.png`, `*.jpg`, `*.jpeg`, `*.mp4` and the
   `client/inputs/`, `client/keyframes/`, `client/outputs/` directories.
2. `.githooks/pre-commit` rejects any commit staging those extensions — this
   catches `git add -f`, which bypasses `.gitignore` entirely.

The hook lives in the repo but `core.hooksPath` is local config and is **not** cloned.
After cloning, run:

```bash
git config core.hooksPath .githooks
```
