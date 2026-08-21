# keyframe-server

Local **Qwen-Image-Edit-2511** inference behind a FastAPI endpoint, plus a CLI client
for building LTX multi-frame conditioning keyframes.

The server mirrors the fal.ai request/response shape, so the same client can target
either a hosted fal endpoint or your own GPU.

- `server.py` — FastAPI wrapper around the Qwen edit pipeline (`POST /edit`, `GET /health`)
- `client/client.py` — edit-based keyframe factory (fal.ai backends)

---

## How to run

### Server

```bash
python server.py --quant nunchaku --port 8188
```

That's the standard invocation: INT4 SVDQuant transformer, 4-step Lightning LoRA,
listening on all interfaces at port 8188.

First run downloads the Qwen-Image-Edit-2511 weights, the Nunchaku transformer, and
the Lightning LoRA from Hugging Face — expect a long startup and tens of GB of cache.
Subsequent runs load from `~/.cache/huggingface`.

**Setup:**

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` does not pin `torch` — install the build matching your CUDA
version first (see pytorch.org), then the rest. `nunchaku` also ships per-GPU-generation
wheels; check huggingface.co/nunchaku-tech for the right one.

**Pre-download the weights** (optional, but recommended — pulls the model outside of
server startup so a slow or interrupted download doesn't look like a hung server):

```bash
pip install -U "huggingface_hub[cli]"
hf download Qwen/Qwen-Image-Edit-2511
```

Everything lands in `~/.cache/huggingface`, which is exactly where the server looks at
load time — so once this finishes, startup is local-disk only. Set `HF_HOME` first if
you want the cache somewhere with more room.

Depending on flags, the server also pulls two smaller repos on first run. Grab them
up front the same way:

```bash
# --quant nunchaku (the default): INT4 SVDQuant transformer
hf download nunchaku-tech/nunchaku-qwen-image-edit-2511

# Lightning 4-step LoRA (on unless you pass --no-lightning)
hf download lightx2v/Qwen-Image-Lightning \
    Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors
```

The download resumes if interrupted, so re-running the same command after a dropped
connection is safe. If the repo is gated, `hf auth login` first.

**Options:**

| Flag | Default | Notes |
|---|---|---|
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8188` | Listen port |
| `--quant` | `nunchaku` | `nunchaku` \| `fp8` \| `none` |
| `--no-lightning` | off | Full 40-step sampling at CFG 4.0 instead of 4-step Lightning |

**Picking a quant mode** (sized for a 24GB 3090):

- `nunchaku` — INT4 SVDQuant. Fastest load, most VRAM headroom. Leaves room to keep
  LTX partially resident alongside it. This is the one you want by default.
- `fp8` — torchao float8 weight-only on the bf16 transformer. Middle ground.
- `none` — plain bf16 with model CPU offload. Slowest, maximum quality.

All three modes call `enable_model_cpu_offload()`, so layers stream to GPU on demand.

**Quality vs. speed:** Lightning LoRA is on by default (4 steps, `true_cfg_scale` 1.0).
For hard compositional edits where 4 steps smears detail, drop it:

```bash
python server.py --quant nunchaku --port 8188 --no-lightning   # 40 steps, CFG 4.0
```

Per-request overrides beat both — `num_inference_steps` and `true_cfg_scale` in the
POST body win over the server defaults.

**Verify it's up:**

```bash
curl -s localhost:8188/health
# {"status":"ok","vram_free_gb":21.4,"vram_total_gb":25.4}
```

`/health` reports live VRAM, which is the fastest way to see whether a model is
resident or whether something else on the box is eating the card.

**Hitting the endpoint directly:**

```bash
curl -s localhost:8188/edit \
  -H 'content-type: application/json' \
  -d '{
    "prompt": "same woman, now holding a glass of water",
    "image_urls": ["data:image/png;base64,iVBORw0KG..."],
    "num_images": 1,
    "seed": 42
  }' | jq -r '.images[0].url' | sed 's/^data:image\/png;base64,//' | base64 -d > out.png
```

`image_urls` takes 1–3 entries, each a **data URI** or an **http(s) URL** — file paths
are rejected. Responses come back as base64 data URIs, not files on disk; the server
never writes to the filesystem.

Optional body fields: `seed`, `num_inference_steps`, `true_cfg_scale`.

**Running it as a background service:**

```bash
nohup python server.py --quant nunchaku --port 8188 > server.log 2>&1 &
tail -f server.log
```

Model load happens *before* uvicorn binds the port, so a refused connection during
the first minute means it's still loading, not that it crashed. `/edit` returns
`503 pipeline not loaded` only if loading failed outright.

### Client

```bash
cd client
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export FAL_KEY="key-id:key-secret"     # fal.ai dashboard -> Keys
```

```bash
# one source frame + an instruction
./client.py -i inputs/pour1.png -p "same woman, now holding a glass of water" -o keyframes/p1.png

# multi-ref: source frame + canonical face reference
./client.py -i inputs/pour1.png -i inputs/face_ref.png \
    -p "same woman from image 1 with the face from image 2, tilting the glass toward her chest" \
    -o keyframes/p2.png

# normalize straight to LTX conditioning size (both dims must be /32)
./client.py -i inputs/pour1.png -p "..." -o keyframes/p3.png --size 512x768

# N variants to pick from -> writes p4_1.png .. p4_4.png
./client.py -i inputs/pour1.png -p "..." -o keyframes/p4.png -n 4
```

Backends via `--model`: `qwen` (default), `nb2`, `pro`, `local`. Local images are
inlined as base64 data URIs, so there's no fal storage upload and no CDN auth.
`--size` resizes-to-cover then center-crops to exact conditioning dimensions.

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
