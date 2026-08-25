# keyframe-server

A stateless image-edit microservice for building **LTX-2.5 multi-frame conditioning
keyframes**. One image in, one image out, no memory between calls — a storyboard is
assembled by the caller chaining requests, not by this service knowing what a
storyboard is.

Two pipelines behind one endpoint, because no single model does both jobs:

| `mode` | engine | for | why not the other |
|---|---|---|---|
| `full` | Qwen-Image-Edit (ComfyUI) | garments, scenes, props, composition | LivePortrait only articulates a face it can already see |
| `face` | LivePortrait ExpressionEditor | expression, gaze, small head rotation | Qwen halves skin texture and de-ages the subject on **every** pass, at every denoise setting |

That split is measured, not stylistic — see [docs/dual-pipeline-design.md](docs/dual-pipeline-design.md).

- `server.py` — the endpoint (`POST /generate`, `GET /health`)
- `client/client.py` — CLI keyframe factory ([client README](client/README.md))
- `docker/` — the runtime: ComfyUI + both pipelines ([docker README](docker/README.md))
- `docs/pipeline-notes.md` — measured findings from the proof-of-concept shots

---

## How to run

Everything runs in the container — ComfyUI, both model stacks, and the adapter.
There is no meaningful "local" mode any more: `server.py` is a thin HTTP adapter
that talks to a ComfyUI instance, and it needs one to talk to.

```bash
docker run -d --name keyframe-server --gpus all --memory 56g \
  -v ~/models:/workspace/models:ro \
  -p 8189:8888 \
  davidjbarnes/keyframe-server:latest
```

The 28 GB Qwen checkpoint is **not** baked into the image — it is bind-mounted from
the host so the model survives image updates. The LivePortrait models (~500 MB) *are*
baked in, since they are small and a runtime download would make the first face
request depend on the network.

Running the adapter by hand against an existing ComfyUI:

```bash
python server.py --comfy-url http://127.0.0.1:8188 --port 8189
```

| Flag | Default | Notes |
|---|---|---|
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8189` | 8188 avoided — ComfyUI usually has it |
| `--comfy-url` | `http://127.0.0.1:8188` | Backend to drive |
| `--ckpt` | `Qwen-Rapid-AIO-NSFW-v23.safetensors` | `mode=full` checkpoint |
| `--wait-for-comfy` | `600` | Seconds to wait for the backend before giving up |

**Verify it's up:**

```bash
curl -s localhost:8189/health
# {"status":"ok","backend":"comfyui","ckpt":"Qwen-Rapid-AIO-NSFW-v23.safetensors",
#  "vram_free_gb":21.4,"vram_total_gb":25.4,"vram_low":false}
```

`/health` returns 503 until ComfyUI answers, so a green health check means the whole
chain is live — not just that the adapter bound a port.

---

## The endpoint

`POST /generate`. `image_urls` takes 1–3 entries, each a **data URI** or an
**http(s) URL** — file paths are rejected. Responses are base64 data URIs; the
server never writes to disk.

### `mode: "full"` — Qwen

```bash
curl -s localhost:8189/generate -H 'content-type: application/json' -d '{
  "prompt": "Change her shirt to a dark green sweater. Keep everything else identical.",
  "mode": "full",
  "image_urls": ["data:image/png;base64,iVBORw0KG..."],
  "seed": 42
}'
```

Fields: `seed`, `num_inference_steps`, `true_cfg_scale`, `negative_prompt`,
`denoise`, `width`/`height`, `num_images`.

Output matches the first input's size unless `width`/`height` say otherwise, capped
at `MAX_MP` (1.2 MP). Output pixels drive the cost: 0.39 MP ≈ 6 s, 1.55 MP ≈ 21 s,
7.09 MP ≈ 186 s — and a raw phone photo is ~7 MP.

**Leave the sampler alone.** 4-step Lightning at cfg 1.0 has beaten every
alternative tested; raising steps or cfg measurably makes things *worse*. If an
edit comes back wrong, re-roll the seed and use imperative phrasing
(`"Change X to Y. Keep everything else identical."`) rather than reaching for the
knobs. Descriptive restatement (`"the same woman, now wearing X"`) can make the
model emit a side-by-side before/after pair. Details in
`docs/pipeline-notes.md` §7b.

### `mode: "face"` — LivePortrait

Three ways to say what the face should do. They land on the same node inputs, and
`expression` wins if you supply it.

**Exact** — the honest contract, and what a UI's sliders should send:

```jsonc
{"mode": "face", "image_urls": ["data:..."],
 "expression": {"smile": 0.4, "blink": -3, "rotate_yaw": 5}}
```

| key | range | key | range |
|---|---|---|---|
| `smile` | -0.3 … 1.3 | `pupil_x` / `pupil_y` | ±15 |
| `blink` | -20 … 5 | `rotate_pitch`/`yaw`/`roll` | ±20 |
| `wink` | 0 … 25 | `aaa` (jaw open) | -30 … 120 |
| `eyebrow` | -10 … 15 | `eee` / `woo` (mouth) | -20 … 15 |

Out-of-range values are a 422, not a silent clamp.

**Prompt** — sugar over a small fixed vocabulary:

```jsonc
{"mode": "face", "image_urls": ["data:..."], "prompt": "soften her smile, look left"}
```

Recognised: smile, grin, laugh, frown, blink, wink, squint, wide/closed eyes,
raised brows, open mouth, purse/pout, look left/right/up/down, turn head
left/right, tilt head, chin up/down. `slight`/`soft`/`soften` scale to 0.5x;
`big`/`very`/`wide` to 1.5x. A face request matching none of these returns a 422
listing them rather than silently returning the input unchanged.

**Driving image** — copy an expression off a reference:

```jsonc
{"mode": "face", "image_urls": ["source", "driver"],
 "sample_ratio": 0.7, "sample_parts": "OnlyExpression"}
```

`sample_parts` ∈ `OnlyExpression | OnlyRotation | OnlyMouth | OnlyEyes | All`;
`sample_ratio` -0.2 … 1.2 scales the transfer.

Also: `face_pad` (crop_factor, 1.5–2.5 — how much context the warp sees; lower is
sharper), `src_ratio` (below 1.0 relaxes the resting expression toward neutral
first), and `detail_restore` (0–1, default 1.0).

**About `detail_restore`.** LivePortrait decodes through a fixed 256×256
bottleneck, so it softens the whole face crop — including the parts it did not
move. Detail restoration blends on `|output − source|`, keeping the node's pixels
only where it actually moved something and taking the original's back everywhere
else. Measured on a 214×292 face: texture 16% → 31% of source, forehead lines and
hair strands visibly recovered, expression untouched. Every output pixel comes
from one of the two real images, so it cannot invent detail or shift apparent age.
Set it to 0 to see the raw node output.

Measured, richmond kf1 (512×768):

| | face texture | drift outside warp | time |
|---|---|---|---|
| source | 1405 | — | — |
| raw | 227 (16%) | **0.0000** | 1.0 s |
| `detail_restore: 1.0` | 441 (31%) | **0.0000** | 1.0 s |
| Qwen `mode=full` | ~45% | whole frame drifts | 44–72 s |

`seed`, `steps`, `cfg` and `denoise` have **no meaning in face mode** and are
ignored — the transform is a deterministic keypoint warp. Same input, same
parameters, same output, every time. Output is always the source's own size, with
every pixel outside the face mask bit-identical to the input.

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
