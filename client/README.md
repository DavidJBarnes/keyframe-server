# client — keyframe generation

`client.py` builds LTX conditioning keyframes by **editing source pixels** — never a fresh
text-to-image. Every frame descends from a real photo, which is what keeps identity and
setting stable across a shot.

Backends: your local Qwen-Image-Edit server (default) or three fal.ai endpoints.

---

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

`requests` and `Pillow` are all you need for the local backend. `fal-client` is imported
lazily and only matters for `--model qwen|nb2|pro`; those also need:

```bash
export FAL_KEY="key-id:key-secret"     # fal.ai dashboard -> Keys
```

---

## Generating keyframes

```bash
# local server (default) — start ../server.py first
./client.py -i inputs/wolf/1.png -p "same woman, now waving at someone off to her right" -o inputs/wolf/3.png

# point at the GPU box — --model 3090 is shorthand for its URL
./client.py --model 3090 -i inputs/wolf/1.png -p "..." -o out.png

# or an explicit server
./client.py --server http://otherhost:8189/edit -i inputs/wolf/1.png -p "..." -o out.png

# fal backends
./client.py --model qwen -i inputs/wolf/1.png -p "..." -o out.png
./client.py --model pro  -i out.png -p "..." -o out-fixed.png

# multi-ref: source frame + canonical face reference
./client.py -i inputs/wolf/1.png -i inputs/face_ref.png \
    -p "same woman from image 1 with the face from image 2, standing with arms open" \
    -o inputs/wolf/4.png

# normalize to LTX conditioning size (both dims must be /32)
./client.py -i inputs/wolf/1.png -p "..." -o out.png --size 512x768

# N variants, reproducible seed -> writes out_1.png .. out_4.png
./client.py -i inputs/wolf/1.png -p "..." -o out.png -n 4 --seed 42

# override the server's sampler for a hard edit (local backend only)
./client.py -i inputs/wolf/1.png -p "..." -o out.png --steps 40 --cfg 4.0
```

**`--steps` / `--cfg` default to the server's own values, deliberately.** The right step
count depends on whether the server's Lightning LoRA actually attached — 4 steps when it
did, 40 when it did not — and only the server knows which. Hardcoding 4 client-side would
silently produce 4-step sampling against a server running without Lightning, which yields
garbage with no error. Override only when you know what the server resolved; it logs
`[pipeline] ready: quant=... steps=... cfg=...` at startup.

Measured on 3090.zero (`--quant fp8`, 1103x1426 input): **44 s** at the 4-step default,
**192 s** at `--steps 40 --cfg 4.0`.

| Flag | Default | Notes |
|---|---|---|
| `-i, --image` | — | Input path or URL, repeatable; **first = source frame** |
| `-p, --prompt` | — | Edit instruction |
| `-o, --output` | — | Output path (`.png`) |
| `-n, --num` | `1` | Variants, 1–4; >1 suffixes `_1`.. `_N` |
| `--model` | `local` | `local`, `3090`, `qwen`, `nb2`, `pro` |
| `--server` | from `--model` | Explicit URL, overrides the host `--model` implies |

**Local backends:** `local` → `http://localhost:8188/edit` (env `QWEN_EDIT_URL`),
`3090` → `http://3090.zero:8189/edit` (env `QWEN_EDIT_URL_3090`). Note the **8189** on the
GPU box — 8188 is ComfyUI there, and nothing serves port 80, so the port is not optional.
| `--seed` | none | Honored by the local backend only |
| `--steps` | server's | Sampling steps, local only. Server decides by default |
| `--cfg` | server's | `true_cfg_scale`, local only. Server decides by default |
| `--size` | none | Resize-to-cover + center-crop to `WxH`, /32 enforced |
| `--quiet` | off | Suppress fal queue logs |

Local images are inlined as base64 data URIs — no fal storage upload, no CDN auth. Errors
are scrubbed so base64 payloads never flood the terminal.

---

## Feeding keyframes to LTX

Keyframes exist to be conditioning inputs for LTX multi-frame generation.

**Before conditioning, every keyframe must be:**

- **Edited from the canonical source frame, never chained.** Each edit pass drifts; chaining
  compounds it. One source, N branches — derive every keyframe from the opening frame or the
  character reference, not from the previous keyframe.
- **At exact generation resolution**, divisible by 32, matched aspect. Either `--size 512x768`
  on `client.py`, or
  `convert in.jpeg -resize 512x768^ -gravity center -extent 512x768 kf.png`.
- **Matched in framing and camera distance.** A close-up waypoint followed by a wide one makes
  the model invent a camera move, and it reads as a lurch. Fixed camera in, fixed camera out.
- **Clean.** Watermarks, logos, and artifacts become literal waypoints and pop in and out at
  each keyframe. Crop or inpaint them first.

Starting recipe (the pour shot) — run from the LTX repo root:

```bash
python packages/ltx-pipelines/src/ltx_pipelines/distilled.py \
  --transformer-path ./models/ltx-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
  --text-encoder-path ./models/ltx-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  --video-vae-path ./models/ltx-2.5/vae/ltx-2.5-video-vae-bf16.safetensors \
  --audio-vae-path ./models/ltx-2.5/vae/ltx-2.5-audio-vae-bf16.safetensors \
  --spatial-upsampler-path ./models/ltx-2.5/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  --prompt "A woman in her thirties with shoulder-length blonde hair, side-swept bangs, and a rose-pink turtleneck sweater sits at a wooden table in a warm, cozy restaurant, eating from a plate of grilled vegetables and chicken. Fixed camera, medium close-up. She smiles at the camera, picks up a glass of water with a lemon slice from the table with her left hand, raises it toward her mouth, then tilts it too far and pours the water down the front of her sweater, the wet stain spreading across the pink fabric as she bursts into laughter, eyes closed. Behind her, out of focus, other diners chat and move naturally at their tables. Warm ambient lighting. Sound of restaurant chatter, clinking cutlery, water pouring, and her bright laughter." \
  --image ./inputs/kf1.png 0 1.0 \
  --image ./inputs/kf2.png 41 0.65 \
  --image ./inputs/kf3.png 73 0.7 \
  --image ./inputs/kf4.png 121 0.85 \
  --output-path ./outputs/ltx-pour-$(date +%Y%m%d_%H%M%S).mp4 \
  --width 512 \
  --height 768 \
  --num-frames 121 \
  --frame-rate 24 \
  --seed 42 \
  --offload cpu
```

### The wolf shot — five keyframes, identity-locked

Same recipe against `inputs/wolf/`. `2-swapped.png` through `5-swapped.png` are generated
keyframes, each run back through FaceFusion to re-anchor identity (see **Identity touch-up**).

`1.png` — the original photograph — is deliberately **not** a conditioning frame. Its face
fills 5.5% of frame area against 2.4–2.8% for the generated set, i.e. a camera roughly 2.2x
closer. Mixing that into the waypoint chain makes LTX invent a camera move on the opening
transition. It serves as the canonical face reference for the swap pass instead.

```bash
python packages/ltx-pipelines/src/ltx_pipelines/distilled.py \
  --transformer-path ./models/ltx-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
  --text-encoder-path ./models/ltx-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  --video-vae-path ./models/ltx-2.5/vae/ltx-2.5-video-vae-bf16.safetensors \
  --audio-vae-path ./models/ltx-2.5/vae/ltx-2.5-audio-vae-bf16.safetensors \
  --spatial-upsampler-path ./models/ltx-2.5/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  --prompt "A woman in her fifties with short blonde hair pulled back from her face, dark sunglasses pushed up on top of her head, fair sun-flushed skin and a warm open smile, wearing a bright blue crew-neck t-shirt with a dark mountain-and-pine-tree graphic across the chest and a faded denim skirt, seated at a black metal mesh patio table. Fixed camera, medium shot, no camera movement. She sits with one arm resting on the table beside a plastic cup of water, smiling at the camera, then turns her head to her right and raises her right hand in a bright open-palmed wave. She pushes back from the table and rises to her feet, opening both arms wide in welcome. A blond boy in a navy t-shirt and khaki trousers walks in from the left foreground and into her arms, and she folds him into a close hug, her eyes closing as she laughs. Behind her, out of focus, another diner leans over a nearby table and greenery stirs beyond the porch railing. Warm bright afternoon daylight, soft shade under a wood-plank porch ceiling, white stucco wall and dark teal window frames. Sound of patio chatter, distant birdsong, a chair scraping against brick, and her warm laughter." \
  --image ./inputs/wolf/2-swapped.png 0 1.0 \
  --image ./inputs/wolf/3-swapped.png 33 0.65 \
  --image ./inputs/wolf/4-swapped.png 73 0.7 \
  --image ./inputs/wolf/5-swapped.png 121 0.9 \
  --output-path ./outputs/ltx-wolf-$(date +%Y%m%d_%H%M%S).mp4 \
  --width 512 \
  --height 768 \
  --num-frames 121 \
  --frame-rate 24 \
  --seed 42 \
  --offload cpu
```

**`--image PATH FRAME WEIGHT`** — where the keyframe lands and how hard it pulls. Multi-frame
injection is a **constraint, not an identity source**: LTX treats each image as a literal
waypoint and interpolates between them. It does not learn the character from frame 0 and
carry it forward. Identity is your job upstream; choreography is LTX's.

**Indices must land on the latent temporal grid — `0`, then `8n+1`:**

```
0, 1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121
```

Off-grid indices get snapped to the nearest grid point, so the guide frame lands somewhere
you didn't put it — which presents as "the keyframe isn't working." Opener goes at `0`, not
`1`. Note `121` **is** on-grid and is the correct terminal index for a 121-frame clip.

**Weights: `1.0` opener, `0.6–0.7` interior, `0.85–0.9` terminal.** Full strength on an
interior waypoint snaps the video onto it visibly; lower strength lets the model blend
through. The terminal frame gets more so the clip lands its final beat. Interior strengths
are also what keep the background alive — when keyframes share a near-identical background,
high strengths freeze the extras into mannequins. Pair low interiors with prompted background
motion ("another diner leans over a nearby table").

**Frame budget follows action duration, weighted to the payoff.** A sit or stand needs
~1.5–2s (36–48 frames at 24fps). The wolf spacing gives 33 frames to settle-and-wave, 40 to
stand, and 48 to the hug — the beat that matters gets the most room.

**One continuous physical action per transition.** Reach, lift, turn, stand — single-stage
limb motions interpolate cleanly. Multi-stage actions morph or produce spaghetti limbs.
Discrete state changes (shoes on → barefoot) have no intermediate at all and belong on
either side of a hard cut, or need a keyframe depicting the mid-state.

**Introduce processes in the keyframes; don't ask the model to invent them.** Anything
entering frame between waypoints must be motivated in the prompt, and is more reliable
still if a keyframe already shows it partway in. The wolf `73 → 121` gap asks for exactly
this — the boy is absent at 73 and embraced at 121 — so it's the transition most likely to
need a fifth keyframe around `97`.

**When keyframe and prompt disagree, the keyframe wins.** Fix the image; don't fight it with
prompt engineering. Where the scene offers a competing target, name it away explicitly
("spilling onto her sweater, missing the plate entirely").

**Iterate with a fixed seed, one variable at a time** — one keyframe, one strength, or one
prompt line per run — otherwise you can't tell what steered the result. Single bad seams are
a `RetakePipeline` pass over that region, not a re-roll. On OOM, reach for
`--quantization fp8-cast` before cutting resolution or frames.

---

## Identity touch-up (FaceFusion)

Generated keyframes drift off-identity. Fix by swapping the original face back in — see
`../../wanly/experiment/faceswap.sh` for the locked video recipe. For stills, on the 3090:

```bash
ssh david@3090.zero
cd ~/projects/facefusion
~/miniconda3/envs/facefusion/bin/python facefusion.py headless-run \
  -s 1.png -t 3.png -o 3-swapped.png \
  --processors face_swapper face_enhancer \
  --face-swapper-model inswapper_128 --face-swapper-pixel-boost 512x512 \
  --face-enhancer-model gfpgan_1.4 \
  --face-mask-types box occlusion \
  --face-selector-mode reference --reference-face-distance 0.6 \
  --face-selector-order large-small --reference-face-position 0 \
  --execution-providers cuda
```

- **`--face-selector-order large-small`** with position `0` targets the biggest face in
  frame. Prefer this over `left-right` whenever bystanders sit near the left edge — the
  wolf frames have exactly that, and left-right ordering grabs the wrong person.
- **`--reference-face-distance 0.6`** isolates one identity. At `1.0` every face in frame
  matches the reference and they all get swapped.
- **`box occlusion` masks** keep the swap clean where a hand or arm crosses the face.
- Holds identity only within **~30–40° of frontal**. Past that the landmark match breaks
  down — use it for identity touch-up, not to rescue extreme head turns.

Identity work happens at **both** ends. Swapping the keyframes before conditioning makes the
waypoints agree with each other on who the subject is; a second pass over the finished MP4
(`../../wanly/experiment/faceswap.sh`) cleans identity in the interpolated frames LTX invents
between them. The canonical reference — `1.png` here — is the source for both and is never
itself a conditioning frame.

---

## Output

Generated media is gitignored repo-wide (`*.png`, `*.jpg`, `*.jpeg`, `*.mp4` plus
`inputs/`, `keyframes/`, `outputs/`), with a `pre-commit` hook backing it up. Nothing in
`inputs/` or `outputs/` will ever be committed. See the root README.
