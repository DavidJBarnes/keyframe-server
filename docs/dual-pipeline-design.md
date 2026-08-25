# Dual-pipeline design: Qwen (macro) + LivePortrait (micro)

Status: **implemented in `server.py`**, container work outstanding.
Source of truth: `PowerHouseMan/ComfyUI-AdvancedLivePortrait@main/nodes.py`, read directly
(the README documents almost nothing). Line references are to that file.

## 1. Why

Measured on 2026-08-25: every Qwen pass halves skin texture (Laplacian variance
1015 -> ~450) regardless of denoise, and de-ages the subject. Denoise gives no
gradation. This is structural — a diffusion edit model regenerates pixels; it does
not preserve them.

LivePortrait warps the source through implicit keypoints and composites the result
back into the original frame. **Identity and age are preserved** — that is the
failure mode that mattered, and it holds.

**Texture is not preserved by construction.** An earlier draft of this document
claimed it was; measurement says otherwise. The node decodes through a fixed
256x256 bottleneck, so it softens the *entire* face crop, including the parts it
did not move. See §7.

It cannot change garments, scenes, or composition. Qwen stays for those.

## 2. What ExpressionEditor actually is

`nodes.py:833`. Confirmed from source:

- **Only `src_image` is needed.** `sample_image` (a driving photo) is optional.
  So the node can be driven **purely by numbers** — no reference image required.
- **Returns a full-frame composite.** `nodes.py:949-951` warps the edited crop back
  with `cv2.warpAffine` and alpha-blends via `mask_ori` against the untouched source.
  Everything outside the face mask is bit-identical to the input.
- Outputs `(IMAGE, EDITOR_LINK, EXP_DATA)`.

### Parameter surface (`nodes.py:846-865`)

| param | range | controls |
|---|---|---|
| `rotate_pitch` / `rotate_yaw` / `rotate_roll` | -20 .. 20 | head orientation |
| `blink` | -20 .. 5 | eyelid open/close |
| `wink` | 0 .. 25 | one eye |
| `eyebrow` | -10 .. 15 | brow raise |
| `pupil_x` / `pupil_y` | -15 .. 15 | gaze direction |
| `aaa` | -30 .. 120 | jaw open |
| `eee` / `woo` | -20 .. 15 | mouth shape |
| `smile` | -0.3 .. 1.3 | smile |
| `crop_factor` | float | face crop padding |
| `src_ratio` | 0 .. 1 | how much of the source expression to keep |

**This is the magnitude control we could not get from denoise.** `smile: 0.3` and
`smile: 0.9` are different amounts of the same edit, deterministically.

### Driving-image mode

Pass `sample_image` plus `sample_parts` in
`{OnlyExpression, OnlyRotation, OnlyMouth, OnlyEyes, All}` and `sample_ratio`
(-0.2 .. 1.2) to transfer an expression from a reference photo at a scalar
intensity. Maps cleanly onto our existing `image_urls[1]`.

## 2b. Measured, 2026-08-25 (richmond kf1, 214x292 face in a 512x768 frame)

| | face texture | % of source | drift outside crop | time |
|---|---|---|---|---|
| source | 1405.1 | — | — | — |
| Qwen `mode=full` | ~450 (different frame) | 45% | whole frame drifts | 44-72 s |
| LivePortrait raw | 226.9 | 16% | **0.0000** | **1.0 s** |
| LivePortrait + detail restore | 440.6 | 31% | **0.0000** | 1.0 s |

Three things this establishes:

1. **Containment is exact.** Mean absolute drift outside the warp region is
   `0.0000` — not "small", zero. Qwen cannot offer that at any denoise.
2. **Identity and age hold.** Visual check on face crops at native resolution:
   no de-aging, no smoothing away of nasolabial folds or under-eye lines. This is
   the thing ArcFace failed to catch on the Qwen chain and it genuinely holds here.
3. **Raw output is soft, and `crop_factor` cannot fix it.** Texture scales
   inversely with crop_factor (1.6 -> 16%, 2.0 -> 11%, 2.5 -> 7%) because a larger
   crop means more downsampling into the fixed 256x256 decode. The node clamps
   crop_factor to 1.5-2.5, so 1.5 is already the sharp end of the available range.

`restore_detail()` in `server.py` answers (3): blend on `|edited - source|` so the
node's pixels are kept only where it actually moved something, and the source's
own pixels come back everywhere else. Every output pixel comes from one of the
two real images, so it cannot invent detail or shift age. Measured 16% -> 31%,
and visibly it recovers forehead lines, hair strands and cheek texture while
leaving the expression intact.

The residual gap to 100% is real and is concentrated where geometry moved
(mouth, cheeks on a smile). That is inherent to the 256x256 decode.

## 3. Models required

Small. Auto-downloaded at first use (`nodes.py:106`), which we should **pre-bake**
rather than inherit as a container cold-start network dependency.

| file | source | goes in |
|---|---|---|
| `appearance_feature_extractor.safetensors` | `Kijai/LivePortrait_safetensors` | `models/liveportrait/` |
| `motion_extractor.safetensors` | same | same |
| `warping_module.safetensors` | same | same |
| `spade_generator.safetensors` | same | same |
| `stitching_retargeting_module.safetensors` | same | same |
| `face_yolov8n.pt` | `Bingsu/adetailer` | `models/ultralytics/` |

~500 MB total, against the 28 GB Qwen checkpoint. Both resolve through
`folder_paths`, so `docker/extra_model_paths.yaml` can map them like `checkpoints`.

## 4. Gotchas found in the source

1. **Do not read the node's UI preview.** `nodes.py:955-957` saves `crop_out` — the
   *face crop only* — as the preview image. The full-frame composite is `result[0]`.
   Our adapter must wire `SaveImage` off the IMAGE output and fetch that, or it will
   silently return face crops instead of keyframes.
2. **Face selection differs from ours.** `detect_face` (`nodes.py:227`) picks the box
   closest to **horizontal centre**; our YuNet path picks **largest**. For multi-person
   frames these disagree. Ours matches the FaceFusion `large-small` choice we settled on.
3. **`ultralytics` is AGPL-3.0** and is a hard dependency of the detector — no
   alternative path exists in the code. Flag for any commercial deployment.
4. **`tyro==0.8.5` is pinned** and `ultralytics` drags in its own torch/opencv
   constraints. Needs a build test against our torch 2.12.1 before trusting it.
5. **`rotate_yaw` is negated on entry** (`nodes.py:872`). Sign convention is the
   node's, not the model's.

## 5. What this changes in `server.py`

`build_workflow` (`server.py:205`) becomes two branches sharing upload/fetch:

- `mode=full` -> current Qwen graph, unchanged.
- `mode=face`  -> `LoadImage -> ExpressionEditor -> SaveImage`. No checkpoint, no
  KSampler, no VAE, no prompt encoder.

Becomes dead code on the face path:

- `composite_face` (`server.py:181`) — the node does its own stitched composite,
  and does it better (mask-based, not a rectangular feather).
- `face_crop_box` (`server.py:171`) — replaced by `crop_factor`.

`detect_face` (`server.py:157`) is worth **keeping** as a cheap preflight so a
faceless input returns a clean 422 instead of an opaque ComfyUI failure.

Also gone on the face path: `seed`, `steps`, `cfg`, `denoise`, `negative_prompt`.
The transform is deterministic — same input, same params, same output, every time.

## 6. VRAM

3090 currently sits at 20.8 / 24 GB with the Qwen checkpoint resident. LivePortrait
adds ~500 MB, so co-residency should fit. If alternating modes causes thrash, the
`/free` endpoint we already call is the lever.

## 7. Open decisions

### D1 — How a caller expresses a facial edit — DECIDED

Numeric primary, prompt sugar, driving image optional. All three land on the same
node inputs.

```jsonc
// exact — the honest contract, and what the web app's sliders will send
{"mode": "face", "image_urls": ["data:..."],
 "expression": {"smile": 0.4, "blink": -3, "rotate_yaw": 5}}

// prompt — sugar over a small fixed vocabulary
{"mode": "face", "image_urls": ["data:..."], "prompt": "soften her smile, look left"}

// driving image — copy an expression off a reference at a scalar intensity
{"mode": "face", "image_urls": ["src", "driver"],
 "sample_ratio": 0.7, "sample_parts": "OnlyExpression"}
```

`expression` wins when supplied. Otherwise the prompt is swept against `_LEXICON`
in `server.py`; the largest magnitude wins per axis, and one global intensity
adverb (`slight`/`soften` -> 0.5x, `big`/`very` -> 1.5x) scales the result before
clamping. A face-mode request with none of the three returns 422 listing the
recognised terms rather than silently returning the input unchanged.

The vocabulary is deliberately small and lives in one list. It is sugar: anything
it cannot say, `expression` can.

### D2 — `mode` stays caller-specified — DECIDED

Inferring it from the prompt would put product judgement inside the service.
The caller knows whether it is changing a garment or a glance.

### D3 — Chaining — OPEN, needs measurement

Face mode should chain far past the measured 2-step limit, since untouched pixels
stay bit-identical and the warped ones are resampled from the source rather than
regenerated. **Unverified.** Needs the 5-step chain test scored on *skin texture*
(Laplacian variance) and visible age, not ArcFace cosine — ArcFace read 0.85 on a
visibly de-aged face and is why the first chain results were wrong.

## 8. Remaining work

- [ ] Bake the custom node, its deps, and the six model files into `docker/Dockerfile`
- [ ] Set `YOLO_CONFIG_DIR=/tmp/ultralytics` (ultralytics warns on a read-only HOME)
- [ ] Rebuild, redeploy, verify `mode=full` is byte-unchanged
- [ ] Run D3's chain test
- [ ] Decide on `ultralytics` AGPL before any commercial deployment
