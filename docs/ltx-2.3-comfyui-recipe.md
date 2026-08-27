# LTX-2.3 via ComfyUI — the working recipe

Status: **working**. First good render 2026-08-27 — portrait 704x1280, 241
frames, 10s, ~256s on the 3090, peak 22.6 GB of 24.5 GB.

This supersedes the CLI approach in [`ltx-keyframe-recipe.md`](ltx-keyframe-recipe.md)
for anything involving LoRAs or matching community output. That document's
*conditioning* rules still hold — they are model behaviour, not interface — but
its command shapes do not.

---

## 1. Why ComfyUI and not the CLI

The CLI cannot reproduce the workflows the community generates with. This is a
**capability gap, not a tuning gap**.

`ltx-pipelines`' argparse exposes no `--sampler`, `--scheduler`, `--shift`,
`--sigmas` or `--denoise` — grepped for each, zero hits — and hardcodes a sampler
per pipeline (`LTX_2_3_HQ`'s own source comment says "Res2s sampler"). The
reference workflows set:

| | workflow | CLI |
|---|---|---|
| sampler | `euler_cfg_pp` / `euler_ancestral_cfg_pp` | not settable |
| scheduler | `linear_quadratic` | not settable |
| shift | `LTXVScheduler` 2.05 / base 0.95 / terminal 0.1 | not settable |
| sigmas | explicit `ManualSigmas` per pass | not settable |

Six CLI renders tuning the four parameters it *does* expose (distill strength,
CFG, STG, steps) could not close the gap. The first ComfyUI render did.

The CLI is a **reduced wrapper** over `ltx-core`; the nodes expose the rest.

---

## 2. The stack

### Container

`ltx-comfy`, built from [`docker/Dockerfile.ltx`](../docker/Dockerfile.ltx).
15.5 GB image, ComfyUI on **port 8191**.

```bash
docker run -d --name ltx-comfy --gpus all --memory 56g --restart unless-stopped \
  -v /home/david/LTX-2/models:/workspace/models:ro \
  -v /home/david/keyframe-server/docker/extra_model_paths.ltx.yaml:/opt/extra_model_paths.yaml:ro \
  -p 8191:8188 ltx-comfy:latest
```

Deliberately **separate** from `keyframe-server`. LTX-2.3 wants essentially the
whole card and Qwen holds ~20 GB once loaded, so they cannot be co-resident on
24 GB regardless — and a broken node install here cannot take the working
Qwen/LivePortrait image down with it.

torch 2.12.1+cu130. Models are bind-mounted, never baked: the set is ~126 GB.

### Node packs

| pack | why |
|---|---|
| `Lightricks/ComfyUI-LTXVideo` | `LTXVScheduler`, `LTXVPreprocess`, `LTXVImgToVideoInplace`, `LTXVLatentUpsampler`, `ManualSigmas`, `GuiderParameters`, `MultimodalGuider` |
| `kijai/ComfyUI-KJNodes` | `ImageResizeKJv2`, `LazySwitchKJ`, `SimpleCalculatorKJ`, Sage-attention patches, `Set`/`GetNode` |
| `rgthree/rgthree-comfy` | Power Lora Loader, Fast Groups Bypasser |
| `Kosinkadink/ComfyUI-VideoHelperSuite` | `VHS_VideoCombine` (mp4 + audio muxing) |
| `city96/ComfyUI-GGUF` | GGUF loaders, for the quantised workflow variants |

`LTXAVTextEncoderLoader`, `LTXVAudioVAELoader`, `LatentUpscaleModelLoader` and
`CheckpointLoaderSimple` are **ComfyUI core**, not custom nodes.

### Model paths

[`docker/extra_model_paths.ltx.yaml`](../docker/extra_model_paths.ltx.yaml).

**Folder keys are exact.** `LatentUpscaleModelLoader` reads
`latent_upscale_models`, which is a *different key* from the `upscale_models`
ordinary ESRGAN upscalers use. A wrong mapping surfaces as an **empty dropdown,
not an error** — invisible until a graph is submitted and validation says
`not in []`.

---

## 3. Models on disk

Under `~/LTX-2/models/`. Total ~126 GB.

### LTX-2.3 (`Lightricks/LTX-2.3`, ungated)

| file | size | role |
|---|---|---|
| `ltx-2.3/diffusion_models/ltx-2.3-22b-dev.safetensors` | 46.15 GB | **the base the content LoRAs name** |
| `ltx-2.3/diffusion_models/ltx-2.3-22b-distilled-1.1.safetensors` | 46.15 GB | fast few-step path |
| `ltx-2.3/loras/ltx-2.3-22b-distilled-lora-384-1.1.safetensors` | 7.61 GB | LTX's own distilled LoRA, stage-2 refinement |
| `ltx-2.3/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | 1.00 GB | 2x latent upscale |
| `ltx-2.3/latent_upscale_models/ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors` | 1.09 GB | alternative |
| `ltx-2.3/latent_upscale_models/ltx-2.3-temporal-upscaler-x2-1.0.safetensors` | 0.26 GB | not yet used |

2.3 ships **monoliths** — one file carrying transformer, both VAEs and the text
projection. That is why `CheckpointLoaderSimple`, `LTXAVTextEncoderLoader` and
`LTXVAudioVAELoader` all take the *same* file.

**Skipped deliberately:** `ltx-2.3-22b-distilled.safetensors` and
`ltx-2.3-22b-distilled-lora-384.safetensors` — pre-1.1 revisions, 53.8 GB of
superseded duplicates.

### Text encoder (`Comfy-Org/ltx-2`, license-gated)

| file | size |
|---|---|
| `ltx-2.3/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors` | 13.21 GB |

**Two Gemma formats exist and they are not interchangeable.** The CLI's
`--gemma-root` wants a 5-shard HF repo directory; ComfyUI's
`LTXAVTextEncoderLoader` wants **one file** from `text_encoders/`. Pointing it at
a shard loads a fifth of the model. Both are on disk; only the single file is
used by ComfyUI.

fp8 on a 3090 (sm_86, Ampere) gives **no speed benefit** — native FP8 tensor
cores arrived with Ada. ComfyUI still stores at fp8 and upcasts per-layer, so the
**VRAM** saving is real, which is the reason to prefer it over the 24 GB bf16
build on a 24 GB card.

### Content LoRAs (`~/LTX-2/models/loras/`)

All record an LTX-2.3-era base. All fuse **100%** into both the 2.3 dev monolith
and the 2.5 transformer — key overlap cannot distinguish the two, only the
weights differ, so fusion coverage proves a LoRA *loaded*, never that it is the
right base.

| file | size | recorded base | triggers |
|---|---|---|---|
| `DR34ML4Y_LT3X_V3.safetensors` | 1.94 GB | (none in metadata; Civitai says LTXV 2.3) | `m15510n4ry` `bl0wj0b` `d0ubl3_bj` `d0gg1e` `c0wg1rl` `r3v3rs3_c0wg1rl` |
| `LTX2-i2v-SexThrust.safetensors` | 0.62 GB | `ltx2` | — |
| `SexGod_Nudity_LTX23_v2_0.safetensors` | 1.23 GB | `ltx2` | `LTXNUDES` |
| `Sulphur_LTX 2.3_better _NSFW_motion.safetensors` | 0.65 GB | `sulphur_dev_bf16.safetensors` | — |
| `2ltsway-breastsway.comfy.safetensors` | 0.20 GB | `ltx-2.3-22b-dev.safetensors` | — |
| `LTX 2.3 _60 FPS_Buttery_Smooth Motion_LoRa.safetensors` | 1.35 GB | (none) | — |

Trigger words are **not** in the LoRA metadata for DR34ML4Y — they come from the
Civitai model page (`/api/v1/models/1811313`). Put the trigger at the **front**
of the prompt.

The DR34ML4Y author's guidance: *"Performs BEST WITHOUT DISTILLATION. The
distillation lora and checkpoint actively fight the nsfw training and account for
body horror… we recommend the full dev checkpoint with between 0.25-0.35 distill
strength."* Note that means distillation **turned down**, not removed — the
reference workflow's own two passes use 0.3 and 0.6.

---

## 4. The workflow catalogue

[`RuneXX/LTX-2.3-Workflows`](https://huggingface.co/RuneXX/LTX-2.3-Workflows) —
~80 workflows. This is the highest-value find of the whole exercise: it is the
practical documentation LTX does not publish.

Local copies live in [`../ltx-runner/workflows/`](../ltx-runner/workflows/).

### Directly relevant to this project

| workflow | why it matters |
|---|---|
| `LTX-2.3_-_I2V_T2V_Basic_for_checkpoint_models.json` | **the one we run.** `CheckpointLoaderSimple` against a monolith = our bf16 weights |
| `LTX-2.3_-_I2V_T2V_Dev_Full-Steps.json` | dev model, full steps — source of the sampler/scheduler/sigma values |
| `First-Last-Frame/LTX-2.3_-_FLF2V_First-Last-Frame.json` | two-keyframe conditioning |
| `First-Last-Frame/LTX-2.3_-_FML2V_First_Middle_Last_Frame_guider.json` | **three keyframes via a guider — closest to the storyboard model** |
| `Movie-Maker/LTX-2.3_-_I2V_Short-Story_PromptRelay-Timeline_multi-image_multi-sequence.json` | **a storyboard renderer in all but name**: multi-image, multi-sequence, timeline-driven |
| `Movie-Maker/Prompt-Relay-Dev-Model/..._Dev-model.json` | the same on the dev checkpoint |

### The rest, by category

- **Long video** (`Long-Video-Experimental/`) — custom audio, looping, single-pass loop
- **Video-to-video** (`Video-2-Video/`) — extend any video, inpainting (SAM2/SAM3 masking), retake a section, outpaint, upscale, viewpoint change, watermark removal, clean-plate people removal, HDR
- **Shot transitions** (`Video-2-Video/Shot-to-Shot-Transition/`) — three approaches
- **Character reference** (`Multi-ref-character-sheet/`) — multi-subject reference, face-ID LoRAs, character-sheet helpers for Flux/Qwen/Krea/Ideogram
- **Audio** (`Custom-Audio/`, `Talking-Avatar-TTS/`) — voice cloning via Fish-Audio / OmniVoice / Qwen-TTS, dubbing, lip-sync to any video, Foley
- **Control** (`Control-reference/`) — body-motion transfer (RealisDance, DWPose, SDPose), camera-motion transfer
- **Music video** (`Music-Video-Creator/`) — multi-scene, per-segment saves, AceStep music generation
- **3-pass experimental** (`3-Pass-Experimental/`)

### Fetching them

```bash
curl -s "https://huggingface.co/api/models/RuneXX/LTX-2.3-Workflows" \
  | python3 -c "import json,sys;[print(s['rfilename']) for s in json.load(sys.stdin)['siblings']]"
curl -sL -o wf.json "https://huggingface.co/RuneXX/LTX-2.3-Workflows/resolve/main/<path>"
```

---

## 5. Running a render

```bash
curl -s http://3090.zero:8191/object_info > objinfo.json
python3 ltx-runner/workflows/ui2api.py <workflow>.json api.json objinfo.json
python3 ltx-runner/workflows/patch_graph.py          # image, prompt, LoRA, orientation
curl -X POST http://3090.zero:8191/prompt -H 'Content-Type: application/json' \
     -d "{\"prompt\": $(cat ltx23_api_patched.json)}"
```

Upload the source image first (`POST /upload/image`), poll `GET /history/<id>`,
and pull the mp4 from `/app/ComfyUI/output/`.

A downloaded workflow is **UI format** and cannot be POSTed — the browser
normally converts it. See
[`../ltx-runner/workflows/README.md`](../ltx-runner/workflows/README.md) for the
six conversion traps, each of which produces a *silently wrong graph rather than
an error*.

### Three corrections every downloaded workflow needs

1. **Prompt goes in the POSITIVE encoder.** Read `LTXVConditioning`'s
   `positive`/`negative` links — do not guess. Guessing by "longest string wins"
   put an entire prompt in the *negative*, so the model was told to avoid exactly
   what was asked for while the positive kept the author's placeholder. The
   result was a slow camera drift over a still subject.
2. **Orientation is the author's, not yours.** These ship landscape
   (WIDTH 1280, HEIGHT 736). A portrait source needs those constants swapped.
3. **The content LoRA is not in the file.** rgthree's Power Lora Loader ships
   empty and holds LoRAs in dynamic widgets the API schema does not declare, so
   it cannot be filled over `/prompt`. Splice an explicit `LoraLoaderModelOnly`
   in after it.

---

## 6. Constraints that still apply

These are model behaviour and carry over from the CLI work:

- **Keyframe indices must be `0` or `1+8k`.** `SpatioTemporalScaleFactors` is
  `(time=8, …)` and the encoder is causal, so latent boundaries land at
  `0, 1, 9, 17…`. Index `0` is special twice over: it is the only legal index
  below 1, and `helpers.py` routes it to `VideoConditionByLatentIndex` — the true
  first-frame anchor — while every other index becomes a keyframe token.
- **Resolution divisible by 64** for two-stage pipelines (`/32` is one-stage
  only). Verified via `assert_resolution(..., is_two_stage=True)`.
- **`num_frames` floors to `k*8+1`**, matching `snap_frames_to_grid`.
- **Keyframes are waypoints, not an identity source.** LTX interpolates between
  them and takes the shortest path; two similar adjacent waypoints mean "hold
  still". A single keyframe leaves everything after it unconstrained, which is
  where identity drift comes from — not a model defect.

## 7. Judging output

**Do not rank renders by frame-to-frame smoothness.** A near-static clip is
maximally smooth, so a "jerk" metric scores an empty render *better* than a
correct one — which is exactly what happened here, and the empty render was
reported as the best result on that basis. Watch the video.
