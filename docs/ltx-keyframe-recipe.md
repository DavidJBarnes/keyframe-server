# LTX-2.5 Multi-Keyframe Conditioning Recipe

> **Superseded for execution.** The command shapes here drive LTX's own CLI,
> which exposes no sampler, scheduler, shift or sigma control and therefore
> cannot reproduce community workflows or use content LoRAs well. For anything
> involving LoRAs or matching published output, see
> [ltx-2.3-comfyui-recipe.md](ltx-2.3-comfyui-recipe.md).
>
> The **conditioning** rules below still hold — they are model behaviour, not
> interface. One correction: §2 says dimensions divisible by 32; that is the
> one-stage rule. Two-stage pipelines require **64**
> (`assert_resolution(..., is_two_stage=True)`).

The core mental model: multi-frame injection is a **constraint, not an identity source**. LTX does not learn the character from frame 0 and propagate it — it treats every conditioned image as a literal waypoint the video must pass through and interpolates motion between them. Identity consistency is your job, upstream; LTX's job is choreography between beats you have already made consistent.

## 1. Keyframe production

**Generate keyframes with an image *editing* model, never fresh text-to-image.** Qwen-Image-Edit / Nano Banana condition on the source pixels via a vision encoder, so identity survives the edit. Fresh generation re-imagines the person from text and drifts.

**Always edit from the canonical source frame — never chain edits.** Each edit pass introduces slight drift; chained edits compound it. One source, N branches: every keyframe derives from the original opening frame (or canonical character reference), not from the previous keyframe.

**Clean frames before conditioning.** Watermarks, logos, and artifacts in a conditioned frame become literal waypoints — they pop in and out at each keyframe. Crop or inpaint them out first.

**Reserve one canonical face reference for the faceswap post-pass** (ReActor + GPEN). It is not a conditioning frame; it's the identity anchor applied after generation.

## 2. Keyframe geometry

**All conditioning frames at the exact generation resolution**, dimensions divisible by 32, matched aspect ratio. Normalize with a resize+center-crop pass before conditioning.

**Matched framing and camera distance across keyframes.** A portrait-crop waypoint followed by a full-body waypoint forces the model to invent a camera move, which reads as a lurch. Fixed camera in, fixed camera out.

**Grid-align every frame index to 8n+1** (0 for the opener, then 41, 73, 97, 121...). Off-grid indices get snapped to the latent temporal grid, landing the guide frames away from where you placed it — which looks like "the keyframe isn't working."

**Opening frame at index 0, not 1.** Index 0 is the true first-frame anchor.

## 3. Strength ramp

**1.0 on the opener, ~0.6–0.7 on interior waypoints, ~0.85–0.9 on the terminal frame.** Full strength on interior keyframes produces a visible snap as the video is forced onto the waypoint; lower strength lets the model blend into and out of it. The terminal frame gets more strength so the clip lands on the intended final beat/expression.

**Interior strengths are also what let the background live.** If keyframes share a pixel-identical background, high strengths freeze the extras into mannequins. Lower interior strengths + prompting background motion ("diners chat and move behind her") gives the model room to animate between waypoints.

## 4. Transition design

**One continuous physical action per transition.** Reach, lift, tilt, sit, turn, walk — single-stage limb motions interpolate cleanly. Multi-stage actions (stand → climb onto table → fold into cross-legged pose) produce morphs, teleports, or spaghetti limbs.

**Discrete state changes are cut boundaries, not transitions.** Shoes on → barefoot has no intermediate; LTX will make the shoes evaporate mid-clip. Break the chain: end clip A, start clip B from a fresh keyframe, hard cut. Cuts are a filmmaking tool, not a failure. Alternatively, insert an intermediate keyframe depicting the mid-state.

**Introduce processes in the keyframes; don't ask the model to invent them.** A pour that already exists as a stream in two consecutive keyframes is a continuation task (reliable); water materializing from nothing is an invention task (unreliable). Same for anything entering frame: if an object appears between keyframes, motivate it in the prompt ("she picks up a glass from the table").

**Frame budget follows action duration, weighted toward the beat that matters.** A sit-down is ~1.5–2s → 49–97 frames minimum at 24fps. Space indices so the payoff beat gets the most frames (e.g. 0 / 41 / 73 / 121 concentrates time on the final action).

## 5. Keyframe/prompt agreement — the highest-leverage rule

**When a keyframe and the prompt disagree, the keyframe wins.** If the keyframe's geometry shows water falling toward the plate, the model pours on the plate no matter what the prompt says. Fix the keyframe (edit it so the visual evidence matches the intent), don't fight it with prompt engineering.

**Name competing attractors explicitly.** When the scene contains a plausible alternative target (a plate under a tilting glass), negative-space phrasing helps: "spilling onto her sweater, missing the plate entirely."

**Pull the terminal keyframe closer to the action** when the model improvises in the gap — fewer unconstrained frames after the last interior waypoint means less room to go off-script.

## 6. Prompt structure

Fixed character block (50–80 words, identical wording every generation) → camera spec → the action as a continuous sequence → background life → lighting → **audio spec**. LTX-2.5 generates sound natively; the audio sentence ("restaurant chatter, clinking cutlery, water pouring, her bright laughter") is the soundtrack spec, and sounds sync to their visual events.

## 7. Iteration discipline

**Same-seed A/B.** Keep the seed fixed; change exactly one variable (one keyframe, one strength, one prompt line) per run. That isolates what's steering.

**Fix segments with Retake, not full re-rolls.** A briefly-detached water stream or one bad seam is a `RetakePipeline` pass over that time region.

**On OOM: `--quantization fp8-cast` before dropping resolution or frames.** On-the-fly downcast, minimal quality cost.

## Reference command shape

```bash
python packages/ltx-pipelines/src/ltx_pipelines/distilled.py \
  --transformer-path .../ltx-2.5-22b-distilled-transformer-bf16.safetensors \
  --text-encoder-path .../gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  --video-vae-path .../ltx-2.5-video-vae-bf16.safetensors \
  --audio-vae-path .../ltx-2.5-audio-vae-bf16.safetensors \
  --spatial-upsampler-path .../ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  --prompt "<character block> <camera> <continuous action> <background life> <lighting> <audio>" \
  --image kf1.png 0 1.0 \
  --image kf2.png 41 0.65 \
  --image kf3.png 73 0.7 \
  --image kf4.png 121 0.85 \
  --width 512 --height 768 \
  --num-frames 121 --frame-rate 24 \
  --seed 42 --offload cpu
```

Pre-flight: `convert in.jpeg -resize 512x768^ -gravity center -extent 512x768 kf.png` per keyframe.
