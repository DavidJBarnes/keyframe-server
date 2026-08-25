# TODO — facial edit control

Open work on making small, controlled facial adjustments while preserving identity.

**Two distinct modes are wanted (David, 2026-08-25), and they need different
machinery:**

* **Whole-scene / fundamental changes — already working well.** Garment swaps, reframes,
  posture changes, multi-reference composition. Keep this path as-is; it is the one the
  LTX keyframe workflow depends on and it is fast (~6-12s) and identity-faithful at
  native aspect. Nothing below should regress it.
* **Micro adjustments, mostly facial — the gap.** This is what the items below address.

Whole-frame regeneration is the wrong architecture for the *second* mode. Every edit currently redraws all 832x1040 pixels to change
something occupying ~4% of them, which is why control is poor and why unrelated parts of
the image are free to drift. **Localised editing (items 2 and 3) is the approach, not the
fallback**, and it also makes the checkpoint choice far less important — v19 vs v23 vs
base 2511 matters when rebuilding a scene, much less when handed a tight face crop.

Suggested order for the facial work: **3, then 2, then 1** (1 stays cheap and worth
trying at any point). All three are *additive* — a face-crop path or a mask parameter
sits alongside the existing whole-frame endpoint rather than replacing it.

**These belong SERVER-side, not in the client (David, 2026-08-25).** The CLI is a
stepping stone: the real consumer will be a web app making a series of calls to assemble
a storyboard for LTX. Any image logic that lives in `client.py` would have to be
reimplemented by every future caller and would drift. The server owns the capability and
clients stay thin.

That reframes what this service is: not a wrapper around ComfyUI, but **the keyframe
service** — it knows how to produce a conditioning frame, whatever that takes internally.

API shape:

```
POST /edit { "mode": "full" | "face", "prompt": ..., "image_urls": [...],
             "face_pad": 1.6, "seed": ..., "denoise": ... }
```

`mode: face` detects the largest face in image_urls[0], crops with padding, edits the
crop, then composites back — returning an image of the ORIGINAL dimensions. `mode: full`
is today's behaviour, unchanged.

Face detection has to run inside the container. OpenCV cascades are free (ComfyUI already
pulls opencv) but unreliable on angled or occluded faces, which the wolf and hailey
sources are full of. insightface buffalo_l is much better and the weights already exist on
the box at ~/.insightface from the FaceFusion work. Start with OpenCV; return the detected
box in the response so failures are visible rather than mysterious.

**Context.** Identity preservation is solid (that was an aspect-ratio bug, since fixed).
What does not work is *controlling the magnitude* of a facial change. `--denoise` was
added for this and barely moves the needle: at 4 steps, denoise 0.15 changes ~74% as
much as denoise 1.0 (mean abs diff 3.01 vs 4.08); at 12 steps, 3.25 vs 5.37. Raising
steps did not restore gradation, so it is not sampler resolution.

Measured on `client/inputs/test/k2_crop.png` (832x1040), prompt "Make her smile slightly
wider...", seed 88, checkpoint Qwen-Rapid-AIO-NSFW-v23.

---

## 1. Try a non-ancestral sampler  — TESTED 2026-08-25, DOES NOT HELP

**Result: `euler` is worse than `euler_ancestral` on every micro measure.** The
noise floor rose on both checkpoints (v23 9.05 -> 11.03) and `macro_scene` face drift
nearly tripled (8.80 -> 24.63) when only the background should have changed.

The reasoning was wrong: ancestral noise fights preservation on a normal img2img model,
but this workflow samples from an EMPTY latent at denoise 1.0 — there is no source latent
for the noise to disturb. Lightning is distilled for 4 steps and tuned for its
recommended sampler, so swapping degrades quality, which shows up as drift.

Full 4-way (v19/v23 x euler_ancestral/euler) in `client/battery/`. Best configuration is
**v23 + euler_ancestral**, which is what we started with.

<details><summary>original reasoning, kept for the record</summary>

`euler_ancestral` **injects fresh noise at every step**, which directly fights latent
preservation: you keep 85% of the source at denoise 0.15 and the sampler re-randomises
as it goes. Non-ancestral samplers (`euler`, `dpmpp_2m`, `ddim`) do not, and are what
img2img workflows normally use.

The author recommends `euler_ancestral` for generation *from scratch* — a different job
from partial denoise.

- Cost: one env var, ~5 min
- Test: `docker run ... -e SAMPLER=euler`, repeat the denoise sweep, compare gradation
</details>

## 2. Masked / inpaint editing  — the real missing capability

Every edit currently redraws the whole frame. Localized changes normally use a mask:
constrain the edit to a region, then denoise hard *inside* it while everything outside
is untouched by construction.

ComfyUI has the nodes (`VAEEncodeForInpaint`, `SetLatentNoiseMask`); the graph in
`build_workflow` simply does not use them.

- Cost: ~40 lines, plus deciding how the mask arrives
- Options: caller supplies a mask image; or auto-derive a face mask server-side
  (insightface is already on the box for the FaceFusion work)
- This is what would give genuine facial control

## 3. Crop-to-face, edit, composite back  — START HERE for micro facial edits

In the test source the face is 184px of an 832x1040 image — about **4% of the pixels**
the model attends to. That alone may explain why facial instructions barely register.

Crop the face at native resolution, edit it as a standalone square, paste back.

- Cost: server-side (see the architecture note above); client passes `mode` through
- Risk: seams at the composite boundary; may need feathering. Colour drift between the
  edited crop and the original is the subtler risk — a tonal edge reads worse than a
  geometric one. Histogram-match the boundary ring if it shows up.
- Pairs naturally with the FaceFusion post-pass already documented in
  `docs/pipeline-notes.md`
- Face detection is already available: OpenCV cascades locally, and insightface
  (buffalo_l) on the 3090 from the FaceFusion work — the same code used for the
  face-pixel-budget measurements
- Sketch: detect face box -> expand ~40% for context -> crop -> edit at 512x512 ->
  resize back -> feathered alpha composite into the original. Nothing outside the
  box is ever regenerated, so drift is structurally impossible rather than merely
  discouraged.

---

## Decisions (David, 2026-08-25)

**Response contract: finished result only.** `mode: face` returns the composited
full-size image and nothing else — no crop, no box. A rejected result is just a retry
with a new seed, which the stateless design already handles.

**Rename `/edit` -> `/generate`.** `/edit` was named to mirror fal's endpoint so the
client could target either; that rationale is gone and the server does considerably more
than "edit". `/generate` says what comes out. Migration is free: add `/generate` as
canonical, keep `/edit` as a deprecated alias hitting the same handler, drop the alias
once the web app is the only caller.

---

## 4. Storyboard coherence check  — quality gate before spending GPU on LTX

The wolf shot failed because its keyframes disagreed with each other, and that was only
discovered *after* the render. A check that scores agreement across a set of keyframes
would catch it beforehand. Every piece already exists:

- **identity agreement** — ArcFace cosine to the anchor frame. Run on the richmond
  keyframes it gave 0.941 / 0.838 / 0.804; the spread flagged real drift
  (`docs/pipeline-notes.md` section 5)
- **framing consistency** — face-box width and centre across frames. Exactly what killed
  the wolf shot: 5.5% face area on the opener vs 2.4% on the rest, which made LTX invent
  a camera move
- **face pixel budget** — the >=180px threshold from the notes, predicting whether output
  will read soft
- **dimension/aspect uniformity** — trivial, and would have caught the hailey drift

Shape: an endpoint taking N image URLs, returning per-frame scores and a verdict, so the
web app can say "frame 3 disagrees" before anyone renders.

The interesting part is what it does with the answer. Reporting is easy; acting on it is
where the value is — "frame 3 is off-identity, regenerate it from the anchor" is the loop
that would have saved the wolf shot.

Related open question: if the web app assembles several keyframes from one source, face
detection should run ONCE and the box be reused, or the crops differ slightly per frame
and the composites drift — reintroducing the exact failure this check is meant to catch.

---

## Also queued

**v19 evaluated 2026-08-25 — v23 is better, stay on it.** v19's noise floor is higher
(12.41/12.17 vs 9.05/7.46 face) and it is worse on identity_large. The author's "best for
consistency in edits" appears to mean consistency of applying an edit, not fidelity to
the source. Visually the two are hard to tell apart; the gap matters only because micro
edits live at the floor.

**Base Qwen-Image-Edit-2511 download: NOT worth it.** The hypothesis was that baked LoRAs
cause the drift, but v19 and v23 carry different LoRA mixes and both floor at 9-12. That
points at whole-frame regeneration itself rather than any particular merge, so ~20GB would
likely buy the same answer.

**Four levers now ruled out for micro edits:** checkpoint version, sampler, denoise, and
step count. Each moves the floor by a point or two where a micro edit needs it near zero.
Face mode (items 2/3) is not the preferred approach, it is the only one that can work —
compositing makes the untouched region bit-identical, which is categorically different
from a model choosing to leave it alone.

<details><summary>superseded: original v19 note</summary>
**AIO v19 evaluation.** Downloading `v19/Qwen-Rapid-AIO-NSFW-v19.safetensors` (28.43 GB).
The author's note: *"v19 is likely best for consistency in edits, while v23 is likely
best for prompt adherence."* Consistency is what we want. Note v19 recommends
**er_sde/beta or euler_ancestral/beta** — and er_sde is also stochastic, so item 1
applies to it too.

Switch with `-e CKPT_NAME=Qwen-Rapid-AIO-NSFW-v19.safetensors` once `extra_model_paths`
covers `qwen/v19/`; no rebuild needed.
</details>

**Base Qwen-Image-Edit-2511 in fp8** (`xms991/Qwen-Image-Edit-2511-fp8-e4m3fn` or
`armychimp/Qwen-Image-Edit-2511-FP8`, ~20 GB) — only if 1-3 and v19 all fail. It would
separate "the baked LoRAs override the latent" from "instruction conditioning inherently
dominates". No baked Lightning, so expect more steps and slower.
