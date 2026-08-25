# TODO — facial edit control

Open work on making small, controlled facial adjustments while preserving identity.

**Context.** Identity preservation is solid (that was an aspect-ratio bug, since fixed).
What does not work is *controlling the magnitude* of a facial change. `--denoise` was
added for this and barely moves the needle: at 4 steps, denoise 0.15 changes ~74% as
much as denoise 1.0 (mean abs diff 3.01 vs 4.08); at 12 steps, 3.25 vs 5.37. Raising
steps did not restore gradation, so it is not sampler resolution.

Measured on `client/inputs/test/k2_crop.png` (832x1040), prompt "Make her smile slightly
wider...", seed 88, checkpoint Qwen-Rapid-AIO-NSFW-v23.

---

## 1. Try a non-ancestral sampler  — cheap, do first

`euler_ancestral` **injects fresh noise at every step**, which directly fights latent
preservation: you keep 85% of the source at denoise 0.15 and the sampler re-randomises
as it goes. Non-ancestral samplers (`euler`, `dpmpp_2m`, `ddim`) do not, and are what
img2img workflows normally use.

The author recommends `euler_ancestral` for generation *from scratch* — a different job
from partial denoise.

- Cost: one env var, ~5 min
- Test: `docker run ... -e SAMPLER=euler`, repeat the denoise sweep, compare gradation
- If it works: expose `sampler`/`scheduler` per-request in `EditRequest` so it can be
  chosen per edit rather than per server

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

## 3. Crop-to-face, edit, composite back  — cheaper than masking, maybe better

In the test source the face is 184px of an 832x1040 image — about **4% of the pixels**
the model attends to. That alone may explain why facial instructions barely register.

Crop the face at native resolution, edit it as a standalone square, paste back.

- Cost: client-side, no server change
- Risk: seams at the composite boundary; may need feathering
- Pairs naturally with the FaceFusion post-pass already documented in
  `docs/pipeline-notes.md`

---

## Also queued

**AIO v19 evaluation.** Downloading `v19/Qwen-Rapid-AIO-NSFW-v19.safetensors` (28.43 GB).
The author's note: *"v19 is likely best for consistency in edits, while v23 is likely
best for prompt adherence."* Consistency is what we want. Note v19 recommends
**er_sde/beta or euler_ancestral/beta** — and er_sde is also stochastic, so item 1
applies to it too.

Switch with `-e CKPT_NAME=Qwen-Rapid-AIO-NSFW-v19.safetensors` once `extra_model_paths`
covers `qwen/v19/`; no rebuild needed.

**Base Qwen-Image-Edit-2511 in fp8** (`xms991/Qwen-Image-Edit-2511-fp8-e4m3fn` or
`armychimp/Qwen-Image-Edit-2511-FP8`, ~20 GB) — only if 1-3 and v19 all fail. It would
separate "the baked LoRAs override the latent" from "instruction conditioning inherently
dominates". No baked Lightning, so expect more steps and slower.
