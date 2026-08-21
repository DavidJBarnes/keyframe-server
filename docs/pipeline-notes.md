# Keyframe pipeline — production notes

Working notes from four proof-of-concept shots (pour, wolf, richmond, hailey) run
2026-08-21. These are the things that actually decided quality, in rough order of
leverage. The LTX conditioning rules live in `client/README.md`; this file covers what
we learned running them.

---

## 1. Face pixel width is the dominant quality lever

Measured median face width in the output video, at 512x768:

| shot | face px | % frame | reads as |
|---|---|---|---|
| wolf | 98 | 2.4% | soft — teeth blur to a smear, no iris detail, hair is a mass not strands |
| pour | 199 | 10.1% | good |
| hailey | 200 | — | good |
| richmond | 274 | 19.2% | best |

**This is set by framing, not by generation resolution.** Richmond hit 274px at plain
512x768 purely because the source was a close selfie. Raising resolution costs VRAM and
time; cropping tighter is free.

Rule of thumb: **target ≥180px.** Below ~140px the face reads mushy no matter what else
is right.

To measure:

```bash
ffmpeg -i out.mp4 -vsync 0 /tmp/f/%03d.png
# then per frame, OpenCV haarcascade_frontalface_default, take the largest box
```

### The framing/action tradeoff

A source photo's framing caps what shots are possible, and face size trades directly
against how much body is in frame:

| framing | face px (hailey source) | what motion reads |
|---|---|---|
| chest-up | 275 | expression, head/shoulder tilt |
| waist-up | 213 | torso lean, shoulder line |
| full seated | 137 | hip shift, hands, posture |

You cannot have a large face *and* full-body motion from one source. Decide which the
shot needs before generating anything. Widening beyond the source means outpainting,
which reintroduces drift — don't.

---

## 1a. Motion budget = pose delta between keyframes

**LTX interpolates between waypoints; it does not invent motion the waypoints don't
imply.** If consecutive keyframes are near-identical, the output is a still image with
blinks — no strength setting rescues it, because there is no delta to blend through.

Learned the hard way on hailey v1: after an over-tilted set looked theatrical, the poses
were capped to "a few degrees, never exaggerated". The five keyframes then differed so
little that the result was a woman sitting still and blinking. Low interior strengths let
the model blend *between* poses; they cannot manufacture a pose that isn't there.

Two failure directions, and they are opposite:

| symptom | cause | fix |
|---|---|---|
| theatrical, snapping poses | keyframe deltas too large for the gap | more keyframes, or smaller deltas |
| static, "she just blinked" | keyframe deltas too small | **bigger deltas**, wider crop so motion is visible |

Sanity-check before running LTX: put the keyframes side by side. **If you can't
immediately name what changed between consecutive frames, there won't be motion.**

A cheap numeric proxy after the fact — track the face-box centre across output frames.
Hailey v1's head barely moved; v2 travelled 7% → 85% of frame width.

### The wider-framing tax

v2 got real motion by widening the crop so hips and legs were in frame, at a cost:

| | face px | motion |
|---|---|---|
| hailey v1 (chest-up) | 200 | head tilt and a blink |
| hailey v2 (full seated) | 134 | genuine weight shift into a recline |

This is the §1 tradeoff in its sharpest form. **Decide first whether the shot is about
her face or about what she does** — you cannot have both from a single source.

---

## 2. Edit models re-frame when you ask for something off-frame

The biggest surprise. Asking Qwen-Image-Edit for "leaning onto one hip" from a chest-up
crop makes it **zoom out** to show the hip — silently breaking framing consistency across
keyframes, which is the wolf failure mode arriving by a new route.

Adding "same camera framing and distance" to the prompt is **not** sufficient.

What works:

1. **Feed the already-cropped anchor**, not the full-resolution source. The crop is the
   contract.
2. **Forbid re-framing explicitly and redundantly**: *"Keep the photograph exactly as it
   is — same camera position, same distance, same crop, same zoom, same background. Do
   not zoom out and do not show more of her body than is already visible."*
3. **Only ask for changes visible inside the current frame.** If the pose requires
   showing something the crop excludes, the model will widen to comply. This is the real
   constraint — 1 and 2 are mitigations for it.

Verify numerically rather than by eye — detect the face box in each keyframe and compare
width and centre. Hailey's locked set held at 190–212px; the unlocked set varied 70–212px.

### Native-resolution edits

Feeding the full source with no `--size` returns ~1808x2288 (about 1.6MP) instead of
512x768. More detail, but the subject's *scale within frame* still changes between
generations, so a uniform post-crop does not give you a fixed camera. Face-anchoring the
crop fixes scale but cancels head motion. Not worth it — lock the framing at generation
time instead.

---

## 3. Pose language leaks

Qwen-Image-Edit over-interprets pose instructions, and the leakage is specific:

- **"shoulders low and loose" → pulled her top off the shoulder.** Pose words that could
  describe clothing will alter clothing. Protect wardrobe explicitly: *"her top stays
  exactly as it is, covering both shoulders fully with the same neckline — do not change,
  move or alter her clothing in any way."*
- **"head tilted" → 30–40° theatrical tilt.** Cap it numerically: *"at most a very slight
  natural tilt of a few degrees, never exaggerated."*
- **"head coming back toward level" → head thrown backwards.** Directional corrections
  get read as new poses, not as less of the current one. State the absolute target pose,
  never a relative adjustment.
- **Unrequested props appear** — a hand-on-hip showed up twice without being asked for.

Generic instruction: describe the **absolute end state**, cap magnitudes explicitly, and
name what must not change.

---

## 4. Constrain the tail

Richmond grew a **phantom second hand** at f090–f098 — a second hand faded in around the
mug and back out, entirely inside the 48-frame gap between the last interior waypoint
(73) and the terminal (121).

Nothing in the prompt said how many hands hold a mug, so two seconds of unconstrained
frames found something to improvise with. Long final gaps are the risk **regardless of
what you expect to go wrong in them** — wolf's flagged-as-risky boy entrance in the same
gap came out clean because it was motivated in the prompt.

Use five keyframes — `0 / 33 / 73 / 97 / 121` — so no gap exceeds ~40 frames. Hailey ran
this way and showed no improvisation artifacts.

---

## 4a. Multi-stage motion still morphs, even with a keyframe per stage

Hailey v2 ran keyframes at every stage of the recline (upright → hip-shift → propped on
a straight arm → down onto the elbow) and still produced a **ghosting morph around
f095–f105**, on the prop-to-recline transition where the whole body drops.

Keyframes at each stage are a mitigation, not a cure. When a transition moves the entire
body rather than a limb, either give it more frames, add another intermediate, or accept
it as a cut boundary. `RetakePipeline` over the bad region is the cheap repair.

---

## 5. Identity: metrics vs. the eye

ArcFace cosine similarity against the source photo, richmond keyframes:

| | pre-swap | post-swap |
|---|---|---|
| kf2 | 0.941 | 0.894 |
| kf3 | 0.838 | 0.861 |
| kf4 | 0.804 | 0.880 |
| **spread** | **0.137** | **0.033** |

FaceFusion **degraded** the best frame and improved the worst. It is a *consistency
normalizer*, not an identity booster — the spread collapsing from 0.137 to 0.033 is the
real effect, and spread is what matters, since drift *between* waypoints is what LTX
interpolates through.

But we shipped the **unswapped** set: visually it read fine, and a GFPGAN pass adds
waxiness plus a skin-texture mismatch against kf1, which is a real photograph. Treat the
numbers as a diagnostic that points at a cause, not as the acceptance criterion.

Big expression changes cost identity — richmond's kf4 dropped to 0.804 when asked for a
broad smile. If identity matters more than the expression beat, keep expressions subtle.

Rule: **uniform treatment.** Either all keyframes get the swap or none do. A mix gives
kf1 real photographic texture and the rest enhanced texture, and that discontinuity shows.

---

## 6. Practical gotchas

**EXIF rotation.** Phone photos carry `Orientation: 6` (rotate 90 CW) with landscape
pixel data. Anything reading raw pixels feeds a sideways face into the encoder. Bake it
in first, don't leave it as metadata:

```bash
magick start.jpg -auto-orient -strip start-oriented.png
```

**Object consistency across keyframes.** Keyframes are independent edits from one source
(never chained), so each will invent a *different* mug/prop and it flickers between
waypoints. Fix with multi-ref: generate the prop's first appearance, then pass **two**
references — the anchor for the person, that keyframe for the prop — and describe the
prop with an identical string in every prompt. Worked cleanly on richmond.

**Seeds don't work on fal.** `client.py` only forwards `--seed` on the `local` backend.
Re-run the single bad keyframe rather than expecting to reproduce a set.

**Expression under-delivery.** "A small warm smile" produced no change at all on
richmond's kf4; it took explicit anatomy — *"corners of her mouth clearly turned up, her
cheeks lifted and her eyes crinkling"* — to land. Under-specified expressions are ignored;
under-specified poses are exaggerated.

---

## 7. Order of operations

1. `magick -auto-orient -strip` the source
2. Choose the crop from the **face-px vs. body-in-frame** tradeoff, before anything else
3. Cut the anchor: `magick src -crop WxH+X+Y +repage -resize 512x768^ -gravity center -extent 512x768 kf1.png`
4. Generate keyframes **from the cropped anchor**, framing locked, multi-ref for props
5. Verify framing numerically (face width + centre across the set) — regenerate outliers
6. Optional FaceFusion pass — all keyframes or none
7. LTX at `0/33/73/97/121`, strengths `1.0/0.65/0.7/0.75/0.9`
8. Optional FaceFusion post-pass on the MP4 (`wanly/experiment/faceswap.sh`)

## 8. Open items for productionalising

- No edit server is deployed. Everything above ran on fal (`--model qwen`). The 3090 has
  a broken partial Qwen download (25 `.incomplete`, 0 finished safetensors) and an empty
  venv at `~/keyframe-server`. See root README for the standup.
- Steps 3 and 5 are mechanical and should be a script — crop-by-face-target and a
  framing-consistency assertion that fails loudly on outliers.
- The richmond phantom hand was never fixed; the prescribed fix (a keyframe at 97) is
  untested on that shot, though hailey's five-keyframe layout is indirect evidence.
- Face-px targets are from four shots on one model at one resolution. Re-baseline if the
  transformer or resolution changes.
