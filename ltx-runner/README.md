# ltx-runner

LTX-2.5 keyframe video generation behind a job API. The other half of the loop:
`server.py` makes the keyframes, this turns a sequence of them into a clip.

```
POST /job            -> {"job_id": ..., "placement": [...]}   returns at once
GET  /job/{id}       -> {"status": "None"|"Processing"|"Done"|"Failed", "video": url|null}
GET  /job/{id}/video -> the mp4
GET  /jobs, /health
```

## Running it

Lives on the GPU host next to a working `LTX-2` checkout. Not in the container —
it shells out to `LTX-2/venv/bin/python`, and it needs the models on local disk.

```bash
./start.sh                    # detached tmux session `ltx-runner`
tmux attach -t ltx-runner     # watch a render scroll; Ctrl-b d to detach
./start.sh --restart          # pick up new code
```

| env | default | |
|---|---|---|
| `LTX_HOME` | `/home/david/LTX-2` | checkout with `packages/` and `models/ltx-2.5` |
| `JOBS_DIR` | `/home/david/ltx-jobs` | keyframes, cmd.txt, ltx.log, out.mp4 per job |
| `KEYFRAME_URL` | `http://127.0.0.1:8189` | for the `/free` handshake |
| `MIN_FREE_GB` | `18.0` | refuse to start below this |
| `PORT` | `8190` | |

## Why it yields the GPU first

keyframe-server's ComfyUI keeps the Qwen checkpoint resident after any edit —
measured 20.4 GB, leaving 888 MB on a 24 GB card — and never releases it. LTX-2.5
wants essentially the whole card, so the two cannot be co-resident. Every job
calls keyframe-server's `POST /free` and refuses to start if the card does not
actually come back, because LTX does not fail gracefully on a short card, it dies
part-way through a model load. The next keyframe edit reloads and pays ~30s.

One GPU means one job: work runs through a single worker thread. Concurrency here
would be an OOM, not a speedup.

## Keyframe placement is not free-form

Read out of the LTX-2 source rather than inferred, because getting it wrong is
silent:

- **Indices must be `0` or `1+8k`.** `SpatioTemporalScaleFactors.default()` is
  `(time=8, …)` and the video encoder is causal, so the first latent frame covers
  one pixel frame and every later one covers eight — boundaries land at
  `0, 1, 9, 17 …`. An off-grid index is **not** snapped for you:
  `VideoConditionByKeyframeIndex.apply_to` does `positions += frame_idx`
  literally, so the guide token sits between two latent slots and smears across
  both. That is what "the keyframe isn't working" looks like. Off-grid is a 422
  naming the nearest legal values; `snap_indices: true` opts into snapping.
- **Resolution divisible by 64, not 32.** We always pass
  `--spatial-upsampler-path`, which makes this the two-stage pipeline, and LTX's
  `assert_resolution(..., is_two_stage=True)` requires 64. The /32 rule is for
  one-stage.
- **`num_frames` floors to `k*8+1`**, matching LTX's own `snap_frames_to_grid`.

Index `0` is special twice over: it is the only legal index below 1, and
`helpers.py` routes it to `VideoConditionByLatentIndex(latent_idx=0)` — the true
first-frame anchor — while every other index becomes a keyframe conditioning.

Omit indices and strengths and the recipe defaults apply: evenly spaced on grid,
`1.0` opener / `0.65` interior / `0.9` terminal, terminal sitting at `num_frames`.
Even spacing is the neutral default, not the good one — see
`docs/ltx-keyframe-recipe.md` §4 on weighting the budget toward the payoff beat.

## Verified

Richmond's keyframes, indices, strengths, prompt and seed submitted through the
API produce a video **byte-for-byte identical** to the hand-run
`run_richmond.sh`. The service adds no drift.
