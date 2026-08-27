#!/usr/bin/env python3
"""ltx-job-runner — LTX-2.5 keyframe video generation behind an HTTP job API.

Implements DavidJBarnes/wanly-console#355. storyboard-ui POSTs a storyboard's
keyframes, gets a job_id back immediately, and polls until a video URL appears.

Shape of the thing:

  POST /job          -> {"job_id": ...}          (returns at once; work is queued)
  GET  /job/{id}     -> {"status": ..., "video": url|null, ...}
  GET  /job/{id}/video -> the mp4 bytes
  GET  /health

Two facts drive the design.

**One GPU, one job.** LTX-2.5 22B wants essentially the whole card, so jobs run
strictly one at a time through a single worker thread. Concurrency here would not
be faster, it would be an OOM.

**The GPU is already occupied.** keyframe-server's ComfyUI holds the Qwen
checkpoint resident after any edit -- measured 20.4 GB, leaving 888 MB free --
and never releases it on its own. Every job therefore calls keyframe-server's
POST /free first and waits for the card to actually come back. Without that step
LTX does not fail gracefully, it dies mid-load.
"""
import argparse
import base64
import json
import os
import queue
import re
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import requests
import uvicorn
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import ltx_grid

LTX_HOME = Path(os.environ.get("LTX_HOME", "/home/david/LTX-2"))
JOBS_DIR = Path(os.environ.get("JOBS_DIR", "/home/david/ltx-jobs"))
KEYFRAME_URL = os.environ.get("KEYFRAME_URL", "http://127.0.0.1:8189")
MODELS = LTX_HOME / "models" / "ltx-2.5"
# Free at least this much VRAM before starting, or refuse rather than OOM deep
# in a model load twenty minutes later.
MIN_FREE_GB = float(os.environ.get("MIN_FREE_GB", "18.0"))
# Two pipelines, both two-stage and both needing /64 dimensions.
#
#   distilled — ltx-2.5-22b-distilled-transformer. One model, no LoRA, no
#               negative prompt, no step count. Fast and what run_richmond.sh
#               used.
#   hq        — the ltx-2.5-22b-DEV transformer with the distilled LoRA applied
#               at stage 2. Takes a negative prompt and an explicit step count,
#               which is the whole reason to reach for it: the distilled model
#               gives you no lever against identity drift, and --negative-prompt
#               is where "identity change, face distortion, different person"
#               goes.
#
# `--distilled-lora` is REQUIRED by the hq parser (args.py), so hq without a
# LoRA is not a configuration that exists.
#   ltx23     — the 2.3 distilled monolith through distilled.py.
#   ltx23-hq  — the 2.3 DEV monolith through ti2vid_two_stages_hq with 2.3's own
#               distilled LoRA at stage 2. This is the one the content LoRAs on
#               this box are actually for: they record ss_sd_model_name
#               "ltx-2.3-22b-dev.safetensors", and LTX's MODELS-LTX-2.3.md says
#               to pair them with a 2.3 checkpoint.
#
# 2.3 ships MONOLITHS — one file carrying transformer, both VAEs and the text
# projection — so the CLI differs from 2.5's five split paths: it takes
# --checkpoint-path (or --distilled-checkpoint-path for distilled.py) plus
# --gemma-root pointing at a Gemma HF directory.
#   ltx23-pure — the 2.3 DEV monolith through ti2vid_one_stage, with NO
#               distillation anywhere. Both two-stage parsers make
#               --distilled-lora required=True, so "dev + HQ" still applies a
#               distilled schedule at stage 2; ti2vid_one_stage is the only path
#               that applies none. Use it when the model author says the weights
#               perform best undistilled.
#
#               The cost is real: one stage means no 2x latent upscale, so every
#               step denoises at the full target resolution rather than a quarter
#               of it, and the undistilled default is 30 steps against 8. Expect
#               it to be several times slower and much heavier on VRAM.
PIPELINES = ("distilled", "hq", "ltx23", "ltx23-hq", "ltx23-pure")
MODELS_23 = LTX_HOME / "models" / "ltx-2.3"
# Content LoRAs live here, NOT under models/ltx-2.5/loras (that holds the
# distilled LoRA the hq pipeline needs). Requests name a file in this directory
# rather than passing a path, so a browser cannot walk the filesystem.
LORA_DIR = LTX_HOME / "models" / "loras"
JOB_TIMEOUT_S = int(os.environ.get("JOB_TIMEOUT_S", "5400"))


class Lora(BaseModel):
    """A content LoRA by filename, resolved inside LORA_DIR."""
    name: str
    strength: float = Field(default=0.6, ge=0.0, le=2.0)


class Keyframe(BaseModel):
    # data URI or http(s) URL, exactly like keyframe-server's image_urls
    image: str
    # Omit both and the recipe's defaults are applied across the whole set.
    index: int | None = None
    strength: float | None = Field(default=None, ge=0.0, le=1.0)


class JobRequest(BaseModel):
    # Empty is allowed and meaningful: with a strong conditioning image and a
    # LoRA carrying the motion, an empty prompt is a real configuration rather
    # than an oversight.
    prompt: str = ""
    # Content LoRAs, applied in order. `--lora` is repeatable on both pipelines.
    loras: list[Lora] = Field(default_factory=list, max_length=4)
    keyframes: list[Keyframe] = Field(min_length=1, max_length=12)
    width: int = 512
    height: int = 768
    num_frames: int = 121
    frame_rate: int = 24
    seed: int | None = None
    # "distilled" or "hq" — see PIPELINES.
    pipeline: str = "distilled"
    # hq only. Ignored by the distilled pipeline, which has no CFG to steer.
    negative_prompt: str | None = None
    num_inference_steps: int | None = Field(default=None, ge=1, le=50)
    lora_strength: float | None = Field(default=None, ge=0.0, le=2.0)
    # Per-stage distilled-LoRA strength on the hq paths.
    #
    # These matter more than the names suggest. LTX defaults stage 1 to 0.25 and
    # stage 2 to 0.50, and the DR34ML4Y author states plainly that the
    # distillation LoRA and checkpoint "actively fight the nsfw training and
    # account for body horror", recommending the full dev checkpoint at 0.25-0.35
    # distill strength. The stage-2 default of 0.5 sits ABOVE that range, so the
    # out-of-the-box hq configuration is already past what the LoRA tolerates.
    lora_strength_stage_1: float | None = Field(default=None, ge=0.0, le=2.0)
    lora_strength_stage_2: float | None = Field(default=None, ge=0.0, le=2.0)
    # CFG guidance. "Adjust steps and CFG accordingly for both passes" is the
    # other half of that advice, and neither was reachable before.
    cfg_scale: float | None = Field(default=None, ge=0.0, le=20.0)
    stg_scale: float | None = Field(default=None, ge=0.0, le=20.0)
    # Keyframes larger than the video are downscaled here, which is the point of
    # decoupling the two: the board can hold 832x1216 images so face edits work
    # on the full-resolution face, while the clip renders at whatever size the
    # GPU and the shot actually want. Upscaling is refused instead, because
    # inventing pixels to feed a conditioning frame is never what was meant.
    allow_upscale: bool = False
    # Off-grid indices are REPORTED, not refused.
    #
    # The recipe says an off-grid index "gets snapped elsewhere". No snapping
    # code exists: VideoConditionByKeyframeIndex.apply_to does
    # `positions += frame_idx` and divides by fps, so the index is an exact
    # continuous time. The reasonable worry is that a keyframe token which does
    # not coincide with a latent slot spreads its influence over the neighbours
    # instead of pinning one -- but that is inference, not measurement, and
    # working commands here use index 8. So off-grid comes back flagged on the
    # job and `strict_grid` opts into refusing it.
    strict_grid: bool = False
    snap_indices: bool = False


@dataclass
class Job:
    id: str
    req: JobRequest
    status: str = "None"          # None -> Processing -> Done | Failed
    video: str | None = None
    error: str | None = None
    placement: list[dict] = field(default_factory=list)
    loras: list[dict] = field(default_factory=list)
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    log_tail: list[str] = field(default_factory=list)

    def view(self, base: str) -> dict:
        d = {
            "job_id": self.id,
            "status": self.status,
            "video": f"{base}/job/{self.id}/video" if self.status == "Done" else None,
            "placement": self.placement,
            "pipeline": self.req.pipeline,
            "loras": self.loras,
            "distill": ({"stage_1": self.req.lora_strength_stage_1,
                         "stage_2": self.req.lora_strength_stage_2,
                         "cfg": self.req.cfg_scale, "stg": self.req.stg_scale}
                        if is_hq(self.req.pipeline) else None),
            "queued_s": round((self.started or time.time()) - self.created, 1),
        }
        if self.started:
            d["elapsed_s"] = round((self.finished or time.time()) - self.started, 1)
        if self.error:
            d["error"] = self.error
            d["log_tail"] = self.log_tail[-25:]
        return d


JOBS: dict[str, Job] = {}
QUEUE: "queue.Queue[str]" = queue.Queue()
_LOCK = threading.Lock()

app = FastAPI(title="ltx-job-runner")


def decode_image(src: str, dest: Path):
    if src.startswith("data:"):
        dest.write_bytes(base64.b64decode(re.sub(r"^data:[^,]+,", "", src)))
    elif src.startswith(("http://", "https://")):
        dest.write_bytes(requests.get(src, timeout=120).content)
    else:
        raise HTTPException(400, "keyframe image must be a data URI or http(s) URL")


def safetensors_header(path: Path) -> dict:
    """Tensor names and metadata, without a tensor framework and without the file.

    The format is 8 bytes of little-endian u64 header length, then that many
    bytes of UTF-8 JSON. So this reads a few hundred KB off a 42 GB checkpoint
    and needs neither torch nor numpy -- `safe_open(framework="pt")` would drag
    torch into this venv purely to enumerate strings, and the whole point of the
    runner having its own venv is that LTX's stays untouched.
    """
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        return json.loads(f.read(n))


_TKEYS: dict[str, set[str]] = {}


def _transformer_keys(transformer: Path) -> set[str]:
    """Weight names as the loader sees them, cached (header read, no tensors)."""
    key = str(transformer)
    if key not in _TKEYS:
        pre = "model.diffusion_model."
        _TKEYS[key] = {k[len(pre):] if k.startswith(pre) else k
                       for k in safetensors_header(transformer) if k != "__metadata__"}
    return _TKEYS[key]


def lora_coverage(lora: Path, transformer: Path) -> tuple[int, int]:
    """(fused, targeted) weights for this LoRA against this transformer.

    A LoRA whose keys do not line up fuses NOTHING and says nothing about it:
    `_affected_weight_keys` in fuse_loras.py matches purely on a
    `.lora_A.weight` naming convention, and `apply_loras` then iterates an empty
    set. No error, no warning, no log line -- the run looks completely normal and
    the LoRA simply is not there. Nothing in LTX logs LoRA loading at INFO, so
    there is otherwise no way to tell from the output of a job.

    Both sides get a prefix strip at load, which is the whole subtlety: the
    transformer loses `model.diffusion_model.` (LTXV_MODEL_COMFY_RENAMING_MAP)
    and the LoRA loses `diffusion_model.` (LTXV_LORA_COMFY_RENAMING_MAP), and
    they meet at `transformer_blocks.*`. Comparing the raw file keys reports 0%
    for every LoRA, which looks like a catastrophe and is just the wrong test.
    """
    pre = "diffusion_model."
    keys = [k[len(pre):] if k.startswith(pre) else k
            for k in safetensors_header(lora) if k != "__metadata__"]
    suffix = ".lora_A.weight"
    affected = {k[: -len(suffix)] + ".weight" for k in keys if k.endswith(suffix)}
    return len(affected & _transformer_keys(transformer)), len(affected)


def lora_base_model(lora: Path) -> str | None:
    """What the LoRA file says it was trained against, if it says anything.

    Trainers record this inconsistently -- `ss_sd_model_name`, `ss_base_model_version`,
    or nothing at all -- so absence proves nothing and this is advisory only.
    """
    meta = safetensors_header(lora).get("__metadata__") or {}
    return meta.get("ss_sd_model_name") or meta.get("ss_base_model_version")


def recorded_version(base: str | None) -> str | None:
    """LTX generation a LoRA claims, or None when the file does not really say.

    Deliberately conservative. Only "2.3" or "2.5" appearing in the recorded base
    counts; "ltx2" and "ltx2_v1" are era-ambiguous and two of the six files here
    record nothing at all. Guessing from those would produce confident warnings
    about files whose provenance is genuinely unknown, and a warning that fires
    on unknowns is one people learn to click through.
    """
    if not base:
        return None
    for v in ("2.5", "2.3"):
        if v in base:
            return v
    return None


def resolve_lora(name: str) -> Path:
    """A filename inside LORA_DIR, never a path.

    Requests come from a browser, so anything path-shaped is refused outright
    rather than sanitised -- there is no legitimate reason for a LoRA reference
    to contain a separator.
    """
    if "/" in name or "\\" in name or name.startswith("."):
        raise HTTPException(422, f"lora must be a filename in {LORA_DIR}, not a path")
    p = (LORA_DIR / name)
    if not p.is_file():
        avail = sorted(f.name for f in LORA_DIR.glob("*.safetensors"))
        raise HTTPException(422, f"no such lora {name!r}. Available: {avail}")
    return p


def normalise(path: Path, width: int, height: int, allow_upscale: bool) -> str | None:
    """Bring a keyframe to the exact generation resolution. Returns what it did.

    Conditioning frames must arrive at the generation resolution, and LTX does
    not refuse a mismatch -- decode.py runs `resize_and_center_crop` on every
    conditioning image. So the resize happens regardless; the only question is
    whether it is done deliberately with a good filter or incidentally with
    whatever torch interpolation LTX reaches for, and whether anyone is told.
    Doing it here makes it LANCZOS, logged, and reported back on the job.

    Upscaling is refused rather than performed: a keyframe smaller than the clip
    means detail is being invented to fill a frame the model will then treat as
    ground truth.
    """
    with Image.open(path) as im:
        sw, sh = im.size
        if (sw, sh) == (width, height):
            return None
        scale = max(width / sw, height / sh)
        if scale > 1.0 and not allow_upscale:
            raise RuntimeError(
                f"{path.name} is {sw}x{sh}, smaller than the {width}x{height} clip. "
                f"Upscaling it would invent detail the model then treats as ground "
                f"truth -- render at or below the keyframe size, or pass "
                f"allow_upscale=true if you really mean it.")
        # Centre-crop to the target aspect first, then one resample. Cropping
        # after would resample content that is about to be thrown away.
        cw, ch = min(sw, round(width / scale)), min(sh, round(height / scale))
        left, top = (sw - cw) // 2, (sh - ch) // 2
        out = im.convert("RGB").crop((left, top, left + cw, top + ch)) \
                .resize((width, height), Image.LANCZOS)
    out.save(path)
    kept = (cw * ch) / (sw * sh)
    return (f"{sw}x{sh} -> {width}x{height} ({width / cw:.2f}x, "
            f"kept {kept:.0%} of frame)")


def plan(req: JobRequest) -> list[dict]:
    """Resolve every keyframe to an on-grid (index, strength). Raises 422 if it cannot."""
    n = len(req.keyframes)
    auto_idx = ltx_grid.auto_place(n, req.num_frames)
    auto_str = ltx_grid.default_strengths(n)
    out = []
    for i, kf in enumerate(req.keyframes):
        idx = auto_idx[i] if kf.index is None else kf.index
        snapped, off_grid = False, False
        if not ltx_grid.is_on_grid(idx):
            if req.strict_grid:
                raise HTTPException(422,
                    f"keyframe {i}: index {idx} is off the latent grid (0 or 1+8k); "
                    f"nearest are {ltx_grid.snap(idx - 4)} and {ltx_grid.snap(idx + 4)}.")
            if req.snap_indices:
                idx, snapped = ltx_grid.snap(idx), True
            else:
                off_grid = True
        if idx > req.num_frames:
            raise HTTPException(422,
                f"keyframe {i}: index {idx} is past num_frames={req.num_frames}")
        out.append({"index": idx,
                    "strength": auto_str[i] if kf.strength is None else kf.strength,
                    "snapped_from": kf.index if snapped else None,
                    "off_grid": off_grid})
    idxs = [o["index"] for o in out]
    if len(set(idxs)) != len(idxs):
        raise HTTPException(422, f"two keyframes share a latent slot: {idxs}. The later "
                                 f"one would silently win.")
    if idxs != sorted(idxs):
        raise HTTPException(422, f"keyframe indices must ascend, got {idxs}")
    return out


def free_the_gpu() -> float:
    """Ask keyframe-server to drop the Qwen checkpoint. Returns free GB."""
    try:
        r = requests.post(f"{KEYFRAME_URL}/free", timeout=180)
        if r.status_code == 200:
            return float(r.json().get("vram_free_gb", 0.0))
        # A 404 means keyframe-server predates the /free endpoint. Say so plainly
        # rather than letting LTX die in a model load.
        raise RuntimeError(f"keyframe-server /free -> {r.status_code} {r.text[:200]}")
    except requests.RequestException as e:
        raise RuntimeError(f"keyframe-server unreachable at {KEYFRAME_URL}: {e}")


DISTILLED_T = "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
DEV_T = "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors"
DISTILLED_LORA = "loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors"

# --- LTX 2.3 monolith layout, mirroring models/ltx-2.5/ --------------------
T23_DEV = MODELS_23 / "diffusion_models/ltx-2.3-22b-dev.safetensors"
T23_DISTILLED = MODELS_23 / "diffusion_models/ltx-2.3-22b-distilled-1.1.safetensors"
GEMMA_23 = MODELS_23 / "text_encoders/gemma-3-12b-it"
LORA_23 = MODELS_23 / "loras/ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
UPSAMPLER_23 = MODELS_23 / "latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"


def is_23(pipeline: str) -> bool:
    return pipeline.startswith("ltx23")


def is_hq(pipeline: str) -> bool:
    """Two-stage paths that require LTX's own distilled LoRA at stage 2."""
    return pipeline in ("hq", "ltx23-hq")


def is_pure(pipeline: str) -> bool:
    """Single-stage, no distilled LoRA, full CFG at full resolution."""
    return pipeline == "ltx23-pure"


def base_transformer(pipeline: str) -> Path:
    """The weights a LoRA would be fused into, for coverage checking."""
    if pipeline == "ltx23":
        return T23_DISTILLED
    if pipeline in ("ltx23-hq", "ltx23-pure"):
        return T23_DEV
    return MODELS / (DEV_T if pipeline == "hq" else DISTILLED_T)


def require_23_assets(pipeline: str) -> None:
    """Fail at submit with something actionable, not four minutes into a run."""
    need = [(GEMMA_23, "Gemma text encoder directory"),
            (T23_DEV if pipeline in ("ltx23-hq", "ltx23-pure") else T23_DISTILLED,
             "2.3 checkpoint")]
    if not is_pure(pipeline):
        # One stage means no latent upscale, so no upsampler is loaded at all.
        need.append((UPSAMPLER_23, "2.3 spatial upsampler"))
    if pipeline == "ltx23-hq":
        need.append((LORA_23, "2.3 distilled LoRA (required by the hq parser)"))
    missing = [f"{d} at {p}" for p, d in need if not p.exists()]
    if missing:
        raise HTTPException(503,
            "LTX 2.3 assets are not on disk yet: " + "; ".join(missing) +
            ". Run LTX-2/download-ltx23.sh (tmux session ltx23-dl).")


def build_argv(job: Job, workdir: Path) -> list[str]:
    """Assemble the CLI. Both pipelines take the same core flags.

    The hq path is `-m ltx_pipelines.ti2vid_two_stages_hq` against the dev
    transformer with the distilled LoRA, recovered from shell history rather
    than from test1.sh -- that script names --checkpoint-path and --gemma-root,
    neither of which exists in args.py, which is why its output directory is
    empty while identity_sweep/ has videos in it.
    """
    py = LTX_HOME / "venv" / "bin" / "python"
    py = str(py if py.exists() else "python")
    pipe = job.req.pipeline
    hq, pure, v23 = is_hq(pipe), is_pure(pipe), is_23(pipe)

    argv = [py]
    if pure:
        argv += ["-m", "ltx_pipelines.ti2vid_one_stage"]
    elif hq:
        argv += ["-m", "ltx_pipelines.ti2vid_two_stages_hq"]
    else:
        argv += ["packages/ltx-pipelines/src/ltx_pipelines/distilled.py"]

    if v23:
        # Monolith: the checkpoint carries the transformer, both VAEs and the
        # text projection, so there is nothing to pass for vae or text-encoder.
        # distilled.py's parser names the monolith --distilled-checkpoint-path;
        # the hq parser names it --checkpoint-path. Same file role, two flags.
        # --distilled-checkpoint-path exists only on distilled.py's parser;
        # everything else names the monolith --checkpoint-path.
        monolith = hq or pure
        argv += [("--checkpoint-path" if monolith else "--distilled-checkpoint-path"),
                 str(T23_DEV if monolith else T23_DISTILLED),
                 "--gemma-root", str(GEMMA_23)]
        # One stage never upscales, so it neither needs nor accepts an upsampler.
        if not pure:
            argv += ["--spatial-upsampler-path", str(UPSAMPLER_23)]
    else:
        argv += [
            "--transformer-path", str(MODELS / (DEV_T if hq else DISTILLED_T)),
            "--text-encoder-path", str(MODELS / "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"),
            "--video-vae-path", str(MODELS / "vae/ltx-2.5-video-vae-bf16.safetensors"),
            "--audio-vae-path", str(MODELS / "vae/ltx-2.5-audio-vae-bf16.safetensors"),
            "--spatial-upsampler-path", str(MODELS / "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"),
        ]
    argv += ["--prompt", job.req.prompt]

    if hq:
        # Required by the hq parser -- hq without a LoRA is not a thing. Each
        # version brings its own; a 2.5 distilled LoRA on a 2.3 base is the very
        # mismatch this pipeline exists to avoid.
        lora = ["--distilled-lora", str(LORA_23 if v23 else MODELS / DISTILLED_LORA)]
        if job.req.lora_strength is not None:
            lora.append(str(job.req.lora_strength))
        argv += lora
        if job.req.lora_strength_stage_1 is not None:
            argv += ["--distilled-lora-strength-stage-1", str(job.req.lora_strength_stage_1)]
        if job.req.lora_strength_stage_2 is not None:
            argv += ["--distilled-lora-strength-stage-2", str(job.req.lora_strength_stage_2)]
        if job.req.negative_prompt:
            argv += ["--negative-prompt", job.req.negative_prompt]
        argv += ["--num-inference-steps", str(job.req.num_inference_steps or 8)]

    if hq or pure:
        if job.req.cfg_scale is not None:
            argv += ["--video-cfg-guidance-scale", str(job.req.cfg_scale)]
        if job.req.stg_scale is not None:
            argv += ["--video-stg-guidance-scale", str(job.req.stg_scale)]

    if pure:
        if job.req.negative_prompt:
            argv += ["--negative-prompt", job.req.negative_prompt]
        # 30 is LTX_2_3_PARAMS' own default for the undistilled schedule. The 8
        # used by the HQ path is a distilled-schedule number and would badly
        # under-denoise here.
        argv += ["--num-inference-steps", str(job.req.num_inference_steps or 30)]

    for lo in job.req.loras:
        argv += ["--lora", str(resolve_lora(lo.name)), str(lo.strength)]

    for i, p in enumerate(job.placement):
        argv += ["--image", str(workdir / f"kf{i + 1}.png"), str(p["index"]), str(p["strength"])]
    argv += [
        "--output-path", str(workdir / "out.mp4"),
        "--width", str(job.req.width), "--height", str(job.req.height),
        "--num-frames", str(job.req.num_frames), "--frame-rate", str(job.req.frame_rate),
        "--seed", str(job.req.seed if job.req.seed is not None else 42),
        "--offload", "cpu",
    ]
    return argv


def run_job(job: Job):
    workdir = JOBS_DIR / job.id
    workdir.mkdir(parents=True, exist_ok=True)
    job.status = "Processing"
    job.started = time.time()
    try:
        for i, kf in enumerate(job.req.keyframes):
            dest = workdir / f"kf{i + 1}.png"
            decode_image(kf.image, dest)
            # Conditioning frames must arrive at the exact generation
            # resolution. LTX does not refuse a mismatch -- decode.py runs
            # `resize_and_center_crop(image, height, width)` on every
            # conditioning image -- so a wrong-sized keyframe is silently
            # rescaled and, if the aspect differs, silently re-framed. A
            # landscape still cropped into a portrait target can lose the
            # subject's head entirely, and nothing in the output says so.
            # storyboard-ui already normalises uploads to the board size, so
            # this only fires when something bypassed it.
            note = normalise(dest, job.req.width, job.req.height, job.req.allow_upscale)
            if note:
                job.placement[i]["resized"] = note
        (workdir / "prompt.txt").write_text(job.req.prompt)

        for lo in job.req.loras:
            hit, total = lora_coverage(resolve_lora(lo.name),
                                       base_transformer(job.req.pipeline))
            job.loras.append({"name": lo.name, "strength": lo.strength,
                              "fused": hit, "targeted": total})
            print(f"[{job.id}] lora {lo.name} @{lo.strength} -> fuses {hit}/{total} weights",
                  flush=True)

        free = free_the_gpu()
        if free < MIN_FREE_GB:
            raise RuntimeError(
                f"only {free:.1f} GB VRAM free after /free (need {MIN_FREE_GB}); "
                f"something else holds the card")

        argv = build_argv(job, workdir)
        (workdir / "cmd.txt").write_text(" \\\n  ".join(argv))
        print(f"[{job.id}] running LTX: {len(job.placement)} keyframes "
              f"@ {[p['index'] for p in job.placement]}, {free:.1f} GB free", flush=True)

        with open(workdir / "ltx.log", "wb") as log:
            proc = subprocess.Popen(argv, cwd=str(LTX_HOME), stdout=log,
                                    stderr=subprocess.STDOUT)
            rc = proc.wait(timeout=JOB_TIMEOUT_S)

        tail = (workdir / "ltx.log").read_text(errors="replace").splitlines()
        job.log_tail = tail
        out = workdir / "out.mp4"
        if rc != 0 or not out.exists():
            raise RuntimeError(f"distilled.py exited {rc} without an output video")
        job.video = str(out)
        job.status = "Done"
        print(f"[{job.id}] done in {time.time() - job.started:.0f}s -> {out}", flush=True)
    except subprocess.TimeoutExpired:
        proc.kill()
        job.status, job.error = "Failed", f"LTX exceeded {JOB_TIMEOUT_S}s"
    except Exception as e:
        job.status, job.error = "Failed", f"{type(e).__name__}: {e}"
        print(f"[{job.id}] FAILED: {job.error}", flush=True)
    finally:
        job.finished = time.time()


def worker():
    while True:
        jid = QUEUE.get()
        job = JOBS.get(jid)
        if job:
            run_job(job)
        QUEUE.task_done()


@app.post("/job")
def submit(req: JobRequest):
    req.num_frames = ltx_grid.valid_num_frames(req.num_frames)
    # 64, not 32. We always pass --spatial-upsampler-path, which makes this the
    # two-stage pipeline, and LTX's own assert_resolution(..., is_two_stage=True)
    # demands 64 (helpers.py). Validating at 32 here would let a 544-wide board
    # through to a ValueError deep inside the run, twenty minutes later.
    for name, v in (("width", req.width), ("height", req.height)):
        if v % 64:
            raise HTTPException(422,
                f"{name}={v} must be divisible by 64. The two-stage distilled "
                f"pipeline (spatial upsampler) requires it; only one-stage "
                f"pipelines accept 32. Nearest: {(v // 64) * 64} or {(v // 64 + 1) * 64}.")
    if req.pipeline not in PIPELINES:
        raise HTTPException(422, f"pipeline must be one of {PIPELINES}, got {req.pipeline!r}")
    if is_23(req.pipeline):
        require_23_assets(req.pipeline)
    if req.pipeline in ("distilled", "ltx23"):
        # Silently ignoring these would be worse: a caller who sent a negative
        # prompt to steer identity drift would think it was doing something.
        unusable = [n for n, v in (("negative_prompt", req.negative_prompt),
                                   ("num_inference_steps", req.num_inference_steps),
                                   ("lora_strength", req.lora_strength),
                                   ("cfg_scale", req.cfg_scale),
                                   ("stg_scale", req.stg_scale),
                                   ("lora_strength_stage_1", req.lora_strength_stage_1),
                                   ("lora_strength_stage_2", req.lora_strength_stage_2))
                    if v is not None]
        if unusable:
            raise HTTPException(422,
                f"{', '.join(unusable)} only apply to pipeline='hq'. The distilled "
                f"transformer has a fixed schedule and no CFG, so there is nothing "
                f"for them to steer.")
    for lo in req.loras:
        path = resolve_lora(lo.name)      # fail at submit, not four minutes in
        hit, total = lora_coverage(path, base_transformer(req.pipeline))
        if total == 0 or hit == 0:
            raise HTTPException(422,
                f"lora {lo.name!r} fuses into 0 of {total} weights on the "
                f"{req.pipeline} transformer -- it would load, log nothing, and have "
                f"no effect. Wrong LoRA format or wrong base model.")
        if hit < total:
            print(f"[warn] lora {lo.name} fuses {hit}/{total} weights "
                  f"({100 * hit / total:.0f}%) -- partial match", flush=True)
        # Every LoRA on this box records a 2.3-era base while the transformers
        # here are 2.5. The shapes line up, so it fuses cleanly and silently --
        # it is a delta trained against different weights, which is tolerable at
        # low strength and destructive at high. Measured the hard way: the same
        # LoRA that produced good video at 0.6 produced unusable output at 1.0.
        base = lora_base_model(resolve_lora(lo.name))
        recorded, target = recorded_version(base), "2.3" if is_23(req.pipeline) else "2.5"
        if recorded and recorded != target and lo.strength > 0.7:
            raise HTTPException(422,
                f"lora {lo.name!r} records base model {base!r} (LTX {recorded}) but this "
                f"pipeline is LTX {target}, and strength {lo.strength} is high for a "
                f"cross-version LoRA. It fuses cleanly and degrades the model rather than "
                f"failing. 0.6 or lower is the range that has produced good output here; "
                f"pipeline 'ltx23-hq' pairs these LoRAs with the base they name.")
    placement = plan(req)
    job = Job(id=uuid.uuid4().hex[:12], req=req, placement=placement)
    with _LOCK:
        JOBS[job.id] = job
    QUEUE.put(job.id)
    print(f"[{job.id}] queued: {len(placement)} keyframes at "
          f"{[p['index'] for p in placement]} / {req.num_frames} frames", flush=True)
    return {"job_id": job.id, "status": job.status, "placement": placement,
            "queue_depth": QUEUE.qsize()}


@app.get("/job/{job_id}")
def status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, f"no such job: {job_id}")
    return job.view(PUBLIC_BASE)


@app.get("/job/{job_id}/video")
def video(job_id: str):
    job = JOBS.get(job_id)
    if not job or not job.video or not Path(job.video).exists():
        raise HTTPException(404, "no video for that job (yet)")
    return FileResponse(job.video, media_type="video/mp4",
                        filename=f"ltx-{job_id}.mp4")


@app.get("/jobs")
def list_jobs():
    return {"jobs": [j.view(PUBLIC_BASE) for j in
                     sorted(JOBS.values(), key=lambda j: -j.created)][:50]}


@app.get("/loras")
def loras():
    """Content LoRAs available to `loras: [{name, strength}]`."""
    if not LORA_DIR.is_dir():
        return {"dir": str(LORA_DIR), "loras": []}
    return {"dir": str(LORA_DIR),
            "loras": sorted(
                ({"name": f.name, "size_gb": round(f.stat().st_size / 1e9, 2),
                  "base_model": lora_base_model(f),
                  **dict(zip(("fused", "targeted"),
                             lora_coverage(f, MODELS / DISTILLED_T)))}
                 for f in LORA_DIR.glob("*.safetensors")),
                key=lambda x: x["name"])}


@app.get("/health")
def health():
    free = None
    try:
        free = requests.get(f"{KEYFRAME_URL}/health", timeout=5).json().get("vram_free_gb")
    except Exception:
        pass
    return {"status": "ok", "ltx_home": str(LTX_HOME),
            "ltx_present": (LTX_HOME / "packages/ltx-pipelines").exists(),
            "models_present": MODELS.exists(),
            "queue_depth": QUEUE.qsize(),
            "running": sum(1 for j in JOBS.values() if j.status == "Processing"),
            "keyframe_server": KEYFRAME_URL, "keyframe_vram_free_gb": free}


PUBLIC_BASE = ""

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8190)
    ap.add_argument("--public-base", default=None,
                    help="base URL clients see, for building video links")
    args = ap.parse_args()
    PUBLIC_BASE = args.public_base or f"http://{args.host}:{args.port}"
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=worker, daemon=True).start()
    print(f"ltx-job-runner on {args.host}:{args.port} | LTX_HOME={LTX_HOME} | "
          f"keyframe-server={KEYFRAME_URL}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)
