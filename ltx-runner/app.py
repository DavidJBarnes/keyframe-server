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
import io
import os
import queue
import re
import shutil
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
JOB_TIMEOUT_S = int(os.environ.get("JOB_TIMEOUT_S", "5400"))


class Keyframe(BaseModel):
    # data URI or http(s) URL, exactly like keyframe-server's image_urls
    image: str
    # Omit both and the recipe's defaults are applied across the whole set.
    index: int | None = None
    strength: float | None = Field(default=None, ge=0.0, le=1.0)


class JobRequest(BaseModel):
    prompt: str = Field(min_length=1)
    keyframes: list[Keyframe] = Field(min_length=1, max_length=12)
    width: int = 512
    height: int = 768
    num_frames: int = 121
    frame_rate: int = 24
    seed: int | None = None
    # Off-grid indices are a hard error by default. The whole point of this
    # service is that the caller cannot get placement subtly wrong and then spend
    # a day wondering why the keyframes "don't work"; set this to snap instead.
    snap_indices: bool = False


@dataclass
class Job:
    id: str
    req: JobRequest
    status: str = "None"          # None -> Processing -> Done | Failed
    video: str | None = None
    error: str | None = None
    placement: list[dict] = field(default_factory=list)
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


def plan(req: JobRequest) -> list[dict]:
    """Resolve every keyframe to an on-grid (index, strength). Raises 422 if it cannot."""
    n = len(req.keyframes)
    auto_idx = ltx_grid.auto_place(n, req.num_frames)
    auto_str = ltx_grid.default_strengths(n)
    out = []
    for i, kf in enumerate(req.keyframes):
        idx = auto_idx[i] if kf.index is None else kf.index
        snapped = False
        if not ltx_grid.is_on_grid(idx):
            if not req.snap_indices:
                raise HTTPException(422,
                    f"keyframe {i}: index {idx} is off the LTX latent grid. The video "
                    f"encoder is causal with temporal scale 8, so the only valid indices "
                    f"are 0 or 1+8k -- nearest legal values are "
                    f"{ltx_grid.snap(idx - 4)} and {ltx_grid.snap(idx + 4)}. An off-grid "
                    f"index is not snapped for you: the guide token lands between two "
                    f"latent slots and smears across both. Pass snap_indices=true to "
                    f"snap automatically.")
            idx, snapped = ltx_grid.snap(idx), True
        if idx > req.num_frames:
            raise HTTPException(422,
                f"keyframe {i}: index {idx} is past num_frames={req.num_frames}")
        out.append({"index": idx,
                    "strength": auto_str[i] if kf.strength is None else kf.strength,
                    "snapped_from": kf.index if snapped else None})
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


def build_argv(job: Job, workdir: Path) -> list[str]:
    py = LTX_HOME / "venv" / "bin" / "python"
    argv = [
        str(py if py.exists() else "python"),
        "packages/ltx-pipelines/src/ltx_pipelines/distilled.py",
        "--transformer-path", str(MODELS / "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"),
        "--text-encoder-path", str(MODELS / "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"),
        "--video-vae-path", str(MODELS / "vae/ltx-2.5-video-vae-bf16.safetensors"),
        "--audio-vae-path", str(MODELS / "vae/ltx-2.5-audio-vae-bf16.safetensors"),
        "--spatial-upsampler-path", str(MODELS / "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"),
        "--prompt", job.req.prompt,
    ]
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
            with Image.open(dest) as im:
                if (im.width, im.height) != (job.req.width, job.req.height):
                    raise RuntimeError(
                        f"keyframe {i + 1} is {im.width}x{im.height} but the clip "
                        f"is {job.req.width}x{job.req.height}. LTX would resize and "
                        f"centre-crop it silently; normalise it first so the crop "
                        f"is your decision, not a side effect.")
        (workdir / "prompt.txt").write_text(job.req.prompt)

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
