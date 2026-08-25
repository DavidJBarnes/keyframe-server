#!/usr/bin/env python3
"""server.py — Qwen-Image-Edit behind a FastAPI endpoint, backed by ComfyUI.

Mirrors the fal request/response shape so client.py can target it unchanged:
  in : {prompt, image_urls (data URIs or http URLs), num_images, ...}
  out: {images: [{url: data URI, content_type, file_name}]}

Why ComfyUI rather than diffusers (the previous implementation is on the
`diffusers` branch):

  * The community distributes Qwen edit models as all-in-one ComfyUI
    checkpoints with accelerators and LoRAs already merged. diffusers cannot
    load them at all — QwenImageEditPlusPipeline has no from_single_file, and
    diffusers' single-file loader has no Qwen entries.
  * A baked checkpoint needs no runtime adapter, which sidesteps the two
    failures that blocked LoRA support on 24GB: adapters and the fp8 base
    together exceed VRAM, and attaching a LoRA stopped CPU offload from
    releasing VRAM between requests.
  * LyCORIS formats (LoKr) that diffusers has no converter for work here.

The model is a merge: transformer + VAE + CLIP in one file, already fp8, with
Lightning baked in — hence 4 steps at cfg 1.
"""
import argparse
import base64
import contextlib
import io
import json
import os
import re
import socket
import sys
import time
import uuid

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

COMFY = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
CKPT = os.environ.get("CKPT_NAME", "Qwen-Rapid-AIO-NSFW-v23.safetensors")
# v23's author-recommended sampler. Earlier versions differ, so this is settable.
SAMPLER = os.environ.get("SAMPLER", "euler_ancestral")
SCHEDULER = os.environ.get("SCHEDULER", "beta")
STEPS = int(os.environ.get("STEPS", "4"))
CFG = float(os.environ.get("CFG", "1.0"))
# Ceiling for the auto-chosen output size. Only applies when the caller does not
# specify width/height.
MAX_MP = float(os.environ.get("MAX_MP", "1.2"))
BIND = ("0.0.0.0", 8189)


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    host, port = BIND
    shown = "localhost" if host in ("0.0.0.0", "127.0.0.1") else host
    print("\n" + "=" * 62, flush=True)
    print("  READY — accepting requests", flush=True)
    print(f"    POST   http://{shown}:{port}/edit", flush=True)
    print(f"    GET    http://{shown}:{port}/health", flush=True)
    print(f"    backend ComfyUI {COMFY}  ckpt={CKPT}", flush=True)
    print(f"    {SAMPLER}/{SCHEDULER}  steps={STEPS}  cfg={CFG}", flush=True)
    print("=" * 62 + "\n", flush=True)
    yield
    print("\n[server] shutting down", flush=True)


app = FastAPI(title="qwen-edit-comfy", lifespan=lifespan)


class EditRequest(BaseModel):
    prompt: str
    image_urls: list[str] = Field(min_length=1, max_length=3)
    num_images: int = Field(default=1, ge=1, le=4)
    seed: int | None = None
    num_inference_steps: int | None = None
    true_cfg_scale: float | None = None
    negative_prompt: str | None = None
    # <1.0 starts sampling from the source image's latent instead of noise, so
    # only part of the original is redrawn. 0.2-0.4 gives small adjustments;
    # 1.0 (default) is a full regeneration conditioned on the reference images,
    # which is how Qwen edit models normally run.
    denoise: float | None = Field(default=None, gt=0.0, le=1.0)
    # Absent, output matches the first input image. The diffusers path chose its
    # own ~1MP size and returned 832x1248 for a 512x768 input, which forced a
    # resize on every keyframe; here the latent is ours to set.
    width: int | None = None
    height: int | None = None


def decode_image(src: str) -> Image.Image:
    if src.startswith("data:"):
        b64 = re.sub(r"^data:[^,]+,", "", src)
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    if src.startswith(("http://", "https://")):
        return Image.open(io.BytesIO(requests.get(src, timeout=60).content)).convert("RGB")
    raise HTTPException(400, f"image_urls entries must be data URIs or http(s) URLs, got: {src[:40]}")


def encode_image(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def comfy_upload(im: Image.Image) -> str:
    """Put an image in ComfyUI's input folder; returns the stored filename."""
    buf = io.BytesIO()
    im.save(buf, "PNG")
    name = f"kfs_{uuid.uuid4().hex}.png"
    r = requests.post(
        f"{COMFY}/upload/image",
        files={"image": (name, buf.getvalue(), "image/png")},
        data={"overwrite": "true"},
        timeout=120,
    )
    r.raise_for_status()
    return r.json().get("name", name)


def build_workflow(req: EditRequest, filenames: list[str], w: int, h: int) -> dict:
    """ComfyUI API-format graph, wired as Phr00t's reference workflow."""
    steps = req.num_inference_steps or STEPS
    cfg = req.true_cfg_scale if req.true_cfg_scale is not None else CFG
    seed = req.seed if req.seed is not None else int.from_bytes(os.urandom(4), "big")
    denoise = req.denoise if req.denoise is not None else 1.0

    wf: dict = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "9": {"class_type": "EmptyLatentImage",
              "inputs": {"width": w, "height": h, "batch_size": req.num_images}},
        "4": {"class_type": "TextEncodeQwenImageEditPlus",
              "inputs": {"prompt": req.negative_prompt or "",
                         "clip": ["1", 1], "vae": ["1", 2]}},
        "3": {"class_type": "TextEncodeQwenImageEditPlus",
              "inputs": {"prompt": req.prompt, "clip": ["1", 1], "vae": ["1", 2]}},
        "2": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["3", 0], "negative": ["4", 0],
                         "latent_image": ["9", 0], "seed": seed, "steps": steps,
                         "cfg": cfg, "sampler_name": SAMPLER,
                         "scheduler": SCHEDULER, "denoise": denoise}},
        "5": {"class_type": "VAEDecode", "inputs": {"samples": ["2", 0], "vae": ["1", 2]}},
        "6": {"class_type": "SaveImage",
              "inputs": {"images": ["5", 0], "filename_prefix": "keyframe"}},
    }
    # Condition images are optional inputs image1..image3 on the positive encoder.
    for i, fn in enumerate(filenames[:3], start=1):
        node = str(100 + i)
        wf[node] = {"class_type": "LoadImage", "inputs": {"image": fn, "upload": "image"}}
        wf["3"]["inputs"][f"image{i}"] = [node, 0]

    if denoise < 1.0:
        # Seed the sampler with the source image rather than noise. Requires the
        # first condition image to already match the output size — the encoder
        # emits a latent at the image's own dimensions, and KSampler cannot mix
        # that with a differently shaped one.
        wf["110"] = {"class_type": "VAEEncode",
                     "inputs": {"pixels": ["101", 0], "vae": ["1", 2]}}
        wf["2"]["inputs"]["latent_image"] = ["110", 0]
        wf.pop("9", None)
    return wf


def run_workflow(wf: dict, timeout_s: int = 900) -> list[dict]:
    """Submit, poll to completion, return the SaveImage output descriptors."""
    r = requests.post(f"{COMFY}/prompt", json={"prompt": wf}, timeout=60)
    if r.status_code != 200:
        # ComfyUI reports graph validation errors here; surfacing the body makes
        # a bad node name or input obvious instead of a generic 500.
        raise HTTPException(502, f"ComfyUI rejected the workflow: {r.text[:600]}")
    pid = r.json()["prompt_id"]

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        h = requests.get(f"{COMFY}/history/{pid}", timeout=30)
        if h.status_code == 200 and (entry := h.json().get(pid)):
            status = entry.get("status", {})
            if status.get("status_str") == "error" or status.get("completed") is False:
                raise HTTPException(502, f"ComfyUI execution failed: {json.dumps(status)[:600]}")
            outs = [img for node in entry.get("outputs", {}).values()
                    for img in node.get("images", [])]
            if outs:
                return outs
        time.sleep(1.0)
    raise HTTPException(504, f"ComfyUI did not finish within {timeout_s}s")


def fetch_output(desc: dict) -> bytes:
    r = requests.get(f"{COMFY}/view", timeout=120, params={
        "filename": desc["filename"],
        "subfolder": desc.get("subfolder", ""),
        "type": desc.get("type", "output"),
    })
    r.raise_for_status()
    return r.content


@app.post("/edit")
def edit(req: EditRequest):
    t0 = time.time()
    rid = uuid.uuid4().hex[:8]
    images = [decode_image(u) for u in req.image_urls]
    t_decode = time.time() - t0

    # Latent defaults to the first input's size, but capped: compute scales with
    # output pixels (0.39MP ~6s, 1.55MP ~21s, 7.09MP ~186s), and a phone photo
    # fed in raw is ~7MP. Aspect ratio is preserved; explicit width/height wins.
    if req.width and req.height:
        w, h = req.width, req.height
    else:
        w, h = images[0].width, images[0].height
        mp = (w * h) / 1e6
        if mp > MAX_MP:
            scale = (MAX_MP / mp) ** 0.5
            w, h = int(w * scale), int(h * scale)
            print(f"[edit] input {images[0].width}x{images[0].height} ({mp:.2f}MP) "
                  f"capped to {w}x{h} ({MAX_MP}MP). Pass width/height to override.",
                  flush=True)
    w, h = max(16, (w // 16) * 16), max(16, (h // 16) * 16)

    if req.denoise is not None and req.denoise < 1.0 and (images[0].width, images[0].height) != (w, h):
        # VAEEncode produces a latent sized to the image it is given, so the
        # source must already be the output size or the sampler shapes disagree.
        images[0] = images[0].resize((w, h), Image.LANCZOS)
        print(f"[edit] denoise<1: resized source to {w}x{h} to match the latent", flush=True)

    srcs = " ".join(f"{im.width}x{im.height}" for im in images)
    print(f"[{rid}] in {len(images)} img ({srcs}) -> out {w}x{h} | "
          f"steps={req.num_inference_steps or STEPS} "
          f"cfg={req.true_cfg_scale if req.true_cfg_scale is not None else CFG} "
          f"denoise={req.denoise if req.denoise is not None else 1.0} "
          f"seed={req.seed if req.seed is not None else 'random'}", flush=True)
    print(f'[{rid}] prompt: {req.prompt[:110]}{"..." if len(req.prompt) > 110 else ""}',
          flush=True)

    t1 = time.time()
    filenames = [comfy_upload(im) for im in images]
    t_upload = time.time() - t1

    t2 = time.time()
    outs = run_workflow(build_workflow(req, filenames, w, h))
    t_run = time.time() - t2

    t3 = time.time()
    payload = {
        "images": [
            {
                "url": encode_image(Image.open(io.BytesIO(fetch_output(o))).convert("RGB")),
                "content_type": "image/png",
                "file_name": f"{uuid.uuid4().hex}.png",
            }
            for o in outs
        ],
        "description": "",
    }
    t_fetch = time.time() - t3

    try:
        free, total = _vram()
        vram = f" | vram {free:.1f}/{total:.1f} GB free"
    except Exception:
        vram = ""
    print(f"[{rid}] done in {time.time() - t0:.1f}s "
          f"(decode {t_decode:.1f} upload {t_upload:.1f} "
          f"generate {t_run:.1f} fetch {t_fetch:.1f}) "
          f"-> {len(outs)} image(s){vram}", flush=True)
    return payload


def _vram() -> tuple[float, float]:
    r = requests.get(f"{COMFY}/system_stats", timeout=5)
    r.raise_for_status()
    dev = (r.json().get("devices") or [{}])[0]
    return dev.get("vram_free", 0) / 1e9, dev.get("vram_total", 0) / 1e9


@app.get("/health")
def health():
    try:
        free, total = _vram()
        return {"status": "ok", "backend": "comfyui", "ckpt": CKPT,
                "vram_free_gb": round(free, 2), "vram_total_gb": round(total, 2),
                "vram_low": free < 2.0}
    except Exception as e:
        raise HTTPException(503, f"ComfyUI backend not ready: {type(e).__name__}")


def assert_port_free(host: str, port: int):
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind((host, port))
    except OSError as e:
        sys.exit(f"port {port} on {host} is already in use ({e.strerror}). "
                 f"Pass a free port with --port.")
    finally:
        probe.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Qwen edit endpoint backed by ComfyUI")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8189)
    ap.add_argument("--comfy-url", default=COMFY, help=f"ComfyUI base URL (default {COMFY})")
    ap.add_argument("--ckpt", default=CKPT, help=f"checkpoint name (default {CKPT})")
    ap.add_argument("--wait-for-comfy", type=int, default=600,
                    help="seconds to wait for ComfyUI before serving")
    args = ap.parse_args()

    COMFY, CKPT = args.comfy_url, args.ckpt
    assert_port_free(args.host, args.port)
    BIND = (args.host, args.port)

    print(f"[server] waiting for ComfyUI at {COMFY} ...", flush=True)
    deadline = time.time() + args.wait_for_comfy
    while time.time() < deadline:
        try:
            if requests.get(f"{COMFY}/system_stats", timeout=5).status_code == 200:
                print("[server] ComfyUI is up", flush=True)
                break
        except Exception:
            pass
        time.sleep(3)
    else:
        sys.exit(f"ComfyUI did not come up at {COMFY} within {args.wait_for_comfy}s")

    uvicorn.run(app, host=args.host, port=args.port)
