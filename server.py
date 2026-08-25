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

import cv2
import numpy as np
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
# Face mode: ExpressionEditor's crop_factor. How far past the detected face box
# to take the warp region — it needs hairline and jaw to anchor against, not the
# bare jaw-to-brow box. This no longer controls a composite seam (the node masks
# its own), so it is purely about how much context the warp sees.
FACE_PAD = float(os.environ.get("FACE_PAD", "1.6"))
# Detail restoration for face mode. ExpressionEditor decodes through a fixed
# 256x256 bottleneck, so it softens the entire crop — including the parts it did
# not move. Measured on a 214x292 face: texture falls to 16% of source, and
# crop_factor cannot fix it (the node clamps to 1.5-2.5, and 1.5 is already the
# sharp end). These thresholds ramp the handover on |output - source|: below LO
# the node changed nothing worth keeping, so the source pixel wins.
DETAIL_LO = float(os.environ.get("DETAIL_LO", "6.0"))
DETAIL_HI = float(os.environ.get("DETAIL_HI", "22.0"))
DETAIL_BLUR = float(os.environ.get("DETAIL_BLUR", "2.0"))
# Fraction of the crop's smaller side blended at the boundary.
# YuNet, a small DNN detector. OpenCV 5 dropped CascadeClassifier from the top
# level, and YuNet is the better tool regardless: Haar cascades miss angled and
# partially occluded faces, which is most of a real keyframe set. The model is a
# ~350KB ONNX file baked into the image.
YUNET_PATH = os.environ.get("YUNET_PATH", "/opt/face_detection_yunet.onnx")
_DETECTOR = None


def _detector(width: int, height: int):
    """YuNet instance sized to the current image (it needs explicit input size)."""
    global _DETECTOR
    if _DETECTOR is None:
        if not os.path.exists(YUNET_PATH):
            raise HTTPException(500, f"face detector model missing at {YUNET_PATH}")
        _DETECTOR = cv2.FaceDetectorYN.create(YUNET_PATH, "", (width, height),
                                              score_threshold=0.6)
    _DETECTOR.setInputSize((width, height))
    return _DETECTOR
BIND = ("0.0.0.0", 8189)


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    host, port = BIND
    shown = "localhost" if host in ("0.0.0.0", "127.0.0.1") else host
    print("\n" + "=" * 62, flush=True)
    print("  READY — accepting requests", flush=True)
    print(f"    POST   http://{shown}:{port}/generate", flush=True)
    print(f"    GET    http://{shown}:{port}/health", flush=True)
    print(f"    mode=full  ComfyUI {COMFY}  ckpt={CKPT}", flush=True)
    print(f"               {SAMPLER}/{SCHEDULER}  steps={STEPS}  cfg={CFG}", flush=True)
    print(f"    mode=face  LivePortrait ExpressionEditor  crop_factor={FACE_PAD}", flush=True)
    print("=" * 62 + "\n", flush=True)
    yield
    print("\n[server] shutting down", flush=True)


app = FastAPI(title="qwen-edit-comfy", lifespan=lifespan)


# --- mode=face: LivePortrait expression parameters ---------------------------
#
# ExpressionEditor warps the source pixels through implicit keypoints and
# composites the result back behind a face mask, so pixels outside the face are
# bit-identical to the input. That is why face mode moved off Qwen: every Qwen
# pass halves skin texture (Laplacian variance 1015 -> ~450) and de-ages the
# subject, at every denoise setting. See docs/dual-pipeline-design.md.
#
# Ranges are the node's own (nodes.py:846-865) and are enforced here so an
# out-of-range value fails as a 422 rather than a ComfyUI validation error.
class Expression(BaseModel):
    rotate_pitch: float = Field(0, ge=-20, le=20)
    rotate_yaw: float = Field(0, ge=-20, le=20)
    rotate_roll: float = Field(0, ge=-20, le=20)
    blink: float = Field(0, ge=-20, le=5)
    eyebrow: float = Field(0, ge=-10, le=15)
    wink: float = Field(0, ge=0, le=25)
    pupil_x: float = Field(0, ge=-15, le=15)
    pupil_y: float = Field(0, ge=-15, le=15)
    aaa: float = Field(0, ge=-30, le=120)      # jaw open
    eee: float = Field(0, ge=-20, le=15)       # wide mouth
    woo: float = Field(0, ge=-20, le=15)       # pursed mouth
    smile: float = Field(0, ge=-0.3, le=1.3)


# Prompt sugar over the numeric contract. Deliberately small and explicit: a
# caller that wants exact control passes `expression` and skips all of this.
# Longer patterns are matched first so "look up" cannot shadow "look upper left".
_LEXICON: list[tuple[str, dict]] = [
    (r"\bwid(e|er) eyes?\b|\beyes? wide\b", {"blink": 4}),
    (r"\bsquint(s|ed|ing)?\b|\bnarrow(ed)? eyes?\b", {"blink": -8}),
    (r"\bclos(e|es|ed|ing) (her |his |their )?eyes?\b|\beyes? clos(ed|ing)\b", {"blink": -18}),
    (r"\bblink(s|ed|ing)?\b", {"blink": -12}),
    (r"\bwink(s|ed|ing)?\b", {"wink": 15}),
    (r"\brais(e|es|ed|ing) (her |his |their )?(eye)?brows?\b|\b(eye)?brows? rais(ed|e)\b", {"eyebrow": 8}),
    (r"\bfurrow(s|ed|ing)?\b|\bfrown(s|ed|ing)?\b", {"eyebrow": -6, "smile": -0.2}),
    (r"\bgrin(s|ning)?\b|\bbig smile\b|\bbroad smile\b", {"smile": 1.0}),
    (r"\bsmil(e|es|ing)\b", {"smile": 0.5}),
    (r"\blaugh(s|ed|ing)?\b", {"smile": 0.9, "aaa": 35}),
    (r"\bmouth open\b|\bopens? (her |his |their )?mouth\b|\bgasp(s|ed|ing)?\b", {"aaa": 45}),
    (r"\bpurs(e|es|ed|ing)\b|\bpout(s|ed|ing)?\b|\bwhistl(e|es|ing)\b", {"woo": 10}),
    (r"\blook(s|ing)? (to (her |his |their )?)?left\b|\bglanc(e|es|ing) left\b", {"pupil_x": -8}),
    (r"\blook(s|ing)? (to (her |his |their )?)?right\b|\bglanc(e|es|ing) right\b", {"pupil_x": 8}),
    (r"\blook(s|ing)? up\b|\bglanc(e|es|ing) up\b|\beyes? up\b", {"pupil_y": 8}),
    (r"\blook(s|ing)? down\b|\bglanc(e|es|ing) down\b|\beyes? down\b|\blower(s|ed|ing)? (her |his |their )?gaze\b", {"pupil_y": -8}),
    (r"\bturn(s|ed|ing)? (her |his |their )?head (to the )?left\b|\bhead left\b", {"rotate_yaw": -12}),
    (r"\bturn(s|ing)? (her |his |their )?head (to the )?right\b|\bhead right\b", {"rotate_yaw": 12}),
    (r"\btilt(s|ed|ing)? (her |his |their )?head\b|\bhead tilt(ed)?\b", {"rotate_roll": 8}),
    (r"\bchin up\b|\blift(s|ed|ing)? (her |his |their )?chin\b|\bhead up\b", {"rotate_pitch": -8}),
    (r"\bchin down\b|\bhead down\b|\bduck(s|ing)? (her |his |their )?head\b", {"rotate_pitch": 8}),
]

# Intensity adverbs scale whatever they precede. Applied globally rather than
# per-phrase: prompts at this length rarely mix intensities, and per-phrase
# scoping would need a parser rather than a regex sweep.
_INTENSITY = [
    (r"\b(slight(ly)?|soft(ly)?|faint(ly)?|soften(ed|s)?|subtle|barely|a little|a bit|gentl[ey])\b", 0.5),
    (r"\b(very|much|strong(ly)?|wide(ly)?|big|broad(ly)?|deep(ly)?|hard)\b", 1.5),
]


def resolve_expression(req: "GenerateRequest") -> tuple[Expression, str]:
    """Numeric expression wins; otherwise read the prompt. Returns (exp, source)."""
    if req.expression is not None:
        return req.expression, "explicit"

    text = (req.prompt or "").lower()
    params: dict[str, float] = {}
    hits: list[str] = []
    for pattern, delta in _LEXICON:
        if re.search(pattern, text):
            hits.append(pattern.split("\\b")[1] if "\\b" in pattern else pattern)
            for k, v in delta.items():
                # Largest magnitude wins when two phrases drive the same axis,
                # so "smiling and grinning" gives one grin, not a summed clamp.
                if abs(v) > abs(params.get(k, 0.0)):
                    params[k] = v

    scale = 1.0
    for pattern, factor in _INTENSITY:
        if re.search(pattern, text):
            scale = factor
            break
    if scale != 1.0:
        params = {k: v * scale for k, v in params.items()}

    if not params and req.image_urls[1:] == []:
        raise HTTPException(
            422,
            "mode=face needs something to apply: an `expression` object, a prompt "
            "using a known term, or a second image to copy an expression from. "
            "Recognised terms: smile, grin, laugh, frown, blink, wink, squint, "
            "wide eyes, closed eyes, raised brows, open mouth, purse/pout, "
            "look left/right/up/down, turn head left/right, tilt head, chin up/down.",
        )

    # Clamp to the node's ranges — an intensity multiplier can overshoot.
    fields = Expression.model_fields
    for k, v in list(params.items()):
        f = fields[k]
        lo = next(m.ge for m in f.metadata if hasattr(m, "ge"))
        hi = next(m.le for m in f.metadata if hasattr(m, "le"))
        params[k] = max(lo, min(hi, v))

    return Expression(**params), ("prompt:" + ",".join(hits) if hits else "driver-only")


class GenerateRequest(BaseModel):
    # Optional in face mode when `expression` is supplied — the transform there
    # is parametric, not textual.
    prompt: str = ""
    # "full": regenerate the whole frame with Qwen. Garments, scenes, props,
    #         composition — anything that needs new pixels invented.
    # "face": warp the face with LivePortrait. Expression, gaze, small head
    #         rotation. Pixels outside the face mask are bit-identical to the
    #         input, and the pixels inside are warped from the source rather
    #         than regenerated, so identity and skin texture survive.
    #
    # The split is not a preference. Qwen cannot do the face case: it halves
    # skin texture on every pass at every denoise setting. LivePortrait cannot
    # do the full case: it only articulates a face it can already see.
    mode: str = Field(default="full", pattern="^(full|face)$")
    face_pad: float | None = Field(default=None, gt=1.0, le=4.0)

    # --- face mode only ---
    expression: Expression | None = None
    # With a second image supplied, its expression is copied onto the first.
    # `sample_ratio` scales the transfer; `sample_parts` limits which channels
    # come across. Numeric `expression` values are added on top.
    sample_ratio: float = Field(default=1.0, ge=-0.2, le=1.2)
    sample_parts: str = Field(
        default="OnlyExpression",
        pattern="^(OnlyExpression|OnlyRotation|OnlyMouth|OnlyEyes|All)$",
    )
    # How much of the subject's own resting expression to retain. Below 1.0 the
    # face relaxes toward neutral before the requested change is applied.
    src_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    # 1.0 restores source texture everywhere the warp did not move anything;
    # 0 returns the node's output untouched. See restore_detail().
    detail_restore: float = Field(default=1.0, ge=0.0, le=1.0)
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


def detect_face(im: Image.Image) -> tuple[int, int, int, int]:
    """Largest face as (x, y, w, h). Raises 422 if none found."""
    bgr = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    _, dets = _detector(im.width, im.height).detect(bgr)
    faces = [] if dets is None else sorted(
        (d[:4] for d in dets), key=lambda b: -(b[2] * b[3]))
    if not len(faces):
        # Deliberately an error rather than a silent fall back to full mode:
        # quietly doing something different is how a caller ends up debugging
        # the wrong thing later.
        raise HTTPException(422, "mode=face: no face detected in the first image")
    return tuple(int(v) for v in faces[0])


def restore_detail(source: Image.Image, edited: Image.Image,
                   strength: float = 1.0) -> Image.Image:
    """Take the source's pixels back wherever LivePortrait did not move anything.

    The node's decoder softens the whole face crop uniformly, but an expression
    change only moves part of it — a smile leaves the forehead alone. Blending
    on |edited - source| keeps the edit where it happened and the original
    texture everywhere else, which measured 16% -> 31% of source texture with no
    visible seam. Alpha is blurred so the handover has no hard edges.

    Not a sharpening filter and not a restoration model: every pixel here comes
    from one of the two real images, so it cannot invent detail or alter age.
    """
    if strength <= 0:
        return edited
    s = np.asarray(source, np.float32)
    e = np.asarray(edited, np.float32)
    if s.shape != e.shape:
        return edited
    d = np.abs(e - s).mean(2)
    a = np.clip((d - DETAIL_LO) / max(1e-6, DETAIL_HI - DETAIL_LO), 0, 1)
    a = cv2.GaussianBlur(a, (0, 0), DETAIL_BLUR)[..., None]
    a = 1.0 - (1.0 - a) * strength      # strength<1 keeps more of the node's output
    return Image.fromarray(np.clip(s * (1 - a) + e * a, 0, 255).astype(np.uint8))


def build_face_workflow(req: GenerateRequest, filenames: list[str],
                        exp: Expression) -> dict:
    """LivePortrait graph for mode=face.

    No checkpoint, no sampler, no VAE — the transform is a keypoint warp, so
    seed/steps/cfg/denoise have no meaning here and are ignored.

    Note the SaveImage wiring: ExpressionEditor is an OUTPUT_NODE whose UI
    preview is the *face crop only* (nodes.py:955). The full-frame composite is
    output 0, so it must be saved explicitly — reading the preview would return
    crops instead of keyframes.
    """
    pad = req.face_pad if req.face_pad is not None else FACE_PAD

    editor = {
        "src_image": ["101", 0],
        "crop_factor": pad,
        "src_ratio": req.src_ratio,
        "sample_ratio": req.sample_ratio,
        "sample_parts": req.sample_parts,
        **exp.model_dump(),
    }

    wf: dict = {
        "101": {"class_type": "LoadImage",
                "inputs": {"image": filenames[0], "upload": "image"}},
        "200": {"class_type": "ExpressionEditor", "inputs": editor},
        "6": {"class_type": "SaveImage",
              "inputs": {"images": ["200", 0], "filename_prefix": "keyframe"}},
    }

    # A second image is a driving reference: its expression is read and applied.
    if len(filenames) > 1:
        wf["102"] = {"class_type": "LoadImage",
                     "inputs": {"image": filenames[1], "upload": "image"}}
        wf["200"]["inputs"]["sample_image"] = ["102", 0]

    return wf


def build_workflow(req: GenerateRequest, filenames: list[str], w: int, h: int) -> dict:
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


def run_workflow(wf: dict, timeout_s: int = 900, save_node: str = "6") -> list[dict]:
    """Submit, poll to completion, return the SaveImage output descriptors.

    Only `save_node`'s images count. Collecting from every output node breaks
    face mode: ExpressionEditor is itself an OUTPUT_NODE and emits a temp
    preview of the *face crop* (nodes.py:955) alongside the real full-frame
    result, so an unfiltered sweep can return a 512x512 crop where the caller
    asked for a 512x768 keyframe.
    """
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
            nodes = entry.get("outputs", {})
            outs = [img for img in nodes.get(save_node, {}).get("images", [])
                    if img.get("type") != "temp"]
            if outs:
                return outs
            # The graph finished but the save node produced nothing: report that
            # rather than silently falling back to another node's preview.
            if nodes and entry.get("status", {}).get("completed"):
                raise HTTPException(
                    502,
                    f"ComfyUI finished but node {save_node} emitted no image "
                    f"(nodes with output: {sorted(nodes)})")
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


@app.post("/generate")
def generate(req: GenerateRequest):
    t0 = time.time()
    rid = uuid.uuid4().hex[:8]
    images = [decode_image(u) for u in req.image_urls]
    t_decode = time.time() - t0

    # --- face mode: LivePortrait, a different pipeline entirely ---------------
    #
    # Returns at the source's own size with everything outside the face mask
    # untouched, so none of the latent-sizing or compositing below applies.
    if req.mode == "face":
        exp, how = resolve_expression(req)
        # Preflight so a faceless frame is a clean 422 rather than an opaque
        # ComfyUI failure. The node's own detector picks the centre-most face
        # while this picks the largest; they only disagree on crowded frames,
        # and this is only ever used to answer "is there a face at all".
        detect_face(images[0])
        nonzero = {k: v for k, v in exp.model_dump().items() if v}
        driver = f" +driver({req.sample_parts}@{req.sample_ratio})" if len(images) > 1 else ""
        print(f"[{rid}] mode=face {images[0].width}x{images[0].height} | "
              f"src={how}{driver} | {nonzero or 'neutral'}", flush=True)

        t1 = time.time()
        filenames = [comfy_upload(im) for im in images[:2]]
        t_upload = time.time() - t1

        t2 = time.time()
        outs = run_workflow(build_face_workflow(req, filenames, exp))
        t_run = time.time() - t2

        t3 = time.time()
        results = [Image.open(io.BytesIO(fetch_output(o))).convert("RGB") for o in outs]
        if req.detail_restore > 0:
            results = [restore_detail(images[0], r, req.detail_restore) for r in results]
        return _respond(rid, t0, results, outs, t_decode, t_upload, t_run, time.time() - t3)

    face_box = None
    original = None

    # Latent defaults to the first input's size, but capped: compute scales with
    # output pixels (0.39MP ~6s, 1.55MP ~21s, 7.09MP ~186s), and a phone photo
    # fed in raw is ~7MP. Aspect ratio is preserved; explicit width/height wins.
    if face_box is not None:
        # Generate at the crop's own size, capped, so the composite is a
        # like-for-like replacement. Caller width/height describe the final
        # frame, which face mode preserves by construction.
        w, h = images[0].width, images[0].height
        mp = (w * h) / 1e6
        if mp > MAX_MP:
            scale = (MAX_MP / mp) ** 0.5
            w, h = int(w * scale), int(h * scale)
    elif req.width and req.height:
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
    print(f"[{rid}] mode={req.mode} in {len(images)} img ({srcs}) -> out {w}x{h} | "
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
    results = [Image.open(io.BytesIO(fetch_output(o))).convert("RGB") for o in outs]
    return _respond(rid, t0, results, outs, t_decode, t_upload, t_run, time.time() - t3)


def _respond(rid, t0, results, outs, t_decode, t_upload, t_run, t_fetch):
    payload = {
        "images": [
            {
                "url": encode_image(im),
                "content_type": "image/png",
                "file_name": f"{uuid.uuid4().hex}.png",
            }
            for im in results
        ],
        "description": "",
    }
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
