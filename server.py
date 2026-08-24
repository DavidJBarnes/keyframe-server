#!/usr/bin/env python3
"""qwen_edit_server.py — local Qwen-Image-Edit-2511 behind a FastAPI endpoint.

Mirrors the fal request/response shape so generate.py can target it with
--model local. Input: {prompt, image_urls (data URIs or http URLs), num_images}.
Output: {images: [{url: data URI, content_type, file_name}]}.

Quantization modes, as measured on a 24GB 3090 (2026-08-22):
  --quant fp8      : DEFAULT and the only mode that works on 24GB. torchao
                     float8 weight-only, ~20GB transformer, ~44-72s per edit.
  --quant none     : bf16 + model CPU offload. OOMs on 24GB (the transformer is
                     ~40GB and offload swaps whole models). Use on 48GB+ cards.
  --quant nunchaku : BROKEN against diffusers 0.40 -- nunchaku calls a changed
                     QwenEmbedRope signature. Also mutually exclusive with the
                     Lightning LoRA, since PEFT cannot patch its INT4 layers.

Lightning LoRA (4-step) is loaded by default; disable with --no-lightning
for full 40-step quality on hard edits.

Run:
  python server.py

Defaults are the validated ones: --quant fp8, --port 8189 (8188 is usually
ComfyUI), 4-step Lightning, and expandable_segments to limit fragmentation.
"""
import argparse
import base64
import contextlib
import gc
import io
import os
import re
import socket
import sys
import uuid

# Must be set before torch creates a CUDA context. fp8 leaves a narrow margin on
# a 24GB card and fragmentation alone can push an edit into OOM, so default it on
# rather than making every caller remember the env var.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

MODEL_ID = "Qwen/Qwen-Image-Edit-2511"

# nunchaku-tech/* does not exist and nunchaku-ai (the real org) has no 2511 build,
# so the INT4 transformer comes from a community quantisation. Filenames there are
# nunchaku_qwen_image_edit_2511_{variant}_{precision}.safetensors — NOT the
# svdq-{precision}_r128-* scheme the official repos use.
#   precision: int4 for pre-Blackwell (3090 = sm_86), fp4 for RTX 50-series
#   variant:   ultimate_speed (11.5GB) | balance (12.7GB) | best_quality (14.2GB)
QUANT_MODE = "fp8"  # resolved at startup; shown in the ready banner
NUNCHAKU_REPO = "QuantFunc/Nunchaku-Qwen-Image-EDIT-2511"
NUNCHAKU_VARIANT = "balance"

# No 2511 Lightning LoRA has been published — lightx2v/Qwen-Image-Lightning stops
# at 2509. The 2509 4-step LoRA is architecturally compatible in principle but is
# NOT officially supported on 2511; load_pipeline falls back to full sampling if
# it refuses to load.
LIGHTNING_LORA = (
    "lightx2v/Qwen-Image-Lightning",
    "Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors",
)

PIPE = None
STEPS = 4
CFG = 1.0
BIND = ("0.0.0.0", 8189)  # filled in from argv; used only for the ready banner


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    # Fires once uvicorn is actually serving. load_pipeline() finishing is NOT
    # the same thing — the port is not listening until after it returns, so a
    # "ready" printed there is a lie you can act on too early.
    host, port = BIND
    shown = "localhost" if host in ("0.0.0.0", "127.0.0.1") else host
    print("\n" + "=" * 62, flush=True)
    print("  READY — accepting requests", flush=True)
    print(f"    POST   http://{shown}:{port}/edit", flush=True)
    print(f"    GET    http://{shown}:{port}/health", flush=True)
    print(f"    quant={QUANT_MODE}  steps={STEPS}  cfg={CFG}", flush=True)
    print("=" * 62 + "\n", flush=True)
    yield
    print("\n[server] shutting down", flush=True)


app = FastAPI(title="qwen-edit-local", lifespan=lifespan)


class EditRequest(BaseModel):
    prompt: str
    image_urls: list[str] = Field(min_length=1, max_length=3)
    num_images: int = Field(default=1, ge=1, le=4)
    seed: int | None = None
    num_inference_steps: int | None = None
    true_cfg_scale: float | None = None
    negative_prompt: str | None = None


def decode_image(src: str) -> Image.Image:
    if src.startswith("data:"):
        b64 = re.sub(r"^data:[^,]+,", "", src)
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    if src.startswith(("http://", "https://")):
        import requests as rq

        return Image.open(io.BytesIO(rq.get(src, timeout=60).content)).convert("RGB")
    raise HTTPException(400, f"image_urls entries must be data URIs or http(s) URLs, got: {src[:40]}")


def encode_image(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.post("/edit")
def edit(req: EditRequest):
    if PIPE is None:
        raise HTTPException(503, "pipeline not loaded")
    images = [decode_image(u) for u in req.image_urls]
    gen = None
    if req.seed is not None:
        gen = torch.Generator(device="cuda").manual_seed(req.seed)
    cfg = req.true_cfg_scale or CFG
    negative = req.negative_prompt

    # diffusers gates guidance on BOTH knobs:
    #     do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
    # so asking for cfg > 1 without a negative prompt silently disables guidance
    # entirely — the request looks honoured and adherence is quietly poor. Supply
    # a blank negative prompt so the knob does what the caller asked. diffusers'
    # own docs note that even " " is enough to switch CFG on.
    if cfg > 1 and not negative:
        negative = " "

    try:
        out = PIPE(
            image=images if len(images) > 1 else images[0],
            prompt=req.prompt,
            negative_prompt=negative,
            num_inference_steps=req.num_inference_steps or STEPS,
            true_cfg_scale=cfg,
            num_images_per_prompt=req.num_images,
            generator=gen,
        )
    except torch.OutOfMemoryError as e:
        # Without this the failed run's allocations stay resident and EVERY
        # subsequent request OOMs too — one transient failure permanently
        # poisons the server until it is restarted.
        gc.collect()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        print(f"[edit] OOM — recovered, vram free {free / 1e9:.1f} / {total / 1e9:.1f} GB",
              flush=True)
        raise HTTPException(
            503,
            f"GPU out of memory ({free / 1e9:.1f} GB free after cleanup). "
            "Another process may be holding VRAM; retry shortly.",
        ) from e
    payload = {
        "images": [
            {
                "url": encode_image(im),
                "content_type": "image/png",
                "file_name": f"{uuid.uuid4().hex}.png",
            }
            for im in out.images
        ],
        "description": "",
    }

    # Release cached blocks between requests. With model CPU offload the weights
    # move off the GPU, but PyTorch's caching allocator keeps the freed blocks,
    # and varying image sizes fragment them. Left alone the process creeps from
    # ~11GB free to nothing over a session and then every request 500s on OOM —
    # which reads like an unrelated failure hours after the real cause.
    del out
    gc.collect()
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    print(f"[edit] done — vram free {free / 1e9:.1f} / {total / 1e9:.1f} GB", flush=True)

    return payload


@app.get("/health")
def health():
    free, total = torch.cuda.mem_get_info()
    return {
        "status": "ok",
        "vram_free_gb": round(free / 1e9, 2),
        "vram_total_gb": round(total / 1e9, 2),
        # Below roughly 2GB free the next edit is likely to OOM. Surfaced so a
        # monitor can catch the creep rather than discovering it as a 500.
        "vram_low": free / 1e9 < 2.0,
    }


# The bf16 transformer is ~40GB and is held in host RAM for CPU offload, so a
# second server on the same box needs another 40GB. On a 61GB machine that is an
# OOM, and an OOM here can take the whole host down — which is unrecoverable if
# nobody can reach the power button.
APPROX_HOST_RAM_GB = 42.0


def assert_enough_ram(quant: str, force: bool = False):
    """Refuse to start if host RAM cannot hold the model.

    Guards specifically against launching a second server alongside a running
    one: the port check below won't catch it (different ports are fine) but they
    share host memory, and losing the box to an OOM is far worse than refusing
    to boot.
    """
    try:
        with open("/proc/meminfo") as f:
            info = {k.split(":")[0]: int(v.split()[0]) for k, v in
                    (line.split(":", 1) for line in f)}
        available_gb = info["MemAvailable"] / 1024 / 1024
    except Exception as e:
        print(f"[preflight] could not read /proc/meminfo ({e}); skipping RAM check",
              flush=True)
        return

    need = APPROX_HOST_RAM_GB
    print(f"[preflight] host RAM available {available_gb:.1f} GB, need ~{need:.0f} GB",
          flush=True)
    if available_gb >= need:
        return

    others = []
    try:
        import glob
        for d in glob.glob("/proc/[0-9]*"):
            try:
                cmd = open(f"{d}/cmdline", "rb").read().replace(b"\0", b" ").decode()
            except OSError:
                continue
            # Must look like *this* server, not any script named server.py —
            # a bare name match flags unrelated projects and misleads.
            argv0 = cmd.split(" ", 1)[0]
            is_ours = (
                "python" in argv0                       # not a shell wrapper quoting us
                and "server.py" in cmd
                and ("--quant" in cmd or "keyframe-server" in cmd or "/opt/server.py" in cmd)
            )
            if is_ours and d.split("/")[-1] != str(os.getpid()):
                others.append(f"    pid {d.split('/')[-1]}: {cmd.strip()[:70]}")
    except Exception:
        pass

    msg = [
        f"REFUSING TO START: only {available_gb:.1f} GB of host RAM available, "
        f"and the {quant} path needs roughly {need:.0f} GB.",
        "",
        "The transformer is held in host RAM for CPU offload. Starting anyway "
        "risks an OOM that can take the whole machine down, not just this process.",
    ]
    if others:
        msg += ["", "Another server appears to be running already:"] + others + [
            "", "Stop it first (or `docker stop keyframe-server`)."]
    else:
        msg += ["", "Free memory, or stop whatever else is holding it."]
    msg += ["", "Override with --allow-low-ram if you are sure."]
    if force:
        print("\n".join(msg), flush=True)
        print("[preflight] --allow-low-ram given; continuing anyway", flush=True)
        return
    sys.exit("\n".join(msg))


def assert_port_free(host: str, port: int):
    """Fail before loading weights, not after.

    uvicorn binds only once load_pipeline() returns, so a port collision
    surfaces several minutes in — the model loads fine and then the process
    dies on "address already in use". ComfyUI habitually occupies 8188 on these
    boxes, which is exactly how this bites.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind((host, port))
    except OSError as e:
        sys.exit(
            f"port {port} on {host} is already in use ({e.strerror}).\n"
            f"Something else is listening — ComfyUI commonly holds 8188.\n"
            f"Pass a free port, e.g. --port 8189, or stop the other process.\n"
            f"Checked before loading the model so you don't wait minutes to find out."
        )
    finally:
        probe.close()


def merge_lokr(transformer, path: str, strength: float = 1.0):
    """Merge a LyCORIS LoKr adapter into transformer weights, in place.

    diffusers has no LoKr converter — load_lora_weights() dies with an unhelpful
    "state_dict should be empty at this point". But LoKr is simply a Kronecker
    factorisation of the weight delta:

        dW = (alpha / dim) * kron(w1, w2)
        W' = W + strength * dW

    so it can be reconstructed with plain torch. Must run on the bf16 weights
    BEFORE fp8 quantisation: torchao replaces them with tensor subclasses that a
    bf16 delta cannot cleanly be added to.

    Key names are matched by stripping all separators rather than by mangling
    underscores back into dots — "to_out_0" -> "to_out.0" but "add_k_proj" keeps
    its underscores, and no rule distinguishes those reliably.
    """
    from safetensors.torch import load_file

    sd = load_file(path)
    groups: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in sd.items():
        base, _, leaf = k.rpartition(".")
        groups.setdefault(base, {})[leaf] = v

    # normalised model param name -> (real name, parameter)
    params = {}
    collisions = 0
    for name, prm in transformer.named_parameters():
        if not name.endswith(".weight"):
            continue
        norm = name[: -len(".weight")].replace(".", "").replace("_", "").lower()
        if norm in params:
            collisions += 1
        params[norm] = (name, prm)
    if collisions:
        print(f"[lokr] WARNING: {collisions} ambiguous parameter names", flush=True)

    merged = skipped = 0
    with torch.no_grad():
        for base, parts in groups.items():
            if "lokr_w1" not in parts or "lokr_w2" not in parts:
                skipped += 1
                continue
            norm = base.replace("lora_unet_", "").replace(".", "").replace("_", "").lower()
            hit = params.get(norm)
            if hit is None:
                skipped += 1
                continue
            name, prm = hit
            w1 = parts["lokr_w1"].to(torch.float32)
            w2 = parts["lokr_w2"].to(torch.float32)
            delta = torch.kron(w1, w2)
            if tuple(delta.shape) != tuple(prm.shape):
                skipped += 1
                continue
            scale = strength
            if "alpha" in parts:
                scale *= float(parts["alpha"].item()) / w1.shape[0]
            prm.add_(delta.to(prm.dtype).to(prm.device) * scale)
            merged += 1

    print(f"[lokr] merged {merged} modules (skipped {skipped}) "
          f"from {path} at strength {strength}", flush=True)
    if merged == 0:
        print("[lokr] WARNING: nothing merged — names did not match the model", flush=True)
    return merged


def adapter_format(path: str) -> str:
    """Peek at the safetensors header to tell LoKr from a standard LoRA."""
    import json as _json
    import struct as _struct

    with open(path, "rb") as f:
        n = _struct.unpack("<Q", f.read(8))[0]
        header = _json.loads(f.read(n))
    header.pop("__metadata__", None)
    keys = list(header)
    if any("lokr_w" in k for k in keys):
        return "lokr"
    if any("hada_w" in k for k in keys):
        return "loha"
    if any(("lora_down" in k or "lora_A" in k) for k in keys):
        return "lora"
    return "unknown"


def load_pipeline(quant: str, lightning: bool, lora: tuple[str, float] | None = None):
    global PIPE, STEPS, CFG, QUANT_MODE
    QUANT_MODE = quant
    # Qwen-Image-Edit-2511's own model_index.json declares
    #     _class_name: QwenImageEditPlusPipeline
    # Loading the plain QwenImageEditPipeline instead "works" — weights load, edits
    # come out — but the Plus pipeline preprocesses the condition image differently
    # (separate 384x384 VL-encoder path, condition_image_sizes, VAE_IMAGE handling).
    # Using the wrong class degrades edit fidelity: identity and hair drift, and
    # structural edits get skipped in favour of a recolour.
    from diffusers import QwenImageEditPlusPipeline

    if quant == "nunchaku":
        from nunchaku import NunchakuQwenImageTransformer2DModel
        from nunchaku.utils import get_precision

        weights = (
            f"{NUNCHAKU_REPO}/"
            f"nunchaku_qwen_image_edit_2511_{NUNCHAKU_VARIANT}_{get_precision()}.safetensors"
        )
        print(f"[nunchaku] loading {weights}")
        transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(weights)
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            MODEL_ID, transformer=transformer, torch_dtype=torch.bfloat16
        )
        pipe.enable_model_cpu_offload()
    elif quant == "fp8":
        from diffusers import QwenImageTransformer2DModel
        from torchao.quantization import quantize_, Float8WeightOnlyConfig

        transformer = QwenImageTransformer2DModel.from_pretrained(
            MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        # A LoKr must be merged into bf16 weights BEFORE quantising — torchao
        # replaces them with tensor subclasses a bf16 delta cannot be added to.
        # A standard LoRA is attached to the built pipeline instead, further down.
        if lora and adapter_format(lora[0]) == "lokr":
            merge_lokr(transformer, lora[0], lora[1])
        quantize_(transformer, Float8WeightOnlyConfig())
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            MODEL_ID, transformer=transformer, torch_dtype=torch.bfloat16
        )
        pipe.enable_model_cpu_offload()
    else:
        pipe = QwenImageEditPlusPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()

    if lora:
        path, strength = lora
        fmt = adapter_format(path)
        if fmt == "lokr":
            if quant != "fp8":
                print(f"[lora] LoKr merging is only wired into the fp8 path; "
                      f"quant={quant} ignored it", flush=True)
        elif fmt == "lora":
            try:
                pipe.load_lora_weights(path, adapter_name="user")
                pipe.set_adapters(["user"], adapter_weights=[strength])
                print(f"[lora] attached {path} at {strength}", flush=True)
            except Exception as e:
                print(f"[lora] FAILED to attach ({type(e).__name__}: {e})", flush=True)
        else:
            print(f"[lora] UNSUPPORTED format '{fmt}' for {path} — "
                  f"diffusers handles LoRA and this server adds LoKr; "
                  f"LoHa is not supported. Continuing without it.", flush=True)

    if lightning:
        repo, weight = LIGHTNING_LORA
        applied = False
        try:
            pipe.load_lora_weights(repo, weight_name=weight)
            # A clean return does NOT mean the LoRA applied. When a target module
            # is unsupported — nunchaku's quantised SVDQW4A4Linear is — diffusers
            # logs "Loading default_0 was unsuccessful" and returns normally. Left
            # unchecked that silently yields 4-step sampling with no Lightning,
            # i.e. garbage. Verify an adapter is actually active.
            try:
                applied = bool(pipe.get_active_adapters())
            except Exception:
                # Older diffusers without the introspection API: fall back to
                # checking whether any LoRA layers were attached.
                applied = bool(getattr(pipe, "peft_config", None))
        except Exception as e:
            print(f"[lightning] load raised ({type(e).__name__}: {e})")

        if applied:
            STEPS, CFG = 4, 1.0
            print(f"[lightning] active -> {STEPS} steps, cfg {CFG}")
        else:
            # Known case: --quant nunchaku. PEFT cannot patch INT4 SVDQuant
            # layers, so Lightning and nunchaku are mutually exclusive today.
            # Full sampling is correct-but-slow; 4 steps here would be wrong.
            print("[lightning] NOT applied (unsupported target modules?)")
            print("[lightning] falling back to full 40-step sampling")
            STEPS, CFG = 40, 4.0
    else:
        STEPS, CFG = 40, 4.0

    print(f"[pipeline] loaded: quant={quant} steps={STEPS} cfg={CFG} "
          f"(not serving yet — waiting for the port)", flush=True)

    PIPE = pipe


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    # 8189 rather than 8188: ComfyUI conventionally owns 8188 on these boxes.
    ap.add_argument("--port", type=int, default=8189)
    # fp8 is the default because it is the only mode verified working on a 24GB
    # card: nunchaku is broken against diffusers 0.40, and bf16 ("none") OOMs
    # because model-level offload cannot fit a ~40GB transformer in 23.5GB.
    ap.add_argument("--quant", choices=["nunchaku", "fp8", "none"], default="fp8")
    ap.add_argument("--no-lightning", action="store_true", help="full 40-step sampling instead of 4-step Lightning")
    ap.add_argument("--lora", nargs="+", metavar=("PATH", "WEIGHT"), default=None,
                    help="adapter to apply at startup: --lora /path/to.safetensors [0.8]. "
                         "Server-side path. LoRA is attached at runtime; LoKr is merged "
                         "into the weights (fp8 path only) and so needs a restart to change. "
                         "Check compatibility first with tools/check_lora.py")
    ap.add_argument("--allow-low-ram", action="store_true",
                    help="start even if host RAM looks insufficient (risks an OOM "
                         "that can take the machine down)")
    ap.add_argument("--nunchaku-variant", default=NUNCHAKU_VARIANT,
                    choices=["ultimate_speed", "balance", "best_quality"],
                    help="INT4 build to load with --quant nunchaku (speed vs quality)")
    args = ap.parse_args()
    NUNCHAKU_VARIANT = args.nunchaku_variant
    lora = None
    if args.lora:
        if len(args.lora) > 2:
            ap.error("--lora takes a path and an optional weight")
        lora_path = os.path.expanduser(args.lora[0])
        if not os.path.exists(lora_path):
            sys.exit(f"--lora: no such file on the server: {lora_path}")
        try:
            lora_weight = float(args.lora[1]) if len(args.lora) == 2 else 1.0
        except ValueError:
            ap.error(f"--lora weight must be a number, got {args.lora[1]!r}")
        lora = (lora_path, lora_weight)
    elif os.environ.get("LOKR_PATH"):          # back-compat with the env form
        lora = (os.environ["LOKR_PATH"], float(os.environ.get("LOKR_WEIGHT", "1.0")))

    assert_enough_ram(args.quant, args.allow_low_ram)
    assert_port_free(args.host, args.port)
    BIND = (args.host, args.port)
    print(f"[server] loading model (quant={args.quant}) — this takes a few minutes",
          flush=True)
    load_pipeline(args.quant, not args.no_lightning, lora)
    uvicorn.run(app, host=args.host, port=args.port)
