#!/usr/bin/env python3
"""qwen_edit_server.py — local Qwen-Image-Edit-2511 behind a FastAPI endpoint.

Mirrors the fal request/response shape so generate.py can target it with
--model local. Input: {prompt, image_urls (data URIs or http URLs), num_images}.
Output: {images: [{url: data URI, content_type, file_name}]}.

Quantization modes (pick per VRAM budget on a 24GB 3090):
  --quant nunchaku : INT4 SVDQuant transformer (fastest load, most headroom;
                     leaves room to keep LTX partially resident)
  --quant fp8      : torchao float8 weight-only on the bf16 transformer
  --quant none     : bf16 + model CPU offload (slowest, max quality)

Lightning LoRA (4-step) is loaded by default; disable with --no-lightning
for full 40-step quality on hard edits.

Run:
  python qwen_edit_server.py --host 0.0.0.0 --port 8188 --quant nunchaku
"""
import argparse
import base64
import io
import re
import uuid

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

MODEL_ID = "Qwen/Qwen-Image-Edit-2511"
# Check huggingface.co/nunchaku-tech for the exact 2511 repo/file name for your
# GPU generation (INT4 for pre-Blackwell like the 3090, FP4 for RTX 50-series).
NUNCHAKU_REPO = "nunchaku-tech/nunchaku-qwen-image-edit-2511"
LIGHTNING_LORA = (
    "lightx2v/Qwen-Image-Lightning",
    "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
)

app = FastAPI(title="qwen-edit-local")
PIPE = None
STEPS = 4
CFG = 1.0


class EditRequest(BaseModel):
    prompt: str
    image_urls: list[str] = Field(min_length=1, max_length=3)
    num_images: int = Field(default=1, ge=1, le=4)
    seed: int | None = None
    num_inference_steps: int | None = None
    true_cfg_scale: float | None = None


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
    out = PIPE(
        image=images if len(images) > 1 else images[0],
        prompt=req.prompt,
        num_inference_steps=req.num_inference_steps or STEPS,
        true_cfg_scale=req.true_cfg_scale or CFG,
        num_images_per_prompt=req.num_images,
        generator=gen,
    )
    return {
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


@app.get("/health")
def health():
    free, total = torch.cuda.mem_get_info()
    return {"status": "ok", "vram_free_gb": round(free / 1e9, 2), "vram_total_gb": round(total / 1e9, 2)}


def load_pipeline(quant: str, lightning: bool):
    global PIPE, STEPS, CFG
    from diffusers import QwenImageEditPipeline

    if quant == "nunchaku":
        from nunchaku import NunchakuQwenImageTransformer2DModel
        from nunchaku.utils import get_precision

        transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
            f"{NUNCHAKU_REPO}/svdq-{get_precision()}_r128-qwen-image-edit-2511.safetensors"
        )
        pipe = QwenImageEditPipeline.from_pretrained(
            MODEL_ID, transformer=transformer, torch_dtype=torch.bfloat16
        )
        pipe.enable_model_cpu_offload()
    elif quant == "fp8":
        from diffusers import QwenImageTransformer2DModel
        from torchao.quantization import quantize_, Float8WeightOnlyConfig

        transformer = QwenImageTransformer2DModel.from_pretrained(
            MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        quantize_(transformer, Float8WeightOnlyConfig())
        pipe = QwenImageEditPipeline.from_pretrained(
            MODEL_ID, transformer=transformer, torch_dtype=torch.bfloat16
        )
        pipe.enable_model_cpu_offload()
    else:
        pipe = QwenImageEditPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()

    if lightning:
        repo, weight = LIGHTNING_LORA
        pipe.load_lora_weights(repo, weight_name=weight)
        STEPS, CFG = 4, 1.0
    else:
        STEPS, CFG = 40, 4.0

    PIPE = pipe


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8188)
    ap.add_argument("--quant", choices=["nunchaku", "fp8", "none"], default="nunchaku")
    ap.add_argument("--no-lightning", action="store_true", help="full 40-step sampling instead of 4-step Lightning")
    args = ap.parse_args()
    load_pipeline(args.quant, not args.no_lightning)
    uvicorn.run(app, host=args.host, port=args.port)
