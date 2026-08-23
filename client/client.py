#!/usr/bin/env python3
"""generate.py — keyframe generation for LTX multi-frame conditioning.

Backends:
  local : your Qwen-Image-Edit-2511 server on localhost:8188      [default]
  3090  : the same server on 3090.zero:8189
  qwen  : fal-ai/qwen-image-edit-2511
  nb2   : fal-ai/nano-banana-2/edit
  pro   : fal-ai/nano-banana-pro/edit

Edit-based keyframe factory: always edits from source pixels (never fresh
t2i), supports multiple reference images, and optionally normalizes output
to LTX conditioning size. Local images are inlined as base64 data URIs —
no fal storage upload, no CDN auth. Errors are scrubbed so base64 payloads
never flood the terminal.

Usage:
  # local server (default) — start qwen_edit_server.py first
  ./generate.py -i opening.png -p "same woman, now holding a glass of water" -o kf2.png

  # the GPU box, without spelling out the URL (port 8189: 8188 is ComfyUI there)
  ./generate.py --model 3090 -i opening.png -p "..." -o kf2.png

  # or an explicit server (default http://localhost:8188/edit, or $QWEN_EDIT_URL)
  ./generate.py --server http://otherhost:8189/edit -i opening.png -p "..." -o kf2.png

  # fal backends (require: export FAL_KEY="key-id:key-secret")
  ./generate.py --model qwen -i opening.png -p "..." -o kf2.png
  ./generate.py --model pro  -i kf3.png -p "..." -o kf3-fixed.png

  # multi-ref: source frame + canonical face reference
  ./generate.py -i opening.png -i face_ref.png \
      -p "same woman from image 1 with the face from image 2, seated at the table" \
      -o kf3.png

  # normalize straight to LTX conditioning size (w x h, /32)
  ./generate.py -i opening.png -p "..." -o kf4.png --size 512x768

  # N variants, reproducible seed (seed honored by local backend)
  ./generate.py -i opening.png -p "..." -o kf2.png -n 4 --seed 42

  # override the server's sampler for a hard edit (local backend only)
  ./generate.py -i opening.png -p "..." -o kf2.png --steps 40 --cfg 4.0
"""
import argparse
import base64
import io
import json
import mimetypes
import os
import sys
from pathlib import Path

import requests

FAL_MODELS = {
    "qwen": "fal-ai/qwen-image-edit-2511",
    "nb2": "fal-ai/nano-banana-2/edit",
    "pro": "fal-ai/nano-banana-pro/edit",
}
DEFAULT_SERVER = os.environ.get("QWEN_EDIT_URL", "http://localhost:8188/edit")

# Named local backends, so the common hosts don't need --server spelled out.
# "3090" is the GPU box: the edit server listens on 8189 there because 8188 is
# already ComfyUI. Nothing serves port 80 on that host, so the port is required.
SERVERS = {
    "local": DEFAULT_SERVER,
    "3090": os.environ.get("QWEN_EDIT_URL_3090", "http://3090.zero:8189/edit"),
    # RunPod pod IDs change every launch, so there is no sensible built-in default;
    # it must come from the environment or --server.
    "runpod": os.environ.get("QWEN_EDIT_URL_RUNPOD", ""),
}


def parse_size(s: str):
    try:
        w, h = (int(x) for x in s.lower().split("x"))
    except ValueError:
        sys.exit(f"--size must look like 512x768, got: {s}")
    if w % 32 or h % 32:
        sys.exit(f"--size {w}x{h}: both dimensions must be divisible by 32 for LTX conditioning")
    return w, h


def to_data_uri(path: Path) -> str:
    """Inline a local image as a base64 data URI."""
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def scrub(obj):
    """Truncate long strings (data URIs, base64) so error output stays readable."""
    if isinstance(obj, str):
        return obj[:80] + f"...[{len(obj)} chars]" if len(obj) > 200 else obj
    if isinstance(obj, dict):
        return {k: scrub(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub(v) for v in obj]
    return obj


def normalize(img_bytes: bytes, size):
    """Resize-to-cover + center-crop to exact conditioning dimensions."""
    from PIL import Image  # lazy: only needed with --size

    w, h = size
    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    scale = max(w / im.width, h / im.height)
    im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
    left, top = (im.width - w) // 2, (im.height - h) // 2
    return im.crop((left, top, left + w, top + h))


def run_local(server: str, payload: dict, api_key: str | None = None) -> dict:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        # 900s: a 40-step edit measured ~192s, and a cold server has to move
        # weights to the GPU on the first request as well.
        r = requests.post(server, json=payload, timeout=900, headers=headers)
    except requests.ConnectionError:
        sys.exit(f"cannot reach local server at {server} — is qwen_edit_server.py running?")
    if r.status_code == 401:
        sys.exit("server rejected the API key (401). Set --api-key or $KEYFRAME_API_KEY "
                 "to match the API_KEY the server was started with.")
    if r.status_code == 503:
        sys.exit("server is up but the model is still loading (503). Try again shortly.")
    if not r.ok:
        try:
            detail = scrub(r.json())
        except ValueError:
            detail = scrub(r.text)
        sys.exit(f"local server error {r.status_code}:\n{json.dumps(detail, indent=2)}")
    return r.json()


def run_fal(model_key: str, payload: dict, quiet: bool) -> dict:
    if not os.environ.get("FAL_KEY"):
        sys.exit("FAL_KEY is not set. Get one at fal.ai dashboard -> Keys, then: export FAL_KEY=...")
    import fal_client  # lazy: only needed for fal backends

    def on_update(update):
        if not quiet and isinstance(update, fal_client.InProgress):
            for log in update.logs:
                print(f"  [fal] {log['message']}", file=sys.stderr)

    try:
        return fal_client.subscribe(
            FAL_MODELS[model_key],
            arguments=payload,
            with_logs=not quiet,
            on_queue_update=on_update,
        )
    except Exception as e:
        detail = None
        resp = getattr(e, "response", None)
        if resp is not None:
            try:
                detail = scrub(resp.json())
            except Exception:
                detail = None
        if detail is None:
            detail = scrub(str(e))
        sys.exit(f"fal request failed:\n{json.dumps(detail, indent=2)}")


def fetch_bytes(url: str) -> bytes:
    if url.startswith("data:"):
        return base64.b64decode(url.split(",", 1)[1])
    return requests.get(url, timeout=120).content


def main():
    ap = argparse.ArgumentParser(description="Keyframe gen: local Qwen server or fal.ai")
    ap.add_argument("-i", "--image", action="append", required=True,
                    help="input image path or URL (repeatable; first = source frame)")
    ap.add_argument("-p", "--prompt", required=True, help="edit instruction")
    ap.add_argument("-o", "--output", required=True, help="output path (.png)")
    ap.add_argument("-n", "--num", type=int, default=1, help="variants to generate (1-4)")
    ap.add_argument("--model", choices=[*SERVERS, *FAL_MODELS], default="local",
                    help="backend: local (default, localhost:8188), 3090 (3090.zero:8189), "
                         "runpod ($QWEN_EDIT_URL_RUNPOD), qwen, nb2, pro")
    ap.add_argument("--api-key", default=os.environ.get("KEYFRAME_API_KEY"),
                    help="Bearer token for a protected server (env KEYFRAME_API_KEY)")
    ap.add_argument("--server", default=None,
                    help="explicit server URL; overrides the host implied by --model "
                         f"(local={SERVERS['local']}, 3090={SERVERS['3090']}; "
                         "env QWEN_EDIT_URL / QWEN_EDIT_URL_3090)")
    ap.add_argument("--seed", type=int, default=None,
                    help="generation seed (honored by the local backend)")
    ap.add_argument("--steps", type=int, default=None, metavar="N",
                    help="sampling steps (local only). Default: let the server decide — "
                         "4 when its Lightning LoRA is active, 40 when it is not. Only "
                         "override when you know which the server resolved.")
    ap.add_argument("--cfg", type=float, default=None, metavar="F",
                    help="true_cfg_scale (local only). Default: server's value, which "
                         "pairs with its step count (1.0 for 4-step, 4.0 for 40-step)")
    ap.add_argument("--size", type=parse_size, default=None,
                    help="normalize output to WxH (e.g. 512x768), /32 enforced")
    ap.add_argument("--quiet", action="store_true", help="suppress queue logs")
    args = ap.parse_args()

    if args.steps is not None and args.steps < 1:
        sys.exit(f"--steps must be >= 1, got {args.steps}")
    if args.cfg is not None and args.cfg <= 0:
        sys.exit(f"--cfg must be > 0, got {args.cfg}")

    image_urls = []
    for src in args.image:
        if src.startswith(("http://", "https://")):
            image_urls.append(src)
        else:
            path = Path(src)
            if not path.exists():
                sys.exit(f"input not found: {src}")
            image_urls.append(to_data_uri(path))

    payload = {
        "prompt": args.prompt,
        "image_urls": image_urls,
        "num_images": max(1, min(args.num, 4)),
    }

    if args.model in SERVERS:
        server = args.server or SERVERS[args.model]
        if not server:
            sys.exit(f"--model {args.model} has no URL. Set "
                     f"$QWEN_EDIT_URL_{args.model.upper()} or pass --server "
                     f"https://<podid>-8888.proxy.runpod.net/edit")
        if args.seed is not None:
            payload["seed"] = args.seed
        # Steps/cfg deliberately default to None rather than a literal: the correct
        # value depends on whether the server's Lightning LoRA actually attached, and
        # only the server knows that. Hardcoding 4 here would silently produce 4-step
        # sampling on a server running without Lightning — garbage, with no error.
        if args.steps is not None:
            payload["num_inference_steps"] = args.steps
        if args.cfg is not None:
            payload["true_cfg_scale"] = args.cfg
        result = run_local(server, payload, args.api_key)
    else:
        # fal endpoints reject unknown keys, and pick their own sampler settings.
        for flag, val in (("--steps", args.steps), ("--cfg", args.cfg),
                          ("--seed", args.seed), ("--server", args.server),
                          ("--api-key", args.api_key)):
            if val is not None:
                print(f"warning: {flag} is ignored by --model {args.model} (local backends only)",
                      file=sys.stderr)
        payload["output_format"] = "png"
        result = run_fal(args.model, payload, args.quiet)

    images = result.get("images", [])
    if not images:
        sys.exit(f"no images returned:\n{json.dumps(scrub(result), indent=2)}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    for idx, img in enumerate(images, start=1):
        data = fetch_bytes(img["url"])
        dest = out if len(images) == 1 else out.with_name(f"{out.stem}_{idx}{out.suffix}")
        if args.size:
            normalize(data, args.size).save(dest, "PNG")
        else:
            dest.write_bytes(data)
        print(dest)


if __name__ == "__main__":
    main()
