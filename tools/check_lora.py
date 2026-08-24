#!/usr/bin/env python3
"""Report whether a .safetensors adapter can be loaded by the edit server.

Civitai's "Qwen" base-model filter is necessary but not sufficient: it says
nothing about the *adapter format*, and diffusers only loads true LoRA. LyCORIS
variants (LoKr, LoHa) are listed on Civitai under the same "LORA" type and fail
at load time with an unhelpful "state_dict should be empty" error.

Reads only the safetensors header, so it needs no torch and no GPU, and can be
pointed at a file anywhere.

    ./tools/check_lora.py ~/models/qwen/some_lora.safetensors
"""
import argparse
import json
import re
import struct
import sys
from collections import Counter

# diffusers' Qwen converter understands these; anything else is unsupported.
LORA_PAIRS = (("lora_down", "lora_up"), ("lora_A", "lora_B"))
UNSUPPORTED = {
    "lokr_w1": "LoKr (Kronecker product, LyCORIS)",
    "lokr_w2": "LoKr (Kronecker product, LyCORIS)",
    "hada_w1_a": "LoHa (Hadamard product, LyCORIS)",
    "hada_w2_a": "LoHa (Hadamard product, LyCORIS)",
}


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def main():
    ap = argparse.ArgumentParser(description="Check adapter compatibility with the edit server")
    ap.add_argument("path")
    args = ap.parse_args()

    try:
        header = read_header(args.path)
    except Exception as e:
        sys.exit(f"cannot read as safetensors: {type(e).__name__}: {e}")

    meta = header.pop("__metadata__", {})
    keys = list(header)
    if not keys:
        sys.exit("no tensors found")

    print(f"file:      {args.path}")
    print(f"tensors:   {len(keys)}")
    print(f"dtypes:    {dict(Counter(v['dtype'] for v in header.values()))}")

    # --- format ---
    fmt, blocker = None, None
    for marker, name in UNSUPPORTED.items():
        if any(marker in k for k in keys):
            fmt, blocker = name, marker
            break
    if fmt is None:
        for a, b in LORA_PAIRS:
            if any(a in k for k in keys) and any(b in k for k in keys):
                fmt = f"LoRA ({a}/{b})"
                break
    if any("dora_scale" in k for k in keys):
        fmt = (fmt or "LoRA") + " + DoRA"
    print(f"format:    {fmt or 'UNKNOWN — no recognised adapter tensors'}")

    # --- naming convention ---
    conv = Counter()
    for k in keys:
        if k.startswith("diffusion_model."):
            conv["diffusion_model.* (ComfyUI)"] += 1
        elif k.startswith("transformer."):
            conv["transformer.* (diffusers-native)"] += 1
        elif k.startswith("lora_unet_"):
            conv["lora_unet_* (kohya)"] += 1
        else:
            conv["other"] += 1
    print(f"naming:    {dict(conv)}")

    # --- architecture fit ---
    blocks = {int(m.group(1)) for k in keys
              if (m := re.search(r"transformer_blocks[._](\d+)", k))}
    print(f"blocks:    {len(blocks)} distinct"
          + (f" (0-{max(blocks)})" if blocks else " — no transformer_blocks keys"))
    if blocks and max(blocks) != 59:
        print("           ! Qwen-Image's transformer has 60 blocks (0-59);"
              " this may target a different architecture")

    ranks = sorted({v["shape"][0] for k, v in header.items()
                    if "lora_down" in k or "lora_A" in k})
    if ranks:
        print(f"rank(s):   {ranks}")
    if meta:
        interesting = {k: v for k, v in meta.items()
                       if any(t in k for t in ("resolution", "network_module",
                                               "network_dim", "network_alpha", "base_model"))}
        if interesting:
            print(f"metadata:  {json.dumps(interesting)[:200]}")

    # --- verdict ---
    print()
    if blocker:
        print(f"VERDICT:   NOT LOADABLE — {fmt}")
        print(f"           diffusers has no converter for this format ('{blocker}' keys).")
        print("           load_lora_weights() will raise "
              "'state_dict should be empty at this point'.")
        print("           Ask for a standard LoRA export, or use it in ComfyUI,"
              " which implements LyCORIS.")
        sys.exit(2)
    if fmt is None:
        print("VERDICT:   UNKNOWN — no lora_down/up, lora_A/B or LyCORIS tensors found.")
        print("           Not an adapter, or a format this checker does not know.")
        sys.exit(2)
    if not blocks:
        print("VERDICT:   PROBABLY NOT — no transformer_blocks keys, so it does not"
              " target the Qwen transformer.")
        sys.exit(2)
    print(f"VERDICT:   LOADABLE — {fmt}, targets {len(blocks)} transformer blocks.")
    print("           Format is supported. Whether it produces good results on"
          " Qwen-Image-Edit-2511 is a separate question:")
    print("           most Civitai Qwen LoRAs are trained on Qwen-Image (t2i),"
          " which shares this transformer backbone.")


if __name__ == "__main__":
    main()
