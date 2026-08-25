#!/usr/bin/env python3
"""Score two battery runs against each other.

Usage:  tools/battery_compare.py client/battery/v23 client/battery/v19

Reports, per case:
  drift  — mean absolute pixel difference from the SOURCE image. For micro cases
           lower is better (the edit should be small); for macro cases a larger
           number is expected and not itself a fault.
  face   — mean absolute difference restricted to the detected face box, which
           is what identity preservation actually turns on.

The `micro_none` case is the noise floor: it asks for no change at all, so its
drift is how much the checkpoint alters an image unbidden. A checkpoint whose
floor is comparable to its micro-edit drift cannot do micro edits.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

# case -> source image it should be compared against
SOURCES = {
    "macro_garment": "inputs/pour/pour1.png",
    "macro_colour": "inputs/pour/pour1.png",
    "macro_scene": "inputs/pour/pour1.png",
    "micro_smile": "inputs/test/k2_crop.png",
    "micro_eyes": "inputs/test/k2_crop.png",
    "micro_none": "inputs/test/k2_crop.png",
    "identity_large": "inputs/test/k2_crop.png",
    "multiref": "inputs/richmond/kf1.png",
}
CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def face_box(im):
    faces = sorted(CASCADE.detectMultiScale(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), 1.05, 5),
                   key=lambda b: -b[2] * b[3])
    return faces[0] if len(faces) else None


def score(src_path: Path, out_path: Path):
    src, out = cv2.imread(str(src_path)), cv2.imread(str(out_path))
    if src is None or out is None:
        return None
    if src.shape != out.shape:
        out = cv2.resize(out, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    drift = float(np.abs(src.astype(float) - out.astype(float)).mean())
    fb = face_box(src)
    face = None
    if fb is not None:
        x, y, w, h = fb
        face = float(np.abs(src[y:y+h, x:x+w].astype(float) - out[y:y+h, x:x+w].astype(float)).mean())
    return drift, face


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    dirs = [Path(a) for a in sys.argv[1:]]
    client = Path(__file__).resolve().parent.parent / "client"
    names = [d.name for d in dirs]

    print(f"  {'case':16s} " + " ".join(f"{n:>18s}" for n in names))
    print(f"  {'':16s} " + " ".join(f"{'drift / face':>18s}" for _ in names))
    for case, rel in SOURCES.items():
        src = client / rel
        if not src.exists():
            continue
        cells = []
        for d in dirs:
            p = d / f"{case}.png"
            s = score(src, p) if p.exists() else None
            if s is None:
                cells.append(f"{'-':>18s}")
            else:
                drift, face = s
                f = f"{face:5.2f}" if face is not None else "  -  "
                cells.append(f"{drift:6.2f} / {f:>6s}")
        print(f"  {case:16s} " + " ".join(cells))
    print()
    print("  micro_none is the noise floor — how much the checkpoint changes an image")
    print("  when asked to change nothing. Micro edits are only meaningful above it.")


if __name__ == "__main__":
    main()
