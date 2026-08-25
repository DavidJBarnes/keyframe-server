"""6-segment storyboard chain, alternating face and full modes.

Each output is the next segment's input, which is how the web app will drive it.
Scored on skin texture (Laplacian variance over a re-detected face box, since
full mode can move her) and on drift, but the contact sheet is the real verdict —
a metric blind to the failure mode is what made the first chain results wrong.
"""
import base64, io, os, requests, numpy as np, cv2
from PIL import Image

URL = os.environ.get("KEYFRAME_URL", "http://3090.zero:8189")
OUT = os.environ.get("OUT_DIR", "client/outputs/chain")
os.makedirs(OUT, exist_ok=True)
det = cv2.FaceDetectorYN.create(os.environ.get("YUNET", "/tmp/yunet.onnx"), "", (512, 768))

def face_box(im):
    det.setInputSize(im.size)
    _, f = det.detect(cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR))
    if f is None: return None
    x, y, w, h = [int(v) for v in max(f, key=lambda r: r[2] * r[3])[:4]]
    return (max(0,x), max(0,y), min(im.width,x+w), min(im.height,y+h))

def texture(im, box):
    if box is None: return float("nan")
    g = np.asarray(im.convert("L"))[box[1]:box[3], box[0]:box[2]]
    return cv2.Laplacian(g, cv2.CV_64F).var()

def uri(im):
    b = io.BytesIO(); im.save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()

def call(im, **body):
    body["image_urls"] = [uri(im)]
    r = requests.post(f"{URL}/generate", json=body, timeout=900)
    r.raise_for_status()
    return Image.open(io.BytesIO(base64.b64decode(
        r.json()["images"][0]["url"].split(",", 1)[1]))).convert("RGB")

STEPS = [
    ("face", "slight smile",        {"mode": "face", "expression": {"smile": 0.35}}),
    ("full", "charcoal turtleneck", {"mode": "full", "seed": 11, "width": 512, "height": 768,
        "prompt": "Change her top to a charcoal grey turtleneck. Keep everything else identical."}),
    ("face", "glance left",         {"mode": "face", "expression": {"pupil_x": -7, "rotate_yaw": 6}}),
    ("full", "sunlit kitchen bg",   {"mode": "full", "seed": 22, "width": 512, "height": 768,
        "prompt": "Change the background to a sunlit kitchen. Keep the woman identical."}),
    ("face", "warmer smile",        {"mode": "face", "expression": {"smile": 0.7, "eyebrow": 3}}),
    ("full", "coffee mug in hand",  {"mode": "full", "seed": 33, "width": 512, "height": 768,
        "prompt": "She is holding a white coffee mug. Keep everything else identical."}),
]

cur = Image.open(os.environ.get("SRC", "client/inputs/richmond/kf1.png")).convert("RGB")
orig = cur.copy()
b0 = face_box(cur); t0 = texture(cur, b0)
cur.save(f"{OUT}/seg0_source.png")
print(f"seg0  source            face {b0} texture {t0:7.1f} (100%)\n")

prev = cur
for i, (mode, label, body) in enumerate(STEPS, 1):
    cur = call(cur, **body)
    bb = face_box(cur)
    t = texture(cur, bb)
    dprev = np.abs(np.asarray(cur, np.int16) - np.asarray(prev, np.int16))
    dorig = (np.abs(np.asarray(cur, np.int16) - np.asarray(orig, np.int16))
             if cur.size == orig.size else None)
    fw = (bb[2]-bb[0]) if bb else 0
    print(f"seg{i}  {mode:4} {label:22} texture {t:7.1f} ({100*t/t0:3.0f}%)  "
          f"face {fw}px  drift vs prev {dprev.mean():5.2f}"
          + (f"  vs orig {dorig.mean():5.2f}" if dorig is not None else ""))
    cur.save(f"{OUT}/seg{i}_{mode}_{label.replace(' ','_')}.png")
    prev = cur
print(f"\nwrote {OUT}")
