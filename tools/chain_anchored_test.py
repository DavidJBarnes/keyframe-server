"""Same 6-segment storyboard, but every segment is rebuilt from segment 0.

State accumulates in the *instructions*, not in the pixels: scene changes fold
into one cumulative full-mode prompt, expression changes into one cumulative
expression object. So generation depth is at most 2 for every segment, no matter
how far into the sequence it sits — instead of growing to 6.
"""
import base64, io, os, requests, numpy as np, cv2
from PIL import Image

URL = os.environ.get("KEYFRAME_URL", "http://3090.zero:8189")
OUT = os.environ.get("OUT_DIR", "client/outputs/anchored")
os.makedirs(OUT, exist_ok=True)
det = cv2.FaceDetectorYN.create(os.environ.get("YUNET", "/tmp/yunet.onnx"), "", (512, 768))

def face_box(im):
    det.setInputSize(im.size)
    _, f = det.detect(cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR))
    if f is None: return None
    x, y, w, h = [int(v) for v in max(f, key=lambda r: r[2]*r[3])[:4]]
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

# Same six beats as the chained run, so the comparison is like-for-like.
BEATS = [
    ("face", "slight smile",        {"smile": 0.35}),
    ("full", "charcoal turtleneck", "Her top is a charcoal grey turtleneck"),
    ("face", "glance left",         {"pupil_x": -7, "rotate_yaw": 6}),
    ("full", "sunlit kitchen bg",   "the background is a sunlit kitchen"),
    ("face", "warmer smile",        {"smile": 0.7, "eyebrow": 3}),
    ("full", "coffee mug in hand",  "she is holding a white coffee mug"),
]

orig = Image.open(os.environ.get("SRC", "client/inputs/richmond/kf1.png")).convert("RGB")
b0 = face_box(orig); t0 = texture(orig, b0)
orig.save(f"{OUT}/seg0_source.png")
print(f"seg0  source                     texture {t0:7.1f} (100%)   depth 0\n")

scene, expr = [], {}
for i, (mode, label, payload) in enumerate(BEATS, 1):
    if mode == "face":
        expr.update(payload)          # latest value per axis wins
    else:
        scene.append(payload)

    cur, depth = orig, 0
    if scene:
        prompt = ("Change this image so that " + ", and ".join(scene) +
                  ". Keep the woman's face and identity identical.")
        cur = call(cur, mode="full", prompt=prompt, seed=11, width=512, height=768)
        depth += 1
    if expr:
        cur = call(cur, mode="face", expression=expr)
        depth += 1

    bb = face_box(cur); t = texture(cur, bb)
    d = np.abs(np.asarray(cur, np.int16) - np.asarray(orig, np.int16))
    print(f"seg{i}  {mode:4} {label:22} texture {t:7.1f} ({100*t/t0:3.0f}%)   "
          f"depth {depth}  drift vs orig {d.mean():5.2f}")
    cur.save(f"{OUT}/seg{i}_{mode}_{label.replace(' ','_')}.png")
print(f"\nwrote {OUT}")
