#!/usr/bin/env bash
# Compare checkpoints on a fixed set of edits, so v19 vs v23 is a measurement
# rather than an impression. Run from client/.
#
#   ../tools/ckpt_battery.sh <tag>
#
# <tag> labels the outputs (e.g. v23, v19). Switch the server's checkpoint
# between runs with:
#   docker rm -f keyframe-server && docker run ... -e CKPT_NAME=<file> ...
#
# Every case uses a FIXED seed and a source already at the target aspect, since
# aspect mismatch destroys identity independently of the checkpoint.
set -uo pipefail
TAG="${1:?usage: ckpt_battery.sh <tag>}"
OUT="battery/$TAG"
mkdir -p "$OUT"
PY=${PY:-python}
SEED=4242

# Face source for the micro cases. richmond/kf1.png is a real photograph already
# normalised to 512x768 with the face at ~19% of frame. Override with FACE=... if
# it is missing — inputs/ is gitignored, so fixtures can vanish.
FACE="${FACE:-inputs/richmond/kf1.png}"
if [ ! -f "$FACE" ]; then
    echo "  face source not found: $FACE" >&2
    echo "  set FACE=<path to a 512x768 portrait> and rerun" >&2
    exit 1
fi
echo "  face source: $FACE"

run() {
    local name="$1"; shift
    local t0=$(date +%s)
    $PY client.py --model 3090 --seed $SEED "$@" -o "$OUT/$name.png" >/dev/null 2>&1 \
        && echo "  $(printf '%-22s' "$name") $(( $(date +%s)-t0 ))s" \
        || echo "  $(printf '%-22s' "$name") FAILED"
}

echo "=== battery: $TAG ==="

# --- MACRO: whole-scene changes (the mode that already works) ---
run macro_garment   -i inputs/pour/pour1.png \
    -p "Change her sweater to a purple tank top. Keep everything else identical."
run macro_colour    -i inputs/pour/pour1.png \
    -p "Change her sweater to a bright yellow sweater. Keep everything else identical."
run macro_scene     -i inputs/pour/pour1.png \
    -p "Change the background to a sunlit garden patio. Keep her and her clothing identical."

# --- MICRO: small facial changes (the gap we are measuring) ---
run micro_smile     -i $FACE \
    -p "Make her smile slightly wider. Keep her face, hair, clothing and background otherwise identical."
run micro_eyes      -i $FACE \
    -p "Open her eyes slightly wider. Keep everything else identical."
run micro_none      -i $FACE \
    -p "Keep this photograph exactly as it is. Change nothing."

# --- IDENTITY: does the person survive a large edit ---
run identity_large  -i $FACE \
    -p "Change her top to a teal bikini top. Keep her face, hair and the background exactly the same."

# --- MULTI-REF: the mechanism the keyframe workflow depends on ---
if [ -f inputs/richmond/kf1.png ] && [ -f inputs/richmond/kf2.png ]; then
    run multiref -i inputs/richmond/kf1.png -i inputs/richmond/kf2.png \
        -p "The woman from image 1 holding the same cream mug shown in image 2, raised to her lips as she sips."
fi
echo "  -> $OUT"
