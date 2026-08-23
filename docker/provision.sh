#!/usr/bin/env bash
# Fetch the models onto the network volume. Idempotent: `hf download` skips files
# that are already present and complete, so a restart costs seconds, not 57.7GB.
#
# Uses the huggingface_hub downloader rather than aria2 on purpose. HF has moved
# many weights to Xet chunked storage, whose presigned URLs carry short-lived
# per-connection byte ranges that aria2's multi-connection mode cannot satisfy,
# giving intermittent 403s mid-download. hf_xet handles Xet and refreshes URLs.
set -uo pipefail

: "${HF_HOME:=/workspace/hf}"
export HF_HOME
mkdir -p "$HF_HOME"

BASE_REPO="Qwen/Qwen-Image-Edit-2511"
LORA_REPO="lightx2v/Qwen-Image-Lightning"
LORA_FILE="Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors"

echo "=== provisioning models into $HF_HOME ==="
df -h /workspace 2>/dev/null | tail -1 || echo "WARNING: /workspace is not a mount — models will NOT survive a restart"

auth=()
[ -n "${HF_TOKEN:-}" ] && auth=(--token "$HF_TOKEN")

echo "--- base model ($BASE_REPO, 57.7GB) ---"
# --max-workers 1: parallel fetches of the four ~10GB transformer shards contend
# for bandwidth and make partial-resume messier without going faster.
hf download "$BASE_REPO" --max-workers 1 "${auth[@]}" || {
    echo "ERROR: base model download failed"; exit 1; }

echo "--- Lightning LoRA (0.85GB) ---"
# No 2511 LoRA has been published; the 2509 4-step LoRA applies cleanly to the
# 2511 transformer and is what gives 4-step sampling instead of 40.
hf download "$LORA_REPO" "$LORA_FILE" --max-workers 1 "${auth[@]}" || {
    echo "WARNING: LoRA download failed — server will run 40-step sampling"; }

echo "=== provisioning complete ==="
du -sh "$HF_HOME" 2>/dev/null
