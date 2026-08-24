#!/usr/bin/env bash
set -uo pipefail
echo "============================================"
echo "  keyframe-server (ComfyUI backend)"
echo "  image build: ${GIT_SHA:-unknown}  $(date -Is)"
echo "============================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "WARNING: no GPU visible"

CKPT_PATH="${MODELS_DIR}/qwen/v23/${CKPT_NAME}"
if [ ! -f "$CKPT_PATH" ]; then
    echo "WARNING: checkpoint not found at $CKPT_PATH"
    echo "         is ${MODELS_DIR} mounted, and has the download finished?"
    ls -la "${MODELS_DIR}/qwen/v23/" 2>/dev/null || echo "         (directory not present)"
else
    echo "checkpoint: $CKPT_PATH ($(du -h "$CKPT_PATH" | cut -f1))"
fi

# Install the fixed text-encode node over ComfyUI's stock one.
if [ -f /app/fixed_nodes_qwen.py ]; then
    cp /app/fixed_nodes_qwen.py /app/ComfyUI/comfy_extras/nodes_qwen.py
    echo "applied Phr00t's fixed TextEncodeQwenImageEditPlus node"
fi

echo "--- starting ComfyUI on 127.0.0.1:8188 (in-container only) ---"
cd /app/ComfyUI
python main.py --listen 127.0.0.1 --port 8188 \
    --extra-model-paths-config /opt/extra_model_paths.yaml \
    --preview-method none \
    > /tmp/comfyui.log 2>&1 &

echo "--- starting auth proxy on ${PROXY_PORT:-8888} ---"
python3 /opt/auth-proxy.py &

echo "--- starting adapter on 127.0.0.1:${SERVER_PORT:-8189} ---"
exec python3 /opt/server.py \
    --host 127.0.0.1 \
    --port "${SERVER_PORT:-8189}" \
    --comfy-url "${COMFY_URL}" \
    --ckpt "${CKPT_NAME}"
