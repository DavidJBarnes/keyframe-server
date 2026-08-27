#!/usr/bin/env bash
set -uo pipefail
echo "============================================"
echo "  ltx-comfy (ComfyUI + LTX-2.3 nodes)"
echo "  image build: ${GIT_SHA:-unknown}  $(date -Is)"
echo "============================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "WARNING: no GPU visible"

echo "--- model roots ---"
for d in diffusion_models loras latent_upscale_models text_encoders; do
    p="${MODELS_DIR}/ltx-2.3/$d"
    if [ -d "$p" ]; then
        echo "  $d: $(ls -1 "$p" 2>/dev/null | head -3 | tr '\n' ' ')"
    else
        echo "  $d: MISSING at $p"
    fi
done

echo "--- custom nodes ---"
ls -1 /app/ComfyUI/custom_nodes | grep -v __pycache__ | sed 's/^/  /'

echo "--- starting ComfyUI on 0.0.0.0:${COMFY_PORT} ---"
cd /app/ComfyUI
exec python main.py --listen 0.0.0.0 --port "${COMFY_PORT}" \
    --extra-model-paths-config /opt/extra_model_paths.yaml \
    --preview-method none
