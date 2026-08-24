#!/usr/bin/env bash
set -uo pipefail
echo "============================================"
echo "  keyframe-server — starting $(date -Is)"
echo "  image build: ${GIT_SHA:-unknown}"
echo "============================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "WARNING: no GPU visible"

if [ "${SKIP_DOWNLOAD:-0}" != "1" ]; then
    /opt/provision.sh || { echo "FATAL: provisioning failed"; exit 1; }
else
    echo "SKIP_DOWNLOAD=1 — assuming models are already on the volume"
fi

echo "--- starting auth proxy on ${PROXY_PORT:-8888} ---"
python3 /opt/auth-proxy.py &

LIGHT_FLAG=""
[ "${LIGHTNING:-1}" = "0" ] && LIGHT_FLAG="--no-lightning"

LORA_FLAG=""
if [ -n "${LORA:-}" ]; then
    if [ -f "$LORA" ]; then
        LORA_FLAG="--lora $LORA ${LORA_WEIGHT:-1.0}"
        echo "--- adapter: $LORA at ${LORA_WEIGHT:-1.0} ---"
    else
        echo "WARNING: LORA=$LORA not found in the container — is the directory mounted?"
    fi
fi

echo "--- starting edit server (quant=${QUANT:-fp8}) on 127.0.0.1:${SERVER_PORT:-8189} ---"
# Bind the model server to loopback only: the auth proxy is the sole public door.
# expandable_segments cuts fragmentation, which matters because fp8 leaves only a
# narrow margin on a 24GB card.
exec env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python3 /opt/server.py \
        --quant "${QUANT:-fp8}" \
        --port "${SERVER_PORT:-8189}" \
        --host 127.0.0.1 \
        $LIGHT_FLAG $LORA_FLAG
