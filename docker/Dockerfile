# keyframe-server — Qwen-Image-Edit-2511 behind an authenticated HTTP endpoint.
#
# The 57.7GB model is deliberately NOT baked in. It is fetched at startup onto a
# RunPod network volume so it survives pod restarts; baking it would make the
# image impractical to push, pull and store.
#
# Versions are pinned to the combination validated on a 3090 (2026-08-22).
# torch 2.12 rather than latest: nunchaku's wheels top out there, so staying on
# 2.12 keeps the door open if nunchaku ever supports diffusers 0.40.
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# torch first, on its own, so a resolver backtrack can't silently pick a
# different build. Default PyPI wheels bundle their own CUDA runtime.
RUN pip install --no-cache-dir "torch==2.12.1" "torchvision==0.27.1"

# peft is REQUIRED and is not a hard dependency of diffusers. Without it the
# Lightning LoRA cannot patch any layer, and load_lora_weights fails with
# "unsupported target modules" — which silently drops the server to 40-step
# sampling instead of 4-step.
RUN pip install --no-cache-dir \
        "diffusers==0.40.0" \
        "transformers==5.15.1" \
        "accelerate==1.14.0" \
        "torchao==0.18.0" \
        "peft==0.20.0" \
        "safetensors" "sentencepiece" "protobuf" \
        "fastapi==0.141.1" "uvicorn==0.52.4" \
        "pillow" "requests" \
        "huggingface_hub[cli]" "hf_xet"

# Model cache lives on the network volume, not in the container.
# LORA is read by start.sh and passed through as --lora. Mount the directory:
#   -v ~/models/qwen:/workspace/loras -e LORA=/workspace/loras/x.safetensors
ENV HF_HOME=/workspace/hf \
    QUANT=fp8 \
    SERVER_PORT=8189 \
    PROXY_PORT=8888 \
    LIGHTNING=1 \
    SKIP_DOWNLOAD=0

# API_KEY is intentionally NOT declared here — it is a runtime secret,
# supplied by the RunPod template. auth-proxy.py defaults it to empty
# and warns loudly when unset.
# Stamped by CI so a running container can be traced to a commit; mismatched
# images are otherwise invisible until behaviour differs.
ARG GIT_SHA=unknown
ENV GIT_SHA=$GIT_SHA

COPY server.py /opt/server.py
COPY docker/provision.sh /opt/provision.sh
COPY docker/auth-proxy.py /opt/auth-proxy.py
COPY docker/start.sh /opt/start.sh
RUN chmod +x /opt/start.sh /opt/provision.sh /opt/auth-proxy.py

EXPOSE 8888

CMD ["/opt/start.sh"]
