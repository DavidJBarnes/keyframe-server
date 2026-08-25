# keyframe-server — Qwen-Image-Edit behind an authenticated HTTP endpoint,
# backed by ComfyUI running inside this container.
#
# The 28GB all-in-one checkpoint is NOT baked in; it is bind-mounted from the
# host, so the image stays small and the model survives image updates.
#
# ComfyUI listens on loopback INSIDE the container only. Nothing but the auth
# proxy is published, so this never contends for host port 8188 (which on
# 3090.zero belongs to the wanly production container).
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv git curl ca-certificates \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# torch first and alone, so a resolver backtrack cannot quietly pick a
# different build. Default PyPI wheels bundle their own CUDA runtime.
RUN pip install --no-cache-dir "torch==2.12.1" "torchvision==0.27.1"

# ComfyUI pinned to a commit — an unpinned clone means the image silently
# changes behaviour between builds.
ARG COMFYUI_COMMIT=master
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /app/ComfyUI \
 && cd /app/ComfyUI \
 && if [ "$COMFYUI_COMMIT" != "master" ]; then git checkout ${COMFYUI_COMMIT}; fi \
 && pip install --no-cache-dir -r requirements.txt

# Phr00t's replacement TextEncodeQwenImageEditPlus. The stock node rescales
# condition images, which is what made edits come back at a different size than
# they went in — the exact problem that forced a resize on every keyframe.
ADD https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO/resolve/main/fixed-textencode-node/nodes_qwen.v2.py \
    /app/fixed_nodes_qwen.py

# Adapter deps (ComfyUI already brings torch/pillow/requests)
# opencv and numpy come in via ComfyUI, but face mode depends on them directly —
# pin them here so a ComfyUI dependency change cannot silently remove detection.
RUN pip install --no-cache-dir "fastapi==0.141.1" "uvicorn==0.52.4" \
        "opencv-python-headless" "numpy"

ENV COMFY_URL=http://127.0.0.1:8188 \
    CKPT_NAME=Qwen-Rapid-AIO-NSFW-v23.safetensors \
    SAMPLER=euler_ancestral \
    SCHEDULER=beta \
    STEPS=4 \
    CFG=1.0 \
    SERVER_PORT=8189 \
    PROXY_PORT=8888 \
    MODELS_DIR=/workspace/models

ARG GIT_SHA=unknown
ENV GIT_SHA=$GIT_SHA

COPY server.py /opt/server.py
COPY docker/auth-proxy.py /opt/auth-proxy.py
COPY docker/start.sh /opt/start.sh
COPY docker/extra_model_paths.yaml /opt/extra_model_paths.yaml
RUN chmod +x /opt/start.sh /opt/auth-proxy.py

EXPOSE 8888
CMD ["/opt/start.sh"]
