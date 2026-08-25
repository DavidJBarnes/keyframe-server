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

# YuNet face detector for mode=face. OpenCV 5 removed CascadeClassifier from the
# top-level API, and YuNet is the better detector anyway — Haar misses angled and
# partially occluded faces, which most real keyframes contain.
ADD https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx \
    /opt/face_detection_yunet.onnx

# --- mode=face: LivePortrait via ComfyUI-AdvancedLivePortrait ----------------
#
# Qwen cannot do micro facial edits: it regenerates pixels, halving skin texture
# (Laplacian variance 1015 -> ~450) and de-aging the subject on every pass, at
# every denoise setting. LivePortrait warps the source pixels instead, so
# identity and texture survive by construction. See docs/dual-pipeline-design.md.
#
# Pinned to a commit for the same reason ComfyUI is: an unpinned clone means the
# image silently changes behaviour between builds.
ARG ALP_COMMIT=main
RUN git clone https://github.com/PowerHouseMan/ComfyUI-AdvancedLivePortrait.git \
        /app/ComfyUI/custom_nodes/ComfyUI-AdvancedLivePortrait \
 && cd /app/ComfyUI/custom_nodes/ComfyUI-AdvancedLivePortrait \
 && if [ "$ALP_COMMIT" != "main" ]; then git checkout ${ALP_COMMIT}; fi

# Verified against torch 2.12.1 + numpy 2.2.6 + cv2 5.0.0: none are disturbed.
# NOTE: ultralytics is AGPL-3.0 and is a hard dependency of the node's face
# detector — there is no alternative path in its source.
RUN pip install --no-cache-dir \
        "numpy>=1.26.4" "imageio-ffmpeg>=0.5.1" "lmdb>=1.4.1" "timm>=1.0.7" \
        "rich>=13.7.1" "albumentations>=1.4.10" ultralytics "tyro==0.8.5" dill

# Baked in rather than left to the node's runtime downloader (nodes.py:106),
# which would otherwise make the first face request depend on the network and
# fail silently to a broken model file. ~500MB total.
ADD https://huggingface.co/Kijai/LivePortrait_safetensors/resolve/main/appearance_feature_extractor.safetensors \
    /app/ComfyUI/models/liveportrait/appearance_feature_extractor.safetensors
ADD https://huggingface.co/Kijai/LivePortrait_safetensors/resolve/main/motion_extractor.safetensors \
    /app/ComfyUI/models/liveportrait/motion_extractor.safetensors
ADD https://huggingface.co/Kijai/LivePortrait_safetensors/resolve/main/warping_module.safetensors \
    /app/ComfyUI/models/liveportrait/warping_module.safetensors
ADD https://huggingface.co/Kijai/LivePortrait_safetensors/resolve/main/spade_generator.safetensors \
    /app/ComfyUI/models/liveportrait/spade_generator.safetensors
ADD https://huggingface.co/Kijai/LivePortrait_safetensors/resolve/main/stitching_retargeting_module.safetensors \
    /app/ComfyUI/models/liveportrait/stitching_retargeting_module.safetensors
ADD https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt \
    /app/ComfyUI/models/ultralytics/face_yolov8n.pt
RUN chmod -R a+r /app/ComfyUI/models/liveportrait /app/ComfyUI/models/ultralytics

ENV COMFY_URL=http://127.0.0.1:8188 \
    CKPT_NAME=Qwen-Rapid-AIO-NSFW-v23.safetensors \
    SAMPLER=euler_ancestral \
    SCHEDULER=beta \
    STEPS=4 \
    CFG=1.0 \
    SERVER_PORT=8189 \
    PROXY_PORT=8888 \
    MODELS_DIR=/workspace/models \
    YOLO_CONFIG_DIR=/tmp/ultralytics

ARG GIT_SHA=unknown
ENV GIT_SHA=$GIT_SHA

COPY server.py /opt/server.py
COPY docker/auth-proxy.py /opt/auth-proxy.py
COPY docker/start.sh /opt/start.sh
COPY docker/extra_model_paths.yaml /opt/extra_model_paths.yaml
RUN chmod +x /opt/start.sh /opt/auth-proxy.py

EXPOSE 8888
CMD ["/opt/start.sh"]
