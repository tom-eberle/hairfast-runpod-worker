# HairFastGAN RunPod serverless worker.
#
# Dependency versions are taken VERBATIM from camenduru/HairFastGAN-replicate
# (cog.yaml) which is the known-working configuration:
#   CUDA 12.1, Python 3.10, torch 2.2.1+cu121 (+ matching torchvision/audio),
#   dlib, face_alignment, CLIP from OpenAI, etc.
#
# Weights are baked in at build time (download_weights.py) so cold start is fast.
# Must be built for linux/amd64.

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HAIRFAST_DIR=/content/HairFastGAN \
    # dlib / ninja build against the GPU image's toolchain.
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"

# ---- System packages (matches camenduru cog.yaml apt list) ----------------- #
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        software-properties-common && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        build-essential gcc g++ \
        aria2 git git-lfs wget curl ca-certificates \
        libgl1 libglib2.0-0 ffmpeg cmake \
        libgtk2.0-0 libopenmpi-dev && \
    rm -rf /var/lib/apt/lists/*

# Make python3/pip point at 3.10 and upgrade pip tooling.
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    python3 -m pip install --upgrade pip setuptools wheel

# ---- PyTorch (exact versions from camenduru cog.yaml) ---------------------- #
RUN pip install \
        torch==2.2.1+cu121 \
        torchvision==0.17.1+cu121 \
        torchaudio==2.2.1+cu121 \
        torchtext==0.17.1 \
        torchdata==0.7.1 \
        --extra-index-url https://download.pytorch.org/whl/cu121

# ---- Clone HairFastGAN (camenduru dev branch, as in cog.yaml) -------------- #
RUN git clone -b dev https://github.com/camenduru/HairFastGAN ${HAIRFAST_DIR}

# ---- HairFastGAN python deps (exact list from camenduru cog.yaml) ---------- #
RUN pip install \
        ninja \
        face_alignment \
        dill==0.2.7.1 \
        addict \
        fpie \
        git+https://github.com/openai/CLIP \
        gdown \
        matplotlib \
        dlib

# ---- Handler runtime deps -------------------------------------------------- #
RUN pip install runpod requests

# ---- Bake the pretrained weights into the image ---------------------------- #
WORKDIR ${HAIRFAST_DIR}
COPY download_weights.py /download_weights.py
RUN python3 /download_weights.py

# ---- Copy the serverless handler ------------------------------------------- #
COPY rp_handler.py ${HAIRFAST_DIR}/rp_handler.py

# Pre-import to surface any import/path errors during build (best-effort).
RUN python3 -c "import sys; sys.path.append('${HAIRFAST_DIR}'); import runpod" || true

CMD ["python3", "-u", "rp_handler.py"]
