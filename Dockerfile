# HairFastGAN RunPod serverless worker — multi-stage, slim.
#
# Dependency versions follow camenduru/HairFastGAN-replicate (cog.yaml), the
# known-working config: CUDA 12.1, Python 3.10, torch 2.2.1+cu121.
#
# Why multi-stage: the single-stage devel image was ~16 GB, which RunPod
# serverless could not finish pulling within its init window (worker stuck
# "initializing" then recycled). Here we COMPILE in a devel builder, then copy
# only the resulting venv into a slim CUDA *runtime* base — roughly halving the
# image. Unused torch packages (audio/text/data) are dropped; the git history
# of the model repo is not baked in.
#
# Weights (~5 GB) are baked at build time (download_weights.py) so cold start
# pays no download cost. Must be built for linux/amd64.

# ============================================================================ #
# Stage 1 — builder: compile dlib + install all python deps into a venv.
# ============================================================================ #
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3.10-venv python3-pip \
        build-essential gcc g++ cmake ninja-build \
        git ca-certificates wget && \
    rm -rf /var/lib/apt/lists/*

# Self-contained, relocatable venv (python3.10 exists at the same path in the
# runtime stage, so the venv's interpreter symlink resolves there too).
RUN python3.10 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --upgrade pip setuptools wheel

# PyTorch — only torch + torchvision (audio/text/data are unused by the hair
# GAN and added ~1 GB). Exact versions from camenduru cog.yaml.
RUN pip install \
        torch==2.2.1+cu121 \
        torchvision==0.17.1+cu121 \
        --extra-index-url https://download.pytorch.org/whl/cu121

# HairFastGAN python deps (list from camenduru cog.yaml) + handler runtime deps.
RUN pip install \
        ninja \
        face_alignment \
        dill==0.2.7.1 \
        addict \
        fpie \
        git+https://github.com/openai/CLIP \
        gdown \
        matplotlib \
        dlib \
        runpod \
        requests

# ============================================================================ #
# Stage 2 — runtime: slim CUDA runtime base + the prebuilt venv + weights.
# ============================================================================ #
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HAIRFAST_DIR=/content/HairFastGAN \
    PATH=/opt/venv/bin:$PATH

# Only RUNTIME system libs (no build toolchain): OpenCV/GL, glib, ffmpeg, and
# libgomp (OpenMP runtime that dlib/torch link against). git for the clone.
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        git ca-certificates \
        libgl1 libglib2.0-0 ffmpeg libgomp1 && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

# Bring over the fully-built venv (torch, dlib, face_alignment, CLIP, ...).
COPY --from=builder /opt/venv /opt/venv

# Clone the model code (shallow, no history) and bake the pretrained weights.
RUN git clone -b dev --depth 1 https://github.com/camenduru/HairFastGAN ${HAIRFAST_DIR} && \
    rm -rf ${HAIRFAST_DIR}/.git
COPY download_weights.py /download_weights.py
WORKDIR ${HAIRFAST_DIR}
RUN python3 /download_weights.py

COPY rp_handler.py ${HAIRFAST_DIR}/rp_handler.py

# Build-time smoke test: validate the venv imports on the runtime base WITHOUT a
# GPU (catches missing shared libs / broken venv copy). Model load needs a GPU
# so it is NOT exercised here — that happens at worker cold start.
RUN python -c "import torch, torchvision, dlib, face_alignment, runpod; print('deps ok torch', torch.__version__)"

CMD ["python3", "-u", "rp_handler.py"]
