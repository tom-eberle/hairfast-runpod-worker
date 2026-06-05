# HairFastGAN RunPod serverless worker — multi-stage, slim, with pre-compiled
# StyleGAN2 CUDA ops.
#
# Dependency versions follow camenduru/HairFastGAN-replicate (cog.yaml), the
# known-working config: CUDA 12.1, Python 3.10, torch 2.2.1+cu121.
#
# Two problems this Dockerfile solves:
#   1. SIZE — a single-stage devel image was ~16 GB and RunPod serverless could
#      not finish pulling it. We COMPILE in a devel builder, then ship only the
#      venv on a slim CUDA *runtime* base (~12-13 GB), which pulls fine.
#   2. StyleGAN2 CUSTOM CUDA OPS — models/stylegan2/op/{fused,upfirdn2d} call
#      torch.utils.cpp_extension.load() at IMPORT time, which JIT-compiles with
#      nvcc. The runtime base has no nvcc, so the build silently produced no .so
#      and the worker crashed at model load ("fused.so: cannot open shared
#      object file"). Fix: PRE-COMPILE those ops in the builder (nvcc present;
#      no GPU needed when TORCH_CUDA_ARCH_LIST is pinned) and copy the resulting
#      torch_extensions cache into the runtime image. TORCH_CUDA_ARCH_LIST is
#      set identically in both stages so torch reuses the cached .so instead of
#      trying (and failing) to recompile at runtime.
#
# Weights (~5 GB) are baked at build (download_weights.py). Build for linux/amd64.

# A40 is sm_86; keep a broad list so the prebuilt ops also run on other pool
# GPUs (A6000 8.6, L40/RTX6000-Ada 8.9). Pinning this makes torch's cpp_extension
# use these arches instead of probing a (build-time absent) GPU.
ARG TORCH_ARCHES="7.5;8.0;8.6;8.9+PTX"

# ============================================================================ #
# Stage 1 — builder: compile dlib + python deps + StyleGAN2 ops into a venv.
# ============================================================================ #
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS builder
ARG TORCH_ARCHES

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="${TORCH_ARCHES}"

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

# PyTorch — only torch + torchvision (audio/text/data are unused). Versions from
# camenduru cog.yaml.
RUN pip install \
        torch==2.2.1+cu121 \
        torchvision==0.17.1+cu121 \
        --extra-index-url https://download.pytorch.org/whl/cu121

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

# torch 2.2.1 was built against NumPy 1.x; several deps above pull NumPy 2.x,
# which breaks torch<->numpy interop at runtime ("Failed to initialize NumPy:
# _ARRAY_API not found"). Pin back to the last 1.x release. Done last so it wins.
RUN pip install "numpy==1.26.4"

# Clone the model code (shallow) and PRE-COMPILE the StyleGAN2 CUDA ops so the
# compiled .so files land in /root/.cache/torch_extensions. nvcc is present in
# this devel stage; no GPU is required because TORCH_CUDA_ARCH_LIST is pinned.
RUN git clone -b dev --depth 1 https://github.com/camenduru/HairFastGAN /content/HairFastGAN && \
    rm -rf /content/HairFastGAN/.git
RUN cd /content/HairFastGAN && \
    python -c "import sys; sys.path.insert(0,'.'); from models.stylegan2.op import fused_act, upfirdn2d; print('StyleGAN2 ops compiled:', fused_act, upfirdn2d)"

# ============================================================================ #
# Stage 2 — runtime: slim CUDA runtime base + venv + compiled ops + weights.
# ============================================================================ #
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
ARG TORCH_ARCHES

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HAIRFAST_DIR=/content/HairFastGAN \
    CUDA_HOME=/usr/local/cuda \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:$PATH \
    # Match the builder so torch can reuse the prebuilt op cache when its build
    # hash hits; if it misses, the toolchain below recompiles for these arches.
    TORCH_CUDA_ARCH_LIST="${TORCH_ARCHES}"

# Runtime system libs + a MINIMAL CUDA compile toolchain. StyleGAN2's ops use
# torch cpp_extension.load() which recomputes a build hash at runtime and may
# choose to (re)compile rather than reuse a copied cache; it needs g++ + nvcc +
# cudart headers to do so. We install just those (not the full devel base), so
# the image stays far smaller than the original ~16 GB devel build while the op
# JIT-compile reliably succeeds (and reuses our baked cache when the hash hits).
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        ca-certificates \
        libgl1 libglib2.0-0 ffmpeg libgomp1 \
        g++ cuda-nvcc-12-1 cuda-cudart-dev-12-1 && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

# Prebuilt venv + the EXACT same model source it compiled ops against + the
# compiled op cache. Copying the repo (rather than re-cloning) guarantees the
# op source hash matches the cache so torch loads the .so instead of recompiling.
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /content/HairFastGAN /content/HairFastGAN
COPY --from=builder /root/.cache/torch_extensions /root/.cache/torch_extensions

# Bake the pretrained weights into the repo dir.
COPY download_weights.py /download_weights.py
WORKDIR ${HAIRFAST_DIR}
RUN python3 /download_weights.py

COPY rp_handler.py ${HAIRFAST_DIR}/rp_handler.py

# Build-time smoke test on the RUNTIME base (no GPU, no nvcc): importing the ops
# must succeed by loading the PREBUILT .so from the copied cache. If torch tried
# to recompile here it would fail — so this catches a broken op-cache before any
# RunPod test. (Loading a CUDA .so needs no GPU; only execution does.)
RUN python -c "import torch, torchvision, dlib, face_alignment, runpod; print('deps ok torch', torch.__version__)" && \
    cd ${HAIRFAST_DIR} && \
    python -c "import sys; sys.path.insert(0,'.'); from models.stylegan2.op import fused_act, upfirdn2d; print('prebuilt StyleGAN2 ops load OK on runtime base')"

CMD ["python3", "-u", "rp_handler.py"]
