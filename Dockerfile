# HairFastGAN RunPod serverless worker — multi-stage, slim, with the StyleGAN2
# CUDA ops compiled in the runtime environment.
#
# Versions follow camenduru/HairFastGAN-replicate (cog.yaml): CUDA 12.1,
# Python 3.10, torch 2.2.1+cu121.
#
# Design notes (hard-won):
#   * SIZE — a single-stage *devel* image was ~16 GB and RunPod serverless could
#     not finish pulling it. We build the venv in a devel builder and ship it on
#     a slim CUDA *runtime* base.
#   * StyleGAN2 OPS — models/stylegan2/op/{fused,upfirdn2d} call torch
#     cpp_extension.load() at import, JIT-compiling with nvcc. That needs nvcc +
#     g++ + the CUDA math-lib HEADERS (cusparse.h, cublas.h, ...). We install the
#     CUDA libraries-dev meta-package (all headers) but delete the large static
#     archives, so the toolchain adds ~1 GB rather than reverting to the devel
#     base. We then COMPILE the ops during build in this runtime image, baking
#     the .so into /root/.cache/torch_extensions — so at worker cold start torch
#     reuses it (and if its build-hash check still triggers a recompile, the
#     toolchain is present to do so).
#
# Weights (~5 GB) are baked at build (download_weights.py). Build for linux/amd64.

# A40 sm_86; A6000 8.6; L40/RTX6000-Ada 8.9. Pinned so nvcc targets these arches
# without probing a (build-time absent) GPU.
ARG TORCH_ARCHES="7.5;8.0;8.6;8.9+PTX"

# ============================================================================ #
# Stage 1 — builder: build the venv (torch + deps) and fetch the model source.
# ============================================================================ #
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS builder
ARG TORCH_ARCHES
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3.10-venv python3-pip \
        build-essential gcc g++ cmake ninja-build \
        git ca-certificates wget && \
    rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --upgrade pip setuptools wheel

# torch + torchvision only (audio/text/data unused). Versions from camenduru.
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

# torch 2.2.1 was built against NumPy 1.x; deps above pull NumPy 2.x which breaks
# torch<->numpy interop at runtime. Pin back, done last so it wins.
RUN pip install "numpy==1.26.4"

RUN git clone -b dev --depth 1 https://github.com/camenduru/HairFastGAN /content/HairFastGAN && \
    rm -rf /content/HairFastGAN/.git

# ============================================================================ #
# Stage 2 — runtime: slim CUDA runtime base + venv + CUDA dev headers + ops.
# ============================================================================ #
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
ARG TORCH_ARCHES

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HAIRFAST_DIR=/content/HairFastGAN \
    CUDA_HOME=/usr/local/cuda \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:$PATH \
    TORCH_CUDA_ARCH_LIST="${TORCH_ARCHES}"

# Runtime libs + a CUDA compile toolchain: g++, nvcc, and the math-lib DEV
# headers (torch's CUDA extension includes pull in cusparse/cublas/cusolver/...).
# The shared libs themselves are already in the cudnn-runtime base; we delete the
# large *static* archives the -dev packages add (~2 GB) to keep the image slim.
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        python3.10 ca-certificates \
        libgl1 libglib2.0-0 ffmpeg libgomp1 \
        g++ cuda-nvcc-12-1 cuda-cudart-dev-12-1 cuda-libraries-dev-12-1 && \
    rm -rf /var/lib/apt/lists/* && \
    find /usr/local/cuda* \( -name '*_static.a' -o -name 'libcublasLt_static.a' \) -delete 2>/dev/null || true && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /content/HairFastGAN /content/HairFastGAN

COPY download_weights.py /download_weights.py
WORKDIR ${HAIRFAST_DIR}
RUN python3 /download_weights.py

COPY rp_handler.py ${HAIRFAST_DIR}/rp_handler.py

# Compile the StyleGAN2 CUDA ops here (runtime image, so the cached .so matches
# the runtime env) and validate deps. This also fails the build loudly if the
# CUDA toolchain/headers are incomplete — before any RunPod test cost.
RUN python -c "import torch, torchvision, dlib, face_alignment, runpod; print('deps ok torch', torch.__version__)" && \
    cd ${HAIRFAST_DIR} && \
    python -c "import sys; sys.path.insert(0,'.'); from models.stylegan2.op import fused_act, upfirdn2d; print('StyleGAN2 ops compiled + loaded on runtime image')"

CMD ["python3", "-u", "rp_handler.py"]
