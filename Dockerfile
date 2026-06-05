# HairFastGAN RunPod serverless worker — single-stage on the CUDA *devel* base.
#
# Versions follow camenduru/HairFastGAN-replicate (cog.yaml): CUDA 12.1,
# Python 3.10, torch 2.2.1+cu121.
#
# Why single-stage devel (after much iteration):
#   * StyleGAN2's ops (models/stylegan2/op) JIT-compile via torch
#     cpp_extension.load() at import and need a COMPLETE, correctly-configured
#     toolchain: nvcc, the CUDA math-lib headers (cusparse/cublas/...), Python.h,
#     AND a real c++ (the /usr/bin/c++ alternative). Cherry-picking these onto a
#     slim runtime base kept missing one piece per cycle (notably `g++` with
#     --no-install-recommends does NOT create /usr/bin/c++; build-essential does).
#     The devel base + build-essential has everything set up properly.
#   * SIZE is fine now because the ~5 GB of weights are NOT baked — the handler
#     downloads them at first cold start from RunPod's network (HF 429-throttles
#     GitHub Actions build IPs). So this image is ~11 GB, smaller than the
#     12.7 GB runtime+weights image that already pulled fine on RunPod.
#
# Build for linux/amd64.

ARG TORCH_ARCHES="7.5;8.0;8.6;8.9+PTX"

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
ARG TORCH_ARCHES

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HAIRFAST_DIR=/content/HairFastGAN \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    TORCH_CUDA_ARCH_LIST="${TORCH_ARCHES}"

# build-essential gives gcc/g++ AND the /usr/bin/c++ alternative torch shells out
# to. The devel base already provides nvcc + all CUDA dev headers.
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        build-essential cmake ninja-build \
        git ca-certificates wget \
        libgl1 libglib2.0-0 ffmpeg libgomp1 && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    python3 -m pip install --upgrade pip setuptools wheel

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

# torch 2.2.1 built against NumPy 1.x; deps pull 2.x which breaks interop. Pin.
RUN pip install "numpy==1.26.4"

RUN git clone -b dev --depth 1 https://github.com/camenduru/HairFastGAN ${HAIRFAST_DIR} && \
    rm -rf ${HAIRFAST_DIR}/.git

# Pre-compile the StyleGAN2 CUDA ops now (build env == runtime env, same image),
# so torch reuses the cached .so at cold start; if it still recompiles, the full
# toolchain is present. Also a hard build-time check that compilation works.
RUN cd ${HAIRFAST_DIR} && \
    python -c "import sys; sys.path.insert(0,'.'); from models.stylegan2.op import fused_act, upfirdn2d; print('StyleGAN2 ops compiled OK')"

# Weights are downloaded at first cold start by the handler (not baked — HF 429s
# the CI build IP). download_weights.py is idempotent.
COPY download_weights.py /download_weights.py
COPY rp_handler.py ${HAIRFAST_DIR}/rp_handler.py
WORKDIR ${HAIRFAST_DIR}

CMD ["python3", "-u", "rp_handler.py"]
