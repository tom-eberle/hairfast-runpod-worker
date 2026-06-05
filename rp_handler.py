"""
RunPod serverless handler for HairFastGAN (virtual hairstyle transfer).

The HairFast model is loaded ONCE at module import (cold start) and reused
across every invocation. Each request supplies a face image plus an optional
hairstyle (shape) reference and an optional color reference. Missing refs
fall back to the face image, matching the official HairFastGAN demo logic
(face required + at least one of shape/color is recommended, but either may
be omitted).

Inference logic mirrors camenduru/HairFastGAN-replicate's predict.py and the
official AIRI-Institute/HairFastGAN hair_swap.HairFast.swap() API.
"""

import os
import sys
import base64
import binascii
import tempfile
import traceback

import requests

# HairFastGAN expects to be run from inside its own repo dir (relative paths
# like "pretrained_models/StyleGAN/ffhq.pt" are baked into get_parser()).
HAIRFAST_DIR = os.environ.get("HAIRFAST_DIR", "/content/HairFastGAN")
sys.path.append(HAIRFAST_DIR)
os.chdir(HAIRFAST_DIR)

# StyleGAN2's ops JIT-compile via torch at first import; torch shells out to
# `which c++` / `nvcc`. RunPod's serverless launcher can start the handler with a
# PATH that omits /usr/bin (and /usr/local/cuda/bin), so force the compiler dirs
# on now — otherwise the compile fails with "which c++ returned non-zero".
os.environ["PATH"] = ":".join([
    "/opt/venv/bin", "/usr/local/cuda/bin",
    "/usr/local/sbin", "/usr/local/bin", "/usr/sbin", "/usr/bin", "/sbin", "/bin",
    os.environ.get("PATH", ""),
]).rstrip(":")
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")

import torch  # noqa: E402
import torchvision.transforms as transforms  # noqa: E402

import runpod  # noqa: E402


# --------------------------------------------------------------------------- #
# Model is built LAZILY on the first request (not at import) so that:
#   1. runpod.serverless.start() registers the worker immediately, and
#   2. any failure building the model is caught inside handler() and returned
#      to the caller as {"error": <traceback>} instead of silently crashing the
#      worker process (which would leave the job stuck IN_QUEUE forever).
# The built model is cached in _MODEL and reused across invocations.
# --------------------------------------------------------------------------- #
_MODEL = None


def _ensure_weights():
    """
    Download the HF pretrained weights on first cold start (they are not baked
    into the image — see Dockerfile). Idempotent: download_weights.py skips any
    file already present, so a warm worker no-ops here. Runs from HAIRFAST_DIR so
    weights land in ./pretrained_models (where HairFast() expects them).
    """
    marker = os.path.join(HAIRFAST_DIR, "pretrained_models", "StyleGAN", "ffhq.pt")
    if os.path.exists(marker) and os.path.getsize(marker) > 0:
        return
    print("[hairfast] Downloading pretrained weights (first cold start)...", flush=True)
    import subprocess
    subprocess.run([sys.executable, "/download_weights.py"], cwd=HAIRFAST_DIR, check=True)
    print("[hairfast] Weights ready.", flush=True)


def _get_model():
    global _MODEL
    if _MODEL is None:
        _ensure_weights()
        print("[hairfast] Loading HairFast model (first request)...", flush=True)
        # Import here too: importing hair_swap pulls in StyleGAN2 / op modules
        # that may compile CUDA extensions, so import errors are also caught.
        from hair_swap import HairFast, get_parser
        args = get_parser().parse_args([])
        _MODEL = HairFast(args)
        print("[hairfast] Model loaded.", flush=True)
    return _MODEL


_TO_PIL = transforms.ToPILImage()
# Accepted values for the (currently demo-only) blending mode field.
_VALID_BLENDING = {"Article", "Alternative_v1", "Alternative_v2"}


def _is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _materialize_image(value, suffix=".png"):
    """
    Turn a URL or base64 string (optionally a data: URI) into a temp file path.
    Returns the path, or None if value is falsy.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Image input must be a string (url or base64), got {type(value)}")
    value = value.strip()
    if not value:
        return None

    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        if _is_url(value):
            resp = requests.get(value, timeout=60)
            resp.raise_for_status()
            data = resp.content
        else:
            # Strip a data URI prefix if present: "data:image/png;base64,...."
            if value.startswith("data:") and "," in value:
                value = value.split(",", 1)[1]
            try:
                data = base64.b64decode(value, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError(f"Input is neither a valid URL nor valid base64: {exc}")
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
    except Exception:
        # Make sure we don't leak the descriptor/file on error.
        try:
            os.close(fd)
        except OSError:
            pass
        if os.path.exists(path):
            os.remove(path)
        raise
    return path


def handler(event):
    """
    RunPod serverless entrypoint.

    Input  (event['input']):
      {
        "face":            <url | base64>,            # required
        "shape":           <url | base64 | null>,     # optional, hairstyle ref
        "color":           <url | base64 | null>,     # optional, color ref
        "blending":        "Article",                 # optional, default "Article"
        "poisson_iters":   0,                          # optional, default 0
        "poisson_erosion": 15,                         # optional, default 15
        "align":           true                        # optional, default true
      }

    Output:
      { "image": "<base64 PNG>" }   on success
      { "error": "<message>" }      on failure
    """
    temp_files = []
    try:
        # Lazy model build (cached). A failure here is reported to the caller.
        model = _get_model()

        inp = event.get("input") or {}

        face = inp.get("face")
        if not face:
            return {"error": "Missing required input field 'face'."}

        shape = inp.get("shape")
        color = inp.get("color")

        blending = inp.get("blending", "Article")
        if blending not in _VALID_BLENDING:
            return {
                "error": f"Invalid 'blending' value {blending!r}; "
                         f"expected one of {sorted(_VALID_BLENDING)}."
            }
        poisson_iters = int(inp.get("poisson_iters", 0))
        poisson_erosion = int(inp.get("poisson_erosion", 15))
        # Auto-align (face crop) is on by default; arbitrary photos need it.
        align = bool(inp.get("align", True))

        face_path = _materialize_image(face)
        temp_files.append(face_path)

        # If shape/color are missing, reuse the face image (so transferring
        # only color keeps original shape, and vice-versa).
        shape_path = _materialize_image(shape)
        if shape_path:
            temp_files.append(shape_path)
        else:
            shape_path = face_path

        color_path = _materialize_image(color)
        if color_path:
            temp_files.append(color_path)
        else:
            color_path = face_path

        with torch.inference_mode():
            result = model.swap(
                face_path,
                shape_path,
                color_path,
                align=align,
                # Forwarded for forward-compat; the current public swap()
                # pipeline ignores unknown kwargs (everything takes **kwargs).
                blending=blending,
                poisson_iters=poisson_iters,
                poisson_erosion=poisson_erosion,
            )

        # When align=True, swap() returns (final_image, *aligned_inputs).
        final_image = result[0] if isinstance(result, tuple) else result

        # Model returns a float [0,1] CxHxW tensor; ToPILImage handles it.
        image_pil = _TO_PIL(final_image.detach().cpu().clamp(0, 1))

        out_fd, out_path = tempfile.mkstemp(suffix=".png")
        os.close(out_fd)
        temp_files.append(out_path)
        image_pil.save(out_path, format="PNG")

        with open(out_path, "rb") as fh:
            encoded = base64.b64encode(fh.read()).decode("utf-8")

        return {"image": encoded}

    except Exception:  # noqa: BLE001 - surface any failure (incl. model load) to the caller
        tb = traceback.format_exc()
        traceback.print_exc()
        return {"error": tb}
    finally:
        for path in temp_files:
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass


runpod.serverless.start({"handler": handler})
