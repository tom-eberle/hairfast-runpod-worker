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

import torch  # noqa: E402
import torchvision.transforms as transforms  # noqa: E402

import runpod  # noqa: E402

from hair_swap import HairFast, get_parser  # noqa: E402


# --------------------------------------------------------------------------- #
# Cold start: build the model exactly once.
# --------------------------------------------------------------------------- #
print("[hairfast] Loading HairFast model (cold start)...", flush=True)
_model_args = get_parser().parse_args([])
HAIR_FAST = HairFast(_model_args)
print("[hairfast] Model loaded.", flush=True)

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
            result = HAIR_FAST.swap(
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

    except Exception as exc:  # noqa: BLE001 - surface any failure to the caller
        traceback.print_exc()
        return {"error": str(exc)}
    finally:
        for path in temp_files:
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass


runpod.serverless.start({"handler": handler})
