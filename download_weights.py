"""
Download all HairFastGAN pretrained weights at *build* time so they are baked
into the image and cold start does not pay a download cost.

Weight list + destination layout is copied verbatim from camenduru's
known-working cog.yaml (HairFastGAN-replicate). Source of truth:
  https://huggingface.co/AIRI-Institute/HairFastGAN/tree/main/pretrained_models

Run from inside the HairFastGAN repo dir, e.g.:
  cd /content/HairFastGAN && python3 /download_weights.py
"""

import os
import sys
import urllib.request

HF_BASE = "https://huggingface.co/AIRI-Institute/HairFastGAN/resolve/main/pretrained_models"

# (relative_dest_path, remote_filename) — dest is relative to pretrained_models/
FILES = [
    ("ArcFace/backbone_ir50.pth", "ArcFace/backbone_ir50.pth"),
    ("ArcFace/backbone_r100.pth", "ArcFace/backbone_r100.pth"),
    ("ArcFace/ir_se50.pth", "ArcFace/ir_se50.pth"),
    ("BiSeNet/face_parsing_79999_iter.pth", "BiSeNet/face_parsing_79999_iter.pth"),
    ("BiSeNet/seg.pth", "BiSeNet/seg.pth"),
    ("Blending/checkpoint.pth", "Blending/checkpoint.pth"),
    ("Blending/checkpoint_old.pth", "Blending/checkpoint_old.pth"),
    ("Blending/checkpoint_old2.pth", "Blending/checkpoint_old2.pth"),
    ("FeatureStyleEncoder/143_enc.pth", "FeatureStyleEncoder/143_enc.pth"),
    ("FeatureStyleEncoder/79999_iter.pth", "FeatureStyleEncoder/79999_iter.pth"),
    ("FeatureStyleEncoder/backbone.pth", "FeatureStyleEncoder/backbone.pth"),
    ("FeatureStyleEncoder/psp_ffhq_encode.pt", "FeatureStyleEncoder/psp_ffhq_encode.pt"),
    ("PostProcess/latent_avg.pt", "PostProcess/latent_avg.pt"),
    ("PostProcess/pp_model.pth", "PostProcess/pp_model.pth"),
    ("Rotate/rotate_best.pth", "Rotate/rotate_best.pth"),
    ("STAR/WFLW_STARLoss_NME_4_02_FR_2_32_AUC_0_605.pkl",
     "STAR/WFLW_STARLoss_NME_4_02_FR_2_32_AUC_0_605.pkl"),
    ("ShapeAdaptor/mask_generator.pth", "ShapeAdaptor/mask_generator.pth"),
    ("ShapeAdaptor/shape_predictor_68_face_landmarks.dat",
     "ShapeAdaptor/shape_predictor_68_face_landmarks.dat"),
    ("StyleGAN/ffhq.pkl", "StyleGAN/ffhq.pkl"),
    ("StyleGAN/ffhq.pt", "StyleGAN/ffhq.pt"),
    ("StyleGAN/ffhq_PCA.npz", "StyleGAN/ffhq_PCA.npz"),
    ("encoder4editing/e4e_ffhq_encode.pt", "encoder4editing/e4e_ffhq_encode.pt"),
    ("sean_checkpoints/CelebA-HQ_pretrained/latest_net_D.pth",
     "sean_checkpoints/CelebA-HQ_pretrained/latest_net_D.pth"),
    ("sean_checkpoints/CelebA-HQ_pretrained/latest_net_G.pth",
     "sean_checkpoints/CelebA-HQ_pretrained/latest_net_G.pth"),
]


def download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"[skip] {dest} already present", flush=True)
        return
    print(f"[get ] {url} -> {dest}", flush=True)
    # HF resolve URLs 302-redirect to a CDN; urllib follows redirects.
    req = urllib.request.Request(url, headers={"User-Agent": "hairfast-runpod-worker"})
    with urllib.request.urlopen(req, timeout=600) as resp, open(dest, "wb") as fh:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)


def main() -> int:
    base_dir = os.path.join(os.getcwd(), "pretrained_models")
    os.makedirs(base_dir, exist_ok=True)
    failures = []
    for rel_dest, remote in FILES:
        url = f"{HF_BASE}/{remote}"
        dest = os.path.join(base_dir, rel_dest)
        try:
            download(url, dest)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {url}: {exc}", flush=True)
            failures.append((url, str(exc)))
    if failures:
        print(f"\n{len(failures)} download(s) FAILED:", flush=True)
        for url, err in failures:
            print(f"  - {url}: {err}", flush=True)
        return 1
    print("\nAll HairFastGAN weights downloaded.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
