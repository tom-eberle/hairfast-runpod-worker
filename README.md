# HairFastGAN RunPod Serverless Worker

A [RunPod serverless](https://docs.runpod.io/serverless/overview) worker that runs
[HairFastGAN](https://github.com/AIRI-Institute/HairFastGAN) (AIRI-Institute) for
virtual hairstyle transfer: take a **face** photo and transfer the **hairstyle
(shape)** and/or **hair color** from reference photos onto it.

Dependency versions and inference logic are based on the known-working
[camenduru/HairFastGAN-replicate](https://github.com/camenduru/HairFastGAN-replicate)
Cog build (CUDA 12.1, Python 3.10, torch 2.2.1+cu121) to minimize build failures.
All pretrained weights are **baked into the image at build time** so cold starts
do not pay a download cost.

The image is built entirely by **GitHub Actions** (no local Docker required) and
pushed to GHCR.

---

## 1. Build the image (GitHub Actions -> GHCR)

1. Create a GitHub repo named `hairfast-runpod-worker` and push these files to `main`.
   ```bash
   git init
   git add .
   git commit -m "HairFastGAN RunPod serverless worker"
   git branch -M main
   git remote add origin https://github.com/<owner>/hairfast-runpod-worker.git
   git push -u origin main
   ```
2. The workflow at `.github/workflows/build.yml` runs on every push to `main`
   (and via "Run workflow" / `workflow_dispatch`). It:
   - logs in to GHCR with the built-in `GITHUB_TOKEN`,
   - builds for `linux/amd64`,
   - pushes `ghcr.io/<owner>/hairfast-runpod-worker:latest` and `:<git-sha>`.
   - The image name is lowercased automatically (GHCR requires lowercase).
3. Watch the **Actions** tab. The first build is large (CUDA base + torch + ~24
   baked weight files) and can take a while.

### Make the GHCR package public

RunPod needs to pull the image. Either make it public or give RunPod registry creds.

- Public (simplest): GitHub -> your profile/org -> **Packages** ->
  `hairfast-runpod-worker` -> **Package settings** -> **Change visibility** ->
  **Public**.
- Or private: in RunPod add **Container Registry Credentials** (username = your
  GitHub username, password = a PAT with `read:packages`) and select them on the
  template.

---

## 2. Create a RunPod serverless template + endpoint

Install and authenticate the CLI:

```bash
# install (see https://github.com/runpod/runpodctl)
runpodctl config --apiKey <YOUR_RUNPOD_API_KEY>
```

Create the template (30 GB container disk to hold the image + baked weights):

```bash
runpodctl create template \
  --name hairfast-worker \
  --image ghcr.io/<owner>/hairfast-runpod-worker:latest \
  --containerDiskSize 30 \
  --ports "" \
  # if your runpodctl version exposes it, add: --serverless
```

> Flag names vary slightly between `runpodctl` releases. Run
> `runpodctl create template --help` to confirm; common forms are
> `--containerDiskSize 30` or `--container-disk-in-gb 30`, and `--serverless`.
> The equivalent in the web console: **Serverless -> Templates -> New Template**,
> Container Image = `ghcr.io/<owner>/hairfast-runpod-worker:latest`,
> Container Disk = 30 GB.

Create the serverless endpoint from the template (note the template id printed above):

```bash
runpodctl create endpoint \
  --name hairfast-endpoint \
  --templateId <TEMPLATE_ID> \
  --gpuType "NVIDIA RTX A5000" \
  --workersMax 1
```

> A 24 GB GPU (RTX A5000 / 3090 / 4090 / L4 / A10G) is recommended; StyleGAN2 +
> the HairFastGAN encoders are memory-hungry at 1024px. Confirm flags with
> `runpodctl create endpoint --help` (older versions use `--templateId`, some use
> `--template-id`). You can also create the endpoint in the web console:
> **Serverless -> New Endpoint -> select the template**.

---

## 3. Handler input / output contract

The handler reads `event["input"]`.

### Input

| Field             | Type                 | Required | Default     | Notes                                                             |
|-------------------|----------------------|----------|-------------|-------------------------------------------------------------------|
| `face`            | URL or base64 PNG/JPG| **yes**  | —           | Source photo the hairstyle is applied to.                         |
| `shape`           | URL/base64 or `null` | no       | `null`      | Hairstyle reference. If omitted, the face image is reused.        |
| `color`           | URL/base64 or `null` | no       | `null`      | Hair color reference. If omitted, the face image is reused.       |
| `blending`        | string               | no       | `"Article"` | `"Article"`, `"Alternative_v1"`, or `"Alternative_v2"`. See note. |
| `poisson_iters`   | int                  | no       | `0`         | See note.                                                         |
| `poisson_erosion` | int                  | no       | `15`        | See note.                                                         |
| `align`           | bool                 | no       | `true`      | Crop/align inputs to faces (needed for arbitrary photos).         |

Base64 may be raw or a `data:image/...;base64,...` URI.

> **Note on `blending` / `poisson_iters` / `poisson_erosion`:** these fields are
> part of the public input contract for compatibility with the official
> HairFastGAN Gradio demo, but the open-source `hair_swap.HairFast.swap()`
> pipeline currently only acts on `align`. The three fields are forwarded as
> kwargs (the pipeline accepts and ignores unknown kwargs) so requests don't
> break, but they have no effect on the current model. They are kept so the
> contract is stable if the upstream API adds them.

### Output

Success:
```json
{ "image": "<base64-encoded PNG of the result>" }
```

Failure:
```json
{ "error": "<message>" }
```

### Example request

Synchronous run:
```bash
curl -s -X POST \
  "https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync" \
  -H "Authorization: Bearer <RUNPOD_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
        "input": {
          "face":  "https://example.com/face.png",
          "shape": "https://example.com/hairstyle.png",
          "color": "https://example.com/color.png",
          "blending": "Article",
          "poisson_iters": 0,
          "poisson_erosion": 15,
          "align": true
        }
      }'
```

Transfer only color (keep the face's own hairstyle): send `face` + `color`, omit `shape`.
Transfer only shape: send `face` + `shape`, omit `color`.

The `image` field is a base64 PNG. Decode it, e.g.:
```bash
echo "<base64>" | base64 -d > result.png
```

---

## Files

- `rp_handler.py` — RunPod serverless handler (loads model once, runs `HairFast.swap`).
- `download_weights.py` — downloads all pretrained weights at build time.
- `Dockerfile` — CUDA 12.1 + Python 3.10 + torch 2.2.1+cu121, clones HairFastGAN, bakes weights.
- `requirements.txt` — handler-only deps (`runpod`, `requests`); model deps are pinned in the Dockerfile.
- `.github/workflows/build.yml` — builds `linux/amd64` and pushes to GHCR.
