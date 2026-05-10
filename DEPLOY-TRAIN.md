# DEPLOY-TRAIN.md

Building and running the yomi training container on RunPod.

## Overview

| Item | Value |
|---|---|
| Dockerfile | `Dockerfile.train` |
| Base image | `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime` |
| Image | `ghcr.io/hbar137/yomi-train:latest` (public) |
| Image size | ~14 GB uncompressed / ~7 GB pulled |
| Recommended GPU | RTX 3090 spot (Path 1) — see `memory/project_training_plan.md` |
| Data shipping | Mounted as volume; not baked into image |

## How the image is built

GitHub Actions builds and pushes on every commit to `main` that touches
`Dockerfile.train`, `pyproject.toml`, `src/`, or the workflow file itself.

- Workflow: `.github/workflows/build-train.yml`
- Auth: workflow's auto-provisioned `GITHUB_TOKEN` (no manual PAT)
- Tags emitted: `latest` (default branch), `main`, `sha-<short>`, `v*` (on tag pushes)
- Cache: `type=gha` for Docker layers — code-only edits rebuild in ~1–2 min;
  full rebuild (dependency bump) ~7 min on GitHub's runners.

To trigger a manual rebuild without a code push:

```sh
gh workflow run build-train.yml
```

To watch a build:

```sh
gh run list --workflow build-train.yml --limit 5
gh run watch <run-id>
```

## Per-run on RunPod

### 1. Spin up a pod

- Template: any (the image brings its own torch).
- GPU: RTX 3090 spot (Path 1, $0.10–0.20/hr) or A40 spot (Path 2 later).
- Volume: at least 30 GB attached to `/workspace` (data + checkpoints).
- Container Disk: 20 GB (image + HF cache).
- Image: `ghcr.io/hbar137/yomi-train:latest`.
- Volume Mount Path: `/workspace`.

The image is **public**, so RunPod pulls it without any registry credentials.

### 2. Upload data to the pod (one-time per dataset version)

From your local box:

```sh
rsync -avz --progress \
  data/heteronyms.json \
  data/train.jsonl data/val.jsonl data/test.jsonl \
  <pod-ssh-host>:/workspace/data/
```

The training image only reads `heteronyms.json` + `train/val.jsonl` by
default; `test.jsonl` is for `scripts/99_eval.py`. Total ~1 GB.

### 3. Launch training

The container's default `CMD` runs Path 1 training. From the pod's web
terminal or via `docker exec`:

```sh
# default args (3 epochs, bs=32, fp16)
docker run --gpus all --rm \
  -v /workspace:/workspace \
  ghcr.io/hbar137/yomi-train:latest

# or override hyperparameters
docker run --gpus all --rm \
  -v /workspace:/workspace \
  ghcr.io/hbar137/yomi-train:latest \
  --epochs 5 --batch-size 64 --lr 3e-5 --run-name v1-bs64
```

Checkpoints land in `/workspace/models/<run-name>/{best,last}/`.

### 4. Pull checkpoints back

```sh
rsync -avz --progress \
  <pod-ssh-host>:/workspace/models/v1/best/ \
  models/v1/best/
```

The local Pipeline picks them up automatically — see
`Pipeline.load_default(model_dir=…)` in `src/yomi/pipeline.py`.

## Local CPU smoke test

Before pushing for a long GPU run, sanity-check on CPU. Pull the GHA-built
image instead of rebuilding locally:

```sh
docker pull ghcr.io/hbar137/yomi-train:latest

docker run --rm \
  -v $PWD/data:/workspace/data \
  -v $PWD/models:/workspace/models \
  ghcr.io/hbar137/yomi-train:latest \
  --epochs 1 --batch-size 4 --no-fp16 --eval-every-frac 0
```

This won't converge but verifies the data loader, model build, and
checkpoint write all work end-to-end. Without `--gpus all`, torch falls
back to CPU automatically.

## Path 2 later

When `train_seq2seq.py` exists, override the entrypoint at `docker run`
time rather than rebuilding the image:

```sh
docker run --gpus all --rm -v /workspace:/workspace \
  --entrypoint python \
  ghcr.io/hbar137/yomi-train:latest \
  -m yomi.train_seq2seq --data-dir /workspace/data --out-dir /workspace/models
```

Same image works for both paths.

## Local manual build (rarely needed)

If you need to build off a local branch or edit before commit:

```sh
docker build -f Dockerfile.train -t yomi-train:dev .
docker run --rm -v $PWD/data:/workspace/data yomi-train:dev --epochs 0
```

A local PAT with `write:packages` is only needed if you want to push a
local image directly to ghcr.io — normally the GHA pipeline handles this.
