# DEPLOY-TRAIN.md

Building and running the yomi training container on RunPod.

## Overview

| Item | Value |
|---|---|
| Dockerfile | `Dockerfile.train` |
| Base image | `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime` |
| Image registry | `ghcr.io/<github-user>/yomi-train` |
| Image size | ~5 GB |
| Recommended GPU | RTX 3090 spot (Path 1) — see `memory/project_training_plan.md` |
| Data shipping | Mounted as volume; not baked into image |

## One-time setup

### 1. GitHub Container Registry login (local)

Generate a PAT at https://github.com/settings/tokens with `write:packages`
scope. Then on the build host:

```sh
echo $GH_PAT | docker login ghcr.io -u <github-user> --password-stdin
```

### 2. Build and push

Replace `<github-user>` with your GitHub username throughout.

```sh
cd ~/yomi
docker build -f Dockerfile.train -t ghcr.io/<github-user>/yomi-train:latest .
docker push ghcr.io/<github-user>/yomi-train:latest
```

First push is slow (~5 GB upload). Subsequent pushes only ship changed
layers — code edits push in seconds, dependency bumps push in minutes.

## Per-run on RunPod

### 1. Spin up a pod

- Template: PyTorch (any). Skip RunPod's torch — we bring our own.
- GPU: RTX 3090 spot (Path 1, $0.10–0.20/hr) or A40 spot (Path 2 later).
- Volume: at least 30 GB attached to `/workspace` (data + checkpoints).
- Container Disk: 20 GB (image cache + HF cache).
- Image: `ghcr.io/<github-user>/yomi-train:latest`.
- Volume Mount Path: `/workspace`.

### 2. Upload data to the pod (one-time per dataset version)

From your local box:

```sh
rsync -avz --progress \
  data/heteronyms.json \
  data/train.jsonl data/val.jsonl data/test.jsonl \
  <pod-ssh-host>:/workspace/data/
```

The training image only reads heteronyms.json + train/val.jsonl by default;
test.jsonl is for `99_eval.py`. Total ~1 GB.

### 3. Launch training

The container's default `CMD` runs Path 1 training. From the pod's web
terminal or via `docker exec`:

```sh
# default args (3 epochs, bs=32, fp16)
docker run --gpus all --rm \
  -v /workspace:/workspace \
  ghcr.io/<github-user>/yomi-train:latest

# or override hyperparameters
docker run --gpus all --rm \
  -v /workspace:/workspace \
  ghcr.io/<github-user>/yomi-train:latest \
  --epochs 5 --batch-size 64 --lr 3e-5 --run-name v1-bs64
```

Checkpoints land in `/workspace/models/<run-name>/{best,last}/`.

### 4. Pull checkpoints back

```sh
rsync -avz --progress \
  <pod-ssh-host>:/workspace/models/v1/best/ \
  models/v1/best/
```

The local Pipeline picks them up automatically — see `pipeline.py`'s
`Pipeline.load_default(model_dir=…)`.

## Local CPU smoke test

Before pushing for a long GPU run, sanity-check on CPU:

```sh
docker run --rm \
  -v $PWD/data:/workspace/data \
  -v $PWD/models:/workspace/models \
  ghcr.io/<github-user>/yomi-train:latest \
  --epochs 1 --batch-size 4 --no-fp16 --eval-every-frac 0
```

This won't converge but verifies the data loader, model build, and
checkpoint write all work end-to-end. Drop the `--gpus all` flag and pip
install will warn about no CUDA — torch falls back to CPU automatically.

## Updating the image

Code-only changes (anything in `src/`):

```sh
docker build -f Dockerfile.train -t ghcr.io/<github-user>/yomi-train:latest .
docker push ghcr.io/<github-user>/yomi-train:latest
```

Only the `COPY src` layer rebuilds; pip install + base model download stay
cached.

Dependency changes (`pyproject.toml`):

The full pip install layer rebuilds (~2 min). Push the larger diff once.

## Path 2 later

When `train_seq2seq.py` exists, override the entrypoint at `docker run` time
rather than rebuilding the image:

```sh
docker run --gpus all --rm -v /workspace:/workspace \
  --entrypoint python \
  ghcr.io/<github-user>/yomi-train:latest \
  -m yomi.train_seq2seq --data-dir /workspace/data --out-dir /workspace/models
```

Same image works for both paths.
