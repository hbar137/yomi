# RUNPOD.md

Operational guide for training yomi on RunPod via `scripts/run_training_pod.py`.
Captures every trap we hit on first deploy (2026-05-10) so the next session
doesn't relive them.

## Quick reference

```sh
# One-time per machine
export RUNPOD_API_KEY=$(cat ~/secrets/runpod/api_key.txt)
runpodctl me                                  # verify auth + balance

# Smoke test (~1–2 min, ~$0.06)
python3 scripts/run_training_pod.py --max-steps 20 --batch-size 4 \
  --run-name smoke --yes

# Real Path 1 training (~30 min, ~$0.40)
python3 scripts/run_training_pod.py --gpu RTX4090 --epochs 3 --run-name v1 --yes
```

## What the orchestrator does

1. Creates a SECURE-cloud pod from RunPod's pre-cached `runpod-torch-v240`
   template. **Not** from our custom `ghcr.io/hbar137/yomi-train` image —
   see "Trap 1" below.
2. Polls SSH directly (TCP-level probe) until the pod is reachable.
3. SSH'es in, `git clone`s the public repo, `pip install -e '.[train]'`.
4. `scp`s `data/heteronyms.json + train.jsonl + val.jsonl` to the pod.
5. Runs `python -m yomi.train ...` over SSH, streaming stdout to your
   terminal.
6. `scp`s `models/<run-name>/best/` (or `last/` for short runs) back home.
7. Deletes the pod, even on Ctrl-C / errors (try/finally).

## One-time setup

### 1. RunPod API key

```sh
mkdir -p ~/secrets/runpod
chmod 700 ~/secrets ~/secrets/runpod
# paste your key from https://www.runpod.io/console/user/settings:
echo 'rpa_xxxxx' > ~/secrets/runpod/api_key.txt
chmod 600 ~/secrets/runpod/api_key.txt
```

Then in any shell:

```sh
export RUNPOD_API_KEY=$(cat ~/secrets/runpod/api_key.txt)
```

### 2. SSH public key

Upload your `~/.ssh/id_ed25519.pub` (or RSA equivalent) at
https://www.runpod.io/console/user/settings → SSH Public Keys.

RunPod injects this into every pod via `PUBLIC_KEY` env var; the pod's
sshd authorizes it automatically. No additional config on the pod side.

### 3. runpodctl

```sh
# Linux: download the binary (already done on this machine)
curl -L https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-linux-amd64 \
  -o /tmp/runpodctl && sudo install /tmp/runpodctl /usr/local/bin/runpodctl
runpodctl me   # should return JSON with your balance
```

## Cost reality

Observed prices on 2026-05-10 (your mileage may vary):

| GPU | Cloud | $/hr | Stock |
|---|---|---|---|
| RTX 3090 | Community | ~$0.22 | Low (and unreliable, see Trap 2) |
| RTX 3090 | Secure | varies | Low |
| RTX 4090 | Secure | **$0.69** | Medium (recommended) |
| A40 (48 GB) | Secure | $0.44 | High |
| A100 80 GB | Secure | $1.80+ | Low |

Path 1 v1 run: ~30 min × $0.69 = **~$0.40 per run**. Smoke test ~$0.06.

Pricing changes; check `runpodctl gpu list` for current secureCloud prices.

## Traps we hit (and how the script avoids them)

### Trap 1: 14 GB custom Docker image stalls forever on community cloud

**Symptom**: pod reports `desiredStatus=RUNNING` and `costPerHr` accrues, but
`uptimeSeconds=0` for 18+ min, sshd never starts. We were billed for nothing.

**Why**: RunPod community workers don't cache custom images from
`ghcr.io/hbar137/...`. Pulling 14 GB across the public internet times out
or stalls without surfacing any error via `runpodctl pod get`. There's no
`runpodctl pod logs` command to debug it.

**Fix**: don't ship a custom training image. Use RunPod's cached
`runpod-torch-v240` template (PyTorch 2.4 + CUDA 12.4 + Ubuntu 22.04) and
`git clone` the public repo + `pip install -e '.[train]'` at pod startup.
Adds ~30s per pod creation but eliminates the stall.

The `Dockerfile.train` and the GHA-built `ghcr.io/hbar137/yomi-train`
image are kept in the repo as a possible future fallback but aren't on
the critical path.

### Trap 2: community cloud workers are unreliable

**Symptom**: pod creation succeeds, then either stalls (Trap 1) OR boots
fine but shows a maintenance notice ("This server will be removed from
the platform... ALL DATA WILL BE LOST. Maintenance starts in N days").

**Why**: RunPod's community cloud is decentralized — anyone can host. Hosts
get decommissioned for maintenance, hardware swaps, etc. Worker pool is
volatile.

**Fix**: orchestrator defaults to `--cloud-type SECURE`. Costs ~30–50%
more but boots reliably and stays alive. RTX 4090 secure stock is
typically Medium; A40 secure is High.

### Trap 3: `runpodctl pod create` flag names

The flag names changed at some point and the older names (`--imageName`,
`--gpuType`, `--containerDiskSize`, `--volumeSize`, `--volumePath`,
`--communityCloud`, `--startSSH`, `--cost`) all return "unknown flag" errors.

**Fix** — current correct flags:

```
--name                      <name>
--template-id               runpod-torch-v240
  (or: --image <image>      if rolling your own — rarely useful, see Trap 1)
--gpu-id                    "NVIDIA GeForce RTX 4090"
--gpu-count                 1
--container-disk-in-gb      20
--volume-in-gb              30
--volume-mount-path         /workspace
--cloud-type                SECURE | COMMUNITY
--public-ip                 (community cloud only — needed for SSH there)
--ssh
--ports                     22/tcp
```

`runpodctl pod create --help` is authoritative; check it if a flag fails.

### Trap 4: `uptimeSeconds` lags 5+ min behind actual SSH readiness

**Symptom**: `runpodctl pod get` reports `uptimeSeconds=0` even though
`ssh root@host -p port` already works.

**Why**: RunPod doesn't update that field promptly. Possibly only updates
on container restart events, not container start.

**Fix**: orchestrator does NOT trust `uptimeSeconds`. Instead it probes
SSH directly with a one-shot `ssh ... true` test. As soon as `ssh` returns
exit code 0, the pod is treated as ready.

### Trap 5: `cl-tohoku/bert-base-japanese-char-v3` has no fast tokenizer

**Symptom**: `python -m yomi.train` crashes mid-epoch:
```
NotImplementedError: return_offset_mapping is not available when using
Python tokenizers.
```

**Why**: passing `use_fast=True` to `AutoTokenizer.from_pretrained` falls
back silently when no fast version is shipped — and this model only has
the slow Python tokenizer. Slow tokenizers can't return offset mappings.

**Fix**: this model is strictly character-level (1 char → 1 token, [UNK]
for unrepresentable chars). `train.py` computes char-to-token offsets
manually: token index `i` ↔ character position `i - 1` (after the leading
[CLS] token). If you swap to a different base model that uses subword
tokenization, this trick breaks — you'd need a fast tokenizer or rewrite
the offset logic.

### Trap 6: `--max-steps` runs don't write `models/<run>/best/`

**Symptom**: short smoke run completes; orchestrator scp fails with
`no such file or directory: /workspace/yomi/models/smoke/best`.

**Why**: `train.py` saves `best/` only on validation accuracy improvement.
With `--max-steps 20` we never reach an eval cycle (default
`--eval-every-frac 0.5` waits for half an epoch ≈ 12k steps).

**Fix**: orchestrator tries `best/` first, falls back to `last/` (which
is always saved at end of training, even on early-exit via `--max-steps`).

### Trap 7: hooks/signals can kill the orchestrator child

We hit `pkill -f <task-id>` matching its own subshell because the eval'd
command line contained the search string. Avoid this by:

- Killing only specific PIDs you've captured, not name-matching with
  `pkill -f`.
- Or use `runpodctl pod delete <id>` directly — pod deletion is
  authoritative regardless of orchestrator state.

If the orchestrator dies but the pod is still up, you'll be charged.
Check `runpodctl pod list` and delete strays manually.

## Manual recovery procedures

### Pod stuck or orchestrator hung

```sh
RUNPOD_API_KEY=$(cat ~/secrets/runpod/api_key.txt) runpodctl pod list
RUNPOD_API_KEY=$(cat ~/secrets/runpod/api_key.txt) runpodctl pod delete <pod-id>
```

### Need to debug an in-progress run

The orchestrator streams the training script's stdout to your terminal,
so the loss curve scrolls live. If you need to peek at the pod itself:

```sh
RUNPOD_API_KEY=$(cat ~/secrets/runpod/api_key.txt) runpodctl pod get <pod-id> \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['ssh']['ssh_command'])"
```

That gives you the `ssh root@host -p port` line. Tail GPU usage with
`watch -n 1 nvidia-smi` on the pod.

### Resume a partially-trained run

The orchestrator's `--keep-pod` flag leaves the pod alive after the run.
You can SSH in, edit code in `/workspace/yomi`, re-launch
`python -m yomi.train ...` manually. Useful for hyperparameter sweeps
without re-paying boot/setup cost.

When done: `runpodctl pod delete <id>` from your local machine.

## What's in the repo for RunPod

- `scripts/run_training_pod.py` — the orchestrator. Read it; it's ~300 LOC.
- `Dockerfile.train` + `.github/workflows/build-train.yml` — kept around
  but not on the critical path. The GHA pipeline still runs on every push
  to main; if RunPod ever caches your custom image fast enough, you can
  flip back by swapping `--template-id` for `--image` in the create call.
- `DEPLOY-TRAIN.md` — older "manual" instructions (predates the
  orchestrator); kept for context but `RUNPOD.md` is authoritative.
- `~/secrets/runpod/api_key.txt` — your API key, chmod 600.
