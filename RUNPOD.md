# RUNPOD.md

Reusable operations notes for training on RunPod. Captures the traps we
hit so future projects (yours or someone else's) don't relive them.

Project-specific pieces are at the bottom; everything before that is
generic.

---

## 1. One-time machine setup

### API key

Get one at https://www.runpod.io/console/user/settings → "API Keys".

```sh
mkdir -p ~/secrets/runpod
chmod 700 ~/secrets ~/secrets/runpod
echo 'rpa_xxxxx' > ~/secrets/runpod/api_key.txt
chmod 600 ~/secrets/runpod/api_key.txt
```

Use it via env var (matches the convention from existing project secrets):

```sh
export RUNPOD_API_KEY=$(cat ~/secrets/runpod/api_key.txt)
```

Add to `~/.bashrc` / `~/.zshrc` if you want it always available.

### SSH key

Upload `~/.ssh/id_ed25519.pub` (or RSA equivalent) at the same settings
page → "SSH Public Keys". RunPod injects this into every pod via the
`PUBLIC_KEY` env var; the pod's sshd authorizes it automatically. No
config needed inside the pod.

### runpodctl

```sh
curl -L https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-linux-amd64 \
  -o /tmp/runpodctl && sudo install /tmp/runpodctl /usr/local/bin/runpodctl
runpodctl me   # smoke test: should return JSON with your balance
```

---

## 2. Choosing GPU + cloud

### Cloud type

| | Secure | Community |
|---|---|---|
| Reliability | High — managed datacenters | Volatile — anyone can host |
| Pricing | Higher (typ. 1.3–2× community) | Cheaper |
| Maintenance notices | Rare | Frequent ("ALL DATA WILL BE LOST in N days") |
| 14 GB+ custom image pulls | Usually fine | Routinely stall 18+ min with no error |

**Default to secure unless you have a specific reason not to.** Community
cloud burned us repeatedly with stalled image pulls and decommissioning
notices. Secure cloud A40 / 4090 boots reliably.

### GPU choice

```sh
RUNPOD_API_KEY=$(cat ~/secrets/runpod/api_key.txt) runpodctl gpu list \
  | python3 -c "
import json,sys
for g in json.load(sys.stdin):
    if g.get('secureCloud') and g.get('available'):
        print(f\"{g['gpuId']:<35} {g['displayName']:<20} stock={g['stockStatus']}\")
"
```

For a typical 1B-param-or-smaller training job:

| GPU | VRAM | FP16 TFLOPS | Why pick |
|---|---|---|---|
| RTX 4090 | 24 GB | ~83 | best $/throughput, usually Medium stock |
| A40 | 48 GB | ~37 | always High stock, ECC, sustained load |
| A100 80 GB | 80 GB | ~312 | when you need capacity (big batches, big seq) |

**Match GPU to workload, not to "what feels powerful"** — RTX 4090 beats
A40 on every dimension except VRAM headroom. If 24 GB is enough, pay 4090.

`runpodctl gpu list` shows current `stockStatus`. If it's Low, expect
"resource not available" errors at create time; pick something else.

---

## 3. Pod lifecycle

### Recommended pattern: ephemeral pod + git clone

```
Create pod from RunPod's cached template
    ↓ (~1 min boot)
SSH ready
    ↓
git clone repo + pip install deps
    ↓ (~30 s if pre-cached, ~2 min for transformers et al.)
scp data files
    ↓
Run training over SSH (stdout streams to your terminal)
    ↓
scp checkpoints back home
    ↓
Delete pod
```

**Don't ship a custom Docker image** unless you have a strong reason.
Pulling a 5–15 GB image from a private registry is unreliable on RunPod's
network and takes longer than installing dependencies fresh from PyPI.

Use one of RunPod's cached templates instead:

```sh
RUNPOD_API_KEY=$(cat ~/secrets/runpod/api_key.txt) runpodctl template search pytorch
```

`runpod-torch-v240` (PyTorch 2.4 + CUDA 12.4 + Ubuntu 22.04) is a good
starting point. Override per-project deps via `pip install` at startup.

### Persistent volume pattern (for repeat runs)

If you'll do many runs over weeks, attach a network volume so deps and
data survive pod deletion:

```sh
RUNPOD_API_KEY=$(cat ~/secrets/runpod/api_key.txt) runpodctl pod create \
  --network-volume-id <id> ...
```

First run: clone + install + upload data. Subsequent runs: skip those
steps (they're already on the volume). Saves ~3 min per run.

---

## 4. Proactive status checks

You don't have to babysit a long-running pod, but here's how to peek
without waiting:

### What pods am I paying for right now?

```sh
runpodctl pod list -o json | python3 -c "
import json,sys
for p in json.load(sys.stdin):
    print(f\"id={p['id']}  name={p.get('name')}  status={p.get('desiredStatus')}  \\\${p.get('costPerHr')}/hr  uptime={p.get('uptimeSeconds')}s\")
"
```

This is the most-important command — run it any time something feels off.
A "deleted" run that left a pod alive will show up here (and you're being
charged).

### What's my current spend rate / balance?

```sh
runpodctl me -o json | python3 -c "
import json,sys
d = json.load(sys.stdin)
print(f\"balance: \${d['clientBalance']:.2f}  spending: \${d['currentSpendPerHr']}/hr\")
"
```

If `currentSpendPerHr` is non-zero you have an active pod. Cross-check
against `pod list`.

### Live training stdout

If you launched the orchestrator in background via your shell (or our
`Bash` tool with `run_in_background`), find the output file and tail it:

```sh
tail -f /tmp/<orchestrator-output-file>.log
```

You'll see the same `loss=… lr=…` lines as the running terminal would.

### Peek inside the pod

```sh
SSH_CMD=$(runpodctl pod get <pod-id> -o json | python3 -c "import json,sys; print(json.load(sys.stdin)['ssh']['ssh_command'])")
$SSH_CMD 'nvidia-smi; ps -ef | grep python | head'
```

Or just paste `$SSH_CMD` into a new terminal and you're in.

### Web UI fallback

https://www.runpod.io/console/pods has a visual list with logs. Useful
for seeing the container's stdout when sshd hasn't come up yet, or when
you want to quickly peek without a CLI.

---

## 5. Common traps (and how to avoid each)

### Trap 1 — Custom Docker images stall on community cloud

**Symptom**: pod reports `desiredStatus=RUNNING` and `costPerHr` accrues,
but `uptimeSeconds=0` for 18+ min, sshd never starts. No error surfaced.

**Why**: RunPod community workers don't cache custom images from your
private registry. The 5–15 GB pull across the public internet stalls or
times out without surfacing any error. There's no `runpodctl pod logs`
command.

**Fix**: don't ship a custom image. Use a cached RunPod template, install
your deps at pod startup. See section 3.

### Trap 2 — Community workers get decommissioned mid-rental

**Symptom**: pod boots fine. Login banner shows
"This server will be removed from the platform... ALL DATA WILL BE LOST.
Maintenance starts in N days."

**Why**: community cloud is volunteer-hosted; hosts get retired
constantly. Even short-lived pods can land on a decommissioning host.

**Fix**: default to `--cloud-type SECURE`. Costs more; saves wasted hours.

### Trap 3 — `runpodctl pod create` flag names changed

**Symptom**: `unknown flag: --imageName` (or similar) on what looks like
the documented invocation.

**Why**: runpodctl renamed flags from camelCase to kebab-case at some
point and dropped some entirely (no more `--cost`, no more `--args` to
override entrypoint).

**Fix**: trust `runpodctl pod create --help` over any older docs / blog
posts. Current correct flags include `--image`, `--gpu-id`,
`--container-disk-in-gb`, `--volume-in-gb`, `--volume-mount-path`,
`--cloud-type SECURE|COMMUNITY`, `--public-ip` (community only), `--ssh`,
`--ports`, `--template-id`. There is **no flag** to override the docker
entrypoint at create time.

### Trap 4 — `uptimeSeconds` lags actual SSH readiness

**Symptom**: `runpodctl pod get` reports `uptimeSeconds=0` even though
`ssh root@host -p port` already works fine.

**Why**: that field is updated lazily, not on container start. We
observed 5+ min lag.

**Fix**: don't rely on `uptimeSeconds` for readiness. Probe SSH directly
with a one-shot `ssh ... true` test in a poll loop. Once it returns
exit 0, the pod is usable.

### Trap 5 — `--max-steps` short runs leave no `best/` checkpoint

**Symptom**: smoke run completes; final scp fails with
`no such file or directory: .../models/<run>/best`.

**Why**: most training scripts save `best/` only on validation
improvement. Short runs (e.g. `--max-steps 20`) never reach an eval cycle.

**Fix**: have your training script ALWAYS save `last/` at end of training,
unconditionally. Have your orchestrator try `best/` first and fall back
to `last/`. Or just always pull both.

### Trap 6 — Validate model + tokenizer compatibility BEFORE paying

**Symptom**: pod boots, training starts, crashes after several minutes
with a model-specific error. You've already paid for setup time.

**Examples we hit**:
- `cl-tohoku/bert-base-japanese-char-v3` has no fast tokenizer. Code that
  blindly calls `return_offsets_mapping=True` crashes.
- `transformers>=4.50` registers custom torch ops with type annotations
  that older torch (e.g. 2.4 base image) can't parse.

**Fix**: smoke-test the dependency stack on a $0 environment first
(local CPU, free Colab tier, etc.) before committing GPU dollars. If you
must validate on RunPod, use `--max-steps 5` to minimize cost while
exercising every code path that could fail.

### Trap 7 — `pkill -f` matches its own subshell

**Symptom**: `pkill -f some-task-id` exits 144 because the shell wrapper
running pkill itself contains "some-task-id" in its command line.

**Fix**: kill specific PIDs you've captured, not name patterns. Or use
`pgrep -f` to find PIDs first then kill those PIDs explicitly. Or
prefer `runpodctl pod delete <id>` — pod deletion is authoritative
regardless of orchestrator state.

### Trap 8 — Charging starts at pod creation, not container start

**Symptom**: pod stuck pulling image for 18 min; you assumed "no GPU
running, no charges." Actually charged for the full 18 min.

**Fix**: assume you're paying from the moment `runpodctl pod create`
returns. Always have the orchestrator delete the pod in a `try/finally`,
even on Ctrl-C / errors. Always check `runpodctl pod list` after a
session for stragglers.

---

## 6. Manual recovery

### Stuck pod / hung orchestrator

```sh
runpodctl pod list                                           # find stragglers
runpodctl pod delete <pod-id>                                # authoritative
```

Delete is idempotent and works regardless of what the orchestrator thinks.

### Resume a partial run on the same pod

If your orchestrator supports `--keep-pod` (ours does), the pod stays
alive after the script exits. Then:

```sh
SSH_CMD=$(runpodctl pod get <id> -o json | python3 -c "import json,sys; print(json.load(sys.stdin)['ssh']['ssh_command'])")
$SSH_CMD 'cd /workspace/<repo>; python -m <module> <new args>'
```

Cheaper than re-paying boot + setup overhead for hyperparameter sweeps.

### Pull files from a pod that's about to die

```sh
SCP_BASE="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
SSH_HOST=$(runpodctl pod get <id> -o json | python3 -c "import json,sys; d=json.load(sys.stdin)['ssh']; print(f\"{d['ip']} -p {d['port']}\")")
scp $SCP_BASE -P <port> -r root@<ip>:/workspace/<path> ./local-dest/
```

(Or set `--keep-pod`, scp at leisure, then delete.)

---

## 7. Project-specific notes (yomi)

The yomi project's orchestrator is `scripts/run_training_pod.py`. It's
~300 LOC and applies all the above patterns:

- Uses `runpod-torch-v240` template, secure cloud, RTX 4090 default.
- Probes SSH directly (Trap 4), tries `best/` then `last/` (Trap 5),
  cleans up pod on exit (Trap 8).
- Project-specific bits: scps `data/heteronyms.json + train.jsonl + val.jsonl`,
  runs `python -m yomi.train`, pulls `models/<run>/<best|last>` back home.

```sh
# Smoke test (~$0.06)
python3 scripts/run_training_pod.py --max-steps 20 --batch-size 4 --run-name smoke --yes

# Full Path 1 training (~$0.40)
python3 scripts/run_training_pod.py --gpu RTX4090 --epochs 3 --run-name v1 --yes
```

For other projects, copy `scripts/run_training_pod.py`, replace the
project-specific lines (look for `yomi`, `data/heteronyms.json`,
`yomi.train`, etc.), and you have a working orchestrator.
