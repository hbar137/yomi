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

`runpodctl gpu list` shows availability + stock but **not prices**. For
prices, hit the GraphQL API directly. This one-liner prints a sorted
secure/community comparison for training-grade GPUs:

```sh
curl -s -X POST https://api.runpod.io/graphql \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query":"{ gpuTypes { id displayName memoryInGb securePrice communityPrice } }"}' \
| python3 -c "
import json,sys
ts = json.load(sys.stdin)['data']['gpuTypes']
keep = [t for t in ts if 'AMD' not in t['id'] and t['memoryInGb'] >= 16
        and (t.get('securePrice') or t.get('communityPrice'))]
keep.sort(key=lambda t: t.get('securePrice') or t.get('communityPrice') or 99)
print(f\"{'GPU':<28} {'VRAM':>5} {'sec/hr':>8} {'comm/hr':>9}\")
print('-' * 54)
for t in keep:
    s = t.get('securePrice') or 0
    c = t.get('communityPrice') or 0
    print(f\"{t['displayName'][:27]:<28} {str(t['memoryInGb'])+'G':>5} \"
          f\"{f'\${s:.2f}' if s else '—':>8} {f'\${c:.2f}' if c else '—':>9}\")
"
```

#### Snapshot (2026-05-10, secure cloud unless noted)

Picking the right tier matters more than picking the most powerful GPU.

| Tier | GPU | VRAM | $/hr | When to pick |
|---|---|---|---|---|
| **Cheap small jobs** | RTX A5000 | 24 G | $0.27 | Fits ≤500M-param models, slow but cheap |
| **Sweet spot** | **RTX 4090** | 24 G | **$0.69** | Best $/throughput for ≤1B-param training (~83 TFLOPS FP16) |
| **VRAM headroom** | A40 | 48 G | $0.44 | Larger batch / longer seq when 24G is tight; ECC; sustained load |
| **Big batches** | A100 80GB PCIe | 80 G | $1.39 | When batches need 40–60G activation memory |
| **Frontier** | H100 SXM | 80 G | $2.99 | Multi-GPU training, FP8, big models |
| **Avoid for small jobs** | B200/B300/H200 | 140G+ | $3.39+ | Overkill unless you actually need the VRAM/compute |

#### Quick decision matrix

| Constraint | Pick |
|---|---|
| Single-GPU model fits in 24 GB at desired batch size | RTX 4090 (best perf/$ in this range) |
| Need >24 GB VRAM, ≤48 GB | A40 (cheapest at 48G) |
| Need >48 GB VRAM | A100 80GB PCIe (cheapest at 80G) |
| Multi-GPU NVLink needed | H100 SXM or A100 SXM |
| Hobby/idle VM, latency-tolerant | RTX A5000 community ($0.16/hr) |
| Production batch job, must complete | secure cloud (any tier) |

**Don't fall for "what feels powerful"** — H100/B200 are 4–8× the $/hr of
4090 but only ~3× the FP16 throughput on small workloads. For ≤1B-param
training, 4090 wins on $/throughput by a wide margin.

Stock matters too: `runpodctl gpu list --include-unavailable` shows
`stockStatus`. If it's Low, your `pod create` will likely fail with
"resource not available"; pick the next-best tier.

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

### Trap 9 — Tokenize at dataset-build time, not in the dataloader

**Symptom**: training is GPU-bound on paper (small model, fast GPU) but
GPU utilization sits at 40-50%, low CPU too, throughput far below what
the GPU should deliver.

**Why**: a slow Python tokenizer (e.g. cl-tohoku BertJapaneseTokenizer,
which has no fast Rust variant) running inside the DataLoader workers
can't supply data fast enough. With `num_workers=8` you're limited to
roughly `8 × tokenizer_throughput_per_thread`, and that's often less
than the GPU can consume. Increasing `num_workers` past ~8 has rapidly
diminishing returns on fork overhead.

Symptom often masquerades as "low GPU util means GPU sync stall" — but
when you check CPU and it's also low, the workers are waiting on GIL or
process-level overhead, not the GPU.

**Fix**: tokenize once at dataset-build time. Store `input_ids` (and
char-offset → token-index mapping if you need it for span tasks) in
the jsonl/parquet alongside `sentence` and labels. `__getitem__` becomes
a dict lookup instead of a tokenizer call.

Cost: jsonl size grows ~2× (input_ids per row); training-time CPU drops
from "thousands of tokenizer calls per second" to "thousands of dict
lookups per second." On the yomi 814k-example training set this took
training from 11 step/s to 16 step/s on RTX 4090 (+45%) and pushed GPU
util from ~48% to higher (untested ceiling). Apply this whenever your
training script imports a tokenizer.

The host-side tokenization (writing pre-tokenized jsonl) ran at ~9k
examples/sec on a single thread — way above what the dataloader's
8 workers were achieving (~400 ex/s). It's not slow tokenization, it's
multiprocess overhead in the dataloader path.

### Trap 10 — Default scp is slow on long-RTT links; pipe through tar+gzip

**Symptom**: home upload speed-tests at 90 MB/s, pod download speed-tests
at 70 MB/s, but `scp` between them sustains 1-5 MB/s. Uploading a 500 MB
training dataset takes 10+ min instead of seconds.

**Why**: scp uses a single TCP connection with limited application-layer
buffering. Over high-RTT links (~80-120 ms to RunPod datacenters) the
TCP window doesn't fill the bandwidth-delay-product unless the kernel
auto-tunes aggressively. The legacy SCP protocol also adds per-file
metadata round-trips. Net result: throughput far below either endpoint's
capability.

**Fix**: use a single tar+gzip pipeline through one ssh connection.

```python
# Replaces multiple scp <file> calls.
import shlex, subprocess
def upload_files_compressed(host, port, src_dir, names, remote_dir):
    files = " ".join(shlex.quote(n) for n in names)
    ssh_opts = "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
    remote = (f"mkdir -p {shlex.quote(remote_dir)} && "
              f"cd {shlex.quote(remote_dir)} && gunzip | tar -xf -")
    cmd = (f"tar -cf - -C {shlex.quote(str(src_dir))} {files} "
           f"| gzip -1 "
           f"| ssh {ssh_opts} -p {port} root@{host} {shlex.quote(remote)}")
    subprocess.run(cmd, shell=True, check=True)
```

Three things going right at once:
- **One TCP connection** lets the window scale to full BDP rather than
  scp re-negotiating per file.
- **gzip -1** compresses jsonl 3-5× (repeated field names compress well);
  fewer wire bytes regardless of bottleneck.
- **No SCP per-file overhead** — tar streams everything as one blob.

`gzip` and `tar` ship with the RunPod pytorch template; no extra deps.

If you need partial / resumable uploads, use `rsync -avz --progress`
instead — same compression idea but with delta encoding and resume.

For inbound (pulling checkpoints from pod), the same pattern works
inverted: ssh into pod, tar+gzip there, untar locally. For small
artifacts (~50 MB) plain scp is fine; switch to tar+gzip for >100 MB.

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
