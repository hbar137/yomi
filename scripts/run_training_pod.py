"""Orchestrate a single training run on RunPod via runpodctl.

End-to-end:
    1. Create an RTX 3090 spot pod from RunPod's PyTorch 2.4 template
       (~1 min boot — cached on their workers, vs ~20 min for our 14 GB
       custom image which RunPod doesn't cache).
    2. Wait for SSH.
    3. On the pod: git clone https://github.com/hbar137/yomi.git +
       `pip install -e '.[train]'`. Idempotent — re-running on a pod with
       persistent volume just `git pull`s.
    4. scp data/heteronyms.json + train.jsonl + val.jsonl onto the pod.
    5. Run `python -m yomi.train ...` over SSH, streaming stdout.
    6. scp models/<run-name>/best/ back home.
    7. Delete the pod.

Cost confirmation: prints estimated $/hr and total before creating the pod;
requires --yes to skip the interactive prompt. Pod is deleted on exit
(success, error, or Ctrl-C — see the try/finally).

Prerequisites:
    - runpodctl installed
    - RUNPOD_API_KEY in env (e.g. `export RUNPOD_API_KEY=$(cat ~/secrets/runpod/api_key.txt)`)
    - SSH pubkey registered at https://www.runpod.io/console/user/settings
      (RunPod injects it into pods on creation)
    - data/heteronyms.json + data/train.jsonl + data/val.jsonl exist locally

Usage:
    python scripts/run_training_pod.py                # interactive, defaults
    python scripts/run_training_pod.py --yes          # skip cost prompt
    python scripts/run_training_pod.py --gpu RTX3090 --epochs 3 \
                                       --run-name v1 --max-steps 20
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

GPU_TYPES = {
    "RTX3090": "NVIDIA GeForce RTX 3090",
    "RTX4090": "NVIDIA GeForce RTX 4090",
    "A40":     "NVIDIA A40",
    "A100":    "NVIDIA A100-SXM4-80GB",
}


def runpodctl(*args, capture: bool = True) -> dict | str:
    """Invoke `runpodctl` and return parsed JSON (capture=True) or raw stdout."""
    cmd = ["runpodctl", *args]
    if capture:
        cmd.extend(["-o", "json"])
    proc = subprocess.run(cmd, capture_output=capture, text=True, check=False)
    if proc.returncode != 0:
        msg = proc.stderr if capture else "(see above)"
        sys.exit(f"runpodctl {' '.join(args)} failed:\n{msg}")
    if not capture:
        return ""
    out = proc.stdout.strip()
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return out


def ensure_runpodctl_ready() -> None:
    if shutil.which("runpodctl") is None:
        sys.exit("runpodctl not on PATH; install from https://docs.runpod.io/runpodctl")
    me = runpodctl("me")
    if isinstance(me, dict) and me.get("error"):
        sys.exit(f"runpodctl auth not configured: {me['error']}")


def ssh_exec(host: str, port: int, cmd: str, *, stream: bool = False) -> int:
    """Run `cmd` on the pod via ssh. Returns exit code. Streams output to
    this process's stdout if stream=True."""
    ssh = [
        "ssh", "-p", str(port),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        f"root@{host}",
        cmd,
    ]
    if stream:
        return subprocess.run(ssh).returncode
    return subprocess.run(ssh, capture_output=True).returncode


def scp_to(host: str, port: int, src: Path, dst: str) -> None:
    cmd = [
        "scp", "-P", str(port),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        str(src), f"root@{host}:{dst}",
    ]
    subprocess.run(cmd, check=True)


def scp_from(host: str, port: int, src: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "scp", "-P", str(port), "-r",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        f"root@{host}:{src}", str(dst),
    ]
    subprocess.run(cmd, check=True)


def wait_for_ssh(pod_id: str, timeout: int = 1200) -> tuple[str, int]:
    """Poll pod status until SSH actually responds. Returns (host, port).

    `runpodctl pod get` reports SSH host/port quickly, but its
    `uptimeSeconds` lags far behind the container's real readiness — on
    secure-cloud A40 we observed `uptime=0` for 5+ min after sshd was
    already accepting connections. So we test SSH directly with a short
    probe and consider the pod ready as soon as a one-shot `echo` works.
    """
    deadline = time.time() + timeout
    host = port = None
    while time.time() < deadline:
        info = runpodctl("pod", "get", pod_id)
        if isinstance(info, dict):
            ssh = info.get("ssh") or {}
            host = ssh.get("ip") or host
            port = ssh.get("port") or port
            if host and port:
                rc = subprocess.run(
                    ["ssh", "-p", str(port),
                     "-o", "StrictHostKeyChecking=no",
                     "-o", "UserKnownHostsFile=/dev/null",
                     "-o", "LogLevel=ERROR",
                     "-o", "ConnectTimeout=8",
                     f"root@{host}", "true"],
                    capture_output=True,
                ).returncode
                if rc == 0:
                    return host, int(port)
                print(f"  ssh {host}:{port} not yet responding...", file=sys.stderr)
            else:
                print(f"  pod status={info.get('desiredStatus','?')}, no ssh info yet",
                      file=sys.stderr)
        time.sleep(10)
    sys.exit(f"timed out after {timeout}s; check `runpodctl pod get {pod_id}` manually")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", choices=GPU_TYPES.keys(), default="A40")
    ap.add_argument("--cloud-type", choices=("SECURE", "COMMUNITY"), default="SECURE",
                    help="SECURE is more reliable; COMMUNITY is cheaper but "
                         "workers are often being decommissioned/replaced "
                         "(observed pull stalls of 18+ min on community)")
    ap.add_argument("--max-cost", type=float, default=0.30,
                    help="$/hr ceiling; pod fails to create if no GPU available below this")
    ap.add_argument("--container-disk", type=int, default=20, help="GB")
    ap.add_argument("--volume-size", type=int, default=30, help="GB persistent")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--run-name", default="v1")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="cap total steps; 0 = no cap (full epochs)")
    ap.add_argument("--extra-args", default="",
                    help="extra args appended to `python -m yomi.train`")
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "models",
                    help="local dir to receive checkpoints")
    ap.add_argument("--template-id", default="runpod-torch-v240",
                    help="RunPod template id; default uses cached pytorch 2.4")
    ap.add_argument("--repo-url", default="https://github.com/hbar137/yomi.git",
                    help="git repo cloned into /workspace/yomi on pod startup")
    ap.add_argument("--name", default=None,
                    help="pod name (default: yomi-train-<run-name>)")
    ap.add_argument("--keep-pod", action="store_true",
                    help="don't delete the pod when done — useful for debug")
    ap.add_argument("--yes", action="store_true",
                    help="skip cost confirmation prompt")
    return ap.parse_args()


def confirm(args: argparse.Namespace) -> None:
    if args.yes:
        return
    print(f"\n=== about to create RunPod pod ===", file=sys.stderr)
    print(f"  GPU:           {args.gpu} ({GPU_TYPES[args.gpu]})", file=sys.stderr)
    print(f"  max $/hr:      ${args.max_cost:.2f}", file=sys.stderr)
    print(f"  container:     {args.image}", file=sys.stderr)
    print(f"  volume:        {args.volume_size} GB persistent", file=sys.stderr)
    print(f"  run-name:      {args.run_name}", file=sys.stderr)
    print(f"  epochs:        {args.epochs}", file=sys.stderr)
    if args.max_steps:
        print(f"  max-steps:     {args.max_steps}", file=sys.stderr)
    print(f"\nestimated cost: ~${args.max_cost:.2f}/hr × ~1 hr = ~${args.max_cost:.2f} per run",
          file=sys.stderr)
    resp = input("\nproceed? [y/N] ").strip().lower()
    if resp != "y":
        sys.exit("aborted.")


def build_training_args(args: argparse.Namespace) -> str:
    parts = [
        "python", "-m", "yomi.train",
        "--data-dir", "/workspace/data",
        "--heteronyms", "/workspace/data/heteronyms.json",
        "--out-dir", "/workspace/models",
        "--run-name", args.run_name,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
    ]
    if args.max_steps:
        parts += ["--max-steps", str(args.max_steps)]
    if args.extra_args:
        parts += shlex.split(args.extra_args)
    return shlex.join(parts)


def main() -> None:
    args = parse_args()
    pod_name = args.name or f"yomi-train-{args.run_name}"

    # Validate local data files exist before paying for compute.
    for fname in ("heteronyms.json", "train.jsonl", "val.jsonl"):
        if not (args.data_dir / fname).exists():
            sys.exit(f"missing {args.data_dir / fname} — run scripts/03–05 first")

    ensure_runpodctl_ready()
    confirm(args)

    print(f"\ncreating pod...", file=sys.stderr)
    create_args = [
        "pod", "create",
        "--name", pod_name,
        "--template-id", args.template_id,
        "--gpu-id", GPU_TYPES[args.gpu],
        "--gpu-count", "1",
        "--container-disk-in-gb", str(args.container_disk),
        "--volume-in-gb", str(args.volume_size),
        "--volume-mount-path", "/workspace",
        "--cloud-type", args.cloud_type,
        "--ssh",
        "--ports", "22/tcp",
    ]
    pod = runpodctl(*create_args)
    pod_id = pod["id"] if isinstance(pod, dict) else None
    if not pod_id:
        sys.exit(f"unexpected runpodctl response: {pod!r}")
    print(f"  pod id: {pod_id}", file=sys.stderr)

    cleanup_done = False

    def cleanup() -> None:
        nonlocal cleanup_done
        if cleanup_done:
            return
        cleanup_done = True
        if args.keep_pod:
            print(f"\n--keep-pod set; pod {pod_id} left running", file=sys.stderr)
            print(f"  ssh -p {port} root@{host}", file=sys.stderr)
            print(f"  delete later: runpodctl pod delete {pod_id}", file=sys.stderr)
            return
        print(f"\nstopping/deleting pod {pod_id}...", file=sys.stderr)
        runpodctl("pod", "delete", pod_id, capture=False)

    def handle_signal(sig, frame):  # noqa: ANN001
        cleanup()
        sys.exit(130)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    host = port = None
    try:
        print(f"waiting for SSH to be reachable...", file=sys.stderr)
        host, port = wait_for_ssh(pod_id)
        print(f"  ssh: root@{host} port {port}", file=sys.stderr)

        # Wait until sshd is actually accepting connections (status RUNNING
        # doesn't always mean ssh is listening yet).
        for attempt in range(12):
            if ssh_exec(host, port, "echo ok") == 0:
                break
            time.sleep(5)
        else:
            sys.exit("ssh not responding after 60s; abort")

        print(f"\ncloning repo + installing training deps on pod...", file=sys.stderr)
        setup_cmd = (
            "set -e && "
            "cd /workspace && "
            f"if [ -d yomi/.git ]; then cd yomi && git pull --ff-only; "
            f"else git clone {shlex.quote(args.repo_url)} yomi && cd yomi; fi && "
            "pip install --quiet -e '.[train]' && "
            "mkdir -p data models && "
            "echo SETUP_DONE"
        )
        rc = ssh_exec(host, port, setup_cmd, stream=True)
        if rc != 0:
            print(f"\nsetup failed with rc={rc}", file=sys.stderr)
            sys.exit(rc)

        print(f"\nstaging data...", file=sys.stderr)
        for fname in ("heteronyms.json", "train.jsonl", "val.jsonl"):
            src = args.data_dir / fname
            print(f"  scp {src.name} ({src.stat().st_size // (1024*1024)} MB)...",
                  file=sys.stderr)
            scp_to(host, port, src, "/workspace/yomi/data/")

        # Training command runs from /workspace/yomi so paths resolve to the
        # repo's data/ + models/ subdirs.
        train_args = build_training_args(args).replace(
            "--data-dir /workspace/data", "--data-dir /workspace/yomi/data"
        ).replace(
            "--heteronyms /workspace/data/heteronyms.json",
            "--heteronyms /workspace/yomi/data/heteronyms.json"
        ).replace(
            "--out-dir /workspace/models", "--out-dir /workspace/yomi/models"
        )
        train_cmd = f"cd /workspace/yomi && {train_args}"
        print(f"\nrunning training:\n  {train_cmd}\n", file=sys.stderr)
        rc = ssh_exec(host, port, train_cmd, stream=True)
        if rc != 0:
            print(f"\ntraining exited with {rc}", file=sys.stderr)
            sys.exit(rc)

        print(f"\npulling checkpoint...", file=sys.stderr)
        # Try `best` first (saved on val-acc improvement); fall back to
        # `last` (always saved at end of training) for short smoke runs
        # that don't reach an eval cycle.
        pulled = False
        for kind in ("best", "last"):
            remote = f"/workspace/yomi/models/{args.run_name}/{kind}"
            local = args.out_dir / args.run_name / kind
            check = ssh_exec(host, port, f"test -d {shlex.quote(remote)}")
            if check != 0:
                continue
            scp_from(host, port, remote, local)
            print(f"  pulled {kind} -> {local}", file=sys.stderr)
            pulled = True
            break
        if not pulled:
            print("  no checkpoint dir found on pod (no val cycle ran?)",
                  file=sys.stderr)

    finally:
        cleanup()

    print(f"\ndone.", file=sys.stderr)


if __name__ == "__main__":
    main()
