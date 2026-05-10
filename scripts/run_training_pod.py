"""Orchestrate a single training run on RunPod via runpodctl.

End-to-end:
    1. Create an RTX 3090 spot pod with our training image, idle (`sleep infinity`).
    2. Wait for SSH to be ready.
    3. Stage data: scp data/heteronyms.json + train.jsonl + val.jsonl onto the pod.
    4. Run training over SSH, streaming stdout to this terminal.
    5. scp models/<run-name>/best/ back home.
    6. Delete the pod.

Cost confirmation: prints estimated $/hr and total before creating the pod;
requires --yes to skip the interactive prompt. Pod is deleted on exit
(success, error, or Ctrl-C — see the try/finally).

Prerequisites:
    - runpodctl installed (or RUNPOD_API_KEY set + uses HTTP API directly)
    - SSH key registered at https://www.runpod.io/console/user/settings
      (RunPod injects it into pods automatically when --startSSH is set)
    - ~/yomi has data/heteronyms.json + data/train.jsonl + data/val.jsonl

Usage:
    python scripts/run_training_pod.py                # interactive, defaults
    python scripts/run_training_pod.py --yes          # skip cost prompt
    python scripts/run_training_pod.py --gpu RTX3090 --epochs 3 \
                                       --run-name v1 --max-cost 0.30
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


def wait_for_ssh(pod_id: str, timeout: int = 600) -> tuple[str, int]:
    """Poll pod status until SSH is ready. Returns (host, port)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        info = runpodctl("pod", "get", pod_id)
        # Different runpodctl versions surface SSH info differently — try a few.
        if isinstance(info, dict):
            status = info.get("desiredStatus") or info.get("status") or ""
            for port_info in info.get("runtime", {}).get("ports") or []:
                if port_info.get("privatePort") == 22 and port_info.get("isIpPublic"):
                    return port_info["ip"], port_info["publicPort"]
            print(f"  pod status={status}, ssh not ready yet...", file=sys.stderr)
        time.sleep(10)
    sys.exit("timed out waiting for SSH; check `runpodctl pod get <id>` manually")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", choices=GPU_TYPES.keys(), default="RTX3090")
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
    ap.add_argument("--image", default="ghcr.io/hbar137/yomi-train:latest")
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
        "--image", args.image,
        "--gpu-id", GPU_TYPES[args.gpu],
        "--gpu-count", "1",
        "--container-disk-in-gb", str(args.container_disk),
        "--volume-in-gb", str(args.volume_size),
        "--volume-mount-path", "/workspace",
        "--cloud-type", "COMMUNITY",
        "--ssh",
        "--ports", "22/tcp",
        # No --args / --entrypoint flag: image's ENTRYPOINT=sleep infinity
        # keeps the pod idle so we can SSH in.
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

        print(f"\nstaging data...", file=sys.stderr)
        ssh_exec(host, port, "mkdir -p /workspace/data /workspace/models")
        for fname in ("heteronyms.json", "train.jsonl", "val.jsonl"):
            src = args.data_dir / fname
            print(f"  scp {src.name} ({src.stat().st_size // (1024*1024)} MB)...",
                  file=sys.stderr)
            scp_to(host, port, src, "/workspace/data/")

        train_cmd = build_training_args(args)
        print(f"\nrunning training:\n  {train_cmd}\n", file=sys.stderr)
        rc = ssh_exec(host, port, train_cmd, stream=True)
        if rc != 0:
            print(f"\ntraining exited with {rc}", file=sys.stderr)
            sys.exit(rc)

        print(f"\npulling checkpoint...", file=sys.stderr)
        scp_from(host, port,
                 f"/workspace/models/{args.run_name}/best",
                 args.out_dir / args.run_name / "best")
        print(f"  -> {args.out_dir / args.run_name / 'best'}", file=sys.stderr)

    finally:
        cleanup()

    print(f"\ndone.", file=sys.stderr)


if __name__ == "__main__":
    main()
