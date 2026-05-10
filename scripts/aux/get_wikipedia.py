"""Download the latest Japanese Wikipedia XML dump (~3.5 GB).

Streams from Wikimedia. This is large; resume support is intentional:
if the .tmp file exists we restart with a Range request to pick up where we
left off.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

URL = "https://dumps.wikimedia.org/jawiki/latest/jawiki-latest-pages-articles.xml.bz2"
DEST = Path("data/raw/wikipedia.xml.bz2")
CHUNK = 4 << 20  # 4 MiB


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists() and DEST.stat().st_size > 0:
        print(f"already present: {DEST} ({DEST.stat().st_size} bytes)", file=sys.stderr)
        return

    tmp = DEST.with_suffix(".bz2.tmp")
    start = tmp.stat().st_size if tmp.exists() else 0
    headers = {"User-Agent": "yomi-research/0.1"}
    if start > 0:
        headers["Range"] = f"bytes={start}-"
        print(f"resuming from {start} bytes", file=sys.stderr)

    print(f"downloading {URL} -> {DEST}", file=sys.stderr)
    req = urllib.request.Request(URL, headers=headers)
    bytes_read = start
    mode = "ab" if start > 0 else "wb"
    with urllib.request.urlopen(req, timeout=120) as resp, tmp.open(mode) as f:
        while True:
            chunk = resp.read(CHUNK)
            if not chunk:
                break
            f.write(chunk)
            bytes_read += len(chunk)
            print(f"\r  {bytes_read / 1024 / 1024:.1f} MiB", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    tmp.rename(DEST)
    print(f"done: {DEST.stat().st_size} bytes", file=sys.stderr)


if __name__ == "__main__":
    main()
