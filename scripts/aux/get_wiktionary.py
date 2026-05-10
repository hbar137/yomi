"""Download the latest Japanese Wiktionary XML dump.

Streams from Wikimedia (~80 MB) into data/raw/wiktionary.xml.bz2.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

URL = "https://dumps.wikimedia.org/jawiktionary/latest/jawiktionary-latest-pages-articles.xml.bz2"
DEST = Path("data/raw/wiktionary.xml.bz2")
CHUNK = 1 << 20  # 1 MiB


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists() and DEST.stat().st_size > 0:
        print(f"already present: {DEST} ({DEST.stat().st_size} bytes)", file=sys.stderr)
        return
    tmp = DEST.with_suffix(".bz2.tmp")
    print(f"downloading {URL} -> {DEST}", file=sys.stderr)
    req = urllib.request.Request(URL, headers={"User-Agent": "yomi-research/0.1"})
    bytes_read = 0
    with urllib.request.urlopen(req, timeout=60) as resp, tmp.open("wb") as f:
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
