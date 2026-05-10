"""Download JMdict (Japanese-Multilingual Dictionary, English glosses) from EDRDG.

Used for: heteronym candidate generation (surfaces with multiple registered readings)
and as a reading-validity reference. License: CC BY-SA 4.0 (EDRDG).
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

# EDRDG's https cert has a hostname mismatch (last verified 2026-05); their
# http endpoint works and the data is public-domain CC BY-SA, so http is fine.
URL = "http://ftp.edrdg.org/pub/Nihongo/JMdict_e.gz"
DEST = Path("data/raw/jmdict.xml.gz")


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists() and DEST.stat().st_size > 0:
        print(f"already present: {DEST} ({DEST.stat().st_size} bytes)", file=sys.stderr)
        return
    print(f"downloading {URL} -> {DEST}", file=sys.stderr)
    req = urllib.request.Request(URL, headers={"User-Agent": "yomi-research/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        DEST.write_bytes(resp.read())
    print(f"done: {DEST.stat().st_size} bytes", file=sys.stderr)


if __name__ == "__main__":
    main()
