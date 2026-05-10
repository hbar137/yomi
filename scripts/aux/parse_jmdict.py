"""Parse JMdict_e → data/jmdict_readings.json.

JMdict is a CC BY-SA Japanese-English dictionary. We use it as a *reference*
for heteronym mining: it tells us all attested readings for a given kanji
surface, so 03_mine_heteronyms.py can union our corpus-derived candidates
with JMdict's curated list.

Output: a yomikata-style JSON: { surface: [reading, ...], ... }.
Readings are in hiragana. We exclude entries marked with re_nokanji (where
the reading isn't actually a reading of the kanji) and re_inf="ok" (rare /
literary forms — too noisy for everyday yomi).
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path,
                    default=Path("data/raw/jmdict.xml.gz"))
    ap.add_argument("--out", type=Path,
                    default=Path("data/jmdict_readings.json"))
    args = ap.parse_args()

    if not args.input.exists():
        sys.exit(f"missing {args.input}")

    surface_to_readings: dict[str, set[str]] = defaultdict(set)
    n_entries = 0
    n_pairs = 0

    with gzip.open(args.input, "rb") as f:
        for _, elem in ET.iterparse(f, events=("end",)):
            if elem.tag != "entry":
                continue
            n_entries += 1
            kebs = [e.text for e in elem.findall("k_ele/keb") if e.text]
            r_eles = elem.findall("r_ele")
            for r in r_eles:
                reb = r.findtext("reb") or ""
                if not reb:
                    continue
                # Skip re_nokanji entries — reading not associated with the kanji.
                if r.find("re_nokanji") is not None:
                    continue
                # Skip "ok" (out-of-date kana) entries.
                infs = [i.text for i in r.findall("re_inf") if i.text]
                if any("ok" in (i or "") for i in infs):
                    continue
                # re_restr restricts which kebs this reading applies to.
                restr = [e.text for e in r.findall("re_restr") if e.text]
                applicable = restr if restr else kebs
                for surface in applicable:
                    surface_to_readings[surface].add(reb)
                    n_pairs += 1
            elem.clear()

    out = {s: sorted(rs) for s, rs in surface_to_readings.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, sort_keys=True)

    multi = sum(1 for rs in out.values() if len(rs) > 1)
    print(f"\nentries: {n_entries}", file=sys.stderr)
    print(f"surfaces: {len(out)}", file=sys.stderr)
    print(f"surfaces with >1 reading: {multi}", file=sys.stderr)
    print(f"surface,reading pairs: {n_pairs}", file=sys.stderr)
    print(f"-> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
