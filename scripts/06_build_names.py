"""Build a MeCab user dictionary from JMnedict.

JMnedict (data/raw/jmnedict.xml.gz) lists Japanese proper nouns with their
readings. We extract entries tagged as personal/place/company names and emit
a CSV in MeCab user-dict source format. The CSV is later compiled to a
binary .dic with `mecab-dict-index` (run as part of the container build, not
this script — keeps build and data-prep concerns separate).

CSV format (UTF-8, MeCab IPADIC-compatible):
    surface,左文脈ID,右文脈ID,コスト,品詞,...,読み,発音

We use these columns:
    surface, 1288, 1288, 5000, 名詞, 固有名詞, <subtype>, *, *, *, surface,
    reading_kata, reading_kata

The 1288 context IDs come from IPADIC's 名詞,固有名詞 row — they're a safe
default for proper nouns. Cost 5000 is conservative: high enough that
ordinary morphological analysis still wins on common surfaces, low enough
that genuinely rare names (the long tail) get picked up.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

# Map JMnedict <name_type> entity codes to a 名詞-固有名詞 subtype.
NAME_TYPE_MAP = {
    "surname": "人名",
    "given": "人名",
    "person": "人名",
    "fem": "人名",
    "masc": "人名",
    "place": "地名",
    "station": "地名",
    "company": "組織",
    "organization": "組織",
    "product": "一般",
    "work": "一般",
    "char": "一般",
    "creat": "一般",
    "dei": "一般",
    "ev": "一般",
    "fict": "一般",
    "leg": "一般",
    "myth": "一般",
    "obj": "一般",
    "oth": "一般",
    "relig": "一般",
    "serv": "一般",
    "ship": "一般",
    "doc": "一般",
    "group": "組織",
    "id": "一般",
}

# Hiragana → katakana mapping (offset 0x60).
HIRA_RANGE = (0x3041, 0x3096)


def hiragana_to_katakana(s: str) -> str:
    out = []
    for ch in s:
        cp = ord(ch)
        if HIRA_RANGE[0] <= cp <= HIRA_RANGE[1]:
            out.append(chr(cp + 0x60))
        else:
            out.append(ch)
    return "".join(out)


def extract_subtype(types: list[str]) -> str:
    """Pick the first recognised name_type tag's subtype, fall back to 一般."""
    for t in types:
        s = NAME_TYPE_MAP.get(t)
        if s:
            return s
    return "一般"


def iter_jmnedict(path: Path):
    """Yield (surface, reading_hira, [name_types]) tuples."""
    with gzip.open(path, "rb") as f:
        # iterparse to avoid loading the whole 100MB file into memory.
        for _, elem in ET.iterparse(f, events=("end",)):
            if elem.tag != "entry":
                continue
            kebs = [e.text for e in elem.findall("k_ele/keb") if e.text]
            rebs = [e.text for e in elem.findall("r_ele/reb") if e.text]
            types: list[str] = []
            for trans in elem.findall("trans"):
                for tag in trans.findall("name_type"):
                    if tag.text:
                        # JMnedict wraps name types in &<entity>; — ElementTree
                        # delivers us the text content directly.
                        types.append(tag.text.strip())
            for surface in kebs or rebs:
                for reading in rebs:
                    yield surface, reading, types
            elem.clear()


# Surface forms below this are likely too short for safe matching as a
# user-dict entry — they'd shadow common words. Skip them.
MIN_SURFACE_LEN = 2

# Drop entries whose reading contains anything outside hira/kata/長音 — typically
# transliteration artifacts.
KANA_ONLY_RE = re.compile(r"^[ぁ-ゖァ-ヺーー]+$")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path,
                    default=Path("data/raw/jmnedict.xml.gz"))
    ap.add_argument("--out", type=Path, default=Path("data/names.csv"))
    ap.add_argument("--multi-reading-out", type=Path,
                    default=Path("data/names_multi_reading.tsv"),
                    help="sidecar TSV listing surfaces where the user-dict "
                         "kept the first JMnedict reading and dropped others. "
                         "Format: surface<TAB>chosen<TAB>alt1<TAB>alt2... "
                         "Used by v2 to plug in better frequency signals.")
    args = ap.parse_args()

    if not args.input.exists():
        sys.exit(f"missing {args.input}; run scripts/aux/get_jmnedict.py first")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Pass 1: aggregate readings + name_types per surface. JMnedict reading
    # order is editorial/historical (NOT frequency-ranked), so for the user
    # dictionary we just emit the *first* reading we see and accept that as
    # the default. Multi-reading proper nouns are an unavoidable accuracy
    # loss for v1; UniDic is generally more reliable on common names.
    surface_readings: dict[str, list[str]] = {}
    surface_types: dict[str, list[str]] = {}
    n_in = 0
    for surface, reading, types in iter_jmnedict(args.input):
        n_in += 1
        if len(surface) < MIN_SURFACE_LEN:
            continue
        if not KANA_ONLY_RE.match(reading):
            continue
        if surface not in surface_readings:
            surface_readings[surface] = [reading]
            surface_types[surface] = list(types)
        else:
            if reading not in surface_readings[surface]:
                surface_readings[surface].append(reading)
            # accumulate name_type signals across entries for the same surface
            for t in types:
                if t not in surface_types[surface]:
                    surface_types[surface].append(t)

    # Pass 2: emit one user-dict row per surface, using the first observed
    # reading. Multi-reading surfaces also write a sidecar TSV row with all
    # alternatives so v2 can plug in a frequency signal later.
    n_out = n_multi = 0
    type_counts: Counter = Counter()

    args.multi_reading_out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f, \
         args.multi_reading_out.open("w", encoding="utf-8", newline="") as mf:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        # TSV: surface, chosen_reading, alt_reading_1, alt_reading_2, ...
        mw = csv.writer(mf, delimiter="\t", quoting=csv.QUOTE_NONE,
                        escapechar="\\")
        for surface in sorted(surface_readings):
            readings = surface_readings[surface]
            reading = readings[0]
            if len(readings) >= 2:
                n_multi += 1
                mw.writerow([surface, reading] + readings[1:])
            subtype = extract_subtype(surface_types[surface])
            type_counts[subtype] += 1
            kata = hiragana_to_katakana(reading)
            # surface, lcid, rcid, cost, 品詞, 細分類1, 細分類2, 細分類3,
            #         活用型, 活用形, 原形, 読み, 発音
            w.writerow([
                surface, 1288, 1288, 5000,
                "名詞", "固有名詞", subtype, "*",
                "*", "*", surface, kata, kata,
            ])
            n_out += 1
            if n_out % 50000 == 0:
                print(f"  wrote {n_out} rows", file=sys.stderr)

    print(f"\nentries seen: {n_in}", file=sys.stderr)
    print(f"surfaces written: {n_out}", file=sys.stderr)
    print(f"  of which multi-reading (only first kept): {n_multi}", file=sys.stderr)
    print(f"by subtype: {dict(type_counts.most_common())}", file=sys.stderr)
    print(f"-> {args.out}", file=sys.stderr)
    print(f"-> {args.multi_reading_out}  ({n_multi} surfaces)", file=sys.stderr)


if __name__ == "__main__":
    main()
