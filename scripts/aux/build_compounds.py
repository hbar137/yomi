"""Build a MeCab user-dict CSV of corpus-attested compounds that UniDic splits.

The hybrid pipeline relies on MeCab to tokenize, then assigns readings per
token. When UniDic over-segments a compound (e.g. 日曜日 -> 日曜/日), we
miss compositional reading shifts like rendaku (日曜日 = にちよう+び, not
にちよう+ひ). Heteronyms.json doesn't help: that's *ambiguous* surfaces.
We need the *unambiguous* compound surfaces UniDic doesn't know.

Strategy: from corpus ruby pairs, aggregate (base, reading) counts. Keep
surfaces that are
  - len >= 2 chars (single-char isn't a compound),
  - frequent enough (>= --min-total-freq, default 50),
  - dominantly one reading (>= --min-dominance, default 0.98 — mirror of
    the heteronym filter; if it's not dominant it's a heteronym),
  - not already in heteronyms.json (heteronym pipeline handles those),
  - actually split by UniDic+names.dic (no need to add an entry MeCab
    already gets right).

Output: data/compounds.csv (MeCab IPADIC-compatible user-dict CSV — same
schema as data/names.csv from 06_build_names.py). Compiled to compounds.dic
at image build time alongside names.dic.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

HIRA_RANGE = (0x3041, 0x3096)
# Hiragana, katakana, prolonged sound mark. Anything else (latin letters,
# kanji-in-reading, etc.) is transliteration noise from Wikipedia
# disambiguators and similar — not a Japanese reading.
KANA_ONLY_RE = re.compile(r"^[ぁ-ゖァ-ヺーー]+$")


def hira_to_kata(s: str) -> str:
    out = []
    for ch in s:
        cp = ord(ch)
        if HIRA_RANGE[0] <= cp <= HIRA_RANGE[1]:
            out.append(chr(cp + 0x60))
        else:
            out.append(ch)
    return "".join(out)


def load_pairs(parquet_paths: list[Path]) -> dict[str, Counter]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for path in parquet_paths:
        if not path.exists():
            print(f"  skip (missing): {path}", file=sys.stderr)
            continue
        t = pq.read_table(path, columns=["pairs"])
        for row in t["pairs"].to_pylist():
            for p in row:
                base = p.get("base")
                reading = p.get("reading")
                if not (base and reading and len(base) >= 2):
                    continue
                if not KANA_ONLY_RE.match(reading):
                    continue  # transliteration / non-kana noise
                counts[base][reading] += 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, action="append",
                    help="parquet file(s) to read; repeatable")
    ap.add_argument("--heteronyms", type=Path,
                    default=Path("data/heteronyms.json"),
                    help="surfaces already covered by the heteronym pipeline")
    ap.add_argument("--names-dic", type=Path,
                    default=Path("data/names.dic"),
                    help="optional pre-compiled names dict to load with the "
                         "Tagger; if missing, MeCab runs with UniDic only")
    ap.add_argument("--out", type=Path, default=Path("data/compounds.csv"))
    ap.add_argument("--min-total-freq", type=int, default=10)
    ap.add_argument("--min-dominance", type=float, default=0.98,
                    help="dominant reading must be >= this fraction of total "
                         "(else it's a heteronym; skip)")
    args = ap.parse_args()

    parquets = args.input or [
        Path("data/aozora.parquet"),
        Path("data/wikipedia.parquet"),
    ]
    print(f"loading {len(parquets)} parquet(s)...", file=sys.stderr)
    counts = load_pairs(parquets)
    print(f"  {len(counts):,} unique multi-char surfaces", file=sys.stderr)

    with args.heteronyms.open(encoding="utf-8") as f:
        heteronyms = set(json.load(f).keys())
    print(f"  heteronyms (will skip): {len(heteronyms):,}", file=sys.stderr)

    # Load MeCab here so we only pay the import cost when we actually need it.
    import fugashi

    tagger_args = ""
    if args.names_dic.exists():
        tagger_args = f"-u {args.names_dic}"
    tagger = fugashi.Tagger(tagger_args) if tagger_args else fugashi.Tagger()

    # Pass 1: filter by frequency + dominance + heteronym membership.
    candidates: dict[str, tuple[str, int]] = {}  # surface -> (reading, count)
    n_too_rare = n_ambiguous = n_heteronym = 0
    for surface, c in counts.items():
        total = sum(c.values())
        if total < args.min_total_freq:
            n_too_rare += 1
            continue
        reading, top = c.most_common(1)[0]
        if top / total < args.min_dominance:
            n_ambiguous += 1
            continue
        if surface in heteronyms:
            n_heteronym += 1
            continue
        candidates[surface] = (reading, total)

    print(f"  rejected: {n_too_rare:,} too rare, {n_ambiguous:,} ambiguous, "
          f"{n_heteronym:,} already heteronym", file=sys.stderr)
    print(f"  candidates after filter: {len(candidates):,}", file=sys.stderr)

    # Pass 2: keep only those UniDic actually splits. Adding entries MeCab
    # already gets right would just bloat the dict.
    kept: list[tuple[str, str, int]] = []
    n_already_one_token = 0
    for surface, (reading, total) in candidates.items():
        toks = list(tagger(surface))
        if len(toks) >= 2:
            kept.append((surface, reading, total))
        else:
            n_already_one_token += 1
    print(f"  rejected: {n_already_one_token:,} already 1 MeCab token",
          file=sys.stderr)
    print(f"  final compound entries: {len(kept):,}", file=sys.stderr)

    # Frequency band breakdown — useful for documenting coverage.
    bands = [(50, 100), (100, 500), (500, 1000), (1000, 5000), (5000, 10**9)]
    for lo, hi in bands:
        n = sum(1 for _, _, c in kept if lo <= c < hi)
        label = f">={lo}" if hi == 10**9 else f"{lo}-{hi}"
        print(f"  freq {label}: {n:,}", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        # Sort by descending frequency, then surface, so the file diffs cleanly
        # when corpus updates and the head shows the highest-impact entries.
        for surface, reading, _ in sorted(kept, key=lambda r: (-r[2], r[0])):
            kata = hira_to_kata(reading)
            # Same IPADIC-style schema as names.csv. Cost 500 — has to beat
            # UniDic's per-char split path for common kanji like 二/日, where
            # each char has node cost ~500-1000 and the transition penalty
            # is small. names.csv stays at 5000 because a per-char split of
            # a rare-kanji name has much higher unknown-word cost.
            w.writerow([
                surface, 1288, 1288, 500,
                "名詞", "一般", "*", "*",
                "*", "*", surface, kata, kata,
            ])

    print(f"\nwrote {args.out}  ({len(kept):,} entries)", file=sys.stderr)
    if kept:
        print("\ntop 15 by frequency:", file=sys.stderr)
        for surface, reading, count in sorted(kept, key=lambda r: -r[2])[:15]:
            print(f"  {surface:>6}  {reading:<10}  ({count:,})", file=sys.stderr)


if __name__ == "__main__":
    main()
