"""Parse Japanese Wiktionary → data/wiktionary_readings.json.

Wiktionary is messy and inconsistent, but it has wide coverage of less-
common surfaces with their readings. We use it as another reference for
heteronym mining (same role as JMdict).

We look for:
    1. {{ja-noun|...|hira=かんじ}}  / {{ja-verb|...|hira=...}} — Wiktionary's
       Japanese headword templates.
    2. Heading-based fallback for entries that don't use the template:
       title is the surface, the next "==読み==" or pronunciation hints in
       the wikitext give the reading.

We aggressively filter to entries whose surface contains kanji (so we drop
purely-kana entries like 「あいうえお」 → の readings list, which would be
useless for the model).

Output: yomikata-style JSON { surface: [reading, ...], ... } in hiragana.
"""

from __future__ import annotations

import argparse
import bz2
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


def _t(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


HAS_KANJI_RE = re.compile(r"[一-鿿]")
KANA_ONLY_RE = re.compile(r"^[ぁ-ゖー]+$")
KATA_TO_HIRA_OFFSET = -0x60


def kata_to_hira(s: str) -> str:
    out = []
    for ch in s:
        cp = ord(ch)
        if 0x30A1 <= cp <= 0x30F6:
            out.append(chr(cp + KATA_TO_HIRA_OFFSET))
        else:
            out.append(ch)
    return "".join(out)


# {{ja-noun|きょうと|...}}    — first param is hiragana
# {{ja-noun|hira=きょうと}}    — named param
JA_HEAD_RE = re.compile(
    r"\{\{\s*ja-(?:noun|verb|adj|adv|pron|proper|interj|particle|prefix|suffix)\b([^}]*)\}\}",
    re.DOTALL,
)
HIRA_PARAM_RE = re.compile(r"\bhira\s*=\s*([ぁ-ゖー]+)")


def extract_readings_from_wikitext(title: str, text: str) -> list[str]:
    readings: set[str] = set()

    # Restrict to ==Japanese== / ==日本語== section if present.
    sec = re.search(r"==\s*(?:Japanese|日本語)\s*==(.*?)(?=\n==[^=]|\Z)",
                    text, re.DOTALL)
    body = sec.group(1) if sec else text

    for m in JA_HEAD_RE.finditer(body):
        params = m.group(1)
        # Try named hira=…
        h = HIRA_PARAM_RE.search(params)
        if h:
            readings.add(h.group(1))
            continue
        # Else first positional param after the | following the template name.
        # params looks like "|きょうと|..." — split on |
        parts = [p.strip() for p in params.split("|") if p.strip()]
        for p in parts:
            if "=" in p:
                continue
            if KANA_ONLY_RE.match(p):
                readings.add(p)
                break

    # Fallback for entries that just have a よみ section.
    yomi_sec = re.search(r"==+\s*(?:読み|よみ|発音)\s*==+\s*\n((?:[^=]|\n)*)",
                         body)
    if yomi_sec:
        for m in re.finditer(r"[ぁ-ゖー]{2,}", yomi_sec.group(1)):
            readings.add(m.group(0))
        for m in re.finditer(r"[ァ-ヺー]{2,}", yomi_sec.group(1)):
            readings.add(kata_to_hira(m.group(0)))

    return sorted(readings)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path,
                    default=Path("data/raw/wiktionary.xml.bz2"))
    ap.add_argument("--out", type=Path,
                    default=Path("data/wiktionary_readings.json"))
    args = ap.parse_args()

    if not args.input.exists():
        sys.exit(f"missing {args.input}")

    out: dict[str, set[str]] = defaultdict(set)
    n_pages = n_kept = 0

    with bz2.open(args.input, "rb") as f:
        for _, elem in ET.iterparse(f, events=("end",)):
            if _t(elem.tag) != "page":
                continue
            n_pages += 1
            try:
                ns_el = elem.find("{*}ns")
                if ns_el is not None and (ns_el.text or "").strip() != "0":
                    continue
                title_el = elem.find("{*}title")
                rev_el = elem.find("{*}revision")
                if rev_el is None or title_el is None:
                    continue
                text_el = rev_el.find("{*}text")
                if text_el is None or not text_el.text:
                    continue
                title = (title_el.text or "").strip()
                if not HAS_KANJI_RE.search(title):
                    continue  # we want kanji surfaces only

                readings = extract_readings_from_wikitext(title, text_el.text)
                if not readings:
                    continue
                for r in readings:
                    out[title].add(r)
                n_kept += 1
            finally:
                elem.clear()

            if n_pages % 100000 == 0:
                print(f"  pages={n_pages} kept={n_kept}", file=sys.stderr)

    flat = {k: sorted(v) for k, v in out.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(flat, f, ensure_ascii=False, indent=2, sort_keys=True)

    multi = sum(1 for v in flat.values() if len(v) > 1)
    print(f"\npages: {n_pages}", file=sys.stderr)
    print(f"surfaces: {len(flat)}", file=sys.stderr)
    print(f"surfaces with >1 reading: {multi}", file=sys.stderr)
    print(f"-> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
