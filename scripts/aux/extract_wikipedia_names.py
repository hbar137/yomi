"""Extract canonical (surface → reading) pairs from JaWiki article openers.

Output: data/wikipedia_names.json   { "中田英寿": "なかたひでとし", ... }

Most JaWiki articles open with the canonical Japanese name bold, followed
by the kana reading in parentheses:

    '''中田 英寿'''(なかた ひでとし、1977年1月22日 - )は、…
    '''東京駅'''(とうきょうえき)は、…
    '''任天堂株式会社'''(にんてんどうかぶしきがいしゃ、…)は、…

Each Wikipedia article is about one entity, so this (surface, reading)
pair is unambiguous — exactly the per-entity signal we lack from JMnedict
or UniDic. Used at inference time as a per-document override source: when
the Pipeline encounters a document containing 「中田英寿」, it shadows the
default reading with this canonical one.

Internal-space splitting:
    If both surface and kana contain a matching internal space (full-width
    or half-width), we also emit each half as its own entry. For
    「中田 英寿 / なかた ひでとし」 we get:
        中田英寿  -> なかたひでとし   (canonical, no spaces)
        中田      -> なかた          (surname inheritance)
        英寿      -> ひでとし        (given-name inheritance)
    This lets later mentions of just「中田」or「英寿」inherit the right
    reading without needing a heuristic surname dictionary.

We do NOT capture katakana-only surfaces (ベルギー, etc.) — those have
unambiguous readings already (they're their own kana).
"""

from __future__ import annotations

import argparse
import bz2
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def _t(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


# Match the article opener: '''<bold surface>'''(<kana>、…
# Surface is permissive — kanji, kana, ヵヶ々〆 and internal spaces. We
# validate post-match.
OPENER_RE = re.compile(
    r"'''\s*([一-鿿々〆ヵヶァ-ヺー　 ]{2,40}?)\s*'''"
    r"\s*[（(]\s*([ぁ-ゖァ-ヺー　 ']{2,80}?)\s*[、,)）]"
)

# A surface must contain at least one kanji; an all-kana surface needs no
# reading override (it already is its own reading).
HAS_KANJI_RE = re.compile(r"[一-鿿々〆]")

# Wikilinks inside bold like '''[[中田英寿]]''' — strip them.
LINK_RE = re.compile(r"\[\[(?:[^|\]]+\|)?([^\]]+)\]\]")

KANA_ONLY_RE = re.compile(r"^[ぁ-ゖァ-ヺー　 ]+$")


def kata_to_hira(s: str) -> str:
    out = []
    for ch in s:
        cp = ord(ch)
        if 0x30A1 <= cp <= 0x30F6:
            out.append(chr(cp - 0x60))
        else:
            out.append(ch)
    return "".join(out)


def _strip_spaces(s: str) -> str:
    return s.replace(" ", "").replace("　", "").strip()


def _split_on_space(s: str) -> list[str]:
    """Split on full-width or half-width spaces. Returns single-element list
    if no internal space."""
    parts = re.split(r"[　 ]+", s.strip())
    return [p for p in parts if p]


DISAMBIG_SUFFIX_RE = re.compile(r"\s*[（(][^)）]+[)）]\s*$")
BOLD_RE = re.compile(r"'''\s*([一-鿿々〆ヵヶァ-ヺー　 ]{2,40}?)\s*'''")
PAREN_KANA_RE = re.compile(r"\s*[（(]\s*([ぁ-ゖァ-ヺー　 ']{2,80}?)\s*[、,)）]")


def normalize_title(title: str) -> str:
    """Strip a parenthesised disambiguator suffix and whitespace, so an
    article titled 「ジョン・ドウ (野球選手)」 normalises to 「ジョン・ドウ」."""
    return _strip_spaces(DISAMBIG_SUFFIX_RE.sub("", title))


def parse_opener(wikitext: str, title: str) -> tuple[str, str] | None:
    """Extract (surface, hiragana_reading) from the article opener.

    Strategy: scan a generous prefix of the wikitext for every '''bold'''
    occurrence whose bold text matches the article title (after stripping
    internal whitespace). Take the FIRST such bold whose immediately-
    following text is `(kana、…)`. This is robust against fat infobox
    templates without needing to walk MediaWiki template syntax — we just
    look for the canonical opener pattern wherever it lands within a 30k
    char window.

    Example:
        Title:    柴崎友香
        Body:     ...(infobox)...'''柴崎 友香'''(しばさき ともか、本名同じ、…)
        → returns ('柴崎 友香', 'しばさきともか')

    Cross-article false-matches (e.g. body of an unrelated article that
    happens to contain `'''日本'''（ごうしゅう、…）`) are rejected because
    the bold text doesn't match the article title.
    """
    expected = normalize_title(title)
    if not expected or not HAS_KANJI_RE.search(expected):
        # Pure-katakana / Latin titles don't need a reading override.
        return None

    head = LINK_RE.sub(lambda m: m.group(1), wikitext[:50000])

    for m_bold in BOLD_RE.finditer(head):
        surface = m_bold.group(1).strip()
        surf_flat = _strip_spaces(surface)
        is_exact = surf_flat == expected
        is_prefix = (not is_exact) and surf_flat.startswith(expected)
        # Accept exact match, or prefix-of-formal-name (handles
        # title=「任天堂」 / bold=「任天堂株式会社」). Reject otherwise.
        if not (is_exact or is_prefix):
            continue
        m_paren = PAREN_KANA_RE.match(head, m_bold.end())
        if not m_paren:
            continue
        # JaWiki commonly bolds the kana readings (`'''にっぽん'''`); strip
        # the apostrophes before validating.
        kana = m_paren.group(1).replace("'", "").strip()
        if not kana or not KANA_ONLY_RE.match(kana):
            continue
        if not HAS_KANJI_RE.search(surface):
            continue
        if is_exact:
            # Preserve any internal whitespace for surname/given-name split.
            return surface, kata_to_hira(kana)
        # Prefix mode: the captured kana is the *title's* short-form reading
        # (JaWiki convention is to list it first inside the parens), so we
        # emit (title -> reading) and discard the longer formal-name bold.
        return expected, kata_to_hira(kana)
    return None


def emit_entries(surface: str, reading: str) -> dict[str, str]:
    """Convert a (surface, reading) opener pair into the entries we'll
    write out: the no-space canonical form, plus per-segment splits if
    surface and reading have matching internal whitespace."""
    out: dict[str, str] = {}
    s_flat = _strip_spaces(surface)
    r_flat = _strip_spaces(reading)
    if not s_flat or not r_flat:
        return out
    out[s_flat] = r_flat

    s_parts = _split_on_space(surface)
    r_parts = _split_on_space(reading)
    if len(s_parts) >= 2 and len(s_parts) == len(r_parts):
        for sp, rp in zip(s_parts, r_parts):
            sp = sp.strip()
            rp = rp.strip()
            # Skip single-kanji split components: a 1-char given name like
            # 三 read ぞう would poison every common-word use of 三. Multi-
            # char surnames/given-names (中田/英寿) are reliable enough to
            # keep. See trace: 「塚本三」 → 三/ぞう poisoned 三 everywhere.
            if (sp and rp and len(sp) >= 2
                    and HAS_KANJI_RE.search(sp)
                    and KANA_ONLY_RE.match(rp)):
                out.setdefault(sp, rp)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path,
                    default=Path("data/raw/wikipedia.xml.bz2"))
    ap.add_argument("--out", type=Path,
                    default=Path("data/wikipedia_names.json"))
    ap.add_argument("--limit-pages", type=int, default=0,
                    help="for smoke tests; 0 = all")
    args = ap.parse_args()

    if not args.input.exists():
        sys.exit(f"missing {args.input}")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Conflict resolution: for each surface, keep the first reading we see
    # but track conflicts so we can report them. A surface mapping to two
    # different readings across two articles is a real ambiguity (homograph
    # entities — different people named 中田).
    chosen: dict[str, str] = {}
    conflict_count: dict[str, int] = {}

    n_pages = n_with_opener = n_entries = 0

    with bz2.open(args.input, "rb") as f:
        for _, elem in ET.iterparse(f, events=("end",)):
            if _t(elem.tag) != "page":
                continue
            n_pages += 1
            try:
                ns_el = elem.find("{*}ns")
                if ns_el is not None and (ns_el.text or "").strip() != "0":
                    continue
                rev_el = elem.find("{*}revision")
                if rev_el is None:
                    continue
                text_el = rev_el.find("{*}text")
                if text_el is None or not text_el.text:
                    continue
                title_el = elem.find("{*}title")
                title = (title_el.text or "").strip() if title_el is not None else ""
                if not title:
                    continue
                opener = parse_opener(text_el.text, title)
                if not opener:
                    continue
                n_with_opener += 1
                surface, reading = opener
                for s, r in emit_entries(surface, reading).items():
                    if s not in chosen:
                        chosen[s] = r
                        n_entries += 1
                    elif chosen[s] != r:
                        conflict_count[s] = conflict_count.get(s, 1) + 1
            finally:
                elem.clear()

            if n_pages % 100000 == 0:
                print(f"  pages={n_pages} with_opener={n_with_opener} "
                      f"unique_surfaces={len(chosen)}", file=sys.stderr)
            if args.limit_pages and n_pages >= args.limit_pages:
                break

    with args.out.open("w", encoding="utf-8") as f:
        json.dump(chosen, f, ensure_ascii=False, sort_keys=True)

    print(f"\npages: {n_pages}", file=sys.stderr)
    print(f"with parseable opener: {n_with_opener}", file=sys.stderr)
    print(f"unique surfaces written: {len(chosen)}", file=sys.stderr)
    print(f"surfaces with conflicting readings (first kept): {len(conflict_count)}",
          file=sys.stderr)
    print(f"-> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
