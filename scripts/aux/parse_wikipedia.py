"""Parse the Japanese Wikipedia XML dump → data/wikipedia.parquet.

Input:  data/raw/wikipedia.xml.bz2
Output: data/wikipedia.parquet (same schema as Aozora / NHK Easy)

We extract sentences that contain at least one ruby annotation. Sources of
ruby annotations in Wikipedia:

    {{Ruby|漢字|かんじ}}        (most common)
    {{ruby|漢字|かんじ}}
    {{読み仮名|漢字|かんじ}}
    {{読み|漢字|かんじ}}
    <ruby>漢字<rt>かんじ</rt></ruby>   (rare but appears)

Wikipedia is also full of `{{lang|ja|...}}`, link markup, refs, tables, and
templates we don't care about. We strip wikitext aggressively after pulling
ruby pairs out — the remaining text is used as the sentence context for
training, so it must be reasonably clean (no `[[...]]` brackets, no `'''`
runs, no template clutter).

Strategy:
    1. Pull ruby pairs out of the raw wikitext. Replace each match with just
       the base text and remember the pair.
    2. Strip surrounding wikitext (links, refs, templates, tables, html).
    3. Recompute ruby pair offsets against the stripped text.
    4. Sentence-split on 。！？ and emit one row per ruby-bearing sentence.

Streaming: the dump is ~4.6 GB compressed. We use bz2.open + iterparse and
clear elements as we go. RSS stays well under a gig.
"""

from __future__ import annotations

import argparse
import bz2
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PA_SCHEMA = pa.schema([
    ("work_id", pa.string()),
    ("author", pa.string()),
    ("title", pa.string()),
    ("sentence_idx", pa.int32()),
    ("text", pa.string()),
    ("pairs", pa.list_(pa.struct([
        ("base", pa.string()),
        ("reading", pa.string()),
        ("start", pa.int32()),
        ("end", pa.int32()),
    ]))),
])


# Strip XML namespace prefix that ElementTree gives us in tag names like
# "{http://www.mediawiki.org/xml/export-0.10/}page".
def _t(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


# --- ruby extractors -------------------------------------------------------

RUBY_TEMPLATE_RE = re.compile(
    r"\{\{\s*(?:[Rr]uby|読み仮名|読み)\s*\|\s*([^|}]+?)\s*\|\s*([^|}]+?)\s*(?:\|[^}]*)?\}\}"
)
RUBY_HTML_RE = re.compile(
    r"<ruby[^>]*>\s*([^<]+?)\s*<rt[^>]*>\s*([^<]+?)\s*</rt>\s*</ruby>",
    re.IGNORECASE,
)


def _extract_ruby(text: str) -> tuple[str, list[tuple[str, str, int, int]]]:
    """Replace ruby templates / tags in `text` with their base form.

    Returns (text_without_ruby_markup, [(base, reading, start, end), ...])
    where the offsets are into the *output* text (post-substitution) but
    BEFORE wikitext stripping. We re-anchor them in stripped text afterwards
    by keeping the base text intact through stripping.

    To survive wikitext stripping, we wrap each base with sentinel
    characters (\x01 ... \x02) so the stripper won't mangle them. We strip
    sentinels (and recompute offsets) at the end.
    """
    pairs: list[tuple[str, str, int, int]] = []
    out: list[str] = []
    i = 0

    # Combine both patterns so we process them in a single left-to-right
    # pass and keep offsets monotone.
    matches: list[tuple[int, int, str, str]] = []
    for m in RUBY_TEMPLATE_RE.finditer(text):
        matches.append((m.start(), m.end(), m.group(1), m.group(2)))
    for m in RUBY_HTML_RE.finditer(text):
        matches.append((m.start(), m.end(), m.group(1), m.group(2)))
    matches.sort()

    cursor = 0
    for start, end, base, reading in matches:
        if start < cursor:
            continue  # overlapping match (rare); skip
        out.append(text[cursor:start])
        # \x01<base>\x02 — sentinels survive strip passes that operate on
        # markup characters but never on these control codes.
        out.append("\x01")
        sentinel_start = sum(len(s) for s in out) - 1
        out.append(base)
        out.append("\x02")
        pairs.append((base, reading, sentinel_start + 1,
                      sentinel_start + 1 + len(base)))
        cursor = end
    out.append(text[cursor:])
    return "".join(out), pairs


# --- wikitext stripping ----------------------------------------------------

# Order matters here. We do refs and tables first (they can contain other
# markup we don't want to walk into), then templates, then links, then
# decoration.

REF_RE = re.compile(r"<ref[^>/]*>.*?</ref>", re.DOTALL | re.IGNORECASE)
REF_SELF_RE = re.compile(r"<ref[^>]*/>", re.IGNORECASE)
COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
HTML_TAG_RE = re.compile(r"</?(?:div|span|small|sub|sup|s|u|center|nowiki|"
                         r"poem|gallery|table|tr|td|th|ul|ol|li|br|p|hr|"
                         r"font|big|cite|code|pre|blockquote|abbr|q|i|b|em|"
                         r"strong|del|ins|var|kbd|samp|tt|dfn|figure|caption"
                         r")[^>]*/?>", re.IGNORECASE)
FILE_LINK_RE = re.compile(r"\[\[(?:File|Image|ファイル|画像):[^\[\]]*"
                          r"(?:\[\[[^\[\]]*\]\][^\[\]]*)*\]\]",
                          re.IGNORECASE)
LINK_RE = re.compile(r"\[\[([^\[\]|]+)(?:\|([^\[\]]+))?\]\]")
EXTLINK_RE = re.compile(r"\[https?://[^\s\]]+\s+([^\]]+)\]")
EXTLINK_BARE_RE = re.compile(r"\[https?://[^\s\]]+\]")
BOLD_ITALIC_RE = re.compile(r"'{2,5}")
HEADER_RE = re.compile(r"^=+[^=\n]+=+\s*$", re.MULTILINE)
TABLE_RE = re.compile(r"\{\|.*?\|\}", re.DOTALL)
LIST_PREFIX_RE = re.compile(r"^[*#:;]+\s*", re.MULTILINE)
WHITESPACE_RE = re.compile(r"[ \t]+")
NEWLINE_COLLAPSE_RE = re.compile(r"[ \t]*\n+[ \t]*")


def _strip_template(text: str) -> str:
    """Remove all `{{...}}` templates, handling nesting."""
    out = []
    depth = 0
    i = 0
    while i < len(text):
        if text[i:i + 2] == "{{":
            depth += 1
            i += 2
        elif text[i:i + 2] == "}}":
            depth = max(0, depth - 1)
            i += 2
        elif depth == 0:
            out.append(text[i])
            i += 1
        else:
            i += 1
    return "".join(out)


def strip_wikitext(text: str) -> str:
    """Best-effort wikitext → plain text. Keeps sentinels (\x01/\x02) intact
    because they're outside every regex character class."""
    text = COMMENT_RE.sub("", text)
    text = REF_RE.sub("", text)
    text = REF_SELF_RE.sub("", text)
    text = TABLE_RE.sub("", text)
    text = FILE_LINK_RE.sub("", text)
    # Plain links: [[anchor]] or [[target|anchor]] → anchor.
    text = LINK_RE.sub(lambda m: m.group(2) or m.group(1), text)
    text = EXTLINK_RE.sub(lambda m: m.group(1), text)
    text = EXTLINK_BARE_RE.sub("", text)
    text = _strip_template(text)
    text = HTML_TAG_RE.sub("", text)
    text = BOLD_ITALIC_RE.sub("", text)
    text = HEADER_RE.sub("", text)            # drop section headers entirely
    text = LIST_PREFIX_RE.sub("", text)
    text = NEWLINE_COLLAPSE_RE.sub(" ", text)  # paragraphs → space (sentinels survive)
    text = WHITESPACE_RE.sub(" ", text)
    return text


def _resolve_pair_offsets(stripped: str) -> tuple[str, list[dict]]:
    """Walk the sentinel-marked stripped text, collect pair offsets in the
    final clean text, and return both."""
    out: list[str] = []
    pairs: list[dict] = []
    i = 0
    cur_base_start: int | None = None
    cur_base_chars: list[str] = []
    while i < len(stripped):
        ch = stripped[i]
        if ch == "\x01":
            cur_base_start = sum(len(s) for s in out)
            cur_base_chars = []
        elif ch == "\x02":
            base = "".join(cur_base_chars)
            # We don't know the reading here (sentinels don't carry it).
            # The caller will zip these with the reading list it collected
            # earlier in document order.
            pairs.append({
                "base": base,
                "reading": "",  # filled in by caller
                "start": cur_base_start or 0,
                "end": (cur_base_start or 0) + len(base),
            })
            cur_base_start = None
            cur_base_chars = []
        else:
            if cur_base_start is not None:
                cur_base_chars.append(ch)
            out.append(ch)
        i += 1
    return "".join(out), pairs


SENT_SPLIT_RE = re.compile(r"(?<=[。！？])")


def parse_page(text: str) -> list[tuple[str, list[dict]]]:
    """Return [(sentence, pairs), ...] for ruby-bearing sentences."""
    marked, ruby_pairs = _extract_ruby(text)
    if not ruby_pairs:
        return []
    stripped = strip_wikitext(marked)
    clean, pairs = _resolve_pair_offsets(stripped)
    # Zip readings back in (document order is preserved through stripping).
    if len(pairs) != len(ruby_pairs):
        # Sentinel imbalance; bail rather than misalign.
        return []
    for p, (_, reading, _, _) in zip(pairs, ruby_pairs):
        p["reading"] = reading

    # Split into sentences and rebase pair positions.
    sentences: list[tuple[str, list[dict]]] = []
    cursor = 0
    for sent in SENT_SPLIT_RE.split(clean):
        if not sent:
            continue
        lo = cursor
        hi = cursor + len(sent)
        sent_pairs = []
        for p in pairs:
            if p["start"] >= lo and p["end"] <= hi:
                sent_pairs.append({
                    "base": p["base"],
                    "reading": p["reading"],
                    "start": p["start"] - lo,
                    "end": p["end"] - lo,
                })
        sent_stripped = sent.strip()
        if sent_pairs and sent_stripped:
            # Adjust for left strip.
            lead = len(sent) - len(sent.lstrip())
            sent_pairs = [
                {**p, "start": p["start"] - lead, "end": p["end"] - lead}
                for p in sent_pairs
            ]
            # Filter pairs that became negative (boundary cases).
            sent_pairs = [p for p in sent_pairs
                          if p["start"] >= 0 and p["end"] <= len(sent_stripped)]
            if sent_pairs:
                # Sanity: base text at offset must match.
                ok = all(sent_stripped[p["start"]:p["end"]] == p["base"]
                         for p in sent_pairs)
                if ok:
                    sentences.append((sent_stripped, sent_pairs))
        cursor = hi
    return sentences


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path,
                    default=Path("data/raw/wikipedia.xml.bz2"))
    ap.add_argument("--out", type=Path, default=Path("data/wikipedia.parquet"))
    ap.add_argument("--limit-pages", type=int, default=0,
                    help="for smoke tests; 0 = all")
    args = ap.parse_args()

    if not args.input.exists():
        sys.exit(f"missing {args.input}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(args.out, PA_SCHEMA, compression="zstd")

    n_pages = n_with_ruby = n_sents = n_pairs = 0
    batch: list[dict] = []
    BATCH_SIZE = 5000

    with bz2.open(args.input, "rb") as f:
        for _, elem in ET.iterparse(f, events=("end",)):
            if _t(elem.tag) != "page":
                continue
            n_pages += 1
            try:
                # Filter to namespace 0 (main articles).
                ns_el = elem.find("{*}ns")
                if ns_el is not None and (ns_el.text or "").strip() != "0":
                    continue
                title_el = elem.find("{*}title")
                id_el = elem.find("{*}id")
                rev_el = elem.find("{*}revision")
                if rev_el is None:
                    continue
                text_el = rev_el.find("{*}text")
                if text_el is None or not text_el.text:
                    continue

                title = (title_el.text or "").strip() if title_el is not None else ""
                page_id = (id_el.text or "").strip() if id_el is not None else ""
                wikitext = text_el.text

                # Cheap pre-filter: skip pages without any ruby template.
                if "{{Ruby" not in wikitext and "{{ruby" not in wikitext \
                        and "{{読み" not in wikitext and "<ruby" not in wikitext:
                    continue

                sents = parse_page(wikitext)
                if not sents:
                    continue
                n_with_ruby += 1
                for idx, (txt, pairs) in enumerate(sents):
                    batch.append({
                        "work_id": f"wiki:{page_id}",
                        "author": "wikipedia",
                        "title": title,
                        "sentence_idx": idx,
                        "text": txt,
                        "pairs": pairs,
                    })
                    n_sents += 1
                    n_pairs += len(pairs)
                    if len(batch) >= BATCH_SIZE:
                        writer.write_table(
                            pa.Table.from_pylist(batch, schema=PA_SCHEMA))
                        batch.clear()
            finally:
                elem.clear()

            if n_pages % 50000 == 0:
                print(f"  pages={n_pages} ruby_pages={n_with_ruby} "
                      f"sents={n_sents} pairs={n_pairs}", file=sys.stderr)
            if args.limit_pages and n_pages >= args.limit_pages:
                break

    if batch:
        writer.write_table(pa.Table.from_pylist(batch, schema=PA_SCHEMA))
    writer.close()

    print(f"\ndone. pages={n_pages} ruby_pages={n_with_ruby} "
          f"sents={n_sents} pairs={n_pairs} -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
