"""Parse Wayback-archived NHK News Web Easy HTML snapshots into a parquet
matching the Aozora schema.

Input:   data/raw/nhk_easy/snapshots/*.html
Output:  data/nhk_easy.parquet  (columns: work_id, author, title,
                                  sentence_idx, text, pairs)

For each snapshot we:
    1. Locate the article body. Across the years NHK has used several
       containers, all detectable by id; we try them in order.
    2. Drop everything that's not the article body — headers, related-news
       boxes, footers, IA toolbars (we use `id_/...` raw URLs so toolbars
       are usually absent, but be defensive).
    3. Linearize the body, preserving ruby pairs and sentence boundaries.
    4. Emit one row per ruby-bearing sentence.
"""

from __future__ import annotations

import argparse
import re
import sys
from html.parser import HTMLParser
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

ARTICLE_BODY_IDS = ("js-article-body", "newsarticle", "main")
ARTICLE_TITLE_IDS = ("js-article-title", "newstitle")
SENT_SPLIT_RE = re.compile(r"(?<=[。！？])")


class _RubyExtractor(HTMLParser):
    """Walk a fragment of HTML keeping a flat (text, pairs) representation.

    Logic:
        - Outside <ruby>: append text data to `out`.
        - Inside <ruby>: append the *base* (text outside <rt>/<rp>) to `out`,
          collect the *reading* from inside <rt>, drop <rp> entirely, and on
          </ruby> emit one Pair using positions relative to `out`.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.out_len = 0
        self.pairs: list[dict] = []

        self._in_ruby = False
        self._rb_chunks: list[str] = []   # base accumulator
        self._rt_chunks: list[str] = []   # reading accumulator
        self._in_rt = False
        self._in_rp = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag == "ruby":
            self._in_ruby = True
            self._rb_chunks.clear()
            self._rt_chunks.clear()
        elif tag == "rt":
            self._in_rt = True
        elif tag == "rp":
            self._in_rp = True
        elif tag in ("br", "p"):
            # Treat as a sentence-friendly break by adding a newline.
            if not self._in_ruby:
                self.out.append("\n")
                self.out_len += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "ruby" and self._in_ruby:
            base = "".join(self._rb_chunks).strip()
            reading = "".join(self._rt_chunks).strip()
            if base and reading:
                start = self.out_len
                self.out.append(base)
                self.out_len += len(base)
                self.pairs.append({
                    "base": base, "reading": reading,
                    "start": start, "end": start + len(base),
                })
            self._in_ruby = False
            self._rb_chunks.clear()
            self._rt_chunks.clear()
        elif tag == "rt":
            self._in_rt = False
        elif tag == "rp":
            self._in_rp = False
        elif tag in ("p", "div", "li"):
            # Block-level close → encourage sentence boundary.
            if not self._in_ruby:
                self.out.append("\n")
                self.out_len += 1

    def handle_data(self, data: str) -> None:
        if self._in_rp:
            return  # rp wraps fallback parens like 「（」「）」 — drop
        if self._in_ruby:
            if self._in_rt:
                self._rt_chunks.append(data)
            else:
                self._rb_chunks.append(data)
        else:
            self.out.append(data)
            self.out_len += len(data)


def _slice_pairs(pairs: list[dict], lo: int, hi: int) -> list[dict]:
    """Return pairs whose [start, end] lies within [lo, hi), with positions
    rebased to `lo`."""
    return [
        {
            "base": p["base"],
            "reading": p["reading"],
            "start": p["start"] - lo,
            "end": p["end"] - lo,
        }
        for p in pairs
        if p["start"] >= lo and p["end"] <= hi
    ]


def _find_article_body(html: str) -> str:
    """Return the article-body HTML fragment, falling back to the whole doc."""
    for body_id in ARTICLE_BODY_IDS:
        m = re.search(
            rf'<div[^>]*id=["\']{re.escape(body_id)}["\'][^>]*>(.*?)</div>',
            html, re.DOTALL,
        )
        if m:
            return m.group(1)
    # Fallback: <article> tag.
    m = re.search(r"<article[^>]*>(.*?)</article>", html, re.DOTALL)
    if m:
        return m.group(1)
    return html


def _find_title(html: str) -> str:
    for title_id in ARTICLE_TITLE_IDS:
        m = re.search(
            rf'<[^>]*id=["\']{re.escape(title_id)}["\'][^>]*>(.*?)</',
            html, re.DOTALL,
        )
        if m:
            # Strip any inner ruby markup for the title — use the base text.
            ext = _RubyExtractor()
            ext.feed(m.group(1))
            return "".join(ext.out).strip()
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def parse_snapshot(html: str) -> tuple[str, list[tuple[str, list[dict]]]]:
    """Return (title, [(sentence_text, pairs), ...])"""
    title = _find_title(html)
    body = _find_article_body(html)
    ext = _RubyExtractor()
    ext.feed(body)
    full_text = "".join(ext.out)

    sentences: list[tuple[str, list[dict]]] = []
    cursor = 0
    sent_idx = 0
    for chunk in re.split(r"\n+", full_text):
        chunk = chunk.strip()
        if not chunk:
            cursor += 1   # account for the newline we collapsed
            continue
        # Locate this chunk in the original linear text so we can slice pairs.
        # Walk forward from `cursor` to skip whitespace/newlines.
        start = full_text.find(chunk, cursor)
        if start < 0:
            cursor += len(chunk)
            continue
        end = start + len(chunk)
        # Within a chunk, split on sentence terminators.
        offset = 0
        for sent in SENT_SPLIT_RE.split(chunk):
            sent_stripped = sent.strip()
            if not sent_stripped:
                offset += len(sent)
                continue
            lead = len(sent) - len(sent.lstrip())
            sent_lo = start + offset + lead
            sent_hi = sent_lo + len(sent_stripped)
            pairs = _slice_pairs(ext.pairs, sent_lo, sent_hi)
            if pairs:
                sentences.append((sent_stripped, pairs))
                sent_idx += 1
            offset += len(sent)
        cursor = end
    return title, sentences


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path,
                    default=Path("data/raw/nhk_easy/snapshots"))
    ap.add_argument("--out", type=Path, default=Path("data/nhk_easy.parquet"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    files = sorted(args.in_dir.glob("*.html"))
    if args.limit:
        files = files[: args.limit]
    print(f"parsing {len(files)} snapshots", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(args.out, PA_SCHEMA, compression="zstd")

    n_ok = n_skip = n_pairs = n_sents = 0
    batch: list[dict] = []
    BATCH_SIZE = 5000

    for path in files:
        try:
            html = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            n_skip += 1
            continue
        try:
            title, sents = parse_snapshot(html)
        except Exception as e:
            print(f"  parse fail {path.name}: {e!r}", file=sys.stderr)
            n_skip += 1
            continue
        work_id = path.stem
        for idx, (text, pairs) in enumerate(sents):
            batch.append({
                "work_id": work_id,
                "author": "NHK",
                "title": title,
                "sentence_idx": idx,
                "text": text,
                "pairs": pairs,
            })
            n_sents += 1
            n_pairs += len(pairs)
            if len(batch) >= BATCH_SIZE:
                writer.write_table(pa.Table.from_pylist(batch, schema=PA_SCHEMA))
                batch.clear()
        n_ok += 1
        if n_ok % 500 == 0:
            print(f"  [{n_ok}/{len(files)}] sents={n_sents} pairs={n_pairs}",
                  file=sys.stderr)

    if batch:
        writer.write_table(pa.Table.from_pylist(batch, schema=PA_SCHEMA))
    writer.close()

    print(f"\ndone. parsed={n_ok} skipped={n_skip} sentences={n_sents} pairs={n_pairs}"
          f" -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
