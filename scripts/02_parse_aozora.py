"""Parse Aozora Bunko text files into data/aozora.parquet.

Reads zips from data/raw/aozora/zips/ + metadata from data/raw/aozora/_index.csv.
For each work:
    1. Decode the .txt inside the .zip (Shift-JIS).
    2. Strip header (everything up to the second '-------' separator line).
    3. Strip footer (everything from the line starting with '底本：').
    4. Strip ［＃...］ and ※［＃...］ editor annotations.
    5. Per non-empty line: split into sentences on 。！？, extract ruby pairs.
    6. Keep only sentences that contain ≥1 ruby pair.

Output schema (one row per ruby-bearing sentence):
    work_id:      str
    author:       str
    title:        str
    sentence_idx: int32   (sequential within a work)
    text:         str     (ruby-stripped — what the model sees at inference)
    pairs:        list<struct{base, reading, start, end}>

Run with:
    python scripts/02_parse_aozora.py [--limit N] [--out PATH]
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# --- regexes --------------------------------------------------------------

# Header is closed by the second line of >=10 hyphens.
SEP_RE = re.compile(r"^-{10,}$")

# Footer markers: "底本：" is the canonical one; some works also use "［＃改ページ］底本".
FOOTER_PREFIXES = ("底本：", "底本:")

# Aozora editor annotations. ※［＃...］ first (because it has a prefix) then ［＃...］.
GAIJI_RE = re.compile(r"※［＃[^］]*］")
ANNOT_RE = re.compile(r"［＃[^］]*］")

# Sentence terminators (Japanese full-width).
SENT_SPLIT_RE = re.compile(r"(?<=[。！？])")

# Kanji range used for ruby base detection (when no ｜ marker is present).
# CJK Unified Ideographs + Extension A + iteration mark + ヶ/ヵ (sometimes used).
KANJI_RE = re.compile(r"[㐀-䶿一-鿿々〆ヵヶ]")


# --- data types -----------------------------------------------------------

@dataclass
class Pair:
    base: str
    reading: str
    start: int
    end: int


# --- ruby parser ----------------------------------------------------------

def is_kanji(ch: str) -> bool:
    return bool(KANJI_RE.match(ch))


def parse_ruby(sentence: str) -> tuple[str, list[Pair]]:
    """Strip ruby markup from a sentence and return (clean_text, ruby_pairs).

    Handles the two Aozora ruby forms:
        漢字《かな》          -- base = trailing kanji run before 《
        ｜任意《かな》        -- base = run between ｜ and 《 (any chars)
    """
    out: list[str] = []
    pairs: list[Pair] = []
    out_len = 0
    i = 0
    n = len(sentence)
    while i < n:
        j = sentence.find("《", i)
        if j == -1:
            out.append(sentence[i:])
            break
        k = sentence.find("》", j)
        if k == -1:
            # Unclosed ruby — bail and keep the rest verbatim.
            out.append(sentence[i:])
            break
        reading = sentence[j + 1:k]

        # Look for a ｜ (full-width pipe) between i and j to delimit the base.
        sep = sentence.rfind("｜", i, j)
        if sep != -1:
            # Text from i..sep is non-ruby prefix.
            prefix = sentence[i:sep]
            base = sentence[sep + 1:j]
            out.append(prefix)
            out_len += len(prefix)
            base_start = out_len
            out.append(base)
            out_len += len(base)
            pairs.append(Pair(base=base, reading=reading,
                              start=base_start, end=base_start + len(base)))
            i = k + 1
            continue

        # No ｜: walk back from j over consecutive kanji to find the base.
        b = j
        while b > i and is_kanji(sentence[b - 1]):
            b -= 1
        prefix = sentence[i:b]
        base = sentence[b:j]
        if not base:
            # Orphan 《》 with no base. Drop the markup, drop the reading.
            out.append(prefix)
            out_len += len(prefix)
            i = k + 1
            continue
        out.append(prefix)
        out_len += len(prefix)
        base_start = out_len
        out.append(base)
        out_len += len(base)
        pairs.append(Pair(base=base, reading=reading,
                          start=base_start, end=base_start + len(base)))
        i = k + 1

    return "".join(out), pairs


# --- file-level parsing ---------------------------------------------------

def find_body_range(lines: list[str]) -> tuple[int, int]:
    """Return (body_start, body_end) line indices.

    body_start: first content line after the second separator (or 0 if absent).
    body_end:   first line that begins the footer (or len(lines)).
    """
    seps = [i for i, line in enumerate(lines) if SEP_RE.match(line.strip())]
    body_start = seps[1] + 1 if len(seps) >= 2 else 0

    body_end = len(lines)
    for i in range(body_start, len(lines)):
        s = lines[i].lstrip()
        if any(s.startswith(p) for p in FOOTER_PREFIXES):
            body_end = i
            break
    return body_start, body_end


def parse_work(raw: str) -> list[tuple[str, list[Pair]]]:
    """Yield (sentence_text, pairs) for every ruby-bearing sentence in the work."""
    lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    start, end = find_body_range(lines)
    out: list[tuple[str, list[Pair]]] = []
    for line in lines[start:end]:
        line = GAIJI_RE.sub("", line)
        line = ANNOT_RE.sub("", line)
        line = line.strip()
        if not line:
            continue
        # Split into sentences but keep terminator with the preceding sentence.
        for sent in SENT_SPLIT_RE.split(line):
            sent = sent.strip()
            if not sent:
                continue
            text, pairs = parse_ruby(sent)
            if pairs:
                out.append((text, pairs))
    return out


# --- driver ---------------------------------------------------------------

def load_metadata(index_csv: Path) -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    with index_csv.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by_id[r["作品ID"]] = r
    return by_id


def open_text_from_zip(zip_path: Path) -> str | None:
    """Open the (only) .txt inside a Aozora work zip, decode Shift-JIS."""
    try:
        with zipfile.ZipFile(zip_path) as z:
            txt_names = [n for n in z.namelist() if n.lower().endswith(".txt")]
            if not txt_names:
                return None
            with z.open(txt_names[0]) as f:
                return f.read().decode("shift_jis", errors="replace")
    except (zipfile.BadZipFile, OSError):
        return None


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw/aozora"))
    ap.add_argument("--out", type=Path, default=Path("data/aozora.parquet"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    index_csv = args.raw_dir / "_index.csv"
    zips_dir = args.raw_dir / "zips"
    if not index_csv.exists():
        sys.exit(f"missing {index_csv}; run 01_download_aozora.py first")

    metadata = load_metadata(index_csv)
    # Iterate over what's actually been downloaded, not what's in the index;
    # this lets the parser run even when the download is partial.
    work_ids = sorted(p.stem for p in zips_dir.glob("*.zip"))
    if args.limit:
        work_ids = work_ids[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(args.out, PA_SCHEMA, compression="zstd")

    n_works = n_sentences = n_pairs = n_skipped = 0
    batch: list[dict] = []
    BATCH_SIZE = 5000

    for wid in work_ids:
        meta = metadata[wid]
        zip_path = zips_dir / f"{wid}.zip"
        if not zip_path.exists():
            n_skipped += 1
            continue
        raw = open_text_from_zip(zip_path)
        if raw is None:
            n_skipped += 1
            continue

        author = (meta.get("姓") or "") + (meta.get("名") or "")
        title = meta.get("作品名") or ""

        for idx, (text, pairs) in enumerate(parse_work(raw)):
            batch.append({
                "work_id": wid,
                "author": author,
                "title": title,
                "sentence_idx": idx,
                "text": text,
                "pairs": [
                    {"base": p.base, "reading": p.reading, "start": p.start, "end": p.end}
                    for p in pairs
                ],
            })
            n_sentences += 1
            n_pairs += len(pairs)

            if len(batch) >= BATCH_SIZE:
                writer.write_table(pa.Table.from_pylist(batch, schema=PA_SCHEMA))
                batch.clear()

        n_works += 1
        if n_works % 200 == 0:
            print(f"[{n_works}/{len(work_ids)}] sentences={n_sentences} pairs={n_pairs}",
                  file=sys.stderr)

    if batch:
        writer.write_table(pa.Table.from_pylist(batch, schema=PA_SCHEMA))
    writer.close()

    print(f"done. works_parsed={n_works} skipped={n_skipped} "
          f"sentences={n_sentences} pairs={n_pairs} -> {args.out}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
