"""Build training examples from corpus parquets + heteronyms.json.

For every ruby pair whose (surface, reading) is in the heteronym table, emit
one training example with the sentence pre-tokenized:

    {
      "sentence":    "親戚や友人の邸に行って...",
      "span_start":  0,
      "span_end":    2,
      "surface":     "親戚",
      "reading":     "しんせき",
      "work_id":     "060224",
      "source":      "aozora",
      "input_ids":   [2, 12345, 67890, ..., 3],
      "token_start": 1,
      "token_end":   3,
      "valid":       true
    }

Pre-tokenization detail: cl-tohoku/bert-base-japanese-char-v3 is char-level
(1 char -> 1 token), so token_start/token_end map directly from
char-position + 1 (after [CLS]). Computing once at dataset-build time
moves the slow Python tokenizer out of the dataloader hot path —
observed ~5x speedup expected on the GPU side.

If a span is truncated out by --max-length we keep the example with
valid=false so dataset shape stays uniform; train.py masks it from the
loss.

Per-heteronym capping: per surface, cap each reading at --max-per-reading
examples (default 2000). This keeps high-frequency forms from dominating.
Pairs whose reading is NOT in heteronyms[surface] are dropped.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

PARSED_SOURCES = {
    "aozora.parquet": "aozora",
    "nhk_easy.parquet": "nhk_easy",
    "wikipedia.parquet": "wikipedia",
    "wiktionary.parquet": "wiktionary",
}


def source_name(parquet_path: Path) -> str:
    return PARSED_SOURCES.get(parquet_path.name, parquet_path.stem)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, action="append",
                    help="parquet file(s); repeatable. default: data/aozora.parquet")
    ap.add_argument("--heteronyms", type=Path, default=Path("data/heteronyms.json"))
    ap.add_argument("--out", type=Path, default=Path("data/examples.jsonl"))
    ap.add_argument("--max-per-reading", type=int, default=2000,
                    help="cap on examples per (surface, reading) pair")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokenizer", default="cl-tohoku/bert-base-japanese-char-v3",
                    help="HF tokenizer name; used to pre-tokenize sentences")
    ap.add_argument("--max-length", type=int, default=128,
                    help="truncation length applied at tokenize time")
    args = ap.parse_args()

    parquets = args.input or [Path("data/aozora.parquet")]
    with args.heteronyms.open(encoding="utf-8") as f:
        heteronyms: dict[str, dict[str, int]] = json.load(f)
    print(f"loaded {len(heteronyms)} heteronyms", file=sys.stderr)

    rng = random.Random(args.seed)

    # First pass: collect all candidate examples in memory.
    # Memory cost is roughly N_examples × ~200 bytes; for ~5M examples
    # that's ~1 GB which is fine on the host. If it ever becomes a problem
    # we can switch to a two-pass disk-spill version.
    candidates: list[dict] = []
    n_pairs_seen = n_kept = n_dropped_unknown_reading = 0

    for path in parquets:
        if not path.exists():
            print(f"  skip (missing): {path}", file=sys.stderr)
            continue
        src = source_name(path)
        t = pq.read_table(path, columns=["work_id", "text", "pairs"])
        rows = t.to_pylist()
        for row in rows:
            for p in row["pairs"]:
                n_pairs_seen += 1
                surface = p["base"]
                reading = p["reading"]
                if surface not in heteronyms:
                    continue
                if reading not in heteronyms[surface]:
                    n_dropped_unknown_reading += 1
                    continue
                candidates.append({
                    "sentence": row["text"],
                    "span_start": p["start"],
                    "span_end": p["end"],
                    "surface": surface,
                    "reading": reading,
                    "work_id": row["work_id"],
                    "source": src,
                })
                n_kept += 1

    print(f"  pairs seen: {n_pairs_seen}", file=sys.stderr)
    print(f"  pairs in heteronym table: {n_kept + n_dropped_unknown_reading}",
          file=sys.stderr)
    print(f"    -> usable (reading in candidates): {n_kept}", file=sys.stderr)
    print(f"    -> dropped (reading not in candidates): {n_dropped_unknown_reading}",
          file=sys.stderr)

    # Per (surface, reading) cap.
    rng.shuffle(candidates)
    by_pair: dict[tuple[str, str], int] = defaultdict(int)
    output: list[dict] = []
    for ex in candidates:
        key = (ex["surface"], ex["reading"])
        if by_pair[key] >= args.max_per_reading:
            continue
        by_pair[key] += 1
        output.append(ex)

    print(f"  after per-(surface,reading) cap of {args.max_per_reading}: "
          f"{len(output)} examples", file=sys.stderr)

    # Pre-tokenize. Slow Python tokenizer is the dataloader bottleneck at
    # train time; doing it once here amortizes across every epoch and run.
    print(f"\ntokenizing with {args.tokenizer} (max_length={args.max_length})...",
          file=sys.stderr)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False)
    n_special_leading = 1  # [CLS]
    n_truncated_span = 0
    for i, ex in enumerate(output):
        enc = tokenizer(
            ex["sentence"], truncation=True, max_length=args.max_length,
            return_tensors=None,
        )
        ex["input_ids"] = enc["input_ids"]
        # Char-level tokenizer: token at index i corresponds to char at
        # position (i - n_special_leading). Special tokens [CLS] at 0,
        # [SEP] at end.
        n_tokens = len(enc["input_ids"])
        token_start = token_end = None
        for ti in range(n_special_leading, n_tokens - 1):
            char_pos = ti - n_special_leading
            if token_start is None and char_pos == ex["span_start"]:
                token_start = ti
            if char_pos + 1 == ex["span_end"]:
                token_end = ti + 1
                break
        if token_start is None or token_end is None or token_end <= token_start:
            n_truncated_span += 1
            ex["token_start"] = 1   # dummy (any non-special index works since masked)
            ex["token_end"] = 2
            ex["valid"] = False
        else:
            ex["token_start"] = token_start
            ex["token_end"] = token_end
            ex["valid"] = True
        if (i + 1) % 100000 == 0:
            print(f"  tokenized {i+1}/{len(output)}", file=sys.stderr)
    print(f"  truncated spans (kept with valid=false): {n_truncated_span}",
          file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for ex in output:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"\nwrote {args.out}", file=sys.stderr)

    # Per-surface support summary.
    by_surface: Counter = Counter(ex["surface"] for ex in output)
    print(f"\nper-surface example counts (top 10 / bottom 10):", file=sys.stderr)
    sorted_s = by_surface.most_common()
    for s, c in sorted_s[:10]:
        print(f"  {s:>6}  {c}", file=sys.stderr)
    print("   ...", file=sys.stderr)
    for s, c in sorted_s[-10:]:
        print(f"  {s:>6}  {c}", file=sys.stderr)


if __name__ == "__main__":
    main()
