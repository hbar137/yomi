"""Mine heteronym candidates from corpus parquets.

Reads any *.parquet files passed via --input (default: data/aozora.parquet),
counts every (surface, reading) pair across all ruby annotations, and emits
data/heteronyms.json — the surfaces with multiple distinct readings that are
common enough to be worth training on.

Output schema:
    {
      "surface": {"reading": count, ...},
      ...
    }

Filter knobs (defaults are conservative; tune later when we see the
distribution):
    --min-total-freq    surface must appear ≥N times overall (default 50)
    --min-minority-pct  second-most-frequent reading must be ≥X of total
                        (default 0.02 — i.e. real ambiguity, not OCR noise)

Optional: --merge-yomikata <path> reads a Yomikata-style list (json
mapping surface → [readings]) and unions its surfaces with ours, ensuring
we cover the curated heteronyms even if our threshold drops them.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq


def load_pairs(parquet_paths: list[Path]) -> dict[str, Counter]:
    """Walk parquets, return {surface: Counter(reading -> count)}."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for path in parquet_paths:
        if not path.exists():
            print(f"  skip (missing): {path}", file=sys.stderr)
            continue
        t = pq.read_table(path, columns=["pairs"])
        for row in t["pairs"].to_pylist():
            for p in row:
                base = p["base"]
                reading = p["reading"]
                if base and reading:
                    counts[base][reading] += 1
    return counts


def filter_heteronyms(
    counts: dict[str, Counter],
    min_total_freq: int,
    min_minority_pct: float,
    min_reading_freq: int,
    min_reading_pct: float,
) -> dict[str, dict[str, int]]:
    """Two-stage filter:
        Stage 1: drop surfaces too rare overall, or where the 2nd reading is
                 a tiny fraction (no real ambiguity to learn).
        Stage 2: within each surviving surface, drop readings that fall below
                 BOTH a frequency floor AND a percentage floor. This trims
                 the long tail of okurigana noise and one-off Edo stylistic
                 variants that would otherwise pollute the classification
                 head.
    """
    out: dict[str, dict[str, int]] = {}
    for surface, c in counts.items():
        if len(c) < 2:
            continue
        total = sum(c.values())
        if total < min_total_freq:
            continue
        ranked = c.most_common()
        if ranked[1][1] / total < min_minority_pct:
            continue
        kept = {
            r: cnt for r, cnt in ranked
            if cnt >= min_reading_freq and (cnt / total) >= min_reading_pct
        }
        # Re-check: still ambiguous after pruning?
        if len(kept) < 2:
            continue
        out[surface] = kept
    return out


def merge_reference(
    heteronyms: dict[str, dict[str, int]],
    counts_full: dict[str, Counter],
    ref_path: Path,
    label: str,
    require_observed: bool = True,
    min_obs_new_surface: int = 5,
    min_obs_new_reading: int = 20,
    min_minority_pct: float = 0.01,
) -> dict[str, dict[str, int]]:
    """Add curated surface→[readings] from `ref_path`.

    Two things are gated by corpus support, because the model can only learn
    readings we have training examples for:
        - new surfaces: only added if observed ≥ `min_obs_new_surface` times
          in the corpus.
        - new readings on existing surfaces: only added if observed
          ≥ `min_obs_new_reading` times AND ≥ `min_minority_pct` of the
          surface's total occurrences. This is the same filter applied to
          corpus-derived readings; we don't relax it just because a reference
          dict listed the reading.
    """
    with ref_path.open(encoding="utf-8") as f:
        ref = json.load(f)
    added = unioned = 0
    for surface, readings in ref.items():
        if not isinstance(readings, list) or len(readings) < 2:
            continue
        if surface in heteronyms:
            existing = heteronyms[surface]
            surface_total = sum(counts_full.get(surface, Counter()).values())
            for r in readings:
                if r in existing:
                    continue
                obs = counts_full.get(surface, {}).get(r, 0)
                if obs < min_obs_new_reading:
                    continue
                if surface_total and (obs / surface_total) < min_minority_pct:
                    continue
                existing[r] = obs
                unioned += 1
        else:
            obs_total = sum(counts_full.get(surface, Counter()).values())
            if require_observed and obs_total < min_obs_new_surface:
                continue
            kept = {}
            for r in readings:
                obs = counts_full.get(surface, {}).get(r, 0)
                if obs >= min_obs_new_reading and (
                    not obs_total or obs / obs_total >= min_minority_pct
                ):
                    kept[r] = obs
            if len(kept) >= 2:
                heteronyms[surface] = kept
                added += 1
    print(f"  merged {label}: +{added} surfaces, +{unioned} readings on existing surfaces",
          file=sys.stderr)
    return heteronyms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, action="append",
                    help="parquet file(s) to read; repeatable")
    ap.add_argument("--out", type=Path, default=Path("data/heteronyms.json"))
    ap.add_argument("--min-total-freq", type=int, default=50)
    ap.add_argument("--min-minority-pct", type=float, default=0.02)
    ap.add_argument("--min-reading-freq", type=int, default=20,
                    help="drop readings under N total occurrences")
    ap.add_argument("--min-reading-pct", type=float, default=0.01,
                    help="drop readings under X fraction of surface total")
    ap.add_argument("--merge-reference", type=Path, action="append", default=[],
                    help="curated dict json (surface -> [readings]); repeatable. "
                         "JMdict, Wiktionary, Yomikata are all consumed via this flag")
    ap.add_argument("--ref-min-obs", type=int, default=5,
                    help="when adding a NEW surface from a reference, require it "
                         "to appear at least N times in the corpus")
    args = ap.parse_args()

    parquets = args.input or [Path("data/aozora.parquet")]
    print(f"loading {len(parquets)} parquet(s)...", file=sys.stderr)
    counts = load_pairs(parquets)
    print(f"  {len(counts)} unique surfaces", file=sys.stderr)

    heteronyms = filter_heteronyms(
        counts,
        min_total_freq=args.min_total_freq,
        min_minority_pct=args.min_minority_pct,
        min_reading_freq=args.min_reading_freq,
        min_reading_pct=args.min_reading_pct,
    )
    print(f"  {len(heteronyms)} surfaces pass thresholds "
          f"(min_freq={args.min_total_freq}, min_minority_pct={args.min_minority_pct}, "
          f"min_reading_freq={args.min_reading_freq}, min_reading_pct={args.min_reading_pct})",
          file=sys.stderr)

    for ref_path in args.merge_reference:
        heteronyms = merge_reference(
            heteronyms, counts, ref_path, label=ref_path.stem,
            require_observed=True,
            min_obs_new_surface=args.ref_min_obs,
            min_obs_new_reading=args.min_reading_freq,
            min_minority_pct=args.min_reading_pct,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(heteronyms, f, ensure_ascii=False, indent=2, sort_keys=True)

    # Print a quick summary so we can eyeball the distribution.
    sizes = Counter(len(v) for v in heteronyms.values())
    print(f"\nreading-count distribution: {dict(sorted(sizes.items()))}", file=sys.stderr)
    sample = sorted(heteronyms.items(), key=lambda kv: -sum(kv[1].values()))[:15]
    print("\ntop 15 by total frequency:", file=sys.stderr)
    for surface, readings in sample:
        readings_s = ", ".join(f"{r}:{c}" for r, c in readings.items())
        print(f"  {surface:>6}  {readings_s}", file=sys.stderr)

    print(f"\nwrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
