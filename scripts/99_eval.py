"""Evaluate Pipeline.run accuracy on data/test.jsonl.

Reads each (sentence, surface, span, reading) example, runs the pipeline,
finds the predicted segment at that span, and compares to the labeled
reading. Reports:

    - Overall accuracy (micro and macro across surfaces)
    - Per-source accuracy (aozora / wikipedia / nhk_easy)
    - Top-20 per-surface failures (worst accuracy × volume)
    - P50 / P95 inference latency on CPU

Designed to work in two modes:
    1. v0 baseline — Pipeline runs without BERT (most-frequent fallback).
       Just point at a built inference image with no model checkpoint.
    2. Trained — point --model-dir at models/v1/best/.

Usage:
    python scripts/99_eval.py                       # uses defaults
    python scripts/99_eval.py --model-dir models/v1/best
    python scripts/99_eval.py --limit 10000         # quick subset
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

# Pipeline lives in src/yomi; respect "import yomi.pipeline" once installed,
# but also support running this script straight from the repo without
# `pip install -e`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from yomi.pipeline import Pipeline
from yomi.render import Segment


def load_examples(path: Path, limit: int = 0) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


def find_segment_at_span(
    segments: list[Segment],
    span_start: int,
    surface: str,
) -> Segment | None:
    """Walk segments left-to-right, accumulating cursor position. Return
    the segment whose surface matches and whose start aligns with the
    labeled span_start.

    Pipeline.run NFKC-normalizes input; for test sentences that happen to
    contain halfwidth/fullwidth variants the cursor positions can drift
    relative to the original. As a fallback, if positional match fails we
    take the first segment whose surface equals the labeled surface (this
    very slightly overcounts accuracy on sentences with multiple instances
    of the same surface, but those are rare in the heteronym test set).
    """
    cursor = 0
    for seg in segments:
        if cursor == span_start and seg.surface == surface:
            return seg
        cursor += len(seg.surface)
    # fallback: first surface match
    for seg in segments:
        if seg.surface == surface:
            return seg
    return None


def eval_pipeline(
    pipeline: Pipeline,
    examples: list[dict],
    warmup: int = 5,
) -> dict:
    """Run pipeline on every example, collect metrics. Returns a dict
    with overall, per-source, and per-surface stats."""
    # Warmup so latency P50 isn't dominated by lazy-import / cache load.
    for ex in examples[:warmup]:
        pipeline.run(ex["sentence"])

    n_total = n_correct = n_missing = 0
    per_source: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [correct, total]
    per_surface: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    latencies: list[float] = []

    for i, ex in enumerate(examples):
        t0 = time.perf_counter()
        segments = pipeline.run(ex["sentence"])
        latencies.append((time.perf_counter() - t0) * 1000.0)

        seg = find_segment_at_span(segments, ex["span_start"], ex["surface"])
        n_total += 1
        per_source[ex.get("source", "unknown")][1] += 1
        per_surface[ex["surface"]][1] += 1

        if seg is None:
            n_missing += 1
            continue
        if seg.reading == ex["reading"]:
            n_correct += 1
            per_source[ex.get("source", "unknown")][0] += 1
            per_surface[ex["surface"]][0] += 1

        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{len(examples)}  acc={n_correct/n_total:.4f}",
                  file=sys.stderr)

    macro_accs = [c / t for c, t in per_surface.values() if t > 0]
    return {
        "n_total": n_total,
        "n_correct": n_correct,
        "n_missing": n_missing,
        "micro_acc": n_correct / max(n_total, 1),
        "macro_acc": sum(macro_accs) / max(len(macro_accs), 1),
        "per_source": dict(per_source),
        "per_surface": dict(per_surface),
        "latencies_ms": latencies,
    }


def fmt_latency(latencies: list[float]) -> str:
    if not latencies:
        return "n/a"
    p50 = statistics.median(latencies)
    p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) > 20 else max(latencies)
    return f"P50={p50:.1f}ms P95={p95:.1f}ms (n={len(latencies)})"


def report(metrics: dict, top_failures: int = 20) -> None:
    print(f"\n=== overall ===")
    print(f"  examples:    {metrics['n_total']:,}")
    print(f"  correct:     {metrics['n_correct']:,}")
    print(f"  missing seg: {metrics['n_missing']:,}  "
          f"(pipeline didn't produce a segment at the labeled span)")
    print(f"  micro acc:   {metrics['micro_acc']:.4f}")
    print(f"  macro acc:   {metrics['macro_acc']:.4f}  "
          f"(equal weight per heteronym)")
    print(f"  latency:     {fmt_latency(metrics['latencies_ms'])}")

    print(f"\n=== per source ===")
    for src, (c, t) in sorted(metrics["per_source"].items(), key=lambda kv: -kv[1][1]):
        acc = c / t if t else 0.0
        print(f"  {src:>14}  n={t:>7,}  acc={acc:.4f}")

    print(f"\n=== top {top_failures} per-surface failures (sorted by impact) ===")
    rows = [
        (s, c, t, c / t if t else 0.0)
        for s, (c, t) in metrics["per_surface"].items()
    ]
    # Sort by miss count descending — what's hurting us most.
    rows.sort(key=lambda r: -(r[2] - r[1]))
    print(f"  {'surface':<8}  {'miss':>6}  {'n':>6}  acc")
    for s, c, t, acc in rows[:top_failures]:
        miss = t - c
        if miss == 0:
            break
        print(f"  {s:<8}  {miss:>6,}  {t:>6,}  {acc:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--split", default="test", choices=("val", "test"))
    ap.add_argument("--model-dir", type=Path, default=Path("models/v1/best"),
                    help="trained checkpoint dir; if missing, runs the v0 "
                         "baseline (no BERT, most-frequent fallback)")
    ap.add_argument("--limit", type=int, default=0,
                    help="max examples to evaluate (0 = all)")
    ap.add_argument("--top-failures", type=int, default=20)
    args = ap.parse_args()

    print(f"loading pipeline (data={args.data_dir}, model={args.model_dir})...",
          file=sys.stderr)
    pipeline = Pipeline.load_default(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
    )
    if pipeline.bert is None:
        print("  (BERT not loaded — running v0 baseline)", file=sys.stderr)

    eval_path = args.data_dir / f"{args.split}.jsonl"
    print(f"loading {eval_path}...", file=sys.stderr)
    examples = load_examples(eval_path, args.limit)
    print(f"  {len(examples):,} examples", file=sys.stderr)

    print(f"\nevaluating...", file=sys.stderr)
    t0 = time.time()
    metrics = eval_pipeline(pipeline, examples)
    elapsed = time.time() - t0
    print(f"\ndone in {elapsed:.1f}s ({len(examples)/elapsed:.1f} ex/s)",
          file=sys.stderr)

    report(metrics, top_failures=args.top_failures)


if __name__ == "__main__":
    main()
