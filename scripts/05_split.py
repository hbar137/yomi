"""Split data/examples.jsonl into train/val/test by **work_id**.

Splitting by work prevents leakage: sentences from the same work share author
voice, vocabulary, era, and editor decisions about which kanji to ruby.
Splitting by sentence would let train and test see the same work and
overstate generalization.

Default split: 70/15/15. Stratification: we don't currently stratify by
author (Aozora has long-tail prolific authors and lots of singletons; trying
to stratify makes the split brittle without a clear win). Re-evaluate if eval
shows author-specific generalization gaps.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data/examples.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("data"))
    ap.add_argument("--train-pct", type=float, default=0.70)
    ap.add_argument("--val-pct", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    test_pct = 1.0 - args.train_pct - args.val_pct
    if test_pct <= 0:
        sys.exit("train+val must be < 1.0")

    # Group examples by work_id.
    by_work: dict[str, list[dict]] = defaultdict(list)
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            by_work[ex["work_id"]].append(ex)
    works = sorted(by_work.keys())
    print(f"loaded {sum(len(v) for v in by_work.values())} examples "
          f"from {len(works)} works", file=sys.stderr)

    rng = random.Random(args.seed)
    rng.shuffle(works)
    n = len(works)
    n_train = int(n * args.train_pct)
    n_val = int(n * args.val_pct)
    train_w = set(works[:n_train])
    val_w = set(works[n_train:n_train + n_val])
    test_w = set(works[n_train + n_val:])

    splits = {"train": train_w, "val": val_w, "test": test_w}
    for name, ws in splits.items():
        out = args.out_dir / f"{name}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            n_ex = 0
            for w in ws:
                for ex in by_work[w]:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    n_ex += 1
        print(f"  {name}: {len(ws)} works, {n_ex} examples -> {out}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
