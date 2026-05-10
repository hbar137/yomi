"""Evaluate the trained pipeline on data/test.jsonl.

Reports:
    - Per-heteronym accuracy (sorted, with counts).
    - Macro and micro accuracy across heteronyms.
    - Sentence-level character error rate, ours vs UniDic-only baseline.
    - P50 / P95 inference latency on CPU.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
