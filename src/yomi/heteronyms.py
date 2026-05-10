"""Heteronym table + longest-match pre-scan for inference.

Defines `Heteronyms`, the data structure that backs both training (to size
classification heads) and inference (to find ambiguous spans before MeCab
ever sees the text).

The pre-scan exists because MeCab can split a heteronym surface into
multiple tokens (e.g. 日曜日 → 日曜 / 日). If MeCab does that, we never
invoke the head for 日曜日 and yomi is wrong. The fix is to find every
heteronym surface in the input *first*, mark those spans for BERT, and
only run MeCab on the gaps between them.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Span:
    """A region of the input string flagged as a heteronym surface."""
    start: int
    end: int
    surface: str


class Heteronyms:
    """Mapping: surface -> {reading: count}.

    Used at training time to size per-surface classification heads, and at
    inference time to (1) decide whether a span is ambiguous and needs BERT,
    and (2) pre-scan input strings to find heteronym surfaces before MeCab
    is invoked.
    """

    def __init__(self, table: dict[str, dict[str, int]]) -> None:
        self.table = table
        # Index surfaces by their first character, with longest first. This
        # gives O(N × max_candidates_for_a_char) scan time, which for ~2,400
        # surfaces averaging 2-4 chars is effectively O(N).
        self._by_first: dict[str, list[str]] = defaultdict(list)
        for surface in table:
            if surface:
                self._by_first[surface[0]].append(surface)
        for ch, surfs in self._by_first.items():
            surfs.sort(key=len, reverse=True)

    @classmethod
    def load(cls, path: str | Path) -> "Heteronyms":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    def is_ambiguous(self, surface: str) -> bool:
        return surface in self.table

    def candidates(self, surface: str) -> list[str]:
        """Readings for `surface`, in their stored order."""
        return list(self.table.get(surface, {}).keys())

    def reading_index(self, surface: str, reading: str) -> int:
        """Position of `reading` in surface's reading list (training label).

        Raises KeyError if either the surface or the reading is unknown."""
        readings = list(self.table[surface].keys())
        return readings.index(reading)

    def scan(self, text: str, min_len: int = 2) -> list[Span]:
        """Find non-overlapping heteronym surfaces in `text`, longest first.

        `min_len` defaults to 2: single-char heteronyms (1,500+ in our table,
        e.g. 中, 英, 日) are NOT pre-scanned, because they over-aggressively
        carve out characters mid-compound (e.g. fragmenting 中田英寿 into
        中/田/英/寿 before MeCab gets a chance to recognise it as one name).
        Single-char heteronyms are instead disambiguated post-tokenization
        in Pipeline._mecab. Multi-char heteronyms still need pre-scan
        because UniDic over-segments ~8% of them (一日, 何時, 一個, …).
        """
        spans: list[Span] = []
        i = 0
        n = len(text)
        while i < n:
            candidates = self._by_first.get(text[i])
            matched = False
            if candidates:
                for surface in candidates:
                    if len(surface) < min_len:
                        # _by_first is sorted by descending length; once we
                        # see a too-short surface, all remaining are shorter.
                        break
                    end = i + len(surface)
                    if end <= n and text[i:end] == surface:
                        spans.append(Span(i, end, surface))
                        i = end
                        matched = True
                        break
            if not matched:
                i += 1
        return spans

    def __len__(self) -> int:
        return len(self.table)

    def __contains__(self, surface: str) -> bool:
        return surface in self.table
