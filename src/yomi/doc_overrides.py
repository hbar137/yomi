"""Per-document reading-override layer.

At inference, before yomi runs, we scan the input document for canonical
named-entity surfaces (from data/wikipedia_names.json) and register their
readings as overrides. The Pipeline consults this map *first*, ahead of
the heteronym BERT head and MeCab defaults.

Why per-document and not per-corpus:
    The same surface 中田 can be なかた (Hidetoshi Nakata) or なかだ
    (someone else's family) in different documents. Within one document,
    however, all mentions of「中田」are typically the same person, so we
    fix the reading once we've identified the entity (e.g. by matching
    its full form「中田英寿」).

Scoping:
    Wikipedia opener extraction already split full names like「中田 英寿」
    into surname / given-name components, so the overrides map *also*
    contains 中田 -> なかた if it co-occurs with 英寿 -> ひでとし in any
    JaWiki article. No surname dictionary needed at inference time —
    splitting was done at extraction time.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


class WikipediaNames:
    """Loads data/wikipedia_names.json and provides a longest-match scan
    against an input document.

    Same algorithm as Heteronyms.scan: index surfaces by first character,
    sort each bucket by descending length, sweep left-to-right keeping
    longest non-overlapping matches.
    """

    def __init__(self, table: dict[str, str]) -> None:
        self.table = table
        self._by_first: dict[str, list[str]] = defaultdict(list)
        for surface in table:
            if surface:
                self._by_first[surface[0]].append(surface)
        for ch, surfs in self._by_first.items():
            surfs.sort(key=len, reverse=True)

    @classmethod
    def load(cls, path: str | Path) -> "WikipediaNames":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    def __len__(self) -> int:
        return len(self.table)

    def scan(self, text: str) -> list[tuple[int, int, str, str]]:
        """Return [(start, end, surface, reading), ...] for canonical names
        present in `text`, longest non-overlapping match. Result order is
        left-to-right by start position.
        """
        out: list[tuple[int, int, str, str]] = []
        i = 0
        n = len(text)
        while i < n:
            candidates = self._by_first.get(text[i])
            matched = False
            if candidates:
                for surface in candidates:
                    end = i + len(surface)
                    if end <= n and text[i:end] == surface:
                        out.append((i, end, surface, self.table[surface]))
                        i = end
                        matched = True
                        break
            if not matched:
                i += 1
        return out


class DocumentOverrides:
    """Mutable per-document map: surface -> reading.

    Built once at the start of a document (via auto_resolve, optionally
    augmented by caller-supplied hints) and consulted on every span the
    Pipeline classifies as ambiguous.
    """

    def __init__(self) -> None:
        self._map: dict[str, str] = {}

    def auto_resolve(
        self,
        document: str,
        names: WikipediaNames,
    ) -> int:
        """Scan `document` for canonical names from `names` and register
        each (surface -> reading) as an override.

        Within-document conflicts (same surface, two different canonical
        readings found in the document — rare) keep the FIRST occurrence,
        matching the document-order assumption that the entity introduced
        first sets the convention. Returns the number of unique surfaces
        registered.
        """
        before = len(self._map)
        for _, _, surface, reading in names.scan(document):
            self._map.setdefault(surface, reading)
        return len(self._map) - before

    def add(self, surface: str, reading: str) -> None:
        """Caller-supplied hint. Overwrites whatever auto_resolve picked."""
        self._map[surface] = reading

    def get(self, surface: str) -> str | None:
        return self._map.get(surface)

    def __contains__(self, surface: str) -> bool:
        return surface in self._map

    def __len__(self) -> int:
        return len(self._map)
