"""Inference pipeline: text -> [Segment(surface, reading, is_kanji)].

Orchestrates the four pieces we built independently:

    1. Heteronyms pre-scan        (yomi.heteronyms.Heteronyms.scan)
    2. Wikipedia name overrides   (yomi.doc_overrides)
    3. MeCab + UniDic + names.dic (fugashi)
    4. Per-surface BERT heads     (yomi.train.YomiBert)

Resolution order for any span/morpheme reading:

    overrides.get(surface)          -- per-document Wikipedia hits win
        ↳ heteronym BERT head       -- ambiguous surfaces in the table
            ↳ MeCab default reading -- everything else

The pre-scan runs first so MeCab never sees a heteronym surface — that
sidesteps UniDic's tendency to over-segment compounds like 日曜日.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from yomi.doc_overrides import DocumentOverrides, WikipediaNames
from yomi.heteronyms import Heteronyms, Span
from yomi.render import Segment

# CJK Unified Ideographs + extension-A + iteration mark + various rare kanji.
_KANJI_RE = re.compile(r"[一-鿿々〆ヵヶ㐀-䶿𠀀-𪛟]")
# Hiragana + katakana + prolonged sound mark; used to disambiguate user-dict
# entries (kana lemma) from native UniDic kanji lemmas.
_KANA_ONLY_RE = re.compile(r"^[ぁ-ゖァ-ヺーー]+$")


def _has_kanji(s: str) -> bool:
    return bool(_KANJI_RE.search(s))


def _kata_to_hira(s: str) -> str:
    out = []
    for ch in s:
        cp = ord(ch)
        if 0x30A1 <= cp <= 0x30F6:
            out.append(chr(cp - 0x60))
        else:
            out.append(ch)
    return "".join(out)


@dataclass
class _Morph:
    """One MeCab morpheme, sliced out of the input string at known offsets."""
    surface: str
    reading_hira: str  # hiragana; equals surface for kana / non-CJK / unknown.


class Pipeline:
    """Hybrid MeCab+UniDic + per-heteronym BERT classifier.

    Construct via `Pipeline.load_default()` for the standard data/ + models/
    layout, or pass components in directly for tests.
    """

    def __init__(
        self,
        heteronyms: Heteronyms,
        names: WikipediaNames | None,
        tagger,                 # fugashi.Tagger
        bert,                   # yomi.train.YomiBert | None  (None = no BERT)
        tokenizer,              # transformers tokenizer | None
        device: str = "cpu",
        max_length: int = 256,
    ) -> None:
        self.heteronyms = heteronyms
        self.names = names
        self.tagger = tagger
        self.bert = bert
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

    # --- construction --------------------------------------------------------

    @classmethod
    def load_default(
        cls,
        data_dir: str | Path = "data",
        model_dir: str | Path = "models/v1/best",
        device: str = "cpu",
    ) -> "Pipeline":
        data_dir = Path(data_dir)
        model_dir = Path(model_dir)

        heteronyms = Heteronyms.load(data_dir / "heteronyms.json")

        names_path = data_dir / "wikipedia_names.json"
        names = WikipediaNames.load(names_path) if names_path.exists() else None

        # MeCab via fugashi. Compiled user-dict (names.dic) is optional; if
        # we haven't built it yet, fall through with the default UniDic.
        import fugashi  # local import so pipeline.py imports cheaply

        # MeCab user dicts: comma-separated paths after a single -u. Order
        # matters only for cost-tied collisions; both are 1288-tagged proper-
        # /common-noun entries, so collisions resolve by `cost` (compounds.dic
        # at 3000 < names.dic at 5000, intentionally — common compounds beat
        # rare proper-noun matches).
        user_dics = [p for p in (data_dir / "names.dic", data_dir / "compounds.dic")
                     if p.exists()]
        tagger_args = f"-u {','.join(str(p) for p in user_dics)}" if user_dics else ""
        tagger = fugashi.Tagger(tagger_args) if tagger_args else fugashi.Tagger()

        # BERT heads are optional: without them we still produce output (just
        # using the most-frequent reading for each heteronym), which is
        # useful for end-to-end smoke tests before training has finished.
        bert = tokenizer = None
        if model_dir.exists() and (model_dir / "heads.pt").exists():
            import torch
            from transformers import AutoTokenizer
            from yomi.train import HeteronymTable, YomiBert

            hetero_for_model = HeteronymTable.load(model_dir / "heteronyms.json")
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir / "bert"),
                                                     use_fast=True)
            bert = YomiBert(str(model_dir / "bert"), hetero_for_model)
            bert.heads.load_state_dict(
                torch.load(model_dir / "heads.pt", map_location=device)
            )
            bert.to(device).eval()

        return cls(heteronyms, names, tagger, bert, tokenizer, device=device)

    # --- inference -----------------------------------------------------------

    def run(self, text: str) -> list[Segment]:
        if not text:
            return []

        # NFKC keeps width-variants from poisoning surface lookup. Done once
        # up front so all downstream offsets refer to the same string.
        text = unicodedata.normalize("NFKC", text)

        overrides = DocumentOverrides()
        if self.names is not None:
            overrides.auto_resolve(text, self.names)

        spans = self.heteronyms.scan(text)
        # Decide which heteronym spans need BERT (those without overrides).
        bert_targets = [
            (i, sp) for i, sp in enumerate(spans)
            if overrides.get(sp.surface) is None
        ]
        bert_readings: dict[int, str] = {}
        if bert_targets and self.bert is not None:
            bert_readings = self._bert_pick(text, bert_targets)

        # Walk text left-to-right, alternating between heteronym spans and
        # MeCab on the gaps.
        segments: list[Segment] = []
        cursor = 0
        for i, sp in enumerate(spans):
            if cursor < sp.start:
                segments.extend(self._mecab(text[cursor:sp.start], overrides))
            reading = (
                overrides.get(sp.surface)
                or bert_readings.get(i)
                or self._heteronym_default(sp.surface)
            )
            segments.append(Segment(
                surface=sp.surface,
                reading=reading,
                is_kanji=_has_kanji(sp.surface),
            ))
            cursor = sp.end
        if cursor < len(text):
            segments.extend(self._mecab(text[cursor:], overrides))
        return segments

    # --- internals -----------------------------------------------------------

    def _heteronym_default(self, surface: str) -> str:
        """Most-frequent reading for a heteronym, used when BERT isn't loaded.

        heteronyms.json was serialised with sort_keys=True, which alphabetised
        the inner reading dicts — so we can't trust insertion order. Pick by
        max count.
        """
        counts = self.heteronyms.table.get(surface, {})
        if not counts:
            return surface
        return max(counts, key=counts.get)

    def _mecab(self, text: str, overrides: DocumentOverrides) -> list[Segment]:
        """MeCab a contiguous slice of text. Returns one Segment per morpheme.

        Reading priority per token:
            1. per-document override (Wikipedia name auto-resolution)
            2. UniDic feature.kana / .pron / .lemma, hiragana-converted

        Single-char heteronyms (~1,500 in heteronyms.json) deliberately
        skip a heteronym-table lookup here: UniDic disambiguates many of
        them based on context (e.g. 日 reads カ in 二十日 but ヒ in 一日中),
        and our blanket "use most-frequent reading" fallback would discard
        that context-aware signal. Without BERT loaded we have nothing
        better than UniDic. Multi-char heteronyms are caught earlier by
        the pre-scan in run(); they don't reach this loop.

        TODO(v1): with BERT loaded, single-char heteronym surfaces should
        route through the per-surface head, taking precedence over
        UniDic's static reading.
        """
        out: list[Segment] = []
        if not text:
            return out
        for word in self.tagger(text):
            surface = word.surface
            if not surface:
                continue
            override = overrides.get(surface)
            if override is not None:
                reading = override
            else:
                reading = self._unidic_reading(word) or surface
            out.append(Segment(
                surface=surface,
                reading=reading,
                is_kanji=_has_kanji(surface),
            ))
        return out

    @staticmethod
    def _unidic_reading(word) -> str | None:
        feat = getattr(word, "feature", None)
        if feat is None:
            return None
        # `kana` is orthographic kana (にちよう); `pron` is pronunciation
        # transcription that uses ー for long vowels (にちよー). Readers
        # expect orthographic, so we try kana first.
        for name in ("kana", "pron", "kanaBase", "pronBase"):
            val = getattr(feat, name, None)
            if val and val != "*":
                return _kata_to_hira(val)
        # User-dict entries (names.csv, compounds.csv) put their reading in
        # the lemma slot because they use IPADIC-style 13-column features
        # rather than UniDic's 17 columns. UniDic native lemmas for kanji
        # words are kanji (e.g. 行く's lemma is 行く), so the kana-only
        # filter cleanly distinguishes the two cases.
        lemma = getattr(feat, "lemma", None)
        if lemma and lemma != "*" and _KANA_ONLY_RE.match(lemma):
            return _kata_to_hira(lemma)
        return None

    def _bert_pick(
        self,
        text: str,
        targets: list[tuple[int, Span]],
    ) -> dict[int, str]:
        """Run YomiBert on every (text, span) pair in `targets` as one batch.

        Returns {span_index: reading_hiragana}. Spans whose surface isn't in
        the model's heteronym table (possible if the model was trained on an
        older table) are silently dropped — caller falls back to the default.
        """
        import torch

        bert = self.bert
        tokenizer = self.tokenizer
        hetero_model = bert.hetero  # train.HeteronymTable

        kept_indices: list[int] = []
        encs = []
        token_starts: list[int] = []
        token_ends: list[int] = []
        surfaces: list[str] = []

        for span_i, sp in targets:
            if sp.surface not in hetero_model.readings:
                continue
            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors=None,
            )
            ts = te = None
            for ti, (lo, hi) in enumerate(enc["offset_mapping"]):
                if lo == hi == 0:
                    continue
                if ts is None and lo <= sp.start < hi:
                    ts = ti
                if lo < sp.end <= hi:
                    te = ti + 1
            if ts is None or te is None or te <= ts:
                # Span got truncated; fall back to default.
                continue
            kept_indices.append(span_i)
            encs.append(enc)
            token_starts.append(ts)
            token_ends.append(te)
            surfaces.append(sp.surface)

        if not kept_indices:
            return {}

        max_len = max(len(e["input_ids"]) for e in encs)
        pad_id = tokenizer.pad_token_id
        input_ids = torch.full((len(encs), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((len(encs), max_len), dtype=torch.long)
        for i, e in enumerate(encs):
            L = len(e["input_ids"])
            input_ids[i, :L] = torch.tensor(e["input_ids"], dtype=torch.long)
            attn[i, :L] = torch.tensor(e["attention_mask"], dtype=torch.long)

        input_ids = input_ids.to(self.device)
        attn = attn.to(self.device)
        ts_t = torch.tensor(token_starts, dtype=torch.long, device=self.device)
        te_t = torch.tensor(token_ends, dtype=torch.long, device=self.device)

        with torch.no_grad():
            out = bert(input_ids, attn, ts_t, te_t, surfaces)

        readings: dict[int, str] = {}
        for i, span_i in enumerate(kept_indices):
            pred_idx = out["preds"][i]
            surface = surfaces[i]
            readings[span_i] = hetero_model.readings[surface][pred_idx]
        return readings
