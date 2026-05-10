# yomi

Japanese text → yomi/furigana service. Hybrid pipeline: MeCab + UniDic for the
unambiguous tokens, a small character-level BERT for heteronym disambiguation.
CPU inference, runs in the standard `eulerai` container layout.

## Layout

```
src/yomi/        runtime code (pipeline, server, training)
scripts/         one-off data prep and eval scripts (numbered in pipeline order)
data/            corpora and processed datasets (gitignored)
models/          ONNX checkpoint(s) (gitignored, baked into image at build)
```

## Pipeline (planned)

1. `01_download_aozora.sh` — pull NDL huriganacorpus-aozora.
2. `02_parse_aozora.py` — TSV → `data/aozora.parquet`.
3. `03_mine_heteronyms.py` → `data/heteronyms.json`.
4. `04_build_examples.py` → `(sentence, span, reading)` rows.
5. `05_split.py` — split by *work*, not by sentence.
6. `06_build_names.py` — JMnedict → MeCab user dictionary.
7. `src/yomi/train.py` — fine-tune `cl-tohoku/bert-base-japanese-char-v3` with
   per-heteronym classification heads. Runs on RunPod.
8. `99_eval.py` — per-heteronym accuracy + sentence CER vs UniDic baseline.

## Build & run

See `DEPLOY.md` for the full deploy procedure.

```sh
docker compose up -d --build
docker compose logs -f --tail 100
```
