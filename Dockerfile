FROM python:3.12-slim

# fugashi wraps libmecab; we also keep mecab-ipadic available as a fallback dictionary.
# build-essential is needed to compile fugashi's C extension; we keep it in the
# image for now (single-stage) and can multi-stage later if image size matters.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libmecab-dev \
        mecab \
        mecab-ipadic-utf8 \
        ca-certificates \
        wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

# Runtime artifacts: heteronym table, Wikipedia name overrides, and the
# JMnedict-derived names CSV. The CSV gets compiled into a MeCab user
# dictionary (names.dic) below; all three files end up under /app/data/.
# .dockerignore whitelists exactly these three paths.
COPY data/heteronyms.json data/wikipedia_names.json data/names.csv data/compounds.csv ./data/

# Compile both user dictionaries against unidic-lite (the dict fugashi uses
# at runtime). mecab-dict-index ships with the `mecab` apt package and
# lives in /usr/lib/mecab/ on Debian (not on PATH).
#   names.dic     — JMnedict proper nouns
#   compounds.dic — corpus-attested compounds UniDic over-segments
RUN UNIDIC_DIR="$(python3 -c 'import unidic_lite, os; print(os.path.dirname(unidic_lite.__file__))')/dicdir" \
    && /usr/lib/mecab/mecab-dict-index \
         -d "$UNIDIC_DIR" -u /app/data/names.dic \
         -f utf-8 -t utf-8 /app/data/names.csv \
    && /usr/lib/mecab/mecab-dict-index \
         -d "$UNIDIC_DIR" -u /app/data/compounds.dic \
         -f utf-8 -t utf-8 /app/data/compounds.csv

# v0 ships without a trained BERT — Pipeline gracefully falls back to the
# most-frequent heteronym reading. Once training lands, drop the model
# directory in here and rebuild:
#     COPY models/v1/best ./models/v1/best

EXPOSE 8080
CMD ["uvicorn", "yomi.server:app", "--host", "0.0.0.0", "--port", "8080"]
