"""Fetch archived NHK News Web Easy article HTML from the Internet Archive.

The live NHK Easy site is gated behind app-only JWT auth (verified 2026-05),
so we go through Wayback. Strategy:

    1. One bulk Wayback CDX query for `www3.nhk.or.jp/news/easy/*` → all known
       archived URLs. Saved to data/raw/nhk_easy/_cdx.json so we don't repeat.
    2. Filter that index to article-shaped URLs only (skip listing pages, query
       strings, redirects).
    3. For each, fetch the raw snapshot HTML via
       `web.archive.org/web/<ts>id_/<url>` and write to
       data/raw/nhk_easy/snapshots/<filename>.html.

Idempotent: skips snapshots already on disk.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

CDX_URL = "https://web.archive.org/cdx/search/cdx"
WAYBACK_RAW_TPL = "https://web.archive.org/web/{ts}id_/{url}"
USER_AGENT = "yomi-research/0.1 (Japanese furigana model training)"
CDX_TIMEOUT = 300.0   # bulk CDX is slow
FETCH_TIMEOUT = 30.0
PER_REQUEST_DELAY_S = 1.0

# Article URLs look like:  /news/easy/k10012323711000/k10012323711000.html
#                          /news/easy/ne2025042512518/ne2025042512518.html
#                          /news/easy/article/disaster_xxx.html
#                          /news/easy/20120515_k10015143231000.html
ARTICLE_RE = re.compile(
    r"/news/easy/(?:article/[A-Za-z0-9_]+|"
    r"\d+_[A-Za-z0-9]+|"
    r"[A-Za-z0-9]+/[A-Za-z0-9]+)\.html$"
)


def http_get(url: str, timeout: float, accept_404: bool = False) -> bytes | None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if accept_404 and e.code == 404:
            return None
        raise


def fetch_cdx_index(out_dir: Path) -> list[tuple[str, str]]:
    """Return [(url, earliest_timestamp), ...]. Cached on disk after first call."""
    cache = out_dir / "_cdx.json"
    if cache.exists() and cache.stat().st_size > 0:
        with cache.open(encoding="utf-8") as f:
            return [tuple(r) for r in json.load(f)]

    print("running bulk Wayback CDX query (slow, ~60s)...", file=sys.stderr)
    q = urllib.parse.urlencode({
        "url": "www3.nhk.or.jp/news/easy/*",
        "output": "json",
        "filter": ["statuscode:200", "mimetype:text/html"],
        "collapse": "urlkey",
        "fl": "urlkey,timestamp,original",
    }, doseq=True)
    raw = http_get(f"{CDX_URL}?{q}", timeout=CDX_TIMEOUT)
    assert raw is not None
    rows = json.loads(raw.decode("utf-8"))
    # Drop header row, keep (original, timestamp).
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for row in rows[1:]:
        if len(row) < 3:
            continue
        _urlkey, ts, original = row[0], row[1], row[2]
        # Strip any port:80 or query strings that crept in.
        original = original.replace(":80", "")
        if "?" in original:
            original = original.split("?", 1)[0]
        # Force https for fetch consistency.
        if original.startswith("http://"):
            original = "https://" + original[len("http://"):]
        if original in seen:
            continue
        seen.add(original)
        pairs.append((original, ts))

    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False)
    print(f"  cached {len(pairs)} (url, ts) pairs to {cache}", file=sys.stderr)
    return pairs


def filter_articles(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Keep only article-shaped URLs."""
    keep: list[tuple[str, str]] = []
    for url, ts in pairs:
        if ARTICLE_RE.search(url):
            keep.append((url, ts))
    return keep


def url_to_filename(url: str) -> str:
    tail = url.split("/news/easy/", 1)[1]
    return tail.replace("/", "__").replace(".html", "") + ".html"


def fetch_one(url: str, ts: str, out_dir: Path) -> str:
    dest = out_dir / url_to_filename(url)
    if dest.exists() and dest.stat().st_size > 0:
        return "skip"
    raw_url = WAYBACK_RAW_TPL.format(ts=ts, url=url)
    try:
        blob = http_get(raw_url, timeout=FETCH_TIMEOUT, accept_404=True)
    except (urllib.error.URLError, TimeoutError) as e:
        return f"fail:{e!r}"
    if not blob:
        return "fail:empty"
    tmp = dest.with_suffix(".html.tmp")
    tmp.write_bytes(blob)
    tmp.rename(dest)
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/raw/nhk_easy"))
    ap.add_argument("--limit", type=int, default=0,
                    help="max articles to fetch (0 = all)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    snaps_dir = args.out / "snapshots"
    snaps_dir.mkdir(parents=True, exist_ok=True)

    cdx = fetch_cdx_index(args.out)
    articles = filter_articles(cdx)
    print(f"CDX entries: {len(cdx)}, article-shaped: {len(articles)}", file=sys.stderr)
    if args.limit:
        articles = articles[: args.limit]

    counts: dict[str, int] = {}
    for i, (url, ts) in enumerate(articles, 1):
        status = fetch_one(url, ts, snaps_dir)
        bucket = status.split(":", 1)[0]
        counts[bucket] = counts.get(bucket, 0) + 1
        if i % 100 == 0 or status.startswith("fail"):
            print(f"[{i}/{len(articles)}] {url[-50:]} -> {status}  "
                  f"(ok={counts.get('ok',0)} skip={counts.get('skip',0)} "
                  f"fail={counts.get('fail',0)})", file=sys.stderr)
        if status != "skip":
            time.sleep(PER_REQUEST_DELAY_S)

    print(f"\ndone. {counts}", file=sys.stderr)


if __name__ == "__main__":
    main()
