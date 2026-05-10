"""Download Aozora Bunko works (新字新仮名 + out-of-copyright) into data/raw/aozora/.

Steps:
    1. Fetch list_person_all_extended_utf8.csv (Aozora's bibliographic index).
    2. Filter to (文字遣い種別 == 新字新仮名, 作品著作権フラグ == なし, has .zip URL).
    3. Write the filtered metadata to data/raw/aozora/_index.csv.
    4. Download each work's text-file zip into data/raw/aozora/zips/<work_id>.zip.
       - Skip if already present (idempotent / resumable).
       - Bounded thread pool with per-request delay so we don't hammer aozora.gr.jp.

Run with:
    python scripts/01_download_aozora.py [--limit N] [--workers W]
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

INDEX_URL = "https://www.aozora.gr.jp/index_pages/list_person_all_extended_utf8.zip"
INDEX_CSV_NAME = "list_person_all_extended_utf8.csv"
USER_AGENT = "yomi-research/0.1 (Japanese furigana model training; contact: chuchu)"
REQUEST_TIMEOUT = 30.0
PER_REQUEST_DELAY_S = 1.0  # per worker, polite to a free public-good server


def open_url(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        return resp.read()


def fetch_index() -> list[dict]:
    print(f"fetching index: {INDEX_URL}", file=sys.stderr)
    blob = open_url(INDEX_URL)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        with z.open(INDEX_CSV_NAME) as f:
            text = f.read().decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def filter_rows(rows: list[dict]) -> list[dict]:
    keep = []
    for r in rows:
        if r.get("文字遣い種別") != "新字新仮名":
            continue
        if r.get("作品著作権フラグ") != "なし":
            continue
        url = (r.get("テキストファイルURL") or "").strip()
        if not url.endswith(".zip"):
            continue
        keep.append(r)
    return keep


def write_index(rows: list[dict], path: Path) -> None:
    cols = [
        "作品ID", "作品名", "姓", "名", "姓ローマ字", "名ローマ字",
        "文字遣い種別", "テキストファイルURL", "テキストファイル符号化方式",
        "公開日", "最終更新日",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def download_one(row: dict, dest_dir: Path) -> tuple[str, str]:
    """Returns (work_id, status). status ∈ {ok, skip, fail:<reason>}"""
    work_id = row["作品ID"]
    url = row["テキストファイルURL"].strip()
    dest = dest_dir / f"{work_id}.zip"
    if dest.exists() and dest.stat().st_size > 0:
        return work_id, "skip"

    last_err = None
    for attempt in range(3):
        try:
            blob = open_url(url)
            tmp = dest.with_suffix(".zip.tmp")
            tmp.write_bytes(blob)
            tmp.rename(dest)
            time.sleep(PER_REQUEST_DELAY_S)
            return work_id, "ok"
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last_err = e
            time.sleep(2 ** attempt)
    return work_id, f"fail:{last_err!r}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="max works to download (0 = all)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("data/raw/aozora"))
    args = ap.parse_args()

    rows = fetch_index()
    print(f"index rows: {len(rows)}", file=sys.stderr)
    filtered = filter_rows(rows)
    print(f"after filter (新字新仮名 / out-of-copyright / has .zip): {len(filtered)}",
          file=sys.stderr)

    args.out.mkdir(parents=True, exist_ok=True)
    write_index(filtered, args.out / "_index.csv")

    zips_dir = args.out / "zips"
    zips_dir.mkdir(parents=True, exist_ok=True)

    todo = filtered[: args.limit] if args.limit > 0 else filtered
    print(f"downloading {len(todo)} works to {zips_dir} with {args.workers} workers",
          file=sys.stderr)

    counts = {"ok": 0, "skip": 0, "fail": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(download_one, r, zips_dir) for r in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            wid, status = fut.result()
            bucket = status.split(":", 1)[0]
            counts[bucket] = counts.get(bucket, 0) + 1
            if i % 100 == 0 or status.startswith("fail"):
                print(f"[{i}/{len(todo)}] {wid} {status}", file=sys.stderr)

    print(f"done. ok={counts.get('ok',0)} skip={counts.get('skip',0)} "
          f"fail={counts.get('fail',0)}", file=sys.stderr)


if __name__ == "__main__":
    main()
