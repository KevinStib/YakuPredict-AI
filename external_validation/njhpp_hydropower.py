"""Descarga reproducible del dataset público NJHPP desde figshare.

Fuente: Yasir Saleem Afridi (2022), Bearing Vibration Dataset of a Hydropower
Project. DOI 10.6084/m9.figshare.21290895. Licencia CC BY 4.0.
"""
from __future__ import annotations
import argparse
import json
import shutil
import urllib.request
from pathlib import Path

ARTICLE_ID = 21290895
API = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}"
DEFAULT_CACHE = Path("data/external/njhpp")


def _json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "YakuPredict-AI/4.0 academic-validation"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read().decode("utf-8"))


def download(cache_dir: Path = DEFAULT_CACHE) -> list[Path]:
    meta = _json(API)
    files = meta.get("files") or []
    if not files:
        raise RuntimeError("figshare no devolvió archivos para el artículo NJHPP")
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for item in files:
        url = item.get("download_url")
        name = item.get("name")
        if not url or not name:
            continue
        dst = cache_dir / name
        if not dst.exists() or dst.stat().st_size == 0:
            req = urllib.request.Request(url, headers={"User-Agent": "YakuPredict-AI/4.0 academic-validation"})
            with urllib.request.urlopen(req, timeout=180) as r, dst.open("wb") as f:
                shutil.copyfileobj(r, f)
        out.append(dst)
    print(json.dumps({"doi": "10.6084/m9.figshare.21290895", "files": [str(p) for p in out]}, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    args = ap.parse_args()
    if not args.download:
        ap.error("Use --download")
    download(args.cache_dir)


if __name__ == "__main__":
    main()
