"""Prikupljanje YouTube komentara u POSEBAN CSV (ne u annotation_template).

Automatski fetch preko YouTube Data API v3 (YOUTUBE_API_KEY).

Kolone: id, source, text, tip, sentiment, sarcasm
  - source = uvek alias "youtube" (ne pun watch URL)
  - tip    = tema (--tip), šema kolona tip
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.collection.base import text_fingerprint
from src.collection.youtube import YouTubeCollector
from src.common.config import ensure_dir, resolve_path
from src.common.io_utils import save_csv
from src.common.language import is_likely_serbian
from src.common.schema import FINAL_COLUMNS, DatasetRecord
from src.preprocessing.clean import preprocess_text
from src.preprocessing.deduplicate import normalize_for_dedup

_ID_PREFIX = "yt"


def load_youtube_urls(path: Path) -> list[str]:
    """TXT: jedan video ID ili URL po liniji; # komentari se ignorišu."""
    if not path.exists():
        return []
    urls: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    return urls


def _next_youtube_id(existing_ids: list[str]) -> int:
    """Sledeći broj: yt-XXXXX ili postojeći sr-XXXXX iz annotation exporta."""
    max_n = 0
    pats = (
        re.compile(rf"^{_ID_PREFIX}-(\d+)$", re.I),
        re.compile(r"^sr-(\d+)$", re.I),
    )
    for raw in existing_ids:
        s = str(raw).strip()
        for pat in pats:
            m = pat.match(s)
            if m:
                max_n = max(max_n, int(m.group(1)))
                break
    return max_n + 1


def _load_existing(out_path: Path) -> list[dict[str, str]]:
    """Učitaj postojeći youtube_comments CSV ili vrati praznu listu."""
    if not out_path.exists():
        return []
    return (
        pd.read_csv(
            out_path,
            encoding="utf-8-sig",
            dtype=str,
            engine="python",
            on_bad_lines="warn",
        )
        .fillna("")
        .to_dict(orient="records")
    )


def _append_texts(
    config: dict[str, Any],
    texts: list[str],
    *,
    tip: str,
    out_path: Path,
    url_note: str = "",
) -> list[dict[str, str]]:
    """Očisti/dedup tekstove i dopiši u out_path; tip = šema kolona (tema)."""
    prep_cfg = config.get("preprocessing", {})
    lang_cfg = config.get("language", {})
    min_len = int(lang_cfg.get("min_text_length", 15))
    max_len = int(lang_cfg.get("max_text_length", 2000))

    existing = _load_existing(out_path)
    existing_texts = {normalize_for_dedup(str(r.get("text", ""))) for r in existing}
    existing_fps = {text_fingerprint(str(r.get("text", ""))) for r in existing}
    next_id = _next_youtube_id([str(r.get("id", "")) for r in existing])

    appended: list[dict[str, str]] = []
    for raw in texts:
        cleaned = preprocess_text(raw, prep_cfg)
        if len(cleaned) < min_len or len(cleaned) > max_len:
            continue
        if not is_likely_serbian(cleaned, lang_cfg):
            continue
        key = normalize_for_dedup(cleaned)
        fp = text_fingerprint(cleaned)
        if key in existing_texts or fp in existing_fps:
            continue
        existing_texts.add(key)
        existing_fps.add(fp)
        appended.append(
            DatasetRecord(
                id=f"{_ID_PREFIX}-{next_id:05d}",
                source="youtube",
                text=cleaned,
                tip=tip,
                sentiment="",
                sarcasm="",
            ).to_dict()
        )
        next_id += 1

    if not appended:
        print("[youtube] Nema novih tekstova posle čišćenja/dedupa.")
        return []

    combined = existing + appended
    save_csv(combined, out_path, columns=FINAL_COLUMNS)
    print(
        f"[youtube] +{len(appended)} (tip={tip}) -> ukupno {len(combined)} | {out_path}"
    )
    if url_note:
        print(f"[youtube] video: {url_note}")
    return appended


def append_youtube_fetch(
    config: dict[str, Any],
    *,
    tip: str,
    url: str | None = None,
    urls_file: str | Path | None = None,
    out_csv: str | Path | None = None,
    max_comments: int = 0,
) -> list[dict[str, str]]:
    """Skini komentare sa YouTube videa i upiši u youtube_comments.csv.

    tip: vrednost šeme kolone tip (tema). out_csv podrazumevano iz config-a.
    """
    tip = (tip or "").strip() or "ostalo"
    yt_cfg = config.get("collection", {}).get("youtube", {})
    default_out = yt_cfg.get("output_csv", "data/processed/sources/youtube_comments.csv")
    default_urls = yt_cfg.get("video_ids_file", "config/sources/youtube_video_ids.txt")
    out_path = resolve_path(out_csv or default_out)
    ensure_dir(out_path.parent)

    urls: list[str] = []
    if url and str(url).strip():
        urls = [str(url).strip()]
    else:
        urls_path = resolve_path(urls_file or default_urls)
        urls = load_youtube_urls(urls_path)
        print(f"[youtube] Učitano {len(urls)} URL/ID-eva iz {urls_path}")

    if not urls:
        print("[youtube] Nema URL-ova / video ID-eva za fetch.")
        return []

    collector = YouTubeCollector(config)
    per_video = int(max_comments) if max_comments > 0 else int(
        yt_cfg.get("max_comments_per_video", 100)
    )
    # Budget: dovoljno da pokrije sve zadate videe
    raw_budget = max(per_video * len(urls), per_video)

    # Privremeno override max_comments_per_video za ovaj fetch
    old_per = yt_cfg.get("max_comments_per_video")
    yt_cfg["max_comments_per_video"] = per_video
    try:
        raw_records = collector.collect_specific_videos(urls, max_records=raw_budget)
    finally:
        if old_per is None:
            yt_cfg.pop("max_comments_per_video", None)
        else:
            yt_cfg["max_comments_per_video"] = old_per

    if not raw_records:
        print("[youtube] API nije vratio komentare.")
        return []

    # Grupiši po video_id radi logovanja
    by_video: dict[str, list[str]] = {}
    for rec in raw_records:
        vid = str((rec.metadata or {}).get("video_id") or "unknown")
        by_video.setdefault(vid, []).append(rec.text)

    all_appended: list[dict[str, str]] = []
    for vid, texts in by_video.items():
        if not texts:
            continue
        note = f"https://www.youtube.com/watch?v={vid}"
        got = _append_texts(
            config, texts, tip=tip, out_path=out_path, url_note=note
        )
        all_appended.extend(got)

    print(f"[youtube] Ukupno dodato u ovoj sesiji: {len(all_appended)}")
    return all_appended


# Alias za stari import / CLI
def append_new_youtube_videos(
    config: dict[str, Any],
    only_video_ids: list[str] | None = None,
    *,
    tip: str = "ostalo",
    out_csv: str | Path | None = None,
    max_comments: int = 0,
) -> list[dict[str, str]]:
    """Kompatibilnost: fetch u youtube_comments.csv (ne u annotation_template)."""
    url = None
    urls_file = None
    if only_video_ids:
        if len(only_video_ids) == 1:
            url = only_video_ids[0]
        else:
            # Privremeni fajl nije potreban — pozovi fetch po jedan, ili spoji
            all_appended: list[dict[str, str]] = []
            for vid in only_video_ids:
                all_appended.extend(
                    append_youtube_fetch(
                        config,
                        tip=tip,
                        url=vid,
                        out_csv=out_csv,
                        max_comments=max_comments,
                    )
                )
            return all_appended
    return append_youtube_fetch(
        config,
        tip=tip,
        url=url,
        urls_file=urls_file,
        out_csv=out_csv,
        max_comments=max_comments,
    )
