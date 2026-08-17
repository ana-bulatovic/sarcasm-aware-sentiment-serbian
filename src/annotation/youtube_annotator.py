"""Logika anotacije YouTube komentara u ``youtube_comments.csv``.

CSV nema video_id — komentari se vezuju za URL tako što se skinu sa API-ja
i upare sa redovima po normalizovanom tekstu.
"""

from __future__ import annotations

import threading
from typing import Any

from src.collection.youtube import YouTubeCollector
from src.common.config import resolve_path
from src.common.io_utils import save_csv
from src.common.schema import FINAL_COLUMNS, SARCASM_VALUES, SENTIMENT_VALUES
from src.dataset.append_youtube import _append_texts, _load_existing, _next_youtube_id
from src.preprocessing.clean import preprocess_text
from src.preprocessing.deduplicate import normalize_for_dedup

_LOCK = threading.Lock()
_ID_PREFIX = "yt"


def youtube_csv_path(config: dict[str, Any]):
    """Putanja do ``youtube_comments.csv`` iz config-a."""
    yt_cfg = config.get("collection", {}).get("youtube", {})
    default = config.get("paths", {}).get(
        "youtube_csv", "data/processed/sources/youtube_comments.csv"
    )
    return resolve_path(yt_cfg.get("output_csv") or default)


def _load_rows(config: dict[str, Any]) -> list[dict[str, str]]:
    """Učitaj sve redove YouTube CSV-a."""
    path = youtube_csv_path(config)
    rows = _load_existing(path)
    out: list[dict[str, str]] = []
    for rec in rows:
        out.append({col: str(rec.get(col, "") or "") for col in FINAL_COLUMNS})
    return out


def _write_rows(config: dict[str, Any], rows: list[dict[str, str]]) -> None:
    """Prepiši YouTube CSV."""
    save_csv(rows, youtube_csv_path(config), columns=FINAL_COLUMNS)


def _is_labeled(row: dict[str, str]) -> bool:
    """True ako su obe labele validne."""
    sent = str(row.get("sentiment", "")).strip()
    sarc = str(row.get("sarcasm", "")).strip()
    return sent in SENTIMENT_VALUES and sarc in SARCASM_VALUES


def csv_overview(config: dict[str, Any]) -> dict[str, Any]:
    """Brojevi u YouTube CSV-u (za status u UI)."""
    rows = _load_rows(config)
    labeled = sum(1 for r in rows if _is_labeled(r))
    return {
        "path": str(youtube_csv_path(config)),
        "total": len(rows),
        "labeled": labeled,
        "unlabeled": len(rows) - labeled,
    }


def _item_from_row(row: dict[str, str], *, in_csv: bool) -> dict[str, Any]:
    """Jedan komentar za UI red."""
    return {
        "csv_id": row.get("id", "") if in_csv else "",
        "text": row.get("text", ""),
        "tip": row.get("tip", ""),
        "sentiment": str(row.get("sentiment", "")).strip(),
        "sarcasm": str(row.get("sarcasm", "")).strip(),
        "in_csv": in_csv,
        "labeled": _is_labeled(row),
    }


def load_queue(
    config: dict[str, Any],
    *,
    url: str,
    tip: str = "",
    unlabeled_only: bool = True,
    only_in_csv: bool = True,
    add_new: bool = False,
    max_comments: int = 0,
) -> dict[str, Any]:
    """Skini komentare sa videa, upari sa CSV-om, vrati red za anotaciju."""
    url = (url or "").strip()
    if not url:
        raise ValueError("Unesite YouTube URL ili video ID.")

    collector = YouTubeCollector(config)
    video_id = collector._normalize_video_id(url)
    if not video_id:
        raise ValueError("Nisam uspeo da izvučem video ID iz URL-a.")

    yt_cfg = config.get("collection", {}).get("youtube", {})
    per_video = int(max_comments) if max_comments > 0 else int(
        yt_cfg.get("max_comments_per_video", 300)
    )
    raw = collector.collect_specific_videos([video_id], max_records=per_video)
    if not raw:
        raise RuntimeError(
            "API nije vratio komentare. Proverite YOUTUBE_API_KEY, URL i da li "
            "su komentari uključeni na videu."
        )

    prep_cfg = config.get("preprocessing", {})
    cleaned_texts: list[str] = []
    seen_clean: set[str] = set()
    for rec in raw:
        cleaned = preprocess_text(rec.text, prep_cfg)
        key = normalize_for_dedup(cleaned)
        if not cleaned or key in seen_clean:
            continue
        seen_clean.add(key)
        cleaned_texts.append(cleaned)

    added = 0
    tip_val = (tip or "").strip()
    with _LOCK:
        if add_new:
            if not tip_val:
                raise ValueError("Za dodavanje novih komentara treba i tip (tema).")
            appended = _append_texts(
                config,
                cleaned_texts,
                tip=tip_val,
                out_path=youtube_csv_path(config),
                url_note=f"https://www.youtube.com/watch?v={video_id}",
            )
            added = len(appended)
        rows = _load_rows(config)

    index: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        key = normalize_for_dedup(row.get("text", ""))
        if key:
            index.setdefault(key, []).append(i)

    used: set[int] = set()
    items: list[dict[str, Any]] = []
    matched = 0
    for text in cleaned_texts:
        key = normalize_for_dedup(text)
        row_i = None
        for cand in index.get(key, []):
            if cand not in used:
                row_i = cand
                break
        if row_i is not None:
            used.add(row_i)
            matched += 1
            item = _item_from_row(rows[row_i], in_csv=True)
        else:
            item = _item_from_row(
                {
                    "id": "",
                    "text": text,
                    "tip": tip_val,
                    "sentiment": "",
                    "sarcasm": "",
                },
                in_csv=False,
            )
        if only_in_csv and not item["in_csv"]:
            continue
        if unlabeled_only and item["labeled"]:
            continue
        items.append(item)

    unlabeled_matched = sum(
        1
        for i in used
        if not _is_labeled(rows[i])
    )
    return {
        "video_id": video_id,
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "fetched": len(raw),
        "unique_cleaned": len(cleaned_texts),
        "matched": matched,
        "added": added,
        "unlabeled_matched": unlabeled_matched,
        "queue": len(items),
        "items": items,
        "csv": csv_overview(config),
    }


def load_unlabeled_csv(config: dict[str, Any]) -> dict[str, Any]:
    """Red svih neanotiranih redova iz YouTube CSV-a (bez URL-a)."""
    with _LOCK:
        rows = _load_rows(config)
    items = [_item_from_row(r, in_csv=True) for r in rows if not _is_labeled(r)]
    return {
        "video_id": "",
        "url": "",
        "fetched": 0,
        "unique_cleaned": 0,
        "matched": len(items),
        "added": 0,
        "unlabeled_matched": len(items),
        "queue": len(items),
        "items": items,
        "csv": csv_overview(config),
    }


def save_labels(
    config: dict[str, Any],
    *,
    csv_id: str,
    sentiment: str,
    sarcasm: str,
    text: str = "",
    tip: str = "",
) -> dict[str, Any]:
    """Sačuvaj labele. Ako ``csv_id`` nedostaje, dodaj novi red."""
    sent = str(sentiment).strip()
    sarc = str(sarcasm).strip()
    if sent not in SENTIMENT_VALUES:
        raise ValueError("Sentiment mora biti 1, 0 ili -1.")
    if sarc not in SARCASM_VALUES:
        raise ValueError("Sarkazam mora biti 0 ili 1.")

    with _LOCK:
        rows = _load_rows(config)
        csv_id = (csv_id or "").strip()
        if csv_id:
            found = False
            for row in rows:
                if str(row.get("id", "")).strip() == csv_id:
                    row["sentiment"] = sent
                    row["sarcasm"] = sarc
                    found = True
                    break
            if not found:
                raise KeyError(f"Nema reda sa id={csv_id}")
        else:
            cleaned = preprocess_text(text, config.get("preprocessing", {}))
            if not cleaned:
                raise ValueError("Prazan tekst — nema šta da se doda.")
            next_n = _next_youtube_id([str(r.get("id", "")) for r in rows])
            csv_id = f"{_ID_PREFIX}-{next_n:05d}"
            rows.append(
                {
                    "id": csv_id,
                    "source": "youtube",
                    "text": cleaned,
                    "tip": (tip or "").strip(),
                    "sentiment": sent,
                    "sarcasm": sarc,
                }
            )
        _write_rows(config, rows)

    return {"csv_id": csv_id, "sentiment": sent, "sarcasm": sarc, "csv": csv_overview(config)}


def delete_row(config: dict[str, Any], csv_id: str) -> dict[str, Any]:
    """Obriši red iz YouTube CSV-a po id-u."""
    csv_id = (csv_id or "").strip()
    if not csv_id:
        raise ValueError("Nema id-a — komentar nije u CSV-u, samo ga preskočite.")
    with _LOCK:
        rows = _load_rows(config)
        kept = [r for r in rows if str(r.get("id", "")).strip() != csv_id]
        if len(kept) == len(rows):
            raise KeyError(f"Nema reda sa id={csv_id}")
        _write_rows(config, kept)
    return {"deleted": csv_id, "csv": csv_overview(config)}
