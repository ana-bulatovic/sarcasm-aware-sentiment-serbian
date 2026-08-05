"""Dodavanje komentara samo sa novih YouTube videa na postojeci annotation CSV."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.collection.base import text_fingerprint
from src.collection.youtube import YouTubeCollector
from src.common.config import ensure_dir, resolve_path
from src.common.io_utils import load_csv, load_jsonl, save_csv, save_jsonl
from src.common.language import is_likely_serbian
from src.common.schema import FINAL_COLUMNS, DatasetRecord
from src.common.source_utils import platform_from_source, youtube_watch_url
from src.preprocessing.clean import preprocess_text
from src.preprocessing.deduplicate import normalize_for_dedup

_ID_RE = re.compile(r"^sr-(\d+)$")


def _parse_next_id(existing_ids: list[str]) -> int:
    max_n = 0
    for raw in existing_ids:
        m = _ID_RE.match(str(raw).strip())
        if m:
            max_n = max(max_n, int(m.group(1)))
    return max_n + 1


def _load_known_video_ids(config: dict[str, Any]) -> set[str]:
    """Video ID-evi koji su vec obradjeni (state + raw metadata)."""
    state_path = resolve_path(config["paths"]["processed_dir"]) / "youtube_collected_ids.txt"
    known: set[str] = set()
    if state_path.exists():
        for line in state_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                known.add(line)

    raw_path = resolve_path(config["paths"]["raw_dir"]) / "youtube" / "raw.jsonl"
    for rec in load_jsonl(raw_path):
        vid = (rec.get("metadata") or {}).get("video_id")
        if vid:
            known.add(str(vid))
    return known


def _save_known_video_ids(config: dict[str, Any], video_ids: set[str]) -> None:
    path = ensure_dir(resolve_path(config["paths"]["processed_dir"])) / "youtube_collected_ids.txt"
    lines = sorted(video_ids)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_new_youtube_videos(
    config: dict[str, Any],
    only_video_ids: list[str] | None = None,
) -> list[dict[str, str]]:
    """Skini samo nove YouTube ID-eve i dopisi ih na annotation_template.csv."""
    annotation_path = resolve_path(config["paths"]["annotation_csv"])
    dataset_path = resolve_path(config["paths"]["dataset_csv"])
    processed_dir = ensure_dir(resolve_path(config["paths"]["processed_dir"]))

    if not annotation_path.exists():
        raise FileNotFoundError(
            f"Nema {annotation_path}. Prvo napravi osnovni dataset "
            "(python scripts/collection/run_pipeline.py), pa tek onda append."
        )

    df = load_csv(annotation_path)
    existing_rows = df.to_dict(orient="records")
    existing_texts = {normalize_for_dedup(str(r.get("text", ""))) for r in existing_rows}
    existing_fps = {text_fingerprint(str(r.get("text", ""))) for r in existing_rows}

    max_total = int(config["dataset"]["max_total_samples"])
    yt_limit = int(config.get("per_source_limits", {}).get("youtube", max_total))
    current_total = len(existing_rows)
    current_yt = sum(
        1
        for r in existing_rows
        if platform_from_source(str(r.get("source", ""))) == "youtube"
    )
    room_total = max(0, max_total - current_total)
    room_yt = max(0, yt_limit - current_yt)
    room = min(room_total, room_yt)

    if room <= 0:
        print(
            f"[append-youtube] Nema mesta (ukupno {current_total}/{max_total}, "
            f"youtube {current_yt}/{yt_limit}). Povecaj limite u config.yaml."
        )
        return []

    collector = YouTubeCollector(config)
    all_ids = collector._load_video_ids()
    known = _load_known_video_ids(config)

    if only_video_ids:
        candidates = [collector._normalize_video_id(v) for v in only_video_ids]
        candidates = [v for v in candidates if v]
    else:
        candidates = all_ids

    new_ids = [vid for vid in candidates if vid not in known]
    deduped: list[str] = []
    seen_new: set[str] = set()
    for vid in new_ids:
        if vid not in seen_new:
            seen_new.add(vid)
            deduped.append(vid)
    new_ids = deduped

    if not new_ids:
        print("[append-youtube] Nema novih video ID-eva za dodavanje.")
        print(f"  Vec poznato: {len(known)} | U listi: {len(all_ids)}")
        return []

    print(f"[append-youtube] Novi videi ({len(new_ids)}): {', '.join(new_ids)}")
    print(f"[append-youtube] Dostupno mesta: {room} (ukupno room={room_total}, yt room={room_yt})")

    # Oversample raw pa filtriraj
    factor = float(config["dataset"].get("raw_oversample_factor", 1.5))
    raw_budget = max(room, int(room * factor))
    raw_records = collector.collect_specific_videos(new_ids, max_records=raw_budget)

    # Sacuvaj/append raw
    raw_path = ensure_dir(resolve_path(config["paths"]["raw_dir"]) / "youtube") / "raw.jsonl"
    existing_raw = load_jsonl(raw_path)
    existing_raw_fps = {
        text_fingerprint(str(r.get("text", ""))) for r in existing_raw
    }
    new_raw_dicts: list[dict[str, Any]] = []
    for rec in raw_records:
        d = rec.to_dict()
        fp = text_fingerprint(str(d.get("text", "")))
        if fp in existing_raw_fps:
            continue
        existing_raw_fps.add(fp)
        new_raw_dicts.append(d)
    if new_raw_dicts:
        _append_jsonl(raw_path, new_raw_dicts)
        print(f"[append-youtube] Raw +{len(new_raw_dicts)} -> {raw_path}")

    prep_cfg = config.get("preprocessing", {})
    lang_cfg = config.get("language", {})
    min_len = int(lang_cfg.get("min_text_length", 15))
    max_len = int(lang_cfg.get("max_text_length", 2000))

    cleaned_new: list[dict[str, Any]] = []
    for rec in raw_records:
        text = preprocess_text(rec.text, prep_cfg)
        if len(text) < min_len or len(text) > max_len:
            continue
        if not is_likely_serbian(text, lang_cfg):
            continue
        key = normalize_for_dedup(text)
        fp = text_fingerprint(text)
        if key in existing_texts or fp in existing_fps:
            continue
        existing_texts.add(key)
        existing_fps.add(fp)
        cleaned_new.append(
            {
                "source": rec.source or youtube_watch_url(
                    str((rec.metadata or {}).get("video_id", ""))
                ),
                "text": text,
                "source_item_id": rec.source_item_id,
                "sentiment": "",
                "sarcasm": "",
                "metadata": rec.metadata or {},
            }
        )
        if len(cleaned_new) >= room:
            break

    interim_path = ensure_dir(resolve_path(config["paths"]["interim_dir"])) / "cleaned.jsonl"
    if cleaned_new:
        _append_jsonl(interim_path, cleaned_new)

    next_id = _parse_next_id([str(r.get("id", "")) for r in existing_rows])
    appended_rows: list[dict[str, str]] = []
    for rec in cleaned_new:
        row = DatasetRecord(
            id=f"sr-{next_id:05d}",
            source=str(rec["source"]),
            text=str(rec["text"]),
            sentiment="",
            sarcasm="",
        ).to_dict()
        appended_rows.append(row)
        next_id += 1

    if not appended_rows:
        # Ipak zapamti video ID-eve da se ne pokeusava beskonacno ako nema srpskog
        known.update(new_ids)
        _save_known_video_ids(config, known)
        print("[append-youtube] Nema novih tekstova posle filtera/dedupa.")
        return []

    combined = existing_rows + appended_rows
    save_csv(combined, annotation_path, columns=FINAL_COLUMNS)
    save_csv(combined, dataset_path, columns=FINAL_COLUMNS)
    save_jsonl(combined, processed_dir / "dataset.jsonl")

    known.update(new_ids)
    _save_known_video_ids(config, known)

    print(
        f"[append-youtube] Dodato {len(appended_rows)} YouTube redova "
        f"(sada ukupno {len(combined)})."
    )
    print(f"[append-youtube] Azurirano: {annotation_path}")
    return appended_rows
