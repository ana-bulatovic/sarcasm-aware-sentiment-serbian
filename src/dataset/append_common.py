"""Zajedničko dopisivanje redova na annotation CSV.

Koriste ga polu-ručni append_* moduli. Argument source može biti pun URL
(raw/annotation pipeline); platforma se izvodi iz URL-a.
Dedup: normalize_for_dedup + text_fingerprint (16 hex), ne pun SHA.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.collection.base import text_fingerprint
from src.common.config import ensure_dir, resolve_path
from src.common.io_utils import load_csv, save_csv, save_jsonl
from src.common.language import is_likely_serbian
from src.common.schema import FINAL_COLUMNS, DatasetRecord
from src.common.source_utils import platform_from_source
from src.preprocessing.clean import preprocess_text
from src.preprocessing.deduplicate import normalize_for_dedup

_ID_RE = re.compile(r"^sr-(\d+)$")


def next_annotation_id(existing_ids: list[str]) -> int:
    """Sledeći numerički sufiks za id oblika sr-XXXXX."""
    max_n = 0
    for raw in existing_ids:
        m = _ID_RE.match(str(raw).strip())
        if m:
            max_n = max(max_n, int(m.group(1)))
    return max_n + 1


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Dopuni JSONL fajl (kreira parent dir po potrebi)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_texts_to_annotation(
    config: dict[str, Any],
    texts: list[str],
    *,
    source: str,
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Očisti, filtriraj, deduplikuj i dopisi tekstove na annotation CSV.

    source: pun URL ili identifikator izvora (platforma via platform_from_source).
    Piše i raw/<platform>/raw.jsonl, interim/cleaned.jsonl, dataset CSV/JSONL.
    """
    annotation_path = resolve_path(config["paths"]["annotation_csv"])
    dataset_path = resolve_path(config["paths"]["dataset_csv"])
    dataset_jsonl = resolve_path(
        config["paths"].get("dataset_jsonl")
        or str(Path(config["paths"]["dataset_csv"]).with_suffix(".jsonl"))
    )
    ensure_dir(annotation_path.parent)
    ensure_dir(dataset_path.parent)
    ensure_dir(dataset_jsonl.parent)

    if not annotation_path.exists():
        raise FileNotFoundError(
            f"Nema {annotation_path}. Prvo napravi osnovni dataset (npr. YouTube)."
        )

    df = load_csv(annotation_path)
    existing_rows = df.to_dict(orient="records")
    existing_texts = {normalize_for_dedup(str(r.get("text", ""))) for r in existing_rows}
    # 16-hex fingerprint (base.text_fingerprint), ne pun SHA iz preprocessing.deduplicate
    existing_fps = {text_fingerprint(str(r.get("text", ""))) for r in existing_rows}

    max_total = int(config["dataset"]["max_total_samples"])
    platform = platform_from_source(source)
    source_limit = int(config.get("per_source_limits", {}).get(platform, max_total))
    current_total = len(existing_rows)
    current_source = sum(
        1 for r in existing_rows if platform_from_source(str(r.get("source", ""))) == platform
    )
    room = min(max(0, max_total - current_total), max(0, source_limit - current_source))
    if room <= 0:
        print(
            f"[append] Nema mesta (ukupno {current_total}/{max_total}, "
            f"{platform} {current_source}/{source_limit})."
        )
        return []

    prep_cfg = config.get("preprocessing", {})
    lang_cfg = config.get("language", {})
    min_len = int(lang_cfg.get("min_text_length", 15))
    max_len = int(lang_cfg.get("max_text_length", 2000))
    meta = metadata or {}

    cleaned: list[dict[str, Any]] = []
    raw_dicts: list[dict[str, Any]] = []
    for text in texts:
        raw_dicts.append(
            {
                "source": source,
                "text": text,
                "source_item_id": None,
                "sentiment": "",
                "sarcasm": "",
                "metadata": meta,
            }
        )
        cleaned_text = preprocess_text(text, prep_cfg)
        if len(cleaned_text) < min_len or len(cleaned_text) > max_len:
            continue
        if not is_likely_serbian(cleaned_text, lang_cfg):
            continue
        key = normalize_for_dedup(cleaned_text)
        fp = text_fingerprint(cleaned_text)
        if key in existing_texts or fp in existing_fps:
            continue
        existing_texts.add(key)
        existing_fps.add(fp)
        cleaned.append(
            {
                "source": source,
                "text": cleaned_text,
                "sentiment": "",
                "sarcasm": "",
                "metadata": meta,
            }
        )
        if len(cleaned) >= room:
            break

    raw_path = ensure_dir(resolve_path(config["paths"]["raw_dir"]) / platform) / "raw.jsonl"
    if raw_dicts:
        append_jsonl(raw_path, raw_dicts)

    interim_path = ensure_dir(resolve_path(config["paths"]["interim_dir"])) / "cleaned.jsonl"
    if cleaned:
        append_jsonl(interim_path, cleaned)

    next_id = next_annotation_id([str(r.get("id", "")) for r in existing_rows])
    appended: list[dict[str, str]] = []
    for rec in cleaned:
        appended.append(
            DatasetRecord(
                id=f"sr-{next_id:05d}",
                source=source,
                text=str(rec["text"]),
                tip=str(meta.get("tip") or ""),
                sentiment="",
                sarcasm="",
            ).to_dict()
        )
        next_id += 1

    if not appended:
        print(f"[append] Nema novih tekstova za source={source} posle filtera/dedupa.")
        return []

    combined = existing_rows + appended
    save_csv(combined, annotation_path, columns=FINAL_COLUMNS)
    save_csv(combined, dataset_path, columns=FINAL_COLUMNS)
    save_jsonl(combined, dataset_jsonl)
    print(
        f"[append] +{len(appended)} ({source}) -> ukupno {len(combined)} | {annotation_path}"
    )
    return appended
