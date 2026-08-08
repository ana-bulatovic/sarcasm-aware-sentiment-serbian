"""Kreiranje annotation template CSV-a iz interim cleaned podataka.

U finalnom datasetu polje source obično čuva pun URL (raw pipeline);
platforma se izvodi preko platform_from_source.

Alternativa: ``build_dataset_from_sources`` spaja
``data/processed/sources/*_comments.csv`` u finalni dataset.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.common.config import ensure_dir, resolve_path
from src.common.io_utils import load_csv, load_jsonl, save_csv, save_jsonl
from src.common.schema import FINAL_COLUMNS, DatasetRecord
from src.common.source_utils import PLATFORM_ORDER, platform_from_source, platform_sort_key

# Redosled spajanja source CSV-ova (youtube → twitter → instagram)
_SOURCE_CSV_KEYS = (
    ("youtube_csv", "data/processed/sources/youtube_comments.csv"),
    ("twitter_csv", "data/processed/sources/twitter_comments.csv"),
    ("instagram_csv", "data/processed/sources/instagram_comments.csv"),
)


def _tip_from_row(row: dict[str, Any]) -> str:
    """Uzmi tip/topic iz reda (YouTube: tip, Twitter/IG: topic)."""
    tip = str(row.get("tip") or "").strip()
    if tip:
        return tip
    return str(row.get("topic") or "").strip()


def build_dataset_from_sources(config: dict[str, Any]) -> list[dict[str, str]]:
    """Spoji youtube/twitter/instagram comments CSV u finalni dataset.

    ``id`` je redni broj: 1, 2, 3, … (bez ``sr-`` prefiksa).
    Piše ``dataset_csv`` i ``dataset_jsonl`` iz config.paths.
    """
    paths = config.get("paths", {})
    dataset_path = resolve_path(paths["dataset_csv"])
    dataset_jsonl = resolve_path(
        paths.get("dataset_jsonl")
        or str(Path(paths["dataset_csv"]).with_suffix(".jsonl"))
    )
    ensure_dir(dataset_path.parent)
    ensure_dir(dataset_jsonl.parent)

    rows: list[dict[str, str]] = []
    for key, default_rel in _SOURCE_CSV_KEYS:
        csv_path = resolve_path(paths.get(key, default_rel))
        if not csv_path.exists():
            print(f"[dataset] Preskačem (nema fajla): {csv_path}")
            continue
        df = load_csv(csv_path)
        for rec in df.to_dict(orient="records"):
            text = str(rec.get("text", "")).strip()
            if not text:
                continue
            rows.append(
                DatasetRecord(
                    id="",  # popunjava se ispod
                    source=str(rec.get("source", "")).strip(),
                    text=text,
                    tip=_tip_from_row(rec),
                    sentiment=str(rec.get("sentiment") or "").strip(),
                    sarcasm=str(rec.get("sarcasm") or "").strip(),
                ).to_dict()
            )

    final_rows: list[dict[str, str]] = []
    for idx, row in enumerate(rows, start=1):
        row["id"] = str(idx)
        final_rows.append(row)

    save_csv(final_rows, dataset_path, columns=FINAL_COLUMNS)
    save_jsonl(final_rows, dataset_jsonl)

    by_source: dict[str, int] = defaultdict(int)
    for row in final_rows:
        by_source[row["source"] or "unknown"] += 1

    print(f"[dataset] Finalnih uzoraka: {len(final_rows)} -> {dataset_path}")
    print("[dataset] Po izvoru:")
    for source, count in sorted(by_source.items(), key=lambda x: (-x[1], x[0])):
        print(f"  - {source}: {count}")
    return final_rows


def _allocate_per_platform(
    records: list[dict[str, Any]],
    per_source_limits: dict[str, int],
    max_total: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Limit po platformi (youtube/tiktok/...), source u zapisu ostaje pun URL."""
    by_platform: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_platform[platform_from_source(str(rec.get("source", "")))].append(rec)

    selected: list[dict[str, Any]] = []
    for platform in PLATFORM_ORDER:
        pool = list(by_platform.get(platform, []))
        if not pool:
            continue
        limit = int(per_source_limits.get(platform, max_total))
        rng.shuffle(pool)
        selected.extend(pool[:limit])

    # Ostale nepoznate platforme
    for platform, pool in by_platform.items():
        if platform in PLATFORM_ORDER:
            continue
        limit = int(per_source_limits.get(platform, max_total))
        items = list(pool)
        rng.shuffle(items)
        selected.extend(items[:limit])

    selected = selected[:max_total]
    # Grupisi: platforma, pa URL
    selected.sort(key=lambda r: platform_sort_key(str(r.get("source", ""))))
    return selected


def build_annotation_dataset(config: dict[str, Any]) -> list[dict[str, str]]:
    """Od interim cleaned.jsonl napravi finalni CSV za anotaciju.

    Piše annotation_csv, dataset_csv i dataset_jsonl iz config.paths.
    """
    interim_path = resolve_path(config["paths"]["interim_dir"]) / "cleaned.jsonl"
    annotation_path = resolve_path(config["paths"]["annotation_csv"])
    dataset_path = resolve_path(config["paths"]["dataset_csv"])
    dataset_jsonl = resolve_path(
        config["paths"].get("dataset_jsonl")
        or str(Path(config["paths"]["dataset_csv"]).with_suffix(".jsonl"))
    )
    ensure_dir(annotation_path.parent)
    ensure_dir(dataset_path.parent)
    ensure_dir(dataset_jsonl.parent)

    records = load_jsonl(interim_path)
    if not records:
        raise FileNotFoundError(
            f"Nema cleaned podataka u {interim_path}. "
            "Prvo pokrenite kolekciju i preprocesiranje."
        )

    max_total = int(config["dataset"]["max_total_samples"])
    per_source_limits = config.get("per_source_limits", {})
    seed = int(config.get("random_seed", 42))
    rng = random.Random(seed)

    chosen = _allocate_per_platform(records, per_source_limits, max_total, rng)

    final_rows: list[dict[str, str]] = []
    for idx, rec in enumerate(chosen, start=1):
        row = DatasetRecord(
            id=f"sr-{idx:05d}",
            source=str(rec.get("source", "")),
            text=str(rec.get("text", "")),
            tip=str(rec.get("tip") or ""),
            sentiment=str(rec.get("sentiment") or ""),
            sarcasm=str(rec.get("sarcasm") or ""),
        ).to_dict()
        final_rows.append(row)

    save_csv(final_rows, annotation_path, columns=FINAL_COLUMNS)
    save_csv(final_rows, dataset_path, columns=FINAL_COLUMNS)
    save_jsonl(final_rows, dataset_jsonl)

    by_url: dict[str, int] = defaultdict(int)
    by_platform: dict[str, int] = defaultdict(int)
    for row in final_rows:
        by_url[row["source"]] += 1
        by_platform[platform_from_source(row["source"])] += 1

    print(f"[dataset] Finalnih uzoraka: {len(final_rows)} (max={max_total})")
    print("[dataset] Po platformi:")
    for platform, count in sorted(by_platform.items()):
        print(f"  - {platform}: {count}")
    print("[dataset] Po URL-u:")
    for url, count in sorted(by_url.items(), key=lambda x: (-x[1], x[0])):
        print(f"  - {count:4d}  {url}")
    print(f"[dataset] Annotation template: {annotation_path}")
    return final_rows
