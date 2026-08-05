"""Kreiranje annotation template CSV-a."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from src.common.config import ensure_dir, resolve_path
from src.common.io_utils import load_jsonl, save_csv, save_jsonl
from src.common.schema import FINAL_COLUMNS, DatasetRecord
from src.common.source_utils import PLATFORM_ORDER, platform_from_source, platform_sort_key


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
    """Od interim cleaned podataka napravi finalni CSV za anotaciju."""
    interim_path = resolve_path(config["paths"]["interim_dir"]) / "cleaned.jsonl"
    processed_dir = ensure_dir(resolve_path(config["paths"]["processed_dir"]))
    annotation_path = resolve_path(config["paths"]["annotation_csv"])
    dataset_path = resolve_path(config["paths"]["dataset_csv"])

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
    save_jsonl(final_rows, processed_dir / "dataset.jsonl")

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
