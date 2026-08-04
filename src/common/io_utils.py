"""Pomoćne funkcije za čitanje/pisanje JSONL i CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.common.schema import FINAL_COLUMNS


def save_jsonl(records: Iterable[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def save_csv(records: Iterable[dict[str, Any]], path: Path, columns: list[str] | None = None) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(records)
    cols = columns or (list(rows[0].keys()) if rows else FINAL_COLUMNS)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in cols})
    return len(rows)


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")


def append_unique_texts(existing: set[str], texts: Iterable[str]) -> set[str]:
    """Dodaj tekstove u skup (za deduplikaciju tokom kolekcije)."""
    for t in texts:
        existing.add(t)
    return existing
