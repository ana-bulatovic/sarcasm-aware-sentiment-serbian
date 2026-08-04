"""Bazna klasa za kolektore podataka."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.common.config import ensure_dir, resolve_path
from src.common.io_utils import load_jsonl, save_jsonl
from src.common.schema import RawRecord


def text_fingerprint(text: str) -> str:
    """Stabilan hash za deduplikaciju (bez čuvanja PII)."""
    normalized = " ".join((text or "").split()).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


class BaseCollector(ABC):
    """Apstraktni kolektor: prikuplja RawRecord zapise i čuva raw JSONL."""

    source_name: str = "base"

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.source_cfg = config.get("collection", {}).get(self.source_name, {})
        raw_root = resolve_path(config["paths"]["raw_dir"])
        self.raw_dir = ensure_dir(raw_root / self.source_name)
        self.raw_path = self.raw_dir / "raw.jsonl"

    @abstractmethod
    def collect(self, max_records: int) -> list[RawRecord]:
        """Prikupi do max_records sirovih zapisa (bez PII)."""

    def save_raw(self, records: list[RawRecord]) -> Path:
        payload = [r.to_dict() for r in records]
        save_jsonl(payload, self.raw_path)
        return self.raw_path

    def load_raw(self) -> list[dict[str, Any]]:
        return load_jsonl(self.raw_path)

    def collect_and_save(self, max_records: int) -> list[RawRecord]:
        records = self.collect(max_records=max_records)
        seen: set[str] = set()
        unique: list[RawRecord] = []
        for rec in records:
            text = (rec.text or "").strip()
            if not text:
                continue
            fp = text_fingerprint(text)
            if fp in seen:
                continue
            seen.add(fp)
            unique.append(rec)
            if len(unique) >= max_records:
                break
        self.save_raw(unique)
        return unique
