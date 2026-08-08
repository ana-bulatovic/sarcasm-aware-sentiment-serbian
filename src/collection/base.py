"""Bazna klasa za kolektore podataka i kratki fingerprint teksta."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.common.config import ensure_dir, resolve_path
from src.common.io_utils import load_jsonl, save_jsonl
from src.common.schema import RawRecord


def text_fingerprint(text: str) -> str:
    """Stabilan kratki hash za deduplikaciju (bez čuvanja PII).

    Vraća prvih 16 hex znakova SHA-256 normalizovanog teksta.
    Za pun SHA u preprocessing pipeline-u vidi preprocessing.deduplicate.
    """
    normalized = " ".join((text or "").split()).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


class BaseCollector(ABC):
    """Apstraktni kolektor: prikuplja RawRecord zapise i čuva raw JSONL.

    Podklase postavljaju source_name; raw ide u paths.raw_dir/<source_name>/raw.jsonl.
    Polje source u RawRecord može biti pun URL ili alias platforme, zavisno od kolektora.
    """

    source_name: str = "base"

    def __init__(self, config: dict[str, Any]):
        """Inicijalizuj putanje raw_dir/raw_path iz config.paths.raw_dir."""
        self.config = config
        self.source_cfg = config.get("collection", {}).get(self.source_name, {})
        raw_root = resolve_path(config["paths"]["raw_dir"])
        self.raw_dir = ensure_dir(raw_root / self.source_name)
        self.raw_path = self.raw_dir / "raw.jsonl"

    @abstractmethod
    def collect(self, max_records: int) -> list[RawRecord]:
        """Prikupi do max_records sirovih zapisa (bez PII)."""

    def save_raw(self, records: list[RawRecord]) -> Path:
        """Upisi zapise u self.raw_path (JSONL) i vrati tu putanju."""
        payload = [r.to_dict() for r in records]
        save_jsonl(payload, self.raw_path)
        return self.raw_path

    def load_raw(self) -> list[dict[str, Any]]:
        """Učitaj postojeći raw.jsonl kao listu dict-ova."""
        return load_jsonl(self.raw_path)

    def collect_and_save(self, max_records: int) -> list[RawRecord]:
        """collect() + dedup po text_fingerprint, pa save_raw; max max_records."""
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
